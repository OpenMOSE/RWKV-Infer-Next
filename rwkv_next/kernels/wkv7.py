# -*- coding: utf-8 -*-
"""RWKV-7 WKV recurrence (forward only), Triton. Replaces rwkvfla's fused_recurrent_dplr_delta_rule.

Per head the state S (K x V, fp32) evolves as

    S_t = diag(d_t) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T          d_t = exp(-exp(w_t))
    o_t = S_t^T r_t

with the RWKV-7 operands a = -kk, b = kk * a_gate. One program owns one (sequence, head, BV-column block)
and keeps its K x BV slice of the state in registers for the whole sequence.

Differences from the rwkvfla kernel it replaces (same math):
  * the decay is taken RAW (w, the log-log decay) and exp(-exp(w)) is computed in fp32 in the kernel; the old
    path computed gk = -exp(w) outside and rounded it to bf16 before the kernel exponentiated it again
  * N_KEEP (rollback of a speculative verify): row b runs only its first n[b] steps, so dropped draft steps
    need no masking pass and no stacked copies; the state can be written back in place (h0 == ht)
  * STORE_O / STORE_HT switch the output and the final state off (rollback needs no o, verify no state)
  * varlen (cu_seqlens) packed prefill with a per-sequence initial state, as before

Layouts: r, w, k, v, a, b (B, T, H*K) contiguous (bf16); h0 / ht (N, H, K, V) fp32; o like v.
Portable: standard tl.* only.

Adapted from flash-linear-attention / rwkvfla (MIT, Copyright (c) 2023-2025 Songlin Yang, Yu Zhang).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _configs():
    return [triton.Config({"BV": bv}, num_warps=w, num_stages=s)
            for bv in (16, 32, 64) for w in (1, 2, 4) for s in (1, 2)]


@triton.jit(do_not_specialize=["T"])
def _wkv7_raw(r, w, k, v, a, b, o, h0, ht, cu, nkeep, T,
              H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
              IS_VARLEN: tl.constexpr, HAS_H0: tl.constexpr, STORE_HT: tl.constexpr, STORE_O: tl.constexpr,
              HAS_N: tl.constexpr):
    i_v = tl.program_id(0).to(tl.int64)
    i_nh = tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu + i_n).to(tl.int64)
        eos = tl.load(cu + i_n + 1).to(tl.int64)
        L = eos - bos
    else:
        bos = i_n * T
        L = T
    if HAS_N:
        L = tl.minimum(L, tl.load(nkeep + i_n).to(tl.int64))
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mk = o_k < K
    mv = o_v < V
    mh = mk[:, None] & mv[None, :]
    base_k = bos * H * K + i_h * K + o_k
    base_v = bos * H * V + i_h * V + o_v
    S = tl.zeros([BK, BV], dtype=tl.float32)
    if HAS_H0:
        S += tl.load(h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :], mask=mh, other=0).to(tl.float32)
    for t in range(0, L):
        ok = base_k + t * H * K
        ov = base_v + t * H * V
        rr = tl.load(r + ok, mask=mk, other=0).to(tl.float32)
        ww = tl.load(w + ok, mask=mk, other=0).to(tl.float32)
        kk = tl.load(k + ok, mask=mk, other=0).to(tl.float32)
        aa = tl.load(a + ok, mask=mk, other=0).to(tl.float32)
        bb = tl.load(b + ok, mask=mk, other=0).to(tl.float32)
        vv = tl.load(v + ov, mask=mv, other=0).to(tl.float32)
        d = tl.exp(-tl.exp(ww))
        S = d[:, None] * S + bb[:, None] * tl.sum(aa[:, None] * S, 0)[None, :] + kk[:, None] * vv[None, :]
        if STORE_O:
            tl.store(o + ov, tl.sum(S * rr[:, None], 0).to(o.dtype.element_ty), mask=mv)
    if STORE_HT:
        tl.store(ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :], S.to(ht.dtype.element_ty), mask=mh)


# Autotuned entry for the normal calls. The in-place replay (h0 IS ht) must NOT go through the autotuner: it
# benchmarks every config on the real arguments, i.e. it would advance the state once per candidate. It uses a
# fixed config instead (it is a tiny call: T <= K_max+1 steps).
_wkv7_fwd = triton.autotune(configs=_configs(), key=["K", "V", "STORE_O", "STORE_HT", "HAS_N", "IS_VARLEN"])(_wkv7_raw)
REPLAY_BV, REPLAY_WARPS = 32, 2


def _launch(r, w, k, v, a, b, h0, ht, cu, nkeep, store_o, H, K, launch_wrap=None):
    B, T, _ = r.shape
    V = K
    N = B if cu is None else cu.numel() - 1
    o = torch.empty_like(v) if store_o else v                    # o unused (never written) when not stored
    args = (r, w, k, v, a, b, o, h0 if h0 is not None else r, ht if ht is not None else r,
            cu if cu is not None else r, nkeep if nkeep is not None else r, T)
    kw = dict(H=H, K=K, V=V, BK=triton.next_power_of_2(K), IS_VARLEN=cu is not None, HAS_H0=h0 is not None,
              STORE_HT=ht is not None, STORE_O=store_o, HAS_N=nkeep is not None)
    if h0 is not None and h0 is ht:                               # in place: fixed config, no autotune
        kern = launch_wrap(_wkv7_raw) if launch_wrap is not None else _wkv7_raw
        kern[(triton.cdiv(V, REPLAY_BV), N * H)](*args, BV=REPLAY_BV, num_warps=REPLAY_WARPS, **kw)
    else:
        kern = launch_wrap(_wkv7_fwd) if launch_wrap is not None else _wkv7_fwd
        kern[lambda meta: (triton.cdiv(V, meta["BV"]), N * H)](*args, **kw)
    return o


try:
    from torch.library import triton_op, wrap_triton
    _HAS_TRITON_OP = True
except ImportError:                                               # pragma: no cover
    _HAS_TRITON_OP = False

if _HAS_TRITON_OP:
    @triton_op("rwkv_next::wkv7_fwd", mutates_args={})
    def _wkv7_op(r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                 h0: torch.Tensor, H: int, K: int, final: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Functional form for torch.compile (o and the final state are fresh outputs). h0 may be empty (0 rows)."""
        B = r.shape[0]
        ht = torch.empty(B, H, K, K, device=r.device, dtype=torch.float32) if final else r.new_empty(0)
        with torch.cuda.device(r.device):
            o = _launch(r, w, k, v, a, b, h0 if h0.numel() else None, ht if final else None, None, None, True, H, K,
                        launch_wrap=wrap_triton)
        return o, ht


def wkv7(r, w, k, v, a, b, state, H, K, cu_seqlens=None, final=True):
    """o, final_state = recurrence over (B, T, H*K) inputs from `state` (B|N, H, K, K) fp32 or None.
    w is RAW (decay = exp(-exp(w))). cu_seqlens (N+1,) int32: packed varlen (B must be 1)."""
    r, w, k, v, a, b = (t.contiguous() for t in (r, w, k, v, a, b))
    if cu_seqlens is None and _HAS_TRITON_OP:
        h0 = state.contiguous() if state is not None else r.new_empty(0, dtype=torch.float32)
        o, ht = _wkv7_op(r, w, k, v, a, b, h0, H, K, final)
        return o, (ht if final else None)
    N = r.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
    ht = torch.empty(N, H, K, K, device=r.device, dtype=torch.float32) if final else None
    with torch.cuda.device(r.device):
        o = _launch(r, w, k, v, a, b, state.contiguous() if state is not None else None, ht, cu_seqlens, None, True, H, K)
    return o, ht


def wkv7_replay(r, w, k, v, a, b, state, n_keep, H, K):
    """Rollback: state (B, H, K, K) fp32 := the recurrence over the first n_keep[b] steps of row b, IN PLACE.
    No output is computed."""
    r, w, k, v, a, b = (t.contiguous() for t in (r, w, k, v, a, b))
    with torch.cuda.device(r.device):
        _launch(r, w, k, v, a, b, state, state, None, n_keep.contiguous(), False, H, K)
    return state
