# -*- coding: utf-8 -*-
"""Portable 6-bit weight-only GEMM/GEMV in Triton (W6A16).

Why 6 bits
----------
Decode is weight-bandwidth bound (see int8_gemm.py), so the step time follows
the weight bytes. int8 with group-128 fp32 scales costs 1.03 bytes per weight;
this format costs 0.766 (6 bits + one fp16 scale per 128 weights), i.e. 26 %
fewer bytes and VRAM: a 14B model drops from ~14.4 GB to ~10.7 GB, which is the
difference between fitting a 16 GB card or not.

Format: bit planes, not packed 6-bit fields
-------------------------------------------
6-bit fields straddle byte boundaries, which makes loads unaligned and the
unpacking branchy. Instead each weight is split into its low 4 bits and its
high 2 bits, stored in two separate planes (the same idea as llama.cpp's Q6_K
ql/qh and FP6-LLM's 4+2 bit split). Both planes keep the (K, N) = (in, out)
layout of int8_gemm, with N contiguous, so every load is a plain coalesced
uint8 tile.

The trick that keeps the Triton kernel simple is WHICH rows share a byte. K is
cut into chunks of 64 rows. Inside a chunk

    lo plane  L[c*32 + j, n]  = lo4(row j)      | lo4(row j + 32) << 4      j < 32
    hi plane  H[c*16 + j, n]  = hi2(row j)      | hi2(row j + 16) << 2
                              | hi2(row j + 32) << 4 | hi2(row j + 48) << 6    j < 16

so a chunk unpacks into four 16-row sub-tiles [0,16) [16,32) [32,48) [48,64)
with nothing but shifts and masks, and each sub-tile is contracted with its own
16-column slice of the activation. No interleave/reshape of register tiles is
ever needed (Triton has no cheap one), and 16 rows is exactly tl.dot's K tile.

Quantisation: symmetric int6 in [-32, 31] per (group of GROUP_K rows, column),
w ~= s * q, fp16 scale. The scale is not simply absmax/31: a few clip ratios are
tried per group and the one with the least squared error wins (what llama.cpp's
make_qx_quants does), which buys ~10 % lower weight error at 6 bits for free at
load time.

Kernel
------
One kernel, two contraction forms (the same split as int8_gemm):
  * USE_DOT=False  broadcast-and-reduce, for M <= RED_MAX_M (= 1, decode). Wide N tiles so
                   each row load is whole cache lines; split-K over chunks to
                   fill the device when N is narrow.
  * USE_DOT=True   tl.dot on the four 16-row sub-tiles, from M = RED_MAX_M + 1 (M-tiled above 64)
                   (batched decode, MTP verify).
Larger M (prefill) uses the dot form with 64-row M tiles; it beats int8 at
every M and stays within ~1.25x of the bf16 GEMM (numbers at MAX_M below).
Partial sums over K slices go to a fp32 (SPLIT_K, B, M, N) buffer, reduced
afterwards (deterministic, no atomics).

Only standard tl.* ops: nothing here is vendor specific.
"""

import threading

import torch
import triton
import triton.language as tl

CHUNK = 64                # rows per packing chunk (4 sub-tiles of 16)
QMIN, QMAX = -32, 31


# --------------------------------------------------------------------------- #
# quantisation / packing
# --------------------------------------------------------------------------- #
@torch.no_grad()
def quantize_int6(w, group_k=128, clip_search=True):
    """(K, N) or (B, K, N) float -> (lo uint8 (.., K/2, N), hi uint8 (.., K/4, N),
    scales fp16 (.., K/group_k, N), group_k).

    Runs on w's device (do it on the GPU: the clip search is a few elementwise
    passes). group_k must be a multiple of 64 and divide K.
    """
    squeeze = (w.dim() == 2)
    if squeeze:
        w = w.unsqueeze(0)
    B, K, N = w.shape
    g = group_k if group_k and 0 < group_k < K else K
    if K % CHUNK or g % CHUNK or K % g:
        raise ValueError(f"int6 needs K % 64 == 0 and group_k % 64 == 0 dividing K (K={K}, group_k={g})")
    G = K // g
    lo = torch.empty((B, K // 2, N), dtype=torch.uint8, device=w.device)
    hi = torch.empty((B, K // 4, N), dtype=torch.uint8, device=w.device)
    sc = torch.empty((B, G, N), dtype=torch.float16, device=w.device)
    step = max(256, (1 << 23) // K)                      # column chunks: bounded fp32 temporaries
    for b, n0 in ((b, n0) for b in range(B) for n0 in range(0, N, step)):
        n1 = min(N, n0 + step); Nc = n1 - n0
        wf = w[b, :, n0:n1].float().reshape(G, g, Nc)
        amax = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)       # (G, 1, N)
        ratios = (1.0, 0.97, 0.94, 0.91, 0.88, 0.85) if clip_search else (1.0,)
        best_s, best_e = None, None
        for r in ratios:
            s = (amax * r / 31.5).half().float()          # the stored (fp16) scale is what the kernel sees
            q = torch.round(wf / s).clamp(QMIN, QMAX)
            e = ((q * s - wf) ** 2).sum(dim=1, keepdim=True)
            if best_s is None:
                best_s, best_e = s, e
            else:
                m = e < best_e
                best_s = torch.where(m, s, best_s); best_e = torch.where(m, e, best_e)
        q = (torch.round(wf / best_s).clamp(QMIN, QMAX) + 32).to(torch.uint8).reshape(K // CHUNK, CHUNK, Nc)
        sc[b, :, n0:n1] = best_s.reshape(G, Nc).half()
        l4, h2 = q & 15, q >> 4                                        # (C, 64, Nc)
        lo[b, :, n0:n1] = (l4[:, :32] | (l4[:, 32:] << 4)).reshape(K // 2, Nc)
        hi[b, :, n0:n1] = (h2[:, 0:16] | (h2[:, 16:32] << 2) | (h2[:, 32:48] << 4) | (h2[:, 48:64] << 6)).reshape(K // 4, Nc)
        del wf, q, l4, h2
    if squeeze:
        lo, hi, sc = lo[0], hi[0], sc[0]
    return lo.contiguous(), hi.contiguous(), sc.contiguous(), g


@torch.no_grad()
def dequantize_int6(lo, hi, sc, group_k, dtype=torch.float32):
    """Inverse of quantize_int6 (reference / prefill path)."""
    squeeze = (lo.dim() == 2)
    if squeeze:
        lo, hi, sc = lo[None], hi[None], sc[None]
    B, K2, N = lo.shape
    K = K2 * 2
    C = K // CHUNK
    L = lo.reshape(B, C, 32, N).to(torch.int16)
    H = hi.reshape(B, C, 16, N).to(torch.int16)
    l4 = torch.cat([L & 15, L >> 4], dim=2)                                       # rows 0..63
    h2 = torch.cat([H & 3, (H >> 2) & 3, (H >> 4) & 3, (H >> 6) & 3], dim=2)
    q = ((l4 | (h2 << 4)) - 32).reshape(B, K // group_k, group_k, N).to(dtype)
    w = (q * sc.to(dtype).unsqueeze(2)).reshape(B, K, N)
    return w[0] if squeeze else w


# --------------------------------------------------------------------------- #
# kernel
# --------------------------------------------------------------------------- #
def _configs():
    out = []
    for bn, w, st in ((128, 4, 2), (256, 4, 2), (256, 8, 2), (512, 8, 2),
                      (128, 4, 3), (256, 4, 3), (256, 8, 3), (512, 8, 3), (64, 4, 3)):
        out.append(triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=st))
    return out


_RED_MAX_ELEMS = 16384     # live (BLOCK_M, 16, BLOCK_N) product of the reduction form


def _prune(configs, named_args, **kwargs):
    use_dot = kwargs.get("USE_DOT", named_args.get("USE_DOT"))
    bm = kwargs.get("BLOCK_M") or named_args.get("BLOCK_M") or 1
    if use_dot:
        keep = [c for c in configs if c.kwargs["BLOCK_N"] <= 256]      # dot accumulators: BM x BN fp32
    else:
        keep = [c for c in configs if bm * 16 * c.kwargs["BLOCK_N"] <= _RED_MAX_ELEMS]
    return keep or [min(configs, key=lambda c: c.kwargs["BLOCK_N"])]


@triton.autotune(configs=_configs(), key=["M_KEY", "N", "K", "GROUP_K", "SPLIT_K", "USE_DOT"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _w6a16(
    A, L, H, S, P,
    M, N, K, M_KEY, NUM_M,
    sa_b, sa_m, sa_k,
    sl_b, sl_k, sl_n,
    sh_b, sh_k, sh_n,
    ss_b, ss_g, ss_n,
    sp_s, sp_b, sp_m, sp_n,
    GROUP_K: tl.constexpr, SPLIT_K: tl.constexpr, USE_DOT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """P[s, b] = A[b, :, chunks s, s+SPLIT_K, ..] @ dequant(W[b]) (fp32, or the output dtype when SPLIT_K == 1)."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_bm = tl.program_id(2)                 # batch x M tiles (M tiling only matters for prefill-sized M)
    pid_b = pid_bm // NUM_M
    pid_m = pid_bm % NUM_M
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, 16)
    m_mask = offs_m < M
    n_mask = offs_n < N
    a_base = A + pid_b * sa_b + offs_m[:, None] * sa_m + offs_r[None, :] * sa_k
    l_base = L + pid_b * sl_b + offs_r[:, None] * sl_k + offs_n[None, :] * sl_n
    h_base = H + pid_b * sh_b + offs_r[:, None] * sh_k + offs_n[None, :] * sh_n
    s_base = S + pid_b * ss_b + offs_n * ss_n
    am = m_mask[:, None]
    wm = n_mask[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for c in range(pid_s, K // 64, SPLIT_K):
        la = tl.load(l_base + (c * 32) * sl_k, mask=wm, other=0).to(tl.int32)
        lb = tl.load(l_base + (c * 32 + 16) * sl_k, mask=wm, other=0).to(tl.int32)
        hh = tl.load(h_base + (c * 16) * sh_k, mask=wm, other=0).to(tl.int32)
        w0 = ((la & 15) | ((hh & 3) << 4)) - 32
        w1 = ((lb & 15) | ((hh & 12) << 2)) - 32
        w2 = ((la >> 4) | (hh & 48)) - 32
        w3 = ((lb >> 4) | ((hh >> 2) & 48)) - 32
        ap = a_base + (c * 64) * sa_k
        a0 = tl.load(ap, mask=am, other=0.0)
        a1 = tl.load(ap + 16 * sa_k, mask=am, other=0.0)
        a2 = tl.load(ap + 32 * sa_k, mask=am, other=0.0)
        a3 = tl.load(ap + 48 * sa_k, mask=am, other=0.0)
        if USE_DOT:
            # int6 -> bf16 is exact; the products accumulate in fp32
            p = tl.dot(a0, w0.to(a0.dtype), out_dtype=tl.float32)
            p = tl.dot(a1, w1.to(a1.dtype), p, out_dtype=tl.float32)
            p = tl.dot(a2, w2.to(a2.dtype), p, out_dtype=tl.float32)
            p = tl.dot(a3, w3.to(a3.dtype), p, out_dtype=tl.float32)
        else:
            p = tl.sum(a0.to(tl.float32)[:, :, None] * w0.to(tl.float32)[None, :, :], axis=1)
            p += tl.sum(a1.to(tl.float32)[:, :, None] * w1.to(tl.float32)[None, :, :], axis=1)
            p += tl.sum(a2.to(tl.float32)[:, :, None] * w2.to(tl.float32)[None, :, :], axis=1)
            p += tl.sum(a3.to(tl.float32)[:, :, None] * w3.to(tl.float32)[None, :, :], axis=1)
        s = tl.load(s_base + ((c * 64) // GROUP_K) * ss_g, mask=n_mask, other=0.0).to(tl.float32)
        acc += p * s[None, :]
    p_ptr = P + pid_s * sp_s + pid_b * sp_b + offs_m[:, None] * sp_m + offs_n[None, :] * sp_n
    tl.store(p_ptr, acc.to(P.dtype.element_ty), mask=am & wm)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
RED_MAX_M = 1             # reduction form only at M=1: measured, the dot form wins from M=2 (cold, graphed)
DOT_MIN_M = 16
TILE_M = 64               # M tile of the dot form above this
# Prefill goes through the same M-tiled kernel. Dequantise-then-GEMM was measured and loses at
# every M (K=4096 N=16384: M=128 0.24 vs 3.56 ms, M=4096 5.9 vs 8.1 ms; int8 10.2, bf16 4.7), so the
# fallback is only reachable by setting RWKV_INT6_MAX_M.
MAX_M = int(__import__("os").environ.get("RWKV_INT6_MAX_M", str(1 << 30)))
_LAUNCH_LOCK = threading.Lock()
_SM = {}


def _sm_count(dev):
    i = dev.index if dev.index is not None else torch.cuda.current_device()
    if i not in _SM:
        _SM[i] = torch.cuda.get_device_properties(i).multi_processor_count
    return _SM[i]


def plan_int6(M, N, K, n_batch, device=None):
    """(split_k, use_dot) for one shape. Split-K is sized so the launch has a few
    programs per SM at BLOCK_N = 256 (narrow N would otherwise starve the device;
    see int8_gemm.choose_split_k for why this is not timed)."""
    use_dot = M > RED_MAX_M
    sm = _sm_count(device if device is not None else torch.device("cuda"))
    blocks = n_batch * -(-N // 256) * (-(-M // TILE_M) if M > TILE_M else 1)
    chunks = K // CHUNK
    sk = 1
    while blocks * sk < 3 * sm and sk * 2 <= 32 and chunks // (sk * 2) >= 2:
        sk *= 2
    return sk, use_dot


def _impl(a, lo, hi, s, group_k, split_k, use_dot, launch_wrap=None):
    batched = (a.dim() == 3)
    A = a if batched else a.unsqueeze(0)
    Lq = lo if lo.dim() == 3 else lo.unsqueeze(0)
    Hq = hi if hi.dim() == 3 else hi.unsqueeze(0)
    S = s if s.dim() == 3 else s.unsqueeze(0)
    B, M, K = A.shape
    Bq, _, N = Lq.shape
    if Lq.shape[1] * 2 != K:
        raise ValueError(f"K mismatch: a has {K}, int6 weight has {Lq.shape[1] * 2}")
    if A.stride(2) != 1:
        A = A.contiguous()
    if M > MAX_M:                                        # prefill: compute bound, dequantise once
        w = dequantize_int6(Lq, Hq, S, group_k, dtype=a.dtype)
        C = torch.bmm(A, w.expand(B, -1, -1)) if (Bq == 1 and B > 1) else torch.bmm(A, w)
        return C if batched else C[0]
    m_key = min(triton.next_power_of_2(M), 2 * TILE_M)
    block_m = (max(DOT_MIN_M, m_key) if M <= TILE_M else TILE_M) if use_dot else m_key
    num_m = triton.cdiv(M, block_m)
    sb = (lambda t: t.stride(0) if Bq == B else 0)
    if split_k > 1:
        P = torch.empty((split_k, B, M, N), device=a.device, dtype=torch.float32)
    else:
        P = torch.empty((1, B, M, N), device=a.device, dtype=a.dtype)
    kern = launch_wrap(_w6a16) if launch_wrap is not None else _w6a16
    kern[lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), split_k, B * num_m)](
        A, Lq, Hq, S, P, M, N, K, m_key, num_m,
        A.stride(0), A.stride(1), A.stride(2),
        sb(Lq), Lq.stride(1), Lq.stride(2),
        sb(Hq), Hq.stride(1), Hq.stride(2),
        sb(S), S.stride(1), S.stride(2),
        P.stride(0), P.stride(1), P.stride(2), P.stride(3),
        GROUP_K=group_k, SPLIT_K=split_k, USE_DOT=use_dot, BLOCK_M=block_m)
    C = P.sum(0).to(a.dtype) if split_k > 1 else P[0]
    return C if batched else C[0]


def _autotune_load():
    from .int8_gemm import _autotune_load as f
    f()


def int6_matmul(a, lo, hi, s, group_k, split_k=None, use_dot=None):
    """a (M, K) | (B, M, K) @ dequant(int6 weight (K, N) | (B, K, N)) -> a.dtype.
    Eager entry point (tests, benches); the model uses int6_matmul_op."""
    _autotune_load()
    B = a.shape[0] if a.dim() == 3 else 1
    M, K, N = a.shape[-2], a.shape[-1], lo.shape[-1]
    if split_k is None or use_dot is None:
        sk, ud = plan_int6(M, N, K, B, a.device)
        split_k = sk if split_k is None else split_k
        use_dot = ud if use_dot is None else use_dot
    with _LAUNCH_LOCK, torch.cuda.device(a.device):
        return _impl(a, lo, hi, s, group_k, split_k, use_dot)


try:
    from torch.library import triton_op, wrap_triton
    _HAS_TRITON_OP = True
except ImportError:                                   # pragma: no cover
    _HAS_TRITON_OP = False

if _HAS_TRITON_OP:
    @triton_op("rwkv_next::int6_matmul", mutates_args={})
    def int6_matmul_op(a: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, s: torch.Tensor,
                       group_k: int, split_k: int, use_dot: bool) -> torch.Tensor:
        """Functional op for torch.compile / CUDA graphs (see int8_gemm.int8_matmul_op).
        The plan (split_k, use_dot) comes from plan_int6, resolved by the caller."""
        _autotune_load()
        with _LAUNCH_LOCK, torch.cuda.device(a.device):
            return _impl(a, lo, hi, s, group_k, split_k, use_dot, launch_wrap=wrap_triton)
else:                                                 # pragma: no cover
    def int6_matmul_op(a, lo, hi, s, group_k, split_k, use_dot):
        return int6_matmul(a, lo, hi, s, group_k, split_k, use_dot)
