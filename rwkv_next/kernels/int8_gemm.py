# -*- coding: utf-8 -*-
"""Portable int8 weight-only GEMM/GEMV in Triton (W8A16).

Why weight-only, and not int8 activations too
---------------------------------------------
Decode in this model is memory bound over the whole batch range we care about:
one step costs essentially the same at B=1 and B=32, because the time goes into
streaming the weights, not into arithmetic. So the thing worth halving is the
WEIGHT bytes. Quantising activations as well would buy almost no bandwidth (the
activation is M x K with M <= 64, versus a K x N weight) while adding a second
source of error and a per-token scale pass. Activations therefore stay bf16, the
product accumulates in fp32, and only the weight is int8.

That gives ~2x on both counts this is meant to fix:
  * VRAM         30L 1.84B: 3.68 GB -> ~1.9 GB
  * bandwidth    the decode step is weight-bandwidth bound, so ~2x there too

Measured behaviour (W7900, gfx1100, group_k=128, COLD cache, GPU time only,
summed over the quantised matmuls of one 30-layer decode step):

    batch    bf16 ms   int8 ms   speedup   int8 tok/s   scaling vs batch 1
        1       5.62      3.94     1.43x          254                 1.0x
        2       5.63      4.45     1.26x          449                 1.8x
        4       5.63      4.89     1.15x          818                 3.2x
        8       5.83      5.06     1.15x         1581                 6.2x
       16       5.92      5.23     1.13x         3058                12.0x
       32       5.99      7.14     0.84x         4482                17.7x

int8 wins from batch 1 to 16 and the step time stays nearly flat there, so
aggregate throughput scales ~12-14x over that range. At 32 it loses: the kernel
stops being bandwidth bound, and rocBLAS's bf16 GEMM is better at that point than
dequantise-then-dot. Run-to-run spread is ~5%, so treat differences below that as
noise (the bf16 baseline itself moved 5.62 -> 5.92 between runs).

Quantisation
------------
Symmetric, no zero point (RWKV weights are centred), grouped along K:

    q[k, n] = round(w[k, n] / s[k // GROUP_K, n])        int8, [-127, 127]
    s[g, n] = max |w[g*GROUP_K : (g+1)*GROUP_K, n]| / 127

`GROUP_K = 0` means one group per output channel (per-channel), which is the
cheapest and is exact enough for most matrices here; smaller groups trade a
little scale traffic for accuracy on the ones that need it. The kernel handles
both through the same code path, with the per-channel case hoisting the scale
out of the K loop.

The weight is stored (K, N) — the same (in, out) layout `pack_for_inference`
already uses — so `n` is the contiguous axis and a block load of the weight tile
is fully coalesced.

Portability
-----------
Only standard `tl.*` operations: no inline asm, no MFMA/WMMA intrinsics, no
architecture-specific hints. Tuned on RDNA3 (gfx1100) but nothing here is AMD
specific.
"""

import threading

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# quantisation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def quantize_int8(w, group_k=0):
    """(K, N) or (B, K, N) float -> (q int8, scales float32).

    Scales come back as (G, N) / (B, G, N) with G = ceil(K / GROUP_K), or G = 1
    for per-channel. Returns `group_k` normalised to the value the kernel needs
    (K when per-channel), so the caller can store it alongside the tensors.
    """
    squeeze = (w.dim() == 2)
    if squeeze:
        w = w.unsqueeze(0)
    B, K, N = w.shape
    if group_k and group_k > 0 and group_k < K:
        if K % group_k:
            raise ValueError(f"group_k={group_k} must divide K={K}")
        g = group_k
    else:
        g = K                                  # per output channel
    G = K // g

    # column chunks: the fp32 temporaries of a whole 65536-wide head are ~4 GB, which is
    # what a model packed next to a nearly full GPU cannot afford (columns are independent)
    q = torch.empty((B, K, N), dtype=torch.int8, device=w.device)
    scales = torch.empty((B, G, N), dtype=torch.float32, device=w.device)
    step = max(256, (1 << 24) // K)
    for b in range(B):
        for n0 in range(0, N, step):
            wf = w[b, :, n0:n0 + step].float().reshape(G, g, -1)
            sc = (wf.abs().amax(dim=1) / 127.0).clamp(min=1e-12)          # (G, n)
            q[b, :, n0:n0 + step] = torch.round(wf / sc.unsqueeze(1)).clamp(-127, 127).to(torch.int8).reshape(K, -1)
            scales[b, :, n0:n0 + step] = sc
            del wf, sc
    if squeeze:
        q, scales = q.squeeze(0), scales.squeeze(0)
    return q.contiguous(), scales.contiguous(), g


@torch.no_grad()
def dequantize_int8(q, scales, group_k):
    """Inverse of quantize_int8, for error checking."""
    squeeze = (q.dim() == 2)
    if squeeze:
        q, scales = q.unsqueeze(0), scales.unsqueeze(0)
    B, K, N = q.shape
    G = K // group_k
    out = (q.float().reshape(B, G, group_k, N)
           * scales.unsqueeze(2)).reshape(B, K, N)
    return out.squeeze(0) if squeeze else out


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #
# SPLIT_K is deliberately absent from the tuned configs. autotune chooses at call
# time, and a split config must accumulate with float32 atomics into a zeroed
# buffer while a non-split one stores straight into the bf16 output. Allowing
# both in one config set would force the fp32 buffer plus a cast pass on EVERY
# call, which costs more than split-K can win at these shapes (N >= 2048 already
# gives 32+ blocks per batch element). If a shape ever turns out to be
# occupancy-starved, split-K belongs in its own entry point, not here.
# The config lists are deliberately short. This kernel is weight-bandwidth bound
# at every shape used here, so the tile choice matters far less than it would for
# a compute-bound GEMM — and autotune re-tunes per (M bucket, N, K, GROUP_K), so
# a wide list turns startup into tens of minutes. Widen only for a shape that
# actually measures badly.
def _gemm_configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                      num_warps=w, num_stages=st)
        for bm, bn, bk, w, st in (
            (16, 128, 64, 4, 2),
            (16, 256, 64, 8, 2),
            (16, 64, 128, 4, 2),
            (32, 128, 64, 4, 2),
            (32, 256, 64, 8, 2),
            (64, 128, 64, 8, 2),
            (64, 64, 64, 4, 4),
            # deeper pipelining, so the int8->bf16 tile conversion overlaps the
            # next load instead of stalling the dot
            (16, 128, 128, 4, 3),
            (16, 256, 64, 8, 3),
            (32, 128, 128, 8, 3),
            (32, 256, 64, 8, 4),
            (64, 128, 128, 8, 3),
            (64, 256, 64, 8, 4),
        )
    ]


@triton.autotune(configs=_gemm_configs(), key=["M_KEY", "N", "K", "GROUP_K"])
@triton.jit
def _w8a16_gemm(
    A, Q, S, C,
    M, N, K, M_KEY,
    sa_b, sa_m, sa_k,
    sq_b, sq_k, sq_n,
    ss_b, ss_g, ss_n,
    sc_b, sc_m, sc_n,
    GROUP_K: tl.constexpr, PER_CHANNEL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C[b] = A[b] @ dequant(Q[b], S[b]).  tl.dot path, for M >= ~16."""
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    a_ptr = A + pid_b * sa_b + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
    q_ptr = Q + pid_b * sq_b + offs_k[:, None] * sq_k + offs_n[None, :] * sq_n
    s_row = S + pid_b * ss_b + offs_n * ss_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k0 = 0
    step = BLOCK_K

    while k0 < K:
        k_mask = (k0 + offs_k) < K
        a = tl.load(a_ptr, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        q = tl.load(q_ptr, mask=k_mask[:, None] & n_mask[None, :], other=0)
        # int8 -> bf16 is exact for |q| <= 127 (bf16 holds integers to 256), so
        # the dot is no less accurate than the bf16 GEMM it replaces.
        p = tl.dot(a, q.to(a.dtype), out_dtype=tl.float32)
        if PER_CHANNEL:
            acc += p                      # one scale for the whole K: apply later
        else:
            s = tl.load(s_row + (k0 // GROUP_K) * ss_g, mask=n_mask, other=0.0)
            acc += p * s[None, :]
        a_ptr += step * sa_k
        q_ptr += step * sq_k
        k0 += step

    if PER_CHANNEL:
        acc *= tl.load(s_row, mask=n_mask, other=0.0)[None, :]

    c_ptr = C + pid_b * sc_b + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    tl.store(c_ptr, acc.to(C.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


def _gemv_configs():
    return [
        triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=2)
        for bn, bk, w in ((64, 64, 4), (128, 64, 4), (256, 64, 8),
                          (128, 32, 4), (64, 128, 4), (256, 32, 8))
    ]


# The reduction path materialises a (BLOCK_M, BLOCK_K, BLOCK_N) intermediate.
# That is fine at BLOCK_M=1 but grows linearly with M: at BLOCK_M=64 the widest
# config would need 64*128*256 = 2M values in registers and spills catastrophically
# (it does not fail, it just runs ~100x slow, which is worse). Cap the live set so
# a bad combination is never even compiled.
_GEMV_MAX_ELEMS = 32768


def _prune_gemv(configs, named_args, **kwargs):
    # Only the reduction form pays BLOCK_M * BLOCK_K * BLOCK_N; the tl.dot form
    # keeps just a BLOCK_M x BLOCK_N accumulator and needs no cap.
    if kwargs.get("USE_DOT") or named_args.get("USE_DOT"):
        return configs
    bm = kwargs.get("BLOCK_M") or named_args.get("BLOCK_M") or 1
    keep = [c for c in configs
            if bm * c.kwargs["BLOCK_K"] * c.kwargs["BLOCK_N"] <= _GEMV_MAX_ELEMS]
    return keep or [min(configs, key=lambda c: c.kwargs["BLOCK_K"] * c.kwargs["BLOCK_N"])]


@triton.autotune(configs=_gemv_configs(), key=["M_KEY", "N", "K", "GROUP_K"],
                 prune_configs_by={"early_config_prune": _prune_gemv})
@triton.jit
def _w8a16_gemv(
    A, Q, S, C,
    M, N, K, M_KEY,
    sa_b, sa_m, sa_k,
    sq_b, sq_k, sq_n,
    ss_b, ss_g, ss_n,
    sc_b, sc_m, sc_n,
    GROUP_K: tl.constexpr, PER_CHANNEL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Same result without tl.dot, for tiny M.

    `tl.dot` cannot take an M below the hardware tile (16), so at M=1 it would do
    16x redundant work on that axis. Here the contraction is an explicit
    broadcast-and-reduce over (BLOCK_M, BLOCK_K, BLOCK_N), which is only sane
    because BLOCK_M is 1-8. Whether this actually beats the padded dot depends on
    the shape, so both are benchmarked and `int8_matmul` picks per call.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    a_ptr = A + pid_b * sa_b + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
    q_ptr = Q + pid_b * sq_b + offs_k[:, None] * sq_k + offs_n[None, :] * sq_n
    s_row = S + pid_b * ss_b + offs_n * ss_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k0 = 0
    step = BLOCK_K

    while k0 < K:
        k_mask = (k0 + offs_k) < K
        a = tl.load(a_ptr, mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0).to(tl.float32)                  # (BM, BK)
        q = tl.load(q_ptr, mask=k_mask[:, None] & n_mask[None, :],
                    other=0).to(tl.float32)                    # (BK, BN)
        p = tl.sum(a[:, :, None] * q[None, :, :], axis=1)      # (BM, BN)
        if PER_CHANNEL:
            acc += p
        else:
            s = tl.load(s_row + (k0 // GROUP_K) * ss_g, mask=n_mask, other=0.0)
            acc += p * s[None, :]
        a_ptr += step * sa_k
        q_ptr += step * sq_k
        k0 += step

    if PER_CHANNEL:
        acc *= tl.load(s_row, mask=n_mask, other=0.0)[None, :]

    c_ptr = C + pid_b * sc_b + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    tl.store(c_ptr, acc.to(C.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


# --------------------------------------------------------------------------- #
# skinny M (decode): wide-N tiles + split-K
# --------------------------------------------------------------------------- #
# The reduction-form GEMV above streams (BLOCK_K, BLOCK_N) tiles whose rows are
# only BLOCK_N = 64-256 bytes long, and the (1, BLOCK_K, BLOCK_N) fp32 product
# it materialises caps the tile at 32K elements — so each program touches a few
# KB and the kernel never gets enough bytes in flight (measured 106-383 GB/s at
# M=1 on the model's shapes). This variant turns the tile sideways: BLOCK_N =
# 256-512 (whole cache lines per row) and BLOCK_K = 8-32 (tiny live product),
# and relies on split-K for program count. Measured at M=1 (W7900, int8 GB/s):
# att out 106 -> 249, ffn value 188 -> 378, ffn key 383 -> 469, head 545 -> 636.
def _wide_configs():
    # kept short: autotune re-runs per (M bucket, N, K, split) and the server
    # warms ~20 such keys at start-up
    return [
        triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=st)
        for bn, bk, w, st in ((256, 16, 4, 2), (256, 32, 4, 2), (512, 16, 8, 2), (512, 8, 8, 2),
                              (256, 16, 4, 4), (512, 16, 8, 4), (256, 32, 8, 4), (512, 32, 8, 4))
    ]


@triton.autotune(configs=_wide_configs(), key=["M_KEY", "N", "K", "GROUP_K", "SPLIT_K"])
@triton.jit
def _w8a16_wide(
    A, Q, S, P,
    M, N, K, M_KEY,
    sa_b, sa_m, sa_k,
    sq_b, sq_k, sq_n,
    ss_b, ss_g, ss_n,
    sp_s, sp_b, sp_m, sp_n,
    GROUP_K: tl.constexpr, PER_CHANNEL: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """P[s, b] = A[b, :, Ks] @ dequant(Q[b, Ks, :]) for K-slice s (fp32 partials)."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    kper = K // SPLIT_K
    k0 = pid_s * kper
    a_ptr = A + pid_b * sa_b + offs_m[:, None] * sa_m + (k0 + offs_k)[None, :] * sa_k
    q_ptr = Q + pid_b * sq_b + (k0 + offs_k)[:, None] * sq_k + offs_n[None, :] * sq_n
    s_row = S + pid_b * ss_b + offs_n * ss_n
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, kper, BLOCK_K):
        k_mask = (k0 + kk + offs_k) < K
        a = tl.load(a_ptr, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        q = tl.load(q_ptr, mask=k_mask[:, None] & n_mask[None, :], other=0).to(tl.float32)
        pr = tl.sum(a[:, :, None] * q[None, :, :], axis=1)
        if PER_CHANNEL:
            acc += pr
        else:
            sc = tl.load(s_row + ((k0 + kk) // GROUP_K) * ss_g, mask=n_mask, other=0.0)
            acc += pr * sc[None, :]
        a_ptr += BLOCK_K * sa_k
        q_ptr += BLOCK_K * sq_k
    if PER_CHANNEL:
        acc *= tl.load(s_row, mask=n_mask, other=0.0)[None, :]
    p_ptr = P + pid_s * sp_s + pid_b * sp_b + offs_m[:, None] * sp_m + offs_n[None, :] * sp_n
    tl.store(p_ptr, acc, mask=m_mask[:, None] & n_mask[None, :])


# triton's Autotuner is NOT thread-safe: `run` stores the call's args in
# `self.nargs` and clears it afterwards, and `_bench` reads it. The server
# launches the same kernel objects from two threads (the encoder's QLinear in
# the prep thread, the prefill's _qlinear in the decode thread); when one of
# them hits a new shape and autotunes while the other launches, the other saw
# `nargs = None` -> "TypeError: 'NoneType' object is not a mapping" (5 of 8
# requests in a burst failed). The lock covers only the Python launch, not the
# GPU work, so encode and decode still overlap on the device.
_LAUNCH_LOCK = threading.Lock()
_autotune_loaded = False


def _autotune_load():
    """Restore persisted autotune choices before the first launch (once).
    See kernels/autotune_cache.py."""
    global _autotune_loaded
    if _autotune_loaded:
        return
    _autotune_loaded = True
    from . import autotune_cache
    autotune_cache.load(verbose=True)

WIDE_MAX_M = 4            # measured: wide wins at M<=4 (4-28%), loses at M=8 (0.8-0.94x)
WIDE_TARGET_PROGRAMS = 128


def choose_wide_split(M, N, K, n_batch, block_n=256, max_split=32):
    """Split-K for the wide kernel: enough programs to cover the CUs. Tuned by
    measurement at M=1: N=2048 -> 16, 6144/8192 -> 4, 65536 -> 1."""
    blocks = n_batch * -(-N // block_n)
    want = -(-WIDE_TARGET_PROGRAMS // max(blocks, 1))
    sk = 1
    while sk < want and sk < max_split and (K // (sk * 2)) >= 64 and (K % (sk * 2)) == 0:
        sk *= 2
    return sk


def _wide_matmul(A, Q, S, group_k, split_k, out_dtype):
    """A (B,M,K), Q (B,K,N) or (1,K,N), S (B,G,N) or (1,G,N) -> (B,M,N)."""
    B, M, K = A.shape
    Bq, _, N = Q.shape
    per_channel = (S.shape[1] == 1)
    m_key = triton.next_power_of_2(M)
    sq_b = Q.stride(0) if Bq == B else 0
    ss_b = S.stride(0) if Bq == B else 0
    P = torch.empty((split_k, B, M, N), device=A.device, dtype=torch.float32)
    # wrap_triton makes the launch traceable inside the torch.library op (fake
    # tensors under torch.compile); outside tracing it is the plain kernel
    launch = wrap_triton(_w8a16_wide) if _HAS_TRITON_OP else _w8a16_wide
    launch[lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), split_k, B)](
        A, Q, S, P, M, N, K, m_key,
        A.stride(0), A.stride(1), A.stride(2),
        sq_b, Q.stride(1), Q.stride(2),
        ss_b, S.stride(1), S.stride(2),
        P.stride(0), P.stride(1), P.stride(2), P.stride(3),
        GROUP_K=group_k, PER_CHANNEL=per_channel, SPLIT_K=split_k,
        BLOCK_M=max(1, m_key))
    return (P.sum(0) if split_k > 1 else P[0]).to(out_dtype)


# --------------------------------------------------------------------------- #
# split-K
# --------------------------------------------------------------------------- #
# Measured need, not a guess. With one program per (BLOCK_M, BLOCK_N) tile the
# block count is set by N, and on this 48-CU card that starves the narrow shapes:
#
#   shape                 blocks   int8 GB/s
#   att W_out  2048x2048      16         118
#   ffn W_v    8192x2048      16         126
#   ffn W_k    2048x8192      64         354
#   head      2048x65536     512         556
#
# Splitting K gives each output tile several programs, which is the only axis
# left when N is small. Partial sums go to a (SPLIT_K, B, M, N) float32 buffer
# and are reduced afterwards — deterministic, unlike atomics, and the reduction
# is trivial next to the weight traffic (M <= 64).
def _splitk_configs():
    # num_stages varies on purpose. With the dot form the K loop is
    # load-convert-dot, and at stages=2 the int8->bf16 conversion of each weight
    # tile sits in the critical path: the dot form plateaued at 340-400 GB/s for
    # M>=8 while the reduction form reached 425-606 GB/s at M=1. Deeper
    # pipelining is what lets the conversion overlap the next tile's load.
    return [
        triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=st)
        for bn, bk, w, st in (
            (32, 64, 4, 2), (64, 64, 4, 2), (32, 128, 4, 2),
            (64, 128, 4, 2), (128, 64, 4, 2), (32, 256, 8, 2),
            (64, 64, 4, 3), (64, 128, 4, 3), (128, 64, 4, 3),
            (128, 128, 8, 3), (64, 256, 8, 3),
            (64, 128, 4, 4), (128, 64, 8, 4), (128, 128, 8, 4),
        )
    ]


@triton.autotune(configs=_splitk_configs(),
                 key=["M_KEY", "N", "K", "GROUP_K", "SPLIT_K", "USE_DOT"],
                 prune_configs_by={"early_config_prune": _prune_gemv})
@triton.jit
def _w8a16_splitk(
    A, Q, S, P,
    M, N, K, M_KEY,
    sa_b, sa_m, sa_k,
    sq_b, sq_k, sq_n,
    ss_b, ss_g, ss_n,
    sp_s, sp_b, sp_m, sp_n,
    GROUP_K: tl.constexpr, PER_CHANNEL: tl.constexpr, USE_DOT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Partial products into P[split, b, m, n] (float32); caller sums over dim 0.

    `USE_DOT` picks the contraction form, and it is chosen by M rather than fixed:
    the broadcast-and-reduce form has no tile minimum (ideal at M=1) but holds a
    (BLOCK_M, BLOCK_K, BLOCK_N) intermediate that spills once M grows, while
    tl.dot has a 16-row minimum but constant register cost. Splitting K is what
    fills the device for narrow N; both forms need it equally, which is why it
    lives here rather than only on the small-M path.
    """
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    k0 = pid_s * BLOCK_K
    a_ptr = A + pid_b * sa_b + offs_m[:, None] * sa_m + (k0 + offs_k)[None, :] * sa_k
    q_ptr = Q + pid_b * sq_b + (k0 + offs_k)[:, None] * sq_k + offs_n[None, :] * sq_n
    s_row = S + pid_b * ss_b + offs_n * ss_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    step = BLOCK_K * SPLIT_K
    while k0 < K:
        k_mask = (k0 + offs_k) < K
        if USE_DOT:
            a = tl.load(a_ptr, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            q = tl.load(q_ptr, mask=k_mask[:, None] & n_mask[None, :], other=0)
            p = tl.dot(a, q.to(a.dtype), out_dtype=tl.float32)
        else:
            a = tl.load(a_ptr, mask=m_mask[:, None] & k_mask[None, :],
                        other=0.0).to(tl.float32)
            q = tl.load(q_ptr, mask=k_mask[:, None] & n_mask[None, :],
                        other=0).to(tl.float32)
            p = tl.sum(a[:, :, None] * q[None, :, :], axis=1)
        if PER_CHANNEL:
            acc += p
        else:
            s = tl.load(s_row + (k0 // GROUP_K) * ss_g, mask=n_mask, other=0.0)
            acc += p * s[None, :]
        a_ptr += step * sa_k
        q_ptr += step * sq_k
        k0 += step

    if PER_CHANNEL:
        acc *= tl.load(s_row, mask=n_mask, other=0.0)[None, :]

    p_ptr = (P + pid_s * sp_s + pid_b * sp_b
             + offs_m[:, None] * sp_m + offs_n[None, :] * sp_n)
    tl.store(p_ptr, acc, mask=m_mask[:, None] & n_mask[None, :])


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
# Measured crossover for the reduction form. At M=8 it is already register
# starved (0.39-0.69x of bf16) and at M=4 it still trails the dot form, which
# only wastes a padded tile on a kernel that is bandwidth bound anyway.
GEMV_MAX_M = 2          # reduction form beats the padded dot at or below this
DOT_MIN_M = 16          # tl.dot's tile minimum

TARGET_BLOCKS = 256     # ~4 waves on a 48-CU card; below this, split K
# The split-K kernel does not tile M: one program covers all M rows with
# BLOCK_M = next_pow2(M). That is the design for decode-sized M; past 64 the
# tile blows up (M=257 -> BLOCK_M=512 faulted with an illegal address on a
# 256x256 weight, 2026-10-31). Larger M goes to the M-tiled GEMM instead.
SPLITK_MAX_M = 64


def choose_split_k(M, N, K, n_batch, block_n=64, max_split=16):
    """Smallest split that gets the launch to roughly fill the device.

    Block count is n_batch * ceil(N/BLOCK_N) * ceil(M/BLOCK_M). It is dominated
    by N, which is why the narrow shapes here (N=2048 -> 16 blocks) ran at a
    third of the bandwidth the wide ones reached; including the M term lets a
    batch that already supplies parallelism stop asking for a split.

    NOT auto-tuned by timing, though that was tried. A timing sweep re-uses one
    weight tensor, so it measures the CACHE-RESIDENT case and concludes SPLIT_K=1
    is best — correct for that measurement and wrong for this model, which
    streams 3.7 GB of distinct weights per token and hits no cache at all. It
    made att W_out 1.00x -> 0.54x. A timing-based selector would have to evict
    between candidates to be meaningful.
    """
    if M > SPLITK_MAX_M:
        return 1
    block_m = DOT_MIN_M if M > GEMV_MAX_M else max(1, M)
    blocks = n_batch * -(-N // block_n) * -(-M // block_m)
    if blocks >= TARGET_BLOCKS:
        return 1
    want = -(-TARGET_BLOCKS // max(blocks, 1))
    sk = 1
    while sk < want and sk < max_split and (K // (sk * 2)) >= 64:
        sk *= 2
    return sk


def _as3d(t, n):
    return t if t.dim() == n + 1 else t.unsqueeze(0)


def int8_matmul(a, q, s, group_k, kernel="auto", out=None, split_k=None):
    """a @ dequant(q, s).

    a : (M, K) or (B, M, K), bf16/fp16/fp32
    q : (K, N) or (B, K, N) int8         — (in, out) layout
    s : (G, N) or (B, G, N) float32
    Returns (M, N) or (B, M, N) in a.dtype.

    Triton launches on the CURRENT device, not the tensors' device: a cuda:1
    tensor with cuda:0 current is an illegal memory access. Every entry point
    therefore runs under the tensors' device context.
    """
    _autotune_load()
    with _LAUNCH_LOCK, torch.cuda.device(a.device):
        return _int8_matmul_impl(a, q, s, group_k, kernel, out, split_k)


def _int8_matmul_impl(a, q, s, group_k, kernel="auto", out=None, split_k=None):
    """a @ dequant(q, s).

    a : (M, K) or (B, M, K), bf16/fp16/fp32
    q : (K, N) or (B, K, N) int8         — (in, out) layout
    s : (G, N) or (B, G, N) float32
    Returns (M, N) or (B, M, N) in a.dtype.
    """
    batched = (a.dim() == 3)
    A = _as3d(a, 2)
    Q = _as3d(q, 2)
    S = _as3d(s, 2)
    B, M, K = A.shape
    Bq, Kq, N = Q.shape
    if Kq != K:
        raise ValueError(f"K mismatch: a has {K}, q has {Kq}")
    if kernel in ("auto", "wide") and M <= WIDE_MAX_M:
        sk = split_k if split_k is not None else choose_wide_split(M, N, K, B)
        C = _wide_matmul(A, Q, S, group_k, sk, a.dtype)
        return C if batched else C.squeeze(0)
    if Bq != B and Bq != 1:
        raise ValueError(f"batch mismatch: a has {B}, q has {Bq}")
    G = S.shape[1]
    if G * group_k != K:
        raise ValueError(f"scales have {G} groups of {group_k} != K={K}")
    per_channel = (G == 1)

    m_key = triton.next_power_of_2(M)

    # ---- split-K path -------------------------------------------------- #
    if kernel == "auto":
        sk = choose_split_k(M, N, K, B)
    elif kernel == "splitk":
        sk = split_k if split_k and split_k > 1 else 4
    else:
        sk = 1
    if sk > 1:
        use_dot = M > GEMV_MAX_M
        block_m = max(DOT_MIN_M, m_key) if use_dot else max(1, m_key)
        P = torch.empty((sk, B, M, N), device=a.device, dtype=torch.float32)
        _w8a16_splitk[lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), sk, B)](
            A, Q, S, P,
            M, N, K, m_key,
            A.stride(0), A.stride(1), A.stride(2),
            Q.stride(0) if Bq == B else 0, Q.stride(1), Q.stride(2),
            S.stride(0) if Bq == B else 0, S.stride(1), S.stride(2),
            P.stride(0), P.stride(1), P.stride(2), P.stride(3),
            GROUP_K=group_k, PER_CHANNEL=per_channel, USE_DOT=use_dot,
            BLOCK_M=block_m, SPLIT_K=sk,
        )
        C = P.sum(0).to(a.dtype) if out is None else out.copy_(P.sum(0))
        return C if batched else C.squeeze(0)

    use_gemv = (kernel == "gemv") or (kernel in ("auto", "splitk") and M <= GEMV_MAX_M)
    fn = _w8a16_gemv if use_gemv else _w8a16_gemm

    if use_gemv:
        def launch(meta):
            return (triton.cdiv(N, meta["BLOCK_N"]), B)
    else:
        def launch(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), B)

    C = out
    if C is None:
        C = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    kw = {}
    if use_gemv:
        # the reduction path has no tile minimum, so M itself sets BLOCK_M
        kw["BLOCK_M"] = max(1, m_key)

    fn[launch](
        A, Q, S, C,
        M, N, K, m_key,
        A.stride(0), A.stride(1), A.stride(2),
        Q.stride(0) if Bq == B else 0, Q.stride(1), Q.stride(2),
        S.stride(0) if Bq == B else 0, S.stride(1), S.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        GROUP_K=group_k, PER_CHANNEL=per_channel,
        **kw,
    )
    return C if batched else C.squeeze(0)


# --------------------------------------------------------------------------- #
# torch.compile entry point
# --------------------------------------------------------------------------- #
# Triton's Python launch path costs ~50 us per call. A decode step issues ~180
# quantised matmuls, so calling `int8_matmul` directly from eager code would add
# ~9 ms of CPU per token — more than the entire GPU step — and int8 would be a
# large net LOSS in wall clock even though every kernel is faster.
#
# Registering as a torch.library op fixes that: inside a torch.compile'd block the
# launch is emitted by inductor's generated code instead of the Python autotuner,
# and dynamo treats the call as one node rather than breaking the graph around it.
# The eager path stays available (and stays slow) for testing.
try:
    from torch.library import triton_op, wrap_triton
    _HAS_TRITON_OP = True
except ImportError:                                   # pragma: no cover
    _HAS_TRITON_OP = False

if _HAS_TRITON_OP:
    @triton_op("rwkv_next::int8_matmul", mutates_args={})
    def int8_matmul_op(a: torch.Tensor, q: torch.Tensor, s: torch.Tensor,
                       group_k: int, split_k: int, use_dot: bool) -> torch.Tensor:
        """Functional wrapper. The dispatch decisions (which form, what split) are
        made by the CALLER and passed in, because they depend only on the shape and
        so can be resolved once at pack time rather than on every token."""
        _autotune_load()
        with _LAUNCH_LOCK, torch.cuda.device(a.device):          # see int8_matmul
            return _int8_matmul_op_impl(a, q, s, group_k, split_k, use_dot)

    def _int8_matmul_op_impl(a, q, s, group_k, split_k, use_dot):
        batched = (a.dim() == 3)
        A = a if batched else a.unsqueeze(0)
        Q = q if q.dim() == 3 else q.unsqueeze(0)
        S = s if s.dim() == 3 else s.unsqueeze(0)
        B, M, K = A.shape
        if M <= WIDE_MAX_M and not use_dot:
            C = _wide_matmul(A, Q, S, group_k, split_k, a.dtype)
            return C if batched else C.squeeze(0)
        Bq, _, N = Q.shape
        per_channel = (S.shape[1] == 1)
        m_key = triton.next_power_of_2(M)
        sq_b = Q.stride(0) if Bq == B else 0
        ss_b = S.stride(0) if Bq == B else 0

        if split_k > 1:
            block_m = max(DOT_MIN_M, m_key) if use_dot else max(1, m_key)
            P = torch.empty((split_k, B, M, N), device=a.device, dtype=torch.float32)
            wrap_triton(_w8a16_splitk)[
                lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), split_k, B)](
                A, Q, S, P, M, N, K, m_key,
                A.stride(0), A.stride(1), A.stride(2),
                sq_b, Q.stride(1), Q.stride(2),
                ss_b, S.stride(1), S.stride(2),
                P.stride(0), P.stride(1), P.stride(2), P.stride(3),
                GROUP_K=group_k, PER_CHANNEL=per_channel, USE_DOT=use_dot,
                BLOCK_M=block_m, SPLIT_K=split_k)
            C = P.sum(0).to(a.dtype)
            return C if batched else C.squeeze(0)

        C = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
        common = (A, Q, S, C, M, N, K, m_key,
                  A.stride(0), A.stride(1), A.stride(2),
                  sq_b, Q.stride(1), Q.stride(2),
                  ss_b, S.stride(1), S.stride(2),
                  C.stride(0), C.stride(1), C.stride(2))
        if use_dot:
            wrap_triton(_w8a16_gemm)[
                lambda meta: (triton.cdiv(M, meta["BLOCK_M"])
                              * triton.cdiv(N, meta["BLOCK_N"]), B)](
                *common, GROUP_K=group_k, PER_CHANNEL=per_channel)
        else:
            wrap_triton(_w8a16_gemv)[lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), B)](
                *common, GROUP_K=group_k, PER_CHANNEL=per_channel,
                BLOCK_M=max(1, m_key))
        return C if batched else C.squeeze(0)
else:                                                 # pragma: no cover
    def int8_matmul_op(a, q, s, group_k, split_k, use_dot):
        return int8_matmul(a, q, s, group_k,
                           kernel=("splitk" if split_k > 1 else
                                   ("gemm" if use_dot else "gemv")),
                           split_k=split_k)


def plan_int8(M, N, K, n_batch):
    """Resolve the dispatch for one shape: (split_k, use_dot). M <= WIDE_MAX_M
    goes to the wide skinny-M kernel (use_dot=False, its own split)."""
    if M <= WIDE_MAX_M:
        return choose_wide_split(M, N, K, n_batch), False
    return choose_split_k(M, N, K, n_batch), M > GEMV_MAX_M
