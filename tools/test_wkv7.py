#!/usr/bin/env python3
"""rwkv_next/kernels/wkv7.py against a pure-PyTorch fp64 recurrence (and rwkvfla, if installed).
Cases: plain (T=1, 4, 300) with / without initial state, the torch.compile op path, varlen packed prefill
with per-sequence initial states, and the rollback replay (first n[b] steps, in place).

  python tools/test_wkv7.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rwkv_next.kernels.wkv7 import wkv7, wkv7_replay


def ref(r, w, k, v, a, b, s0):
    """fp32 loop. inputs (B, T, H, K); s0 (B, H, K, V) or None -> (o, sT)."""
    B, T, H, K = r.shape
    S = torch.zeros(B, H, K, K, dtype=torch.float64, device=r.device) if s0 is None else s0.double().clone()
    o = torch.empty(B, T, H, K, dtype=torch.float64, device=r.device)
    for t in range(T):
        rr, ww, kk, vv, aa, bb = (x[:, t].double() for x in (r, w, k, v, a, b))
        d = torch.exp(-torch.exp(ww))
        S = d[..., None] * S + bb[..., None] * (aa[..., None] * S).sum(-2, keepdim=True) + kk[..., None] * vv[..., None, :]
        o[:, t] = (S * rr[..., None]).sum(-2)
    return o, S


def inputs(B, T, H, K, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device=dev, generator=g)
    r, k, v = rn(B, T, H, K) * 0.5, rn(B, T, H, K) * 0.5, rn(B, T, H, K) * 0.5
    w = -torch.nn.functional.softplus(-rn(B, T, H, K)) - 0.5                  # RWKV-7 range
    kk = torch.nn.functional.normalize(rn(B, T, H, K), dim=-1)
    ag = torch.sigmoid(rn(B, T, H, K))
    a, b = -kk, kk * ag
    return [x.bfloat16() for x in (r, w, k, v, a, b)]


def rel(x, y):
    return ((x.double() - y.double()).norm() / y.double().norm().clamp_min(1e-12)).item()


def main():
    dev = "cuda"
    H, K = 8, 64
    bad = 0
    flat = lambda x: x.reshape(x.shape[0], x.shape[1], H * K)
    for B, T, use_s0 in ((3, 1, True), (2, 4, True), (2, 300, False), (1, 300, True)):
        ins = inputs(B, T, H, K, dev, B * 1000 + T)
        s0 = torch.randn(B, H, K, K, device=dev) * 0.3 if use_s0 else None
        o_ref, s_ref = ref(*ins, s0)
        for mode in ("eager", "op"):
            if mode == "op":
                o, s = torch.compile(lambda *x: wkv7(*x, s0, H, K))(*map(flat, ins))
            else:
                o, s = wkv7(*map(flat, ins), s0, H, K)
            eo, es = rel(o.view(B, T, H, K), o_ref), rel(s, s_ref)
            ok = eo < 2e-2 and es < 1e-3
            bad += not ok
            print(f"B={B} T={T:3d} h0={use_s0!s:5} {mode:5s}: o rel {eo:.2e}  state rel {es:.2e}  {'ok' if ok else 'FAIL'}")
        try:
            from rwkvfla.ops.generalized_delta_rule import fused_recurrent_dplr_delta_rule as fla
            r, w, k, v, a, b = ins
            of, sf = fla(q=r, k=k, v=v, a=a, b=b, gk=-torch.exp(w.float()), scale=1.0, initial_state=s0, output_final_state=True)
            print(f"      rwkvfla (fp32 gk)   : o rel {rel(of, o_ref):.2e}  state rel {rel(sf, s_ref):.2e}  (reference for the old path)")
        except ImportError:
            pass
    # varlen with per-sequence initial states
    lens = [5, 1, 37, 300]
    seqs = [inputs(1, L, H, K, dev, 7 + L) for L in lens]
    s0 = torch.randn(len(lens), H, K, K, device=dev) * 0.3
    cat = [torch.cat([sq[j] for sq in seqs], 1) for j in range(6)]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
    o, s = wkv7(*map(flat, cat), s0, H, K, cu_seqlens=cu)
    off = 0
    for i, L in enumerate(lens):
        o_ref, s_ref = ref(*seqs[i], s0[i:i + 1])
        eo, es = rel(o[:, off:off + L].view(1, L, H, K), o_ref), rel(s[i:i + 1], s_ref)
        ok = eo < 2e-2 and es < 1e-3; bad += not ok; off += L
        print(f"varlen seq {i} (T={L:3d}): o rel {eo:.2e}  state rel {es:.2e}  {'ok' if ok else 'FAIL'}")
    # rollback replay: keep the first n[b] of T=4 steps, in place
    B, T = 4, 4
    ins = inputs(B, T, H, K, dev, 99)
    s0 = torch.randn(B, H, K, K, device=dev) * 0.3
    n = torch.tensor([1, 2, 4, 3], device=dev)
    st = s0.clone()
    wkv7_replay(*map(flat, ins), st, n, H, K)
    for i in range(B):
        _, s_ref = ref(*(x[i:i + 1, :int(n[i])] for x in ins), s0[i:i + 1])
        es = rel(st[i:i + 1], s_ref); ok = es < 1e-3; bad += not ok
        print(f"replay row {i} keep {int(n[i])}/{T}: state rel {es:.2e}  {'ok' if ok else 'FAIL'}")
    print("ALL PASSED" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
