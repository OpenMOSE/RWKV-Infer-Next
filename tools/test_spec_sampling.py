#!/usr/bin/env python3
"""The spec step's in-graph sampling against the eager Sampler, on random logits (no model needed):

1. penalties: a spec step (penalize_positions -> greedy picks -> commit_counts) emits the same tokens and leaves
   the same penalty table as the plain sampler run token by token over the same per-position logits;
2. full-vocabulary rows: sample_graph's Gumbel-max matches the softmax distribution (with a min-p cut).

  python tools/test_spec_sampling.py [--device cuda]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    from rwkv_next.sampler import Sampler
    from rwkv_next.spec import penalize_positions, commit_counts, sample_graph
    dev = torch.device(a.device)
    g = torch.Generator(device="cpu").manual_seed(0)
    ok = True
    # 1. penalties, greedy: tokens and table identical to the sequential plain sampler
    V, B, K = 50, 6, 3
    T = K + 1
    bad = 0
    for trial in range(200):
        base = torch.randn(B, T, V, generator=g).to(dev) * 2
        pres = torch.rand(B, generator=g).to(dev) * 1.5
        freq = torch.rand(B, generator=g).to(dev) * 1.5
        pres[0] = freq[0] = 0                                     # a row without penalties
        decay = torch.tensor([1.0, 0.99, 0.9, 1.0, 0.5, 0.996], device=dev)
        pen = (pres != 0) | (freq != 0)
        c0 = torch.randint(0, 3, (B, V), generator=g).float().to(dev)
        # drafts: mostly the plain greedy choice (so some get accepted), sometimes random
        S = Sampler(B, V, dev); S.counts.copy_(c0)
        params = [{"temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0, "presence_penalty": float(pres[b]),
                   "frequency_penalty": float(freq[b]), "penalty_decay": float(decay[b])} for b in range(B)]
        # reference: plain decoding over the positions, stopping at the first draft mismatch
        drafts = torch.randint(0, V, (B, K), generator=g).to(dev)
        ref_tokens = [[] for _ in range(B)]
        for b in range(B):
            Sb = Sampler(1, V, dev); Sb.counts.copy_(c0[b:b + 1])
            for i in range(T):
                t = Sb(base[b:b + 1, i], [params[b]], [0]).item()
                if i < K and torch.rand(1, generator=g).item() < 0.7:
                    drafts[b, i] = t                               # make this draft right
                ref_tokens[b].append(t)
                if i == K or t != drafts[b, i].item():
                    break
            if not pen[b]:
                Sb.counts.copy_(c0[b:b + 1])                       # rows without penalties keep their table
            S.counts[b] = Sb.counts[0]
        # the spec step
        toks = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=dev), drafts], 1)
        c = c0.clone()
        lg, cs = penalize_positions(base.clone(), c, toks, pres, freq, decay)
        picks = lg.argmax(-1)
        j = (picks[:, :K] == drafts).long().cumprod(1).sum(1)
        commit_counts(c, cs, picks, j, decay, pen)
        for b in range(B):
            got = picks[b, :j[b] + 1].tolist()
            if got != ref_tokens[b] or not torch.allclose(c[b], S.counts[b], atol=1e-5):
                bad += 1
    print(f"{'PASS' if bad == 0 else 'FAIL'}  penalties: spec step == sequential plain sampler ({200 * B} rows, {bad} mismatches)")
    ok &= bad == 0
    # 2. full-vocabulary Gumbel-max vs softmax (temperature 0.8, min_p 0.05)
    V, M = 40, 200000
    x = torch.randn(1, V, generator=g).to(dev) * 2
    temp, mp = 0.8, 0.05
    xs = x.expand(M, V).contiguous()
    f = lambda v, dt=torch.float32: torch.full((M,), v, device=dev, dtype=dt)
    out = sample_graph(xs, f(temp), f(1.0), f(0, torch.long), f(mp), torch.ones(M, dtype=torch.bool, device=dev))
    emp = torch.bincount(out, minlength=V).float() / M
    p = torch.softmax(x[0] / temp, -1)
    p = torch.where(p >= mp * p.max(), p, torch.zeros_like(p)); p /= p.sum()
    tv = 0.5 * (emp - p).abs().sum().item()
    good = tv < 0.01 and emp[p == 0].sum().item() == 0
    print(f"{'PASS' if good else 'FAIL'}  full-vocabulary sampling: total variation {tv:.4f} vs softmax with min-p cut")
    ok &= good
    print("ALL PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
