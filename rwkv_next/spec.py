# -*- coding: utf-8 -*-
"""One speculative step as ONE CUDA graph per rank (batch size Bg, K drafts).

rank 0                                         ranks > 0
  drafts = drafter.chain(H, t, K)                 .
  toks = [t | drafts]  --- NCCL broadcast --->    toks
  verify: K+1 tokens per row (compiled layers, all-reduces inside, identical on every rank)
  picks = sampler at every position (in-graph)    .
  j = accepted drafts, n = j + 1  -- bcast -->    n
  rollback(n) (masked wkv replay + shifts)        rollback(n)
  drafter commit of the accepted drafts           .

The workers get the drafts and the accepted counts through collectives inside their graph, so they
need no host round trip at all; rank 0 reads picks / n once per step (one sync). The emitted tokens
are the sampler's choices at each position, so the output distribution is the plain decode's.
Rows with presence / frequency penalties take the plain path instead (their penalty state would have
to advance position by position).
"""

from __future__ import annotations

import time

import torch

PREFILTER = 1024


def sample_graph(x, temp, top_p, top_k, min_p):
    """Graph-safe batched sampling: x (M, V) fp32, per-row params (M,). temp <= 0 -> greedy."""
    M = x.shape[0]
    K = min(PREFILTER, x.shape[-1])
    v, idx = x.topk(K, dim=-1)
    greedy = temp <= 1e-6
    t = torch.where(greedy, torch.ones_like(temp), temp).view(M, 1)
    pr = torch.softmax(v / t, dim=-1)
    ar = torch.arange(K, device=x.device).view(1, K)
    keep = ar < torch.where(top_k > 0, top_k, torch.full_like(top_k, K)).view(M, 1)
    keep &= (pr.cumsum(-1) - pr) < top_p.view(M, 1)
    keep &= pr >= min_p.view(M, 1) * pr[:, :1]
    keep[:, 0] = True
    pr = pr * keep
    cdf = pr.cumsum(-1)
    u = torch.rand(M, 1, device=x.device) * cdf[:, -1:]
    j = (cdf < u).sum(-1).clamp(max=K - 1)
    out = idx.gather(1, j.view(M, 1)).view(M)
    return torch.where(greedy, idx[:, 0], out)


class SpecGraphs:
    """All (Bg, K) spec-step graphs of one rank. Rank 0 passes the DraftPool, the others None."""

    def __init__(self, model, pool):
        self.m, self.pool = model, pool
        self.tp = model.tp
        self.graphs = {}
        S, dev = model.slots, model.device
        self.t_in = torch.zeros(S, dtype=torch.long, device=dev)
        self.temp = torch.zeros(S, device=dev); self.top_p = torch.ones(S, device=dev)
        self.top_k = torch.zeros(S, dtype=torch.long, device=dev); self.min_p = torch.zeros(S, device=dev)
        self.toks = {}          # K -> (S, K+1) long
        self.n_buf = torch.ones(S, dtype=torch.long, device=dev)
        self.mem = None

    def _body(self, B, K):
        m, pool = self.m, self.pool
        T = K + 1
        toks = self.toks[K][:B]
        if pool is not None:
            drafts = pool.draft(B, self.t_in[:B], K)
            toks[:, 0] = self.t_in[:B]
            toks[:, 1:] = drafts
        self.tp.broadcast(toks)
        h, H, rec = m.forward_verify(toks, B, pool.taps if pool is not None else ())
        picks = j = None
        if pool is not None:
            lg = m.head(h.reshape(B * T, m.C)).float()
            rep = lambda v: v[:B].repeat_interleave(T)
            picks = sample_graph(lg, rep(self.temp), rep(self.top_p), rep(self.top_k), rep(self.min_p)).view(B, T)
            j = (picks[:, :K] == drafts).long().cumprod(1).sum(1)
            self.n_buf[:B].copy_(j + 1)
        self.tp.broadcast(self.n_buf[:B])
        m.rollback(rec, self.n_buf[:B], B)
        if pool is not None:
            pool.accept_graph(B, H, drafts, j)
        return picks

    def capture(self, B, K, verbose=False):
        if K not in self.toks:
            self.toks[K] = torch.zeros(self.m.slots, K + 1, dtype=torch.long, device=self.m.device)
        t0 = time.time()
        m = self.m
        snap = (m.wkv[:, :B].clone(), m.att_shift[:, :B].clone(), m.ffn_shift[:, :B].clone())
        psnap = self.pool.snapshot(B) if self.pool is not None else None
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._body(B, K)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.mem):
            picks = self._body(B, K)
        self.mem = g.pool()
        self.graphs[(B, K)] = (g, picks)
        m.wkv[:, :B].copy_(snap[0]); m.att_shift[:, :B].copy_(snap[1]); m.ffn_shift[:, :B].copy_(snap[2])
        if psnap is not None:
            self.pool.restore(B, psnap)
        torch.cuda.synchronize()
        if verbose:
            print(f"[graph] spec step B={B} K={K} captured in {time.time() - t0:.1f}s", flush=True)

    def pick(self, B, K, limit):
        """Smallest captured Bg >= B for this K whose padding stays below `limit`, or None."""
        return next((b for (b, k) in sorted(self.graphs) if k == K and b >= B and (b == B or b <= limit)), None)

    def run_rank0(self, Bg, K, B, ids, params):
        """Stage inputs, replay. -> (picks (B, K+1) list, n (B,) list)."""
        dev = self.m.device
        self.t_in[:B].copy_(torch.tensor(ids, dtype=torch.long), non_blocking=True)
        if Bg > B:
            self.t_in[B:Bg].zero_()
        self.temp[:Bg].copy_(torch.tensor([max(p["temperature"], 0.0) for p in params] + [0.0] * (Bg - B)), non_blocking=True)
        self.top_p[:Bg].copy_(torch.tensor([p["top_p"] if 0 < p["top_p"] < 1 else 1.0 for p in params] + [1.0] * (Bg - B)), non_blocking=True)
        self.top_k[:Bg].copy_(torch.tensor([p["top_k"] or 0 for p in params] + [0] * (Bg - B)), non_blocking=True)
        self.min_p[:Bg].copy_(torch.tensor([p["min_p"] for p in params] + [0.0] * (Bg - B)), non_blocking=True)
        g, picks = self.graphs[(Bg, K)]
        g.replay()
        out = torch.cat([picks[:B], self.n_buf[:B].unsqueeze(1)], 1).tolist()      # one sync
        return [r[:-1] for r in out], [r[-1] for r in out]

    def run_worker(self, Bg, K):
        self.graphs[(Bg, K)][0].replay()
