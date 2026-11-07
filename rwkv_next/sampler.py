# -*- coding: utf-8 -*-
"""Batched token selection on rank 0: greedy / temperature / top-k / top-p / min-p, with
presence & frequency penalties over a decaying per-slot count table.

One pass for the whole batch, one host sync for the result (the caller's .tolist()).
With top-p < 1 or top-k <= PREFILTER the candidates are pre-filtered with topk(PREFILTER) instead of a
full sort; a batch containing a row that samples from the whole vocabulary (no top-p, no top-k) uses
every logit, so sampling is exact in both cases.
"""

from __future__ import annotations

import torch

PREFILTER = 1024


def needs_full_vocab(p):
    """a sampled row whose top-p / top-k do not truncate below PREFILTER candidates"""
    if p["temperature"] <= 1e-6:
        return False
    tp, tk = p["top_p"], p["top_k"] or 0
    return not (0 < tp < 1) and not (0 < tk <= PREFILTER)


class Sampler:
    def __init__(self, slots: int, vocab: int, device):
        self.counts = torch.zeros(slots, vocab, device=device, dtype=torch.float32)
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(0x5EED)

    def reset(self, slots):
        self.counts[slots] = 0

    def move(self, src, dst):
        s = torch.as_tensor(src, device=self.device); d = torch.as_tensor(dst, device=self.device)
        self.counts[d] = self.counts[s]

    @torch.no_grad()
    def __call__(self, logits, params, slots=None):
        """logits (B, V); params: list of dicts (temperature, top_p, top_k, min_p, presence_penalty,
        frequency_penalty, penalty_decay); slots: their slot rows (penalty table). Returns (B,) long."""
        B = logits.shape[0]
        x = logits.float()
        dev = x.device
        pen = [p for p in params if p["presence_penalty"] or p["frequency_penalty"]]
        if pen and slots is not None:
            sl = torch.as_tensor(slots, device=dev)
            c = self.counts.index_select(0, sl)
            pp = torch.tensor([p["presence_penalty"] for p in params], device=dev).view(B, 1)
            fp = torch.tensor([p["frequency_penalty"] for p in params], device=dev).view(B, 1)
            x = x - pp * (c > 0).float() - fp * c
        temp = torch.tensor([max(p["temperature"], 0.0) for p in params], device=dev)
        greedy = temp <= 1e-6
        if bool(greedy.all()):
            out = x.argmax(-1)
        else:
            K = x.shape[-1] if any(needs_full_vocab(p) for p in params) else min(PREFILTER, x.shape[-1])
            v, idx = x.topk(K, dim=-1)                                            # sorted, descending
            t = torch.where(greedy, torch.ones_like(temp), temp).view(B, 1)
            pr = torch.softmax(v / t, dim=-1)
            topk = torch.tensor([p["top_k"] if p["top_k"] and p["top_k"] > 0 else K for p in params], device=dev).view(B, 1)
            ar = torch.arange(K, device=dev).view(1, K)
            keep = ar < topk
            topp = torch.tensor([p["top_p"] if 0 < p["top_p"] < 1 else 1.0 for p in params], device=dev).view(B, 1)
            keep &= (pr.cumsum(-1) - pr) < topp
            minp = torch.tensor([p["min_p"] for p in params], device=dev).view(B, 1)
            keep &= pr >= minp * pr[:, :1]
            keep[:, 0] = True
            pr = pr * keep
            cdf = pr.cumsum(-1)
            u = torch.rand(B, 1, device=dev, generator=self.gen) * cdf[:, -1:]
            j = (cdf < u).sum(-1).clamp(max=K - 1)
            out = idx.gather(1, j.view(B, 1)).view(B)
            out = torch.where(greedy, x.argmax(-1), out)
        if pen and slots is not None:
            dec = torch.tensor([p["penalty_decay"] for p in params], device=dev).view(B, 1)
            self.counts[sl] = c * dec                                             # decay, then count this token
            self.counts[sl, out] += 1
        return out
