#!/usr/bin/env python3
"""Quantisation quality vs bf16, TP1: CE / KL(bf16 || q) / top-1 agreement over token windows.
The bf16 reference runs on --ref_dev, the quantised configs one after another on --dev.

  python tools/test_quality.py --ckpt <.pth> --windows <npz of int windows> --configs int6,int6+lora8,int8
  (config = quant[+lora8][+att8]; lora8 = RWKV_LORA_QUANT=int8, att8 = time-mix kept int8)"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F


def windows(path, n, T):
    z = np.load(path)
    rows = [r[:T] for k in sorted(z.files) for r in z[k] if z[k].ndim == 2 and z[k].shape[1] >= T]
    idx = np.random.default_rng(0).permutation(len(rows))[:n]
    return [[int(x) for x in rows[i]] for i in idx]


def logits_all(m, ids):
    dev = m.device
    t = torch.tensor(ids, device=dev)
    cu = torch.tensor([0, len(ids)], dtype=torch.int32, device=dev)
    h = m.forward_prefill(t, cu, torch.tensor([0], device=dev), torch.tensor([True], device=dev),
                          logits_positions=torch.arange(len(ids) - 1, device=dev))
    return m.logits(h).float().cpu()


def load(ckpt, dev, cfg):
    from rwkv_next import tp as tpm
    from rwkv_next.model import RWKV7TP
    parts = cfg.split("+")
    os.environ["RWKV_LORA_QUANT"] = "int8" if "lora8" in parts else ""
    m = RWKV7TP.load(ckpt, tpm.TP(0, 1), dev, quant=parts[0], keep_int8=("att",) if "att8" in parts else (), verbose=False)
    m.alloc_slots(1)
    gb = sum(b.numel() * b.element_size() for b in m.buffers()) / 1e9
    return m, gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--windows", required=True)
    ap.add_argument("--n", type=int, default=64); ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--configs", default="int6,int6+lora8"); ap.add_argument("--dev", default="cuda:0"); ap.add_argument("--ref_dev", default="cuda:1")
    a = ap.parse_args()
    W = windows(a.windows, a.n, a.T)
    torch.cuda.set_device(torch.device(a.ref_dev))
    ref, _ = load(a.ckpt, torch.device(a.ref_dev), "bf16")
    R = [logits_all(ref, w) for w in W]
    del ref; torch.cuda.empty_cache()
    ce_ref = float(np.mean([F.cross_entropy(r, torch.tensor(w[1:])).item() for r, w in zip(R, W)]))
    print(f"{len(W)} windows x {a.T}; bf16 CE {ce_ref:.4f}", flush=True)
    torch.cuda.set_device(torch.device(a.dev))
    for c in a.configs.split(","):
        m, gb = load(a.ckpt, torch.device(a.dev), c)
        ce, kl, top = [], [], []
        for r, w in zip(R, W):
            q = logits_all(m, w)
            ce.append(F.cross_entropy(q, torch.tensor(w[1:])).item())
            pr, pq = F.log_softmax(r, -1), F.log_softmax(q, -1)
            kl.append((pr.exp() * (pr - pq)).sum(-1).mean().item()); top.append((r.argmax(-1) == q.argmax(-1)).float().mean().item())
        print(f"{c:18s} dCE {np.mean(ce) - ce_ref:+.4f}  KL {np.mean(kl):.5f}  top1 {np.mean(top) * 100:.2f}%  weights {gb:.2f} GB", flush=True)
        del m; torch.cuda.empty_cache()
    os._exit(0)


if __name__ == "__main__":
    main()
