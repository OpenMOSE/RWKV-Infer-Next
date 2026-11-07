#!/usr/bin/env python3
"""Correctness and speed of the tensor-parallel model.

  python tools/test_tp.py --ckpt <plain RWKV-7 .pth> --check            # tp1 graphs vs eager, tp2 vs tp1
                                                                         # (+ tp1 vs an external reference model when
                                                                         #  RWKV_REF_HARNESS=<RWKV-Communication-Harness dir>)
  python tools/test_tp.py --ckpt <...> --bench --tp 2 --quant int6      # decode ms/step per batch size (graphs)

The check runs: packed prefill of 3 prompts (one of them continued in a second chunk), then 24 greedy
decode steps eager, then the same with graphs; it compares the logits of every step."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.multiprocessing as mp

PROMPTS = ["User: 日本の首都はどこですか？\n\nAssistant:", "The quick brown fox jumps over the lazy dog. " * 6,
           "def fib(n):\n    if n < 2:\n        return n\n"]


def run(rank, world, a, port, out):
    from rwkv_next import tp as tpm
    from rwkv_next.model import RWKV7TP
    from rwkv_next.tokenizer import RWKVTokenizer
    T = tpm.init(rank, world, port, rank)
    dev = torch.device("cuda", rank)
    m = RWKV7TP.load(a.ckpt, T, dev, quant=a.quant, verbose=True)
    m.alloc_slots(8)
    tok = RWKVTokenizer()
    ids = [tok.encode(p) for p in PROMPTS]
    res = {"prefill": None, "eager": [], "graph": []}
    # prefill: prompt 0 and 1 fresh; prompt 2 in two chunks (fresh, then continued)
    cut = len(ids[2]) // 2
    seqs = [ids[0], ids[1], ids[2][:cut]]
    flat = torch.tensor(sum(seqs, []), device=dev)
    cu = torch.tensor([0] + list(torch.tensor([len(s) for s in seqs]).cumsum(0)), device=dev, dtype=torch.int32)
    h = m.forward_prefill(flat, cu, torch.tensor([0, 1, 2], device=dev), torch.tensor([True, True, True], device=dev))
    flat2 = torch.tensor(ids[2][cut:], device=dev); cu2 = torch.tensor([0, len(ids[2]) - cut], device=dev, dtype=torch.int32)
    h2 = m.forward_prefill(flat2, cu2, torch.tensor([2], device=dev), torch.tensor([False], device=dev))
    h = torch.cat([h[:2], h2])
    lg = m.logits(h)
    snap = (m.wkv.clone(), m.att_shift.clone(), m.ffn_shift.clone())
    if rank == 0:
        res["prefill"] = lg.float().cpu()
    for mode in ("eager", "graph"):
        m.wkv.copy_(snap[0]); m.att_shift.copy_(snap[1]); m.ffn_shift.copy_(snap[2])
        if mode == "graph":
            if a.compile:
                m.compile([4])
            m.capture_decode([4])
        t = lg.argmax(-1) if rank == 0 else torch.zeros(3, dtype=torch.long, device=dev)
        if os.environ.get("TP_DEBUG"):
            print(rank, mode, "start", flush=True)
        for step in range(24):
            # lockstep like the engine (rank 0 syncs every step, workers wait on the command channel): a rank
            # running many steps ahead through graph replays + eager collectives can hang NCCL's SHM transport
            torch.cuda.synchronize()
            if os.environ.get("TP_DEBUG"):
                print(rank, mode, "step", step, flush=True)
            T.broadcast(t)
            if mode == "eager":
                h = m.forward_decode(t, 3); l = m.logits(h)
            else:
                l, h = m.decode(t, 3)
            if rank == 0:
                res[mode].append(l.float().cpu()); t = l.argmax(-1)
    if os.environ.get("TP_DEBUG"):
        print(rank, "loops done", flush=True)
    if rank == 0:
        torch.save(res, out)
    if os.environ.get("TP_DEBUG"):
        print(rank, "saved, exiting", flush=True)
    tpm.shutdown(T)
    os._exit(0)


def harness_reference(a, out):
    """optional: an independent implementation (the RWKV-Communication-Harness model) as the reference"""
    sys.path.insert(0, os.environ["RWKV_REF_HARNESS"])
    from fast_asr.rwkv_lm import RWKVLM
    lm = RWKVLM.from_checkpoint(a.ckpt, device="cuda:0", quant=a.quant if a.quant != "int6" else "int6", verbose=False)
    ids = [lm.encode(p) for p in PROMPTS]
    lg = []
    sts = []
    for s in ids:
        l, st = lm.prefill([s]); lg.append(l.float().cpu()); sts.append(st)
    from fast_asr.rwkv7 import RWKVState
    st = RWKVState.stack(sts)
    t = torch.stack([x[0] for x in lg]).argmax(-1).cuda()
    steps = []
    for _ in range(24):
        l, st = lm.step(t, st); steps.append(l.float().cpu()); t = l.argmax(-1)
    torch.save({"prefill": torch.cat(lg), "eager": steps}, out)


def compare(name, A, B):
    def one(x, y):
        d = (x - y).abs().max().item(); agree = (x.argmax(-1) == y.argmax(-1)).float().mean().item()
        return d, agree
    d, ag = one(A["prefill"], B["prefill"])
    worst, agrees = d, [ag]
    for k in ("eager", "graph"):
        if k in A and k in B and A[k] and B[k]:
            for x, y in zip(A[k], B[k]):
                d, ag = one(x, y); worst = max(worst, d); agrees.append(ag)
    ok = min(agrees) == 1.0
    print(f"{name}: max |dlogit| {worst:.4f}, argmax agreement min {min(agrees)*100:.1f}% -> {'OK' if ok else 'DIFF'}")
    return ok


def spawn(world, a, port, out):
    mp.start_processes(run, args=(world, a, port, out), nprocs=world, start_method="spawn", join=True)


def bench_run(rank, world, a, port):
    from rwkv_next import tp as tpm
    from rwkv_next.model import RWKV7TP
    T = tpm.init(rank, world, port, rank)
    dev = torch.device("cuda", rank)
    m = RWKV7TP.load(a.ckpt, T, dev, quant=a.quant, verbose=True, keep_int8=tuple(x for x in a.keep_int8.split(",") if x))
    Bs = [int(x) for x in a.B.split(",")]
    m.alloc_slots(max(Bs))
    if a.compile:
        m.compile(Bs)
    m.capture_decode(Bs)
    for B in Bs:
        ids = torch.zeros(B, dtype=torch.long, device=dev)
        for _ in range(5): m.decode(ids, B)
        torch.cuda.synchronize(); T.barrier()
        n = 100; t0 = time.perf_counter()
        for _ in range(n): m.decode(ids, B)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / n * 1e3
        if rank == 0:
            print(f"tp={world} quant={a.quant} B={B:3d}: {ms:6.2f} ms/step  {B / ms * 1e3:7.0f} tok/s", flush=True)
    mem = torch.cuda.max_memory_allocated(dev) / 1e9
    print(f"[rank {rank}] peak VRAM {mem:.2f} GB", flush=True)
    tpm.shutdown(T)
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--quant", default="bf16")
    ap.add_argument("--check", action="store_true"); ap.add_argument("--bench", action="store_true")
    ap.add_argument("--tp", type=int, default=2); ap.add_argument("--B", default="1,4,16")
    ap.add_argument("--compile", action="store_true"); ap.add_argument("--keep_int8", default="")
    ap.add_argument("--port", type=int, default=29613); ap.add_argument("--out", default="/tmp/rwkv_next_tp")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.check:
        ref = os.path.join(a.out, "harness.pt"); o1 = os.path.join(a.out, "tp1.pt"); o2 = os.path.join(a.out, "tp2.pt")
        use_ref = bool(os.environ.get("RWKV_REF_HARNESS"))
        if use_ref:
            p = mp.get_context("spawn").Process(target=harness_reference, args=(a, ref)); p.start(); p.join()
        else:
            print("(no RWKV_REF_HARNESS: skipping the comparison against the external reference model)")
        spawn(1, a, a.port, o1)
        spawn(2, a, a.port + 1, o2)
        A1, A2 = torch.load(o1), torch.load(o2)
        ok = compare("tp1 eager vs reference", {"prefill": A1["prefill"], "eager": A1["eager"]}, torch.load(ref)) if use_ref else True
        ok &= compare("tp1 graph vs tp1 eager", {"prefill": A1["prefill"], "eager": A1["graph"]}, {"prefill": A1["prefill"], "eager": A1["eager"]})
        ok &= compare("tp2 vs tp1", A2, A1)
        print("ALL OK" if ok else "SOME DIFF (bf16: small logit drift can flip near-ties; check the magnitudes)")
    if a.bench:
        mp.start_processes(bench_run, args=(a.tp, a, a.port + 2), nprocs=a.tp, start_method="spawn", join=True)
