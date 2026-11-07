#!/usr/bin/env python3
"""Scheduler correctness (in-process, TP1): many greedy requests with random prompt lengths arrive
at random times into a small slot pool (forces waiting, chunked prefill, slot moves, graph padding,
aborts). Every finished request is then re-run ALONE and compared token by token.

bf16 results depend on the batch composition (the matmul for M=1 and M=8 rounds differently), so an
exact match is not required: at the first differing token the SOLO run's top-2 logit margin must be
small (a near-tie). A slot/state bug shows up as divergence at large margins.

  python tools/test_engine.py --ckpt <plain RWKV-7 .pth> [--quant bf16] [--slots 6] [--n 24]
"""
import argparse, os, queue, random, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--quant", default="bf16")
    ap.add_argument("--slots", type=int, default=6); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=48); ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--graphs", default="1,2,4,8")
    ap.add_argument("--mtp", default="", help="sidecar: batched run uses speculative decoding, solo runs don't")
    ap.add_argument("--mtp_k", default="8:3")
    ap.add_argument("--chat", action="store_true", help="real chat prompts instead of random words")
    ap.add_argument("--compile", action="store_true"); ap.add_argument("--eager_spec", action="store_true")
    a = ap.parse_args()
    from rwkv_next import tp as tpm
    from rwkv_next.model import RWKV7TP
    from rwkv_next.sampler import Sampler
    from rwkv_next.engine import Executor, Scheduler
    from rwkv_next.tokenizer import RWKVTokenizer
    T = tpm.init(0, 1, 0, 0)
    dev = torch.device("cuda", 0)
    m = RWKV7TP.load(a.ckpt, T, dev, quant=a.quant)
    m.alloc_slots(a.slots)
    pool = None
    gs = [int(x) for x in a.graphs.split(",") if int(x) <= a.slots]      # a graph wider than the pool cannot exist
    if a.mtp:
        from rwkv_next.drafter import DraftPool
        from rwkv_next.mtp import parse_k_rules
        rules = parse_k_rules(a.mtp_k)
        pool = DraftPool(a.mtp, m, a.slots, K_max=max(k for _, k in rules), compile=a.compile)
        m.tap_layers = pool.taps
    if a.compile:
        m.compile(gs)
    m.capture_decode(gs, verbose=False)
    specg = None
    if pool is not None and not a.eager_spec:
        from rwkv_next.spec import SpecGraphs
        specg = SpecGraphs(m, pool)
        for b in gs:
            specg.capture(b, rules[0][1], verbose=True)
    tok = RWKVTokenizer()
    rng = random.Random(1)
    words = "the of and to in a is that for it as was with be by on not he this are or his from at which but have an they you were her she there".split()
    prompts = [tok.encode(" ".join(rng.choice(words) for _ in range(rng.randint(3, 160)))) for _ in range(a.n)]
    if a.chat:
        qs = ["日本の四季について説明してください。", "Explain how a refrigerator works.", "Pythonで二分探索を書いてください。",
              "What are the main causes of the French Revolution?", "次の英文を日本語に翻訳してください。\nThe committee decided to postpone the vote.",
              "東京から大阪への行き方を教えてください。", "Write a short poem about the sea.", "RWKVとは何ですか？"]
        prompts = [tok.encode(f"User: {qs[i % len(qs)]}\n\nAssistant: <think>\n</think>") for i in range(a.n)]
    greedy = {"temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0, "presence_penalty": 0, "frequency_penalty": 0,
              "penalty_decay": 1, "max_tokens": a.max_new, "stop_token_ids": [-1]}

    def run(reqs, aborts=(), arrivals=None, spec=False):
        """reqs: list of (rid, prompt). Returns {rid: tokens}."""
        in_q, out_q = queue.Queue(), queue.Queue()
        cfg = {"PREFILL_CHUNK": a.chunk, "MAX_PROMPT_TOKENS": 10 ** 6, "MTP_K": a.mtp_k if spec else "0:0"}
        ex = Executor(m, Sampler(a.slots, m.V, dev), pool if spec else None)
        ex.specg = specg if spec else None
        sch = Scheduler(ex, None, cfg, in_q, out_q, 0)
        out = {rid: [] for rid, _ in reqs}
        done = set()

        def feeder():
            for k, (rid, p) in enumerate(reqs):
                in_q.put(("add", rid, p, dict(greedy)))
                if arrivals:
                    time.sleep(arrivals[k])
                if rid in aborts:
                    time.sleep(0.05); in_q.put(("abort", rid))
        th = threading.Thread(target=feeder); th.start()
        t = threading.Thread(target=sch.run); t.start()
        want = {rid for rid, _ in reqs} - set(aborts)
        while not want <= done:
            msg = out_q.get(timeout=120)
            if msg[0] == "tokens":
                for rid, ids, fin, _ in msg[1]:
                    out[rid] += ids
                    if fin:
                        done.add(rid)
        th.join(); in_q.put(("stop",)); t.join()
        run.last = sch
        return out

    rids = [f"r{i}" for i in range(a.n)]
    aborts = {r for r in (rids[3:4] + rids[10:11])} if a.n > 10 else set()
    arrivals = [rng.choice([0, 0, 0.002, 0.02]) for _ in rids]
    t0 = time.time()
    batched = run(list(zip(rids, prompts)), aborts, arrivals, spec=bool(a.mtp))
    print(f"batched run{' (MTP ' + a.mtp_k + ')' if a.mtp else ''}: {a.n} requests into {a.slots} slots, chunk {a.chunk}: {time.time() - t0:.1f}s", flush=True)
    if a.mtp:
        sc = run.last
        print(f"  mtp tokens/row/step: {sc.mtp_total[1] / max(1, sc.mtp_total[0]):.2f} over {sc.mtp_total[0]} row-steps", flush=True)

    # solo reference with logits for the margin check
    bad = exact = near = 0
    for rid, p in zip(rids, prompts):
        if rid in aborts:
            continue
        solo = run([(rid, p)])[rid]
        b = batched[rid]
        if b == solo:
            exact += 1
            continue
        j = next((k for k in range(min(len(b), len(solo))) if b[k] != solo[k]), min(len(b), len(solo)))
        # margin of the solo distribution at the divergence: replay solo greedily up to j
        m.reset_slots(torch.tensor([0], device=dev))
        ids = p + solo[:j]
        h = Executor(m).prefill(ids, [len(ids)], [0], [True])
        lg = m.logits(h)[0].float()
        top2 = lg.topk(2).values
        margin = (top2[0] - top2[1]).item()
        tag = "near-tie" if margin < 0.5 else "BUG?"
        near += margin < 0.5; bad += margin >= 0.5
        print(f"{rid}: prompt {len(p):4d}, diverges at token {j}: solo margin {margin:.3f} -> {tag}", flush=True)
    print(f"exact {exact}, near-tie divergences {near}, large-margin divergences {bad}", flush=True)
    print("PASS" if bad == 0 else "FAIL", flush=True)
    os._exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
