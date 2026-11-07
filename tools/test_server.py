#!/usr/bin/env python3
"""Concurrency smoke test against a running server: N mixed requests (short, long-prompt = chunked
prefill, greedy determinism, aborted streams), checks every one finishes and greedy outputs match
a solo run.   python tools/test_server.py [--url http://localhost:8000] [--n 12]"""
import argparse, json, threading, time, urllib.request

def post(url, body, timeout=600, abort_after=None):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not body.get("stream"):
            return json.loads(r.read()), time.time() - t0
        n = 0
        for line in r:
            if line.startswith(b"data:") and b'"content"' in line:
                n += 1
                if abort_after and n >= abort_after:
                    return {"aborted_after": n}, time.time() - t0
        return {"stream_chunks": n}, time.time() - t0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://localhost:8000"); ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    q = lambda c, **k: dict({"messages": [{"role": "user", "content": c}], "max_tokens": 120, "temperature": 0}, **k)
    solo, _ = post(a.url, q("RWKV について一文で説明してください。", thinking="off"))
    long_text = "以下の文章を三行で要約してください。\n" + ("吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。" * 120)
    jobs = []
    for i in range(a.n):
        if i % 4 == 0: jobs.append(("greedy-same", q("RWKV について一文で説明してください。", thinking="off")))
        elif i % 4 == 1: jobs.append(("long-prompt", q(long_text, max_tokens=80, thinking="off")))
        elif i % 4 == 2: jobs.append(("abort", dict(q("1 から 100 まで数えてください。", max_tokens=400), stream=True, temperature=1.0)))
        else: jobs.append(("sample", q(f"好きな食べ物を{i}個挙げてください。", temperature=1.0, top_p=0.5)))
    res = [None] * len(jobs)
    def run(i, kind, body):
        try:
            r, dt = post(a.url, body, abort_after=5 if kind == "abort" else None); res[i] = (kind, r, dt)
        except Exception as e:
            res[i] = (kind, {"error": str(e)}, 0)
    t0 = time.time()
    th = [threading.Thread(target=run, args=(i, k, b)) for i, (k, b) in enumerate(jobs)]
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.time() - t0
    ok = True
    for kind, r, dt in res:
        if "error" in r: ok = False; print(f"{kind:12s} ERROR {r['error']}"); continue
        if kind == "abort": print(f"{kind:12s} aborted after {r['aborted_after']} chunks ({dt:.1f}s)"); continue
        u, c = r["usage"], r["choices"][0]["message"]["content"]
        same = ""
        if kind == "greedy-same":
            s = c == solo["choices"][0]["message"]["content"]; same = " same-as-solo" if s else " DIFFERS-from-solo"
        print(f"{kind:12s} prompt {u['prompt_tokens']:5d} out {u['completion_tokens']:4d} {r['rwkv']['tok_s'] or 0:6.1f} tok/s ttft {r['rwkv']['ttft_s']:.2f}s{same} | {c[:50]!r}")
    print(f"wall {wall:.1f}s; status:", json.loads(urllib.request.urlopen(a.url + "/api/status").read())["stats"])
    print("OK" if ok else "FAILED")

if __name__ == "__main__":
    main()
