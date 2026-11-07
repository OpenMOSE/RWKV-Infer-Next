#!/usr/bin/env python3
"""Virtual OpenAI client: end-to-end tests of a running RWKV-Infer-Next server (stdlib only).

Covers what an agent / chat client relies on:
  basic chat (non-stream / stream), thinking on / off and every way to ask for it, reasoning_content placement,
  tool calls (non-stream / stream, arguments as JSON strings, finish_reason), the full tool round trip
  (assistant tool_calls + role=tool results -> final answer), parallel tool calls, tool_choice none / named,
  multi-turn history with reasoning, stop / max_tokens, the raw completions endpoint, client disconnects,
  concurrency, and error handling.

  python tools/test_client.py [--url http://localhost:8000] [--key <api key>] [--only name,...] [-v]
"""
import argparse, json, sys, threading, time, urllib.error, urllib.request

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                                                     "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression.",
                                      "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]


class Client:
    def __init__(self, url, key=None):
        self.url, self.key = url.rstrip("/"), key

    def _req(self, path, body=None, method=None):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        data = json.dumps(body).encode() if body is not None else None
        return urllib.request.Request(self.url + path, data=data, headers=h, method=method or ("POST" if body is not None else "GET"))

    def get(self, path):
        return json.loads(urllib.request.urlopen(self._req(path), timeout=60).read())

    def chat(self, **body):
        return json.loads(urllib.request.urlopen(self._req("/v1/chat/completions", body), timeout=600).read())

    def stream(self, path="/v1/chat/completions", abort_after=None, **body):
        """-> list of parsed chunks (data: lines), stops early after abort_after content chunks."""
        body["stream"] = True
        out = []
        with urllib.request.urlopen(self._req(path, body), timeout=600) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                d = line[5:].strip()
                if d == b"[DONE]":
                    out.append("[DONE]"); break
                out.append(json.loads(d))
                if abort_after and len(out) >= abort_after:
                    break
        return out


def assemble(chunks):
    """OpenAI stream chunks -> (content, reasoning, tool_calls, finish_reason, usage)."""
    content = reasoning = ""
    calls, fin, usage = {}, None, None
    for c in chunks:
        if c == "[DONE]" or "choices" not in c:
            continue
        ch = c["choices"][0]
        d = ch.get("delta") or {}
        content += d.get("content") or ""
        reasoning += d.get("reasoning_content") or ""
        for tc in d.get("tool_calls") or []:
            slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
            slot["id"] = tc.get("id") or slot["id"]
            f = tc.get("function") or {}
            slot["name"] += f.get("name") or ""
            slot["arguments"] += f.get("arguments") or ""
        fin = ch.get("finish_reason") or fin
        usage = c.get("usage") or usage
    return content, reasoning, [calls[i] for i in sorted(calls)], fin, usage


TESTS = []
SOFT = set()          # model-behaviour checks: reported as WARN (quality of the model), not engine failures


def test(fn):
    TESTS.append(fn)
    return fn


def soft(fn):
    SOFT.add(fn.__name__)
    return fn


def user(t):
    return [{"role": "user", "content": t}]


G = dict(temperature=0, max_tokens=200)


@test
def models(c):
    m = c.get("/v1/models")
    assert m["object"] == "list" and m["data"] and m["data"][0]["id"], m
    return m["data"][0]["id"]


@test
def chat_basic(c):
    r = c.chat(messages=user("What is the capital of France? Answer in one word."), **G)
    msg = r["choices"][0]["message"]
    assert msg["role"] == "assistant" and "paris" in msg["content"].lower(), msg
    assert r["choices"][0]["finish_reason"] == "stop", r["choices"][0]
    assert "reasoning_content" not in msg, "thinking is off by default: no reasoning expected"
    assert r["usage"]["completion_tokens"] > 0
    return msg["content"]


@test
def chat_stream(c):
    ch = c.stream(messages=user("Count from 1 to 5, separated by commas."), **G)
    content, reasoning, calls, fin, usage = assemble(ch)
    assert ch[-1] == "[DONE]" and fin == "stop" and usage, (fin, usage)
    assert "1" in content and "5" in content and not reasoning and not calls, content
    assert ch[0]["choices"][0]["delta"].get("role") == "assistant"
    return content


@test
def thinking_on(c):
    r = c.chat(messages=user("Is 91 a prime number? Answer yes or no."), thinking=True, temperature=0, max_tokens=600)
    msg = r["choices"][0]["message"]
    assert msg.get("reasoning_content"), f"expected reasoning, got {msg}"
    assert "<think" not in (msg["content"] or ""), msg
    return {"reasoning": msg["reasoning_content"][:80], "content": (msg["content"] or "")[:80], "finish": r["choices"][0]["finish_reason"]}


@test
def thinking_stream_order(c):
    ch = c.stream(messages=user("What is 17 * 3?"), thinking=True, temperature=0, max_tokens=600)
    kinds = []
    for x in ch:
        if x == "[DONE]" or "choices" not in x:
            continue
        d = x["choices"][0]["delta"]
        if d.get("reasoning_content"):
            kinds.append("r")
        if d.get("content"):
            kinds.append("c")
    assert kinds and kinds[0] == "r", kinds[:10]
    if "c" in kinds:
        assert "r" not in kinds[kinds.index("c"):], "reasoning after content"
    content, reasoning, _, fin, _ = assemble(ch)
    assert reasoning and "<think" not in content, (reasoning[:60], content[:60])
    return {"reasoning_chunks": kinds.count("r"), "content_chunks": kinds.count("c"), "finish": fin}


@test
def thinking_switches(c):
    """every spelling of 'thinking on/off' a client may send"""
    out = {}
    for name, kw, want in [("thinking=false", {"thinking": False}, False),
                           ("chat_template_kwargs.thinking", {"chat_template_kwargs": {"thinking": True}}, True),
                           ("chat_template_kwargs.enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}, False),
                           ("reasoning_effort=high", {"reasoning_effort": "high"}, True),
                           ("reasoning_effort=none", {"reasoning_effort": "none"}, False)]:
        r = c.chat(messages=user("Say hi."), temperature=0, max_tokens=8, **kw)
        got = r["rwkv"]["thinking"]
        assert got == want, (name, got)
        out[name] = got
    return out


@test
def tool_call(c):
    r = c.chat(messages=user("What's the weather in Tokyo right now?"), tools=TOOLS, **G)
    ch = r["choices"][0]
    assert ch["finish_reason"] == "tool_calls", ch
    tc = ch["message"]["tool_calls"]
    assert tc and tc[0]["type"] == "function" and tc[0]["id"].startswith("call_"), tc
    assert tc[0]["function"]["name"] == "get_weather", tc
    args = json.loads(tc[0]["function"]["arguments"])                 # arguments must be a JSON STRING
    assert "tokyo" in args.get("city", "").lower(), args
    return tc[0]["function"]


@test
def tool_call_stream(c):
    ch = c.stream(messages=user("Use the calculator to compute 1234 * 5678."), tools=TOOLS, **G)
    content, reasoning, calls, fin, _ = assemble(ch)
    assert fin == "tool_calls" and calls, (fin, content)
    assert calls[0]["name"] == "calculator" and calls[0]["id"], calls
    args = json.loads(calls[0]["arguments"])
    assert "1234" in args["expression"].replace(" ", "") and "5678" in args["expression"].replace(" ", ""), args
    assert "```" not in content, "the tool-call block must not leak into content"
    return calls[0]


@test
def tool_round_trip(c):
    msgs = user("What's the weather in Paris?")
    r = c.chat(messages=msgs, tools=TOOLS, **G)
    m = r["choices"][0]["message"]
    assert m.get("tool_calls"), m
    msgs = msgs + [m, {"role": "tool", "tool_call_id": m["tool_calls"][0]["id"],
                        "content": json.dumps({"city": "Paris", "temperature_c": 21, "condition": "sunny"})}]
    r2 = c.chat(messages=msgs, tools=TOOLS, **G)
    ch = r2["choices"][0]
    assert ch["finish_reason"] == "stop" and not ch["message"].get("tool_calls"), ch
    assert "21" in ch["message"]["content"], ch["message"]["content"]
    return ch["message"]["content"][:120]


@test
def tool_choice_none(c):
    r = c.chat(messages=user("What's the weather in Tokyo?"), tools=TOOLS, tool_choice="none", **G)
    ch = r["choices"][0]
    assert ch["finish_reason"] != "tool_calls" and not ch["message"].get("tool_calls"), ch
    return ch["message"]["content"][:80]


@test
def tool_choice_named(c):
    r = c.chat(messages=user("Please compute 2 + 2 for me."), tools=TOOLS,
               tool_choice={"type": "function", "function": {"name": "calculator"}}, **G)
    tc = r["choices"][0]["message"].get("tool_calls")
    assert tc and tc[0]["function"]["name"] == "calculator", r["choices"][0]
    return tc[0]["function"]


@test
@soft
def no_tool_needed(c):
    r = c.chat(messages=user("Say hello in French."), tools=TOOLS, **G)
    ch = r["choices"][0]
    assert not ch["message"].get("tool_calls") and ch["message"]["content"], ch
    return ch["message"]["content"][:60]


@test
def multi_turn_with_reasoning_history(c):
    msgs = [{"role": "user", "content": "My name is Aiko. Remember it."},
            {"role": "assistant", "content": "Got it, Aiko.", "reasoning_content": "The user told me their name."},
            {"role": "user", "content": "What is my name?"}]
    r = c.chat(messages=msgs, **G)
    t = r["choices"][0]["message"]["content"]
    assert "aiko" in t.lower(), t
    p = c._req("/api/tokenize", {"messages": msgs, "return_prompt": True})
    prompt = json.loads(urllib.request.urlopen(p).read())["prompt"]
    assert "The user told me their name" not in prompt, "reasoning from history must not be sent back"
    return t[:60]


@test
@soft
def system_prompt(c):
    r = c.chat(messages=[{"role": "system", "content": "Always answer in Japanese."}] + user("What color is the sky?"), **G)
    t = r["choices"][0]["message"]["content"]
    assert any("぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in t), t
    return t[:60]


@test
def stop_and_length(c):
    r = c.chat(messages=user("List the numbers 1 to 20, one per line."), stop=["7"], **G)
    t, f = r["choices"][0]["message"]["content"], r["choices"][0]["finish_reason"]
    assert "7" not in t and "6" in t and f == "stop", (t, f)
    r = c.chat(messages=user("Write a long story about a cat."), temperature=0, max_tokens=12)
    assert r["choices"][0]["finish_reason"] == "length" and r["usage"]["completion_tokens"] == 12, r["usage"]
    return {"stop": t[-12:], "length": r["usage"]}


@test
def raw_completion(c):
    body = {"prompt": "The capital of Japan is", "max_tokens": 5, "temperature": 0}
    r = json.loads(urllib.request.urlopen(c._req("/v1/completions", body)).read())
    assert "tokyo" in r["choices"][0]["text"].lower(), r
    ch = c.stream("/v1/completions", prompt="1, 2, 3, 4,", max_tokens=6, temperature=0)
    t = "".join(x["choices"][0]["text"] for x in ch if x != "[DONE]")
    assert "5" in t, t
    return r["choices"][0]["text"]


@test
def disconnect_then_ok(c):
    c.stream(messages=user("Write a very long essay about the ocean."), temperature=1.0, max_tokens=2000, abort_after=5)
    time.sleep(0.5)
    st = c.get("/api/status")
    r = c.chat(messages=user("Say OK."), temperature=0, max_tokens=5)
    assert r["choices"][0]["message"]["content"], r
    return st["stats"]


@test
def concurrency(c):
    jobs = [("plain", dict(messages=user(f"Name a fruit that starts with the letter {ch}."), **G)) for ch in "ABCDEF"]
    jobs += [("tool", dict(messages=user(f"What's the weather in {city}?"), tools=TOOLS, **G)) for city in ("Rome", "Seoul")]
    jobs += [("think", dict(messages=user("What is 6*7?"), thinking=True, temperature=0, max_tokens=400))]
    res = [None] * len(jobs)

    def go(i, body):
        try:
            res[i] = c.chat(**body)
        except Exception as e:
            res[i] = e
    th = [threading.Thread(target=go, args=(i, b)) for i, (_, b) in enumerate(jobs)]
    t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
    for (kind, _), r in zip(jobs, res):
        assert not isinstance(r, Exception), r
        m = r["choices"][0]["message"]
        if kind == "tool":
            assert m.get("tool_calls"), m
        elif kind == "think":
            assert m.get("reasoning_content"), m
        else:
            assert m["content"], m
    return {"requests": len(jobs), "wall_s": round(time.time() - t0, 2)}


@test
def errors(c):
    for body, code in [({"messages": [{"role": "user", "content": "x" * 10}], "max_tokens": 2}, 200)]:
        try:
            urllib.request.urlopen(c._req("/v1/chat/completions", body)); got = 200
        except urllib.error.HTTPError as e:
            got = e.code
        assert got == code, (body, got)
    try:
        urllib.request.urlopen(c._req("/v1/chat/completions", {"messages": "not a list"}))
        raise AssertionError("bad messages accepted")
    except urllib.error.HTTPError as e:
        assert e.code in (400, 422), e.code
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000"); ap.add_argument("--key", default=None)
    ap.add_argument("--only", default=""); ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    c = Client(a.url, a.key)
    only = set(x for x in a.only.split(",") if x)
    fails = warns = 0
    for fn in TESTS:
        if only and fn.__name__ not in only:
            continue
        t0 = time.time()
        try:
            out = fn(c)
            print(f"PASS  {fn.__name__:34s} {time.time() - t0:5.1f}s  {json.dumps(out, ensure_ascii=False)[:150] if a.v else ''}", flush=True)
        except AssertionError as e:
            if fn.__name__ in SOFT:
                warns += 1
                print(f"WARN  {fn.__name__:34s} {time.time() - t0:5.1f}s  (model behaviour) {str(e)[:300]}", flush=True)
                continue
            fails += 1
            print(f"FAIL  {fn.__name__:34s} {time.time() - t0:5.1f}s  {str(e)[:400]}", flush=True)
        except Exception as e:
            fails += 1
            print(f"ERROR {fn.__name__:34s} {time.time() - t0:5.1f}s  {type(e).__name__}: {str(e)[:300]}", flush=True)
    n = len([f for f in TESTS if not only or f.__name__ in only])
    print(f"{n - fails - warns} passed, {fails} failed, {warns} model-behaviour warnings")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
