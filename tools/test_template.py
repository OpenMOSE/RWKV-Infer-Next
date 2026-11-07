#!/usr/bin/env python3
"""Chat template + tokenizer + output parser, no GPU.

1. Our Jinja rendering == transformers' tokenizer.apply_chat_template (the model folder's own tokenizer, loaded with
   trust_remote_code) over conversations covering system prompts, multi-turn, thinking on/off, tools, assistant
   tool calls (arguments as objects and as JSON strings), tool results and content-part lists.
2. Our trie tokenizer == the model's tokenizer on every rendered prompt (token ids).
3. The output parser on reasoning / content / tool-call completions, including streaming in small pieces.

  python tools/test_template.py --model /path/to/RWKV7-G1l-7.2B-20261031
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rwkv_next.template import ChatTemplate
from rwkv_next.parse import StreamParser, extract_tool_calls, split_reasoning
from rwkv_next.tokenizer import RWKVTokenizer

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression",
                                      "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]
CASES = [
    ("single", [{"role": "user", "content": "Hello!"}], None),
    ("system", [{"role": "system", "content": "You are terse.  "}, {"role": "user", "content": "  日本の首都は？ "}], None),
    ("multi-turn", [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"},
                    {"role": "user", "content": "Tell me a joke.\n\nA short one."}], None),
    ("tools", [{"role": "user", "content": "What's the weather in Tokyo?"}], TOOLS),
    ("tools+system", [{"role": "system", "content": "Use tools when helpful."}, {"role": "user", "content": "12*7?"}], TOOLS),
    ("tool round trip (dict args)", [
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"temp_c": 18, "sky": "clear"}'}], TOOLS),
    ("tool round trip (string args, 2 calls, text)", [
        {"role": "user", "content": "Weather in Tokyo and Osaka?"},
        {"role": "assistant", "content": "Let me check.", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}},
            {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Osaka"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "18C clear"}, {"role": "tool", "tool_call_id": "c2", "content": "20C rain"}], TOOLS),
    ("content parts", [{"role": "user", "content": [{"type": "text", "text": "Describe "}, {"type": "text", "text": "RWKV."}]}], None),
]


def check_render(model):
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    ours = ChatTemplate.load(model)
    tok = RWKVTokenizer()
    bad = 0
    for name, msgs, tools in CASES:
        for thinking in (False, True):
            for agp in (True, False):
                kw = dict(tools=tools, add_generation_prompt=agp, thinking=thinking)
                # transformers flattens content parts itself only for some templates: give it the same text form
                hf_msgs = [dict(m, content="".join(p["text"] for p in m["content"])) if isinstance(m.get("content"), list) else m for m in msgs]
                ref = hf.apply_chat_template(hf_msgs, tokenize=False, **kw)
                got = ours.render(msgs, **kw)
                ids_ref = hf(ref, add_special_tokens=False)["input_ids"]       # the model's own tokenizer
                ids = tok.encode(got)
                ok = got == ref and list(ids) == list(ids_ref)
                bad += not ok
                if not ok or (thinking and agp and name in ("single", "tools")):
                    print(f"[{'ok' if ok else 'DIFF'}] {name} thinking={thinking} gen={agp}" + ("" if ok else f"\n  ours: {got!r}\n  hf  : {ref!r}\n  ids equal: {list(ids) == list(ids_ref)}"))
    print(f"render: {len(CASES) * 4 - bad}/{len(CASES) * 4} identical to transformers (text and token ids)")
    print("example (tools, thinking off):\n" + ours.render(CASES[3][1], tools=TOOLS, thinking=False))
    return bad


def check_parser():
    bad = 0

    def stream(prefix, text, tools, step=3):
        p = StreamParser(prefix, tools)
        R = C = ""
        for i in range(0, len(text), step):
            dr, dc = p.feed(text[i:i + step]); R += dr; C += dc
        dr, dc, calls, r, c = p.finish(); R += dr; C += dc
        return R, C, calls, r, c

    cases = [
        # (name, prefix, completion, tools, expect_reasoning, expect_content, expect_calls)
        ("thinking on", " <think", ">\nThe user greets me.\n</think>\nHello there!", None, "The user greets me.", "Hello there!", []),
        ("thinking off", " <think></think>\n", "Hello there!", None, None, "Hello there!", []),
        ("thinking cut by max_tokens", " <think", ">\nStill thinking about it", None, "Still thinking about it", "", []),
        ("no think tag", "", " Plain answer.", None, None, "Plain answer.", []),
        ("tool call", " <think></think>\n", '```json\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n```', TOOLS, None, "",
         [("get_weather", {"city": "Tokyo"})]),
        ("text + tool call", " <think></think>\n", 'Let me check.\n```json\n{"name":"calculator","arguments":{"expression":"12*7"}}\n```', TOOLS, None,
         "Let me check.", [("calculator", {"expression": "12*7"})]),
        ("two tool calls", " <think></think>\n", '```json\n{"name":"get_weather","arguments":{"city":"Tokyo"}}\n```\n```json\n{"name":"get_weather","arguments":{"city":"Osaka"}}\n```',
         TOOLS, None, "", [("get_weather", {"city": "Tokyo"}), ("get_weather", {"city": "Osaka"})]),
        ("thinking + tool call", " <think", '>\nNeed weather.\n</think>\n```json\n{"name":"get_weather","arguments":{"city":"Paris"}}\n```', TOOLS,
         "Need weather.", "", [("get_weather", {"city": "Paris"})]),
        ("unknown function stays content", " <think></think>\n", '```json\n{"name":"delete_all","arguments":{}}\n```', TOOLS, None,
         '```json\n{"name":"delete_all","arguments":{}}\n```', []),
        ("json without tools stays content", " <think></think>\n", '```json\n{"name":"get_weather","arguments":{"city":"Tokyo"}}\n```', None, None,
         '```json\n{"name":"get_weather","arguments":{"city":"Tokyo"}}\n```', []),
        ("code block is content", " <think></think>\n", "Here:\n```python\nprint(1)\n```\nDone.", TOOLS, None, "Here:\n```python\nprint(1)\n```\nDone.", []),
        ("hermes tag", " <think></think>\n", '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>', TOOLS, None, "",
         [("get_weather", {"city": "Paris"})]),
        ("text + hermes tag", " <think></think>\n", 'Sure.\n<tool_call>\n{"name":"calculator","arguments":{"expression":"2+2"}}\n</tool_call>', TOOLS, None,
         "Sure.", [("calculator", {"expression": "2+2"})]),
        ("bare json object", " <think></think>\n", '{"name": "calculator", "arguments": {"expression": "2+2"}}', TOOLS, None, "", [("calculator", {"expression": "2+2"})]),
    ]
    for name, prefix, text, tools, er, ec, ecalls in cases:
        for step in (1, 3, 1000):
            R, C, calls, r, c = stream(prefix, text, tools, step)
            got_calls = [(x["function"]["name"], json.loads(x["function"]["arguments"])) for x in calls]
            ok = (r == er) and (c == ec) and (got_calls == ecalls) and (R == (er or "")) and (C == ec)
            bad += not ok
            if not ok:
                print(f"[DIFF] parser {name} (step {step}): reasoning {r!r} / streamed {R!r}, content {c!r} / streamed {C!r}, calls {got_calls}")
    print(f"parser: {len(cases) * 3 - bad}/{len(cases) * 3} cases ok (each streamed in 1-char, 3-char and whole pieces)")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
    a = ap.parse_args()
    b = check_parser() + check_render(a.model)
    print("ALL PASSED" if not b else f"{b} FAILED")
    sys.exit(1 if b else 0)
