# -*- coding: utf-8 -*-
"""Model output -> OpenAI message parts: reasoning, content, tool calls. Incremental (streaming) and final.

The G1 chat template opens the assistant turn with ' <think' (thinking on: the model writes '>' + reasoning +
'</think>' + answer) or ' <think></think>\\n' (thinking off). The parser sees prefix + completion, so both cases
and templates that open no think block at all are handled the same way.

Tool calls follow the template's instruction (Hermes / Qwen style <tool_call>{...}</tool_call> tags are accepted
too: the model produces them now and then): a fenced block
    ```json
    {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    ```
One or more such blocks become `tool_calls`; text around them stays `content`. A block that does not parse, or
names a function that was not offered, is left in the content untouched. Without tools in the request nothing is
treated as a tool call.

Streaming: `feed(text)` returns the new reasoning / content deltas; content inside an unfinished ```json fence,
and a trailing piece that could still become '</think>' or a fence, is held back until it is decided.
`finish()` returns the last deltas and the tool calls.
"""

from __future__ import annotations

import json
import re
import uuid

THINK_OPEN = "<think"
THINK_CLOSE = "</think>"
FENCE_RE = re.compile(r"```(?:json|JSON|tool_call|tool)?[ \t]*\n(.*?)\n?```", re.S)
# the format the G1 template asks for is the fence above; models also produce the Hermes / Qwen style tag
TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
OPENERS = ("```", "<tool_call>")


def _holdback(text, tokens):
    """Length of the longest suffix of text that is a proper prefix of one of tokens."""
    best = 0
    for t in tokens:
        for k in range(min(len(t) - 1, len(text)), 0, -1):
            if text.endswith(t[:k]):
                best = max(best, k)
                break
    return best


def split_reasoning(text):
    """-> (reasoning or None, content, reasoning_closed). `text` = generation prefix + completion."""
    s = text.lstrip()
    if not s.startswith(THINK_OPEN):
        return None, text.lstrip(), True
    s = s[len(THINK_OPEN):]
    if s.startswith(">"):
        s = s[1:]
    j = s.find(THINK_CLOSE)
    if j < 0:
        return s.strip("\n"), "", False
    return s[:j].strip(), s[j + len(THINK_CLOSE):].lstrip(), True


def _as_call(obj, names):
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
    name = fn.get("name")
    if not isinstance(name, str) or (names and name not in names):
        return None
    args = fn.get("arguments", fn.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}}


def extract_tool_calls(content, tools):
    """-> (content without the tool-call blocks, [tool_calls])."""
    if not tools:
        return content, []
    names = {(t.get("function") or t).get("name") for t in tools if isinstance(t, dict)}
    calls, keep, pos = [], [], 0
    blocks = sorted(list(FENCE_RE.finditer(content)) + list(TAG_RE.finditer(content)), key=lambda m: m.start())
    for m in blocks:
        if m.start() < pos:
            continue
        body = m.group(1).strip()
        got = []
        try:
            obj = json.loads(body)
            for o in (obj if isinstance(obj, list) else [obj]):
                c = _as_call(o, names)
                if c is None:
                    got = []
                    break
                got.append(c)
        except Exception:
            got = []
        if got:
            keep.append(content[pos:m.start()]); pos = m.end(); calls += got
    keep.append(content[pos:])
    if not calls:                                  # a bare JSON object as the whole answer
        s = content.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                c = _as_call(json.loads(s), names)
                if c is not None:
                    return "", [c]
            except Exception:
                pass
        return content, []
    return "".join(keep).strip(), calls


class StreamParser:
    def __init__(self, prefix="", tools=None, reasoning=True, lstrip=True):
        """reasoning=False, lstrip=False: a continued assistant message (continue_final_message) -- the text
        goes on from the old content as is, leading whitespace included."""
        self.prefix, self.tools, self.reasoning, self.lstrip = prefix, tools or [], reasoning, lstrip
        self.text = ""
        self.sent_r = 0          # reasoning chars already emitted
        self.sent_c = 0          # content chars already emitted

    def _view(self, final):
        r, c, closed = split_reasoning(self.prefix + self.text) if self.reasoning else (None, self.text.lstrip() if self.lstrip else self.text, True)
        if r is not None and not closed and not final:
            r = r[:len(r) - _holdback(r, [THINK_CLOSE])].rstrip()   # trailing space may precede '</think>'
        if self.tools and not final:
            k = c.rfind("```")
            t = c.rfind("<tool_call>")
            if k >= 0 and c.count("```") % 2 == 1:      # inside an open fence: hold it until it closes
                c = c[:k]
            elif t >= 0 and c.find("</tool_call>", t) < 0:   # inside an open <tool_call> tag
                c = c[:t]
            elif not c.endswith("```"):
                c = c[:len(c) - _holdback(c, ["```", "<tool_call>"])]   # a trailing piece may become an opener
            c = c.rstrip()                              # whitespace before a tool block is dropped with it
            if c.lstrip().startswith("{"):              # may be a bare JSON tool call: decide at the end
                c = ""
        return (r or ""), c, closed

    def feed(self, piece):
        """-> (reasoning_delta, content_delta)"""
        self.text += piece
        r, c, _ = self._view(final=False)
        if self.tools:
            c, _ = extract_tool_calls(c, self.tools)
        dr, dc = r[self.sent_r:], c[self.sent_c:]
        # content already emitted is never retracted: only extend
        if not c.startswith(c[:self.sent_c]):
            dc = ""
        self.sent_r = max(self.sent_r, len(r)); self.sent_c = max(self.sent_c, len(c))
        return dr, dc

    def finish(self):
        """-> (reasoning_delta, content_delta, tool_calls, full_reasoning or None, full_content)"""
        r, c, closed = self._view(final=True)
        calls = []
        if self.tools:
            c, calls = extract_tool_calls(c, self.tools)
        dr, dc = r[self.sent_r:], c[self.sent_c:]
        return dr, dc, calls, (r if (self.reasoning and r.strip()) else None), c
