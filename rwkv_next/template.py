# -*- coding: utf-8 -*-
"""Chat templates (Jinja), rendered exactly like transformers' apply_chat_template.

Source of the template (setting RWKV_CHAT_TEMPLATE):
    auto      chat_template.jinja in the model folder, else tokenizer_config.json["chat_template"], else builtin
    builtin   the RWKV-7 G1 template shipped with this engine (identical to the G1l release's chat_template.jinja)
    <path>    a .jinja file

The environment matches transformers (ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks, loopcontrols,
the `{% generation %}` block tag, tojson without ASCII escaping, raise_exception, strftime_now), so a template
that works with apply_chat_template renders byte-for-byte the same here (tools/test_template.py checks this
against transformers itself).
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .weights import model_dir

BUILTIN = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "chat_template.jinja"), encoding="utf-8").read()


class _Generation(jinja2.ext.Extension):
    """`{% generation %}...{% endgeneration %}` (transformers' assistant-mask marker): renders its body."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_render"), [], [], body).set_lineno(lineno)

    def _render(self, caller):
        return caller()


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _raise(message):
    raise jinja2.exceptions.TemplateError(message)


def _env():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[_Generation, jinja2.ext.loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env


def find_template(model_path, setting="auto"):
    """-> (template source, where it came from)."""
    setting = (setting or "auto").strip()
    if setting == "builtin":
        return BUILTIN, "builtin"
    if setting != "auto":
        p = os.path.expanduser(setting)
        return open(p, encoding="utf-8").read(), p
    if model_path:
        d = model_dir(model_path)
        p = os.path.join(d, "chat_template.jinja")
        if os.path.exists(p):
            return open(p, encoding="utf-8").read(), p
        tc = os.path.join(d, "tokenizer_config.json")
        if os.path.exists(tc):
            ct = json.load(open(tc)).get("chat_template")
            if isinstance(ct, str) and ct.strip():
                return ct, tc
    return BUILTIN, "builtin"


def normalize_messages(messages):
    """OpenAI request messages -> plain dicts the template understands (content parts joined to text;
    tool_call arguments kept as given: the G1 template accepts JSON strings and objects)."""
    out = []
    for m in messages or []:
        m = dict(m)
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type", "text") in ("text", "input_text"))
        elif c is None:
            m["content"] = ""
        out.append(m)
    return out


class ChatTemplate:
    def __init__(self, source: str, origin: str = "builtin", bos_token: str = ""):
        self.source, self.origin, self.bos_token = source, origin, bos_token
        self._t = _env().from_string(source)

    @classmethod
    def load(cls, model_path, setting="auto"):
        src, origin = find_template(model_path, setting)
        return cls(src, origin)

    def render(self, messages, tools=None, thinking=False, add_generation_prompt=True, **kwargs) -> str:
        return self._t.render(messages=normalize_messages(messages), tools=tools or [], thinking=bool(thinking),
                              add_generation_prompt=add_generation_prompt, bos_token=self.bos_token, **kwargs)

    @staticmethod
    def generation_prefix(prompt: str) -> str:
        """What the template already wrote of the assistant turn (e.g. ' <think' or ' <think></think>\\n'):
        the output parser prepends it to the completion so the reasoning block can be found."""
        i = prompt.rfind("Assistant:")
        return prompt[i + len("Assistant:"):] if i >= 0 else ""
