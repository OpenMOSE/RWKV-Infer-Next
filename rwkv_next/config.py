# -*- coding: utf-8 -*-
"""Settings: one schema for .env, the process environment and the WebUI settings page.

Precedence (low -> high): schema default < .env file < process environment at launch.
The WebUI writes the .env file (only the keys it manages, comments and other lines kept)
and, for keys marked `restart`, asks the front to restart the engine processes; the
other keys apply to the next request immediately.

Every key is `RWKV_<NAME>`. The .env path is RWKV_ENV_FILE or ./.env.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Setting:
    key: str
    type: str                 # str | int | float | bool | choice | text
    default: Any
    group: str
    label: str
    help: str = ""
    choices: tuple = ()
    restart: bool = False     # True: needs an engine restart to apply
    secret: bool = False
    server: bool = False      # True: read only when the server process starts (python -m rwkv_next)


S = Setting
SCHEMA: list[Setting] = [
    # ---- model ----
    S("MODEL_PATH", "str", "", "model", "Model",
      "A Hugging Face repository id (e.g. RWKV/RWKV7-G1l-7.2B-20261031, optionally @revision; downloaded into the HF "
      "cache), a local model folder (safetensors + chat_template.jinja), a .safetensors file or a plain RWKV-7 .pth", restart=True),
    S("MODEL_NAME", "str", "", "model", "Model name", "Name reported by the API (`model` field). Empty: the folder / file name"),
    S("QUANT", "choice", "int6", "model", "Weight quantisation",
      "int6: 74% of int8's memory and bandwidth. int8: near-lossless. bf16: none", choices=("int6", "int8", "bf16"), restart=True),
    S("INT6_KEEP_INT8", "str", "", "model", "Parts kept at int8 under int6",
      "att / ffn / att,ffn. Keeping att at int8 removes ~60% of int6's error (+0.6 GB on a 7B)", restart=True),
    S("QUANT_GROUP_K", "choice", "128", "model", "Quantisation group size", "64 is slightly more accurate (+0.1 GB on a 7B)",
      choices=("128", "64"), restart=True),
    # ---- parallel ----
    S("TP", "int", 1, "parallel", "Tensor parallel size", "GPUs per model; must divide the head count and the FFN width", restart=True),
    S("DEVICES", "str", "", "parallel", "GPUs", "e.g. 0,1. Empty: 0..TP-1", restart=True),
    # ---- engine ----
    S("MAX_BATCH", "int", 32, "engine", "Concurrent sequences (slots)", "Rows of the state pool", restart=True),
    S("GRAPH_BATCH_SIZES", "str", "1,2,4,8,16,32", "engine", "CUDA graph batch sizes",
      "Decode steps of these sizes replay one graph; other sizes round up", restart=True),
    S("COMPILE", "bool", True, "engine", "torch.compile", "Fuses the element-wise kernels (start-up takes longer)", restart=True),
    S("PREFILL_CHUNK", "int", 2048, "engine", "Prefill chunk (tokens)", "Long prompts are prefilled in chunks of this size", restart=True),
    S("PREFILL_CHUNK_BUSY", "int", 512, "engine", "Prefill chunk while decoding",
      "Chunk size while other rows are decoding: smaller keeps their streams smooth", restart=True),
    S("MAX_PROMPT_TOKENS", "int", 32768, "engine", "Max prompt tokens", "", restart=True),
    # ---- mtp ----
    S("MTP_PATH", "str", "auto", "mtp", "MTP sidecar",
      "auto: the model folder's mtp/ (or *.mtp.safetensors / *.mtp.pt), else the one for this model in MTP_REPO; a path or "
      "an HF repository id; empty / off to disable", restart=True),
    S("MTP_FORCE", "bool", False, "mtp", "Force a sidecar from another model",
      "For new models without their own sidecar: take the closest sidecar of the same width from MTP_REPO and accept a "
      "layer-count mismatch. The output is unchanged (drafts are verified); only the speed-up depends on the match", restart=True),
    S("MTP_REPO", "str", "OpenMOSE/RWKV7-MTP-Exp", "mtp", "MTP sidecar repository",
      "Hugging Face repository searched by MTP_PATH=auto when the model folder has no sidecar. Empty: never download", restart=True),
    S("MTP_K", "str", "4:3,16:2,0:0", "mtp", "Drafts per step (K) by batch size",
      "rows<=4: K=3, rows<=16: K=2, larger: off (0)", restart=True),
    S("MTP_DRAFT_VOCAB", "int", 16384, "mtp", "Draft vocabulary", "", restart=True),
    # ---- chat ----
    S("CHAT_TEMPLATE", "str", "auto", "chat", "Chat template",
      "auto: the model folder's chat_template.jinja, else tokenizer_config.json's chat_template, else the builtin RWKV-7 G1 template; builtin; or a path to a .jinja file"),
    S("THINKING_DEFAULT", "bool", False, "chat", "Thinking by default",
      "Used when a request does not say (thinking / chat_template_kwargs.thinking / reasoning_effort)"),
    S("REASONING_FORMAT", "choice", "separate", "chat", "Reasoning in responses",
      "separate: `reasoning_content` field (OpenAI-compatible clients that understand reasoning); inline: <think>...</think> in content",
      choices=("separate", "inline")),
    S("SYSTEM_PROMPT", "text", "", "chat", "Default system prompt", "Added when a request has no system message"),
    S("STOP", "str", "\\n\\nUser:", "chat", "Stop strings", "Comma separated, \\n escaped"),
    # ---- sampling defaults ----
    S("TEMPERATURE", "float", 1.0, "sampling", "temperature", "Used when a request does not set it"),
    S("TOP_P", "float", 0.5, "sampling", "top_p", ""),
    S("TOP_K", "int", 0, "sampling", "top_k", "0 = off"),
    S("MIN_P", "float", 0.0, "sampling", "min_p", ""),
    S("PRESENCE_PENALTY", "float", 0.0, "sampling", "presence penalty", ""),
    S("FREQUENCY_PENALTY", "float", 0.0, "sampling", "frequency penalty", ""),
    S("PENALTY_DECAY", "float", 0.996, "sampling", "penalty decay", "Per-token decay of the penalty counts"),
    S("MAX_TOKENS", "int", 4096, "sampling", "max_tokens", "Generation limit when a request does not set one"),
    # ---- server ----
    S("HOST", "str", "0.0.0.0", "server", "Listen address", "Applied when the server process is restarted", server=True),
    S("PORT", "int", 8000, "server", "Port", "Applied when the server process is restarted", server=True),
    S("API_KEY", "str", "", "server", "API key", "Empty: no auth. Set: requests need `Authorization: Bearer <key>`", secret=True),
    S("ALLOW_REMOTE_SETTINGS", "bool", False, "server", "Allow settings changes from other hosts",
      "Off: saving settings and restarting the engine are only allowed from localhost"),
]
BY_KEY = {s.key: s for s in SCHEMA}
GROUPS = [("model", "Model"), ("parallel", "Parallelism"), ("engine", "Engine"), ("mtp", "MTP speculative decoding"),
          ("chat", "Chat format"), ("sampling", "Sampling defaults"), ("server", "Server")]


def env_path() -> str:
    return os.environ.get("RWKV_ENV_FILE") or os.path.join(os.getcwd(), ".env")


def _parse_env_file(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$", line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        else:
            v = v.split(" #")[0].strip()
        out[k] = v
    return out


def _coerce(s: Setting, v):
    if v is None:
        return s.default
    if s.type == "int":
        return int(v)
    if s.type == "float":
        return float(v)
    if s.type == "bool":
        return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
    if s.type == "text":
        return str(v).replace("\\n", "\n")          # stored escaped in .env, held decoded
    if s.type == "choice":
        v = str(v)
        if v not in s.choices:
            raise ValueError(f"{s.key}: {v!r} not in {s.choices}")
        return v
    return str(v)


class Settings:
    """Current values (thread-safe). `launch_env` = RWKV_* keys present in the process
    environment at start: they win over the .env file and are shown as locked in the UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self.launch_env = {k[5:]: v for k, v in os.environ.items() if k.startswith("RWKV_") and k[5:] in BY_KEY}
        self.values = {}
        self.reload()

    def reload(self):
        f = _parse_env_file(env_path())
        vals = {}
        for s in SCHEMA:
            raw = self.launch_env.get(s.key, f.get("RWKV_" + s.key))
            try:
                vals[s.key] = _coerce(s, raw)
            except Exception:
                vals[s.key] = s.default
        with self._lock:
            self.values = vals
        return vals

    def get(self, key):
        with self._lock:
            return self.values[key]

    def __getitem__(self, key):
        return self.get(key)

    def snapshot(self):
        with self._lock:
            return dict(self.values)

    def text(self, key):
        """A str setting with \\n escapes decoded (separators, stop words)."""
        return self.get(key).encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")

    def stops(self):
        return [x for x in (p.encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")
                            for p in self.get("STOP").split(",")) if x]

    def engine_view(self):
        """The values the engine processes need (the restart-keys)."""
        v = self.snapshot()
        return {k: v[k] for k in v if BY_KEY[k].restart}

    def schema_json(self):
        v = self.snapshot()
        out = []
        for s in SCHEMA:
            val = v[s.key]
            out.append({"key": s.key, "type": s.type, "default": s.default, "group": s.group, "label": s.label,
                        "help": s.help, "choices": list(s.choices), "restart": s.restart, "server": s.server, "secret": s.secret,
                        "value": ("" if s.secret and val else val) if s.secret else val,
                        "is_set": bool(val) if s.secret else None,
                        "locked": s.key in self.launch_env})
        return {"groups": [{"id": g, "label": l} for g, l in GROUPS], "settings": out, "env_file": env_path()}

    def save(self, updates: dict) -> dict:
        """Validate and write `updates` (key -> value) into the .env file. Returns
        {"changed": [...], "restart": bool}. Keys locked by the launch environment are refused."""
        clean = {}
        for k, val in updates.items():
            s = BY_KEY.get(k)
            if s is None:
                raise ValueError(f"unknown setting {k}")
            if k in self.launch_env:
                raise ValueError(f"{k} is set in the process environment at launch and cannot be changed here")
            if s.secret and val == "":
                continue                      # empty secret field = keep the current one
            clean[k] = _coerce(s, val)
        old = self.snapshot()
        changed = [k for k, v in clean.items() if old.get(k) != v]
        if not changed:
            return {"changed": [], "restart": False, "server_restart": False}
        path = env_path()
        lines = open(path, encoding="utf-8").read().splitlines() if os.path.exists(path) else \
            ["# RWKV-Infer-Next settings (written by the WebUI; edit freely)"]
        done = set()
        for i, line in enumerate(lines):
            m = re.match(r"\s*(?:export\s+)?RWKV_([A-Za-z0-9_]+)\s*=", line)
            if m and m.group(1) in clean:
                lines[i] = f"RWKV_{m.group(1)}={_fmt(clean[m.group(1)])}"
                done.add(m.group(1))
        for k in changed:
            if k not in done:
                lines.append(f"RWKV_{k}={_fmt(clean[k])}")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, path)
        self.reload()
        return {"changed": changed, "restart": any(BY_KEY[k].restart for k in changed),
                "server_restart": any(BY_KEY[k].server for k in changed)}


def _fmt(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    s = str(v).replace("\r\n", "\n").replace("\n", "\\n")
    return f'"{s}"' if (s == "" or any(c in s for c in " #'\"\t")) else s
