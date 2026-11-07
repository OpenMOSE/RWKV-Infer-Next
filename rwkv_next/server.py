# -*- coding: utf-8 -*-
"""HTTP front: OpenAI-compatible API, WebUI, settings and engine lifecycle.

The front never touches CUDA. It tokenises, builds prompts, detokenises and applies stop
strings; the engine processes (one per GPU, see engine.py) do the model work. Restarting the
engine (new model, TP, quantisation, ...) kills and respawns those processes; the front and
its connections stay up.

    python -m rwkv_next            # reads ./.env (RWKV_ENV_FILE), serves http://HOST:PORT/
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import queue
import socket
import threading
import time
import traceback
import uuid
from collections import deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse

from .chat import Detok
from .parse import StreamParser
from .template import ChatTemplate
from .config import BY_KEY, Settings
from .tokenizer import RWKVTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBUI = os.path.join(ROOT, "webui")


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


# --------------------------------------------------------------------------- #
# engine lifecycle
# --------------------------------------------------------------------------- #
class EngineManager:
    def __init__(self, settings: Settings):
        self.st = settings
        self.ctx = mp.get_context("spawn")
        self.procs, self.chan = [], None
        self.in_q = self.out_q = None
        self.state, self.info, self.error, self.stats = "stopped", {}, "", {}
        self.started_at = None
        self.cfg = {}
        self.streams: dict[str, tuple] = {}          # rid -> (loop, asyncio.Queue)
        self.lock = threading.Lock()
        self.log = deque(maxlen=200)
        self._reader = None
        self.gen = 0                                  # increments per (re)start: stale events are dropped

    def _note(self, msg):
        self.log.append(f"{time.strftime('%H:%M:%S')} {msg}")
        print(f"[front] {msg}", flush=True)

    def start(self):
        cfg = self.st.engine_view()
        self.cfg = cfg                   # as configured (compared with the settings for "pending restart")
        if not cfg.get("MODEL_PATH"):
            self.state, self.error = "stopped", "RWKV_MODEL_PATH is not set (set it on the Settings page)"
            self._note(self.error)
            return
        self.state, self.error, self.info, self.stats = "loading", "", {}, {}
        self.started_at = time.time()
        try:
            cfg = self._resolve(cfg)     # what the ranks get: local model path, resolved sidecar
        except Exception as e:
            self.state, self.error = "error", f"{type(e).__name__}: {e}"
            self._note(self.error)
            return
        self.model_dir = cfg["MODEL_PATH"]
        self.gen += 1
        world = int(cfg["TP"])
        self.in_q, self.out_q = self.ctx.Queue(), self.ctx.Queue()
        chan_name = None
        if world > 1:
            from .channel import Channel
            self.chan = Channel(None, world - 1, create=True)
            chan_name = self.chan.name
        port = _free_port()
        from .engine import rank_main
        self.procs = [self.ctx.Process(target=rank_main, args=(r, world, cfg, chan_name, self.in_q, self.out_q, port), daemon=True)
                      for r in range(world)]
        for p in self.procs:
            p.start()
        self._note(f"engine starting: tp={world} quant={cfg['QUANT']} model={cfg['MODEL_LABEL']}"
                   + (f" mtp={cfg['MTP_PATH']}" if cfg["MTP_PATH"] else " (no MTP)"))
        self._reader = threading.Thread(target=self._read, args=(self.gen, self.out_q), daemon=True)
        self._reader.start()

    def _resolve(self, cfg):
        """Hugging Face repository ids -> local paths (downloads into the HF cache), and the MTP sidecar: the
        model folder first (MTP_PATH=auto), then the sidecar repository MTP_REPO."""
        from .hub import resolve_model, find_hub_sidecar, model_label
        from .drafter import find_sidecar
        ref = cfg["MODEL_PATH"]
        path = resolve_model(ref, note=self._note)
        if not os.path.exists(path):
            raise FileNotFoundError(f"model not found: {ref}")
        setting = (cfg.get("MTP_PATH") or "").strip()
        force = bool(cfg.get("MTP_FORCE"))
        shape = None
        if force:
            from .weights import model_shape
            shape = model_shape(path)
        mtp = find_sidecar(path, setting)
        if mtp is None and setting.lower() == "auto" and (cfg.get("MTP_REPO") or "").strip():
            mtp = find_hub_sidecar(ref, cfg["MTP_REPO"], note=self._note, force=force, shape=shape)
        elif mtp is not None and setting.lower() != "auto":
            from .hub import parse_repo
            if parse_repo(mtp):                       # an explicit 'org/name[@rev]' sidecar repository
                mtp = find_hub_sidecar(ref, mtp, note=self._note, force=force, shape=shape)
        return dict(cfg, MODEL_PATH=path, MTP_PATH=mtp or "off", MODEL_LABEL=model_label(ref))

    def stop(self):
        if self.in_q is not None and self.state == "ready":
            try:
                self.in_q.put(("stop",))
            except Exception:
                pass
        t0 = time.time()
        for p in self.procs:
            p.join(timeout=max(0.1, 5 - (time.time() - t0)))
        for p in self.procs:
            if p.is_alive():
                p.kill()
        for p in self.procs:
            p.join(timeout=5)
        self.procs = []
        if self.chan is not None:
            self.chan.close(unlink=True); self.chan = None
        self.gen += 1
        with self.lock:
            for rid, (loop, q) in list(self.streams.items()):
                loop.call_soon_threadsafe(q.put_nowait, ([], "error:engine_restarted", {}))
            self.streams.clear()
        self.state = "stopped"
        self._note("engine stopped")

    def restart(self):
        self.stop()
        self.st.reload()
        self.start()

    def _read(self, gen, out_q):
        while gen == self.gen:
            try:
                msg = out_q.get(timeout=0.5)
            except queue.Empty:
                if self.state in ("loading", "ready") and self.procs and not all(p.is_alive() for p in self.procs):
                    if gen == self.gen:
                        was, self.state = self.state, "error"
                        self.error = self.error or f"engine process exited while {was} (see console)"
                        self._note(self.error); self._fail_streams(self.error)
                    return
                continue
            except (EOFError, OSError):
                return
            kind = msg[0]
            if kind == "tokens":
                with self.lock:
                    for rid, ids, fin, info in msg[1]:
                        s = self.streams.get(rid)
                        if s is not None:
                            s[0].call_soon_threadsafe(s[1].put_nowait, (ids, fin, info))
            elif kind == "stats":
                self.stats = msg[1]
            elif kind == "info":
                self.info = msg[1]
            elif kind == "ready":
                self.state = "ready"
                self._note(f"engine ready in {time.time() - self.started_at:.0f}s")
            elif kind == "error":
                self.state, self.error = "error", msg[1]
                self._note("engine error: " + msg[1].strip().splitlines()[-1])
                self._fail_streams("engine error: " + msg[1].strip().splitlines()[-1])

    def _fail_streams(self, why):
        """the engine is gone: end every open request now instead of letting clients wait for tokens"""
        with self.lock:
            streams = list(self.streams.values()); self.streams.clear()
        for loop, q in streams:
            loop.call_soon_threadsafe(q.put_nowait, ([], "error:" + why, {}))
        for p in self.procs:          # the surviving ranks would wait in NCCL forever, spinning their GPU
            if p.is_alive():
                p.kill()
        if self.chan is not None:     # nothing reads the command channel any more
            try:
                self.chan.close(unlink=True)
            except Exception:
                pass
            self.chan = None

    # ---- requests ----------------------------------------------------------
    def submit(self, prompt_ids, params):
        if self.state != "ready":
            raise HTTPException(503, f"engine is {self.state}" + (f": {self.error}" if self.error else ""))
        rid = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        q = asyncio.Queue()
        with self.lock:
            self.streams[rid] = (loop, q)
        self.in_q.put(("add", rid, prompt_ids, params))
        return rid, q

    def release(self, rid, abort):
        with self.lock:
            self.streams.pop(rid, None)
        if abort and self.in_q is not None and self.state == "ready":
            self.in_q.put(("abort", rid))


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
settings = Settings()
tokenizer = RWKVTokenizer()
engine = EngineManager(settings)
app = FastAPI(title="RWKV-Infer-Next")


@app.on_event("startup")
async def _startup():
    threading.Thread(target=engine.start, daemon=True).start()


@app.on_event("shutdown")
async def _shutdown():
    engine.stop()


def _auth(req: Request):
    key = settings["API_KEY"]
    if key and req.headers.get("authorization", "") != f"Bearer {key}":
        raise HTTPException(401, "invalid API key")


def _local(req: Request):
    if settings["ALLOW_REMOTE_SETTINGS"]:
        return
    host = req.client.host if req.client else ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(403, "settings can only be changed from localhost (RWKV_ALLOW_REMOTE_SETTINGS=1 to allow)")


def _params(body):
    g = lambda k, d: body.get(k) if body.get(k) is not None else d
    st = settings
    p = {"temperature": float(g("temperature", st["TEMPERATURE"])), "top_p": float(g("top_p", st["TOP_P"])),
         "top_k": int(g("top_k", st["TOP_K"])), "min_p": float(g("min_p", st["MIN_P"])),
         "presence_penalty": float(g("presence_penalty", st["PRESENCE_PENALTY"])),
         "frequency_penalty": float(g("frequency_penalty", st["FREQUENCY_PENALTY"])),
         "penalty_decay": float(g("penalty_decay", st["PENALTY_DECAY"])),
         "max_tokens": int(g("max_tokens", g("max_completion_tokens", st["MAX_TOKENS"])))}
    stop = body.get("stop")
    stops = ([] if body.get("_no_default_stop") else list(st.stops())) + ([stop] if isinstance(stop, str) else list(stop or []))
    return p, [s for s in stops if s]


async def _generate(prompt, body, req):
    """Yields (text_piece, finish_reason|None, stats). Applies stop strings."""
    params, stops = _params(body)
    ids = tokenizer.encode(prompt)
    rid, q = engine.submit(ids, params)
    det = Detok(tokenizer)
    pending = ""                       # text held back because it might be the start of a stop string
    n_out, t0, t_first = 0, time.time(), None
    engine_done = False                # the engine finished the row itself (else release() aborts it)
    try:
        while True:
            try:
                toks, fin, _ = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if await req.is_disconnected():
                    return
                continue
            if fin:
                engine_done = True
            if t_first is None:
                t_first = time.time()
            n_out += len(toks)
            text = det.push(toks) if toks else ""
            if fin:
                text += det.flush()
            pending += text
            cut = None
            for s in stops:
                k = pending.find(s)
                if k >= 0 and (cut is None or k < cut):
                    cut = k
            stats = lambda: {"prompt_tokens": len(ids), "completion_tokens": n_out,
                             "ttft_s": (t_first - t0) if t_first else None,
                             "tok_s": (n_out - 1) / max(time.time() - t_first, 1e-6) if t_first and n_out > 1 else None}
            if cut is not None:
                if pending[:cut]:
                    yield pending[:cut], None, None
                yield "", "stop", stats()          # a stop string: the engine row is aborted in finally
                return
            if fin:
                if pending:
                    yield pending, None, None
                reason = fin if not fin.startswith("error:") else fin
                yield "", reason, stats()
                return
            # hold back only a suffix that could still grow into a stop string
            keep = 0
            for s in stops:
                for k in range(min(len(s) - 1, len(pending)), 0, -1):
                    if pending.endswith(s[:k]):
                        keep = max(keep, k)
                        break
            out, pending = (pending[:-keep], pending[-keep:]) if keep else (pending, "")
            if out:
                yield out, None, None
    finally:
        engine.release(rid, abort=not engine_done)


_CLIENT_ERRORS = ("prompt_too_long", "empty_prompt")


def _raise_engine_error(fin):
    """non-streaming responses: an engine-side failure is a 503, a bad request a 400"""
    if fin and fin.startswith("error:"):
        why = fin[6:]
        raise HTTPException(400 if why in _CLIENT_ERRORS else 503, why)


def _chunk(cid, model, delta, fin=None, usage=None, obj="chat.completion.chunk"):
    d = {"id": cid, "object": obj, "created": int(time.time()), "model": model,
         "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
    if usage:
        d["usage"] = _usage(usage)
        d["rwkv"] = {"ttft_s": usage["ttft_s"], "tok_s": usage["tok_s"]}
    return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"


def _usage(st):
    return {"prompt_tokens": st["prompt_tokens"], "completion_tokens": st["completion_tokens"],
            "total_tokens": st["prompt_tokens"] + st["completion_tokens"]}


# ---- chat templates ------------------------------------------------------------------------------------- #
_tmpl = {"key": None, "t": None}


def running_model_path():
    """MODEL_PATH of the engine that is running (MODEL_PATH is a restart key: a saved but not yet applied value
    must not change the template or the name the front uses)."""
    return (engine.cfg or {}).get("MODEL_PATH") or settings["MODEL_PATH"]


def running_model_dir():
    """the local folder of the running model (a repository id resolved to its HF cache snapshot)"""
    return getattr(engine, "model_dir", None) or running_model_path()


def template():
    """The chat template for the running model / setting (reloaded when either changes)."""
    key = (running_model_dir(), settings["CHAT_TEMPLATE"])
    if _tmpl["key"] != key:
        _tmpl["t"] = ChatTemplate.load(key[0], settings["CHAT_TEMPLATE"])
        _tmpl["key"] = key
        engine._note(f"chat template: {_tmpl['t'].origin}")
    return _tmpl["t"]


def _thinking(body):
    """thinking on/off from the request: `thinking` (bool | 'on'/'off'/'auto'), chat_template_kwargs.thinking /
    enable_thinking (vLLM / SGLang style), or reasoning_effort ('none' = off, anything else = on)."""
    v = body.get("thinking")
    ctk = body.get("chat_template_kwargs") or {}
    if v is None:
        v = ctk.get("thinking", ctk.get("enable_thinking"))
    if v is None and body.get("reasoning_effort") is not None:
        v = str(body["reasoning_effort"]).lower() not in ("none", "off", "minimal", "false")
    if isinstance(v, str):
        v = {"on": True, "true": True, "off": False, "false": False}.get(v.lower(), None)
    return settings["THINKING_DEFAULT"] if v is None else bool(v)


def _tools(body):
    tools = body.get("tools") or []
    choice = body.get("tool_choice", "auto")
    if not tools or choice == "none":
        return []
    if isinstance(choice, dict):                      # a specific function: offer only that one
        name = (choice.get("function") or {}).get("name")
        tools = [t for t in tools if (t.get("function") or t).get("name") == name] or tools
    return tools


def _render(body):
    msgs = list(body.get("messages") or [])
    if settings["SYSTEM_PROMPT"].strip() and not any(m.get("role") == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": settings["SYSTEM_PROMPT"]})
    tools = _tools(body)
    thinking = _thinking(body)
    # continue_final_message (as in transformers' apply_chat_template): the last assistant message is left open
    # and the model writes on from its end, instead of a new assistant turn being started after it
    cont = bool(body.get("continue_final_message")) and bool(msgs) and msgs[-1].get("role") == "assistant"
    try:
        prompt = template().render(msgs, tools=tools, thinking=thinking, add_generation_prompt=not cont)
        bare = None if cont else template().render(msgs, tools=tools, thinking=thinking, add_generation_prompt=False)
    except Exception as e:
        raise HTTPException(400, f"chat template error: {e}")
    # what the generation prompt itself wrote of the assistant turn (e.g. ' <think' / ' <think></think>\n'): the
    # parser prepends it to the completion. Taken from the difference with / without add_generation_prompt, so
    # an 'Assistant:' inside the conversation can never be mistaken for it.
    prefix = ""
    if not cont:
        suffix = prompt[len(bare):] if prompt.startswith(bare) else None
        if suffix is None:
            prefix = ChatTemplate.generation_prefix(prompt)
        elif "Assistant:" in suffix:
            prefix = suffix[suffix.rfind("Assistant:") + len("Assistant:"):]
        elif "<think" in suffix:
            prefix = suffix[suffix.rfind("<think"):]
    if cont:
        last = msgs[-1].get("content") or ""
        if isinstance(last, list):
            last = "".join(p.get("text", "") for p in last if isinstance(p, dict))
        last = last.strip()
        i = prompt.rfind(last) if last else prompt.rfind("Assistant:")
        if i < 0:
            raise HTTPException(400, "continue_final_message: the final message was not found in the rendered prompt")
        prompt = prompt[:i + (len(last) if last else len("Assistant:"))]
        raw = msgs[-1].get("content") if isinstance(msgs[-1].get("content"), str) else ""
        prompt += raw[len(raw.rstrip()):]        # the template trims the content; keep its trailing whitespace
    return prompt, tools, thinking, prefix


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    _auth(req)
    body = await req.json()
    prompt, tools, thinking, prefix = _render(body)
    msgs_ = body.get("messages") or []
    cont = bool(body.get("continue_final_message")) and bool(msgs_) and msgs_[-1].get("role") == "assistant"
    mkparser = (lambda: StreamParser("", tools, reasoning=False, lstrip=False)) if cont else (lambda: StreamParser(prefix, tools))
    inline = settings["REASONING_FORMAT"] == "inline"
    model = body.get("model") or model_name()
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    if body.get("stream"):
        async def sse():
            parser = mkparser()
            yield _chunk(cid, model, {"role": "assistant", "content": ""})
            state = {"open": False}

            def deltas(dr, dc, closing=False):
                out = []
                if inline:
                    txt = ""
                    if dr:
                        if not state["open"]:
                            txt += "<think>"; state["open"] = True
                        txt += dr
                    if state["open"] and (dc or closing):
                        txt += "</think>\n"; state["open"] = False
                    txt += dc
                    if txt:
                        out.append({"content": txt})
                else:
                    if dr:
                        out.append({"reasoning_content": dr})
                    if dc:
                        out.append({"content": dc})
                return out
            try:
                async for piece, fin, st in _generate(prompt, body, req):
                    if piece:
                        for d in deltas(*parser.feed(piece)):
                            yield _chunk(cid, model, d)
                    if fin:
                        dr, dc, calls, _, _ = parser.finish()
                        for d in deltas(dr, dc, closing=True):
                            yield _chunk(cid, model, d)
                        if calls:
                            yield _chunk(cid, model, {"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
                        err = fin.startswith("error:")
                        reason = "stop" if err else ("tool_calls" if calls else fin)
                        yield _chunk(cid, model, {}, reason, st)
                        if err:
                            yield "data: " + json.dumps({"error": {"message": fin[6:]}}) + "\n\n"
            except HTTPException as e:
                yield "data: " + json.dumps({"error": {"message": e.detail}}, ensure_ascii=False) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    text, fin, st = "", None, None
    async for piece, f, s_ in _generate(prompt, body, req):
        text += piece
        if f:
            fin, st = f, s_
    _raise_engine_error(fin)
    parser = mkparser()
    parser.feed(text)
    _, _, calls, reasoning, content = parser.finish()
    msg = {"role": "assistant", "content": content if (content or not calls) else None}
    if reasoning is not None:
        if inline:
            msg["content"] = f"<think>{reasoning}</think>\n" + (content or "")
        else:
            msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = calls
    return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else fin}],
            "usage": _usage(st), "rwkv": {"ttft_s": st["ttft_s"], "tok_s": st["tok_s"], "thinking": thinking}}


@app.post("/v1/completions")
async def completions(req: Request):
    """Raw text completion (no template). `stop` defaults to none here."""
    _auth(req)
    body = await req.json()
    prompt = body.get("prompt") or ""
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    model = body.get("model") or model_name()
    cid = "cmpl-" + uuid.uuid4().hex[:24]
    body = dict(body, _no_default_stop=True)
    if body.get("stream"):
        async def sse():
            async for piece, fin, st in _generate(prompt, body, req):
                err = bool(fin) and fin.startswith("error:")
                d = {"id": cid, "object": "text_completion", "created": int(time.time()), "model": model,
                     "choices": [{"index": 0, "text": piece, "finish_reason": "stop" if err else fin}]}
                if st:
                    d["usage"] = _usage(st)
                yield "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"
                if err:
                    yield "data: " + json.dumps({"error": {"message": fin[6:]}}, ensure_ascii=False) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")
    text, fin, st = "", None, None
    async for piece, f, s_ in _generate(prompt, body, req):
        text += piece
        if f:
            fin, st = f, s_
    _raise_engine_error(fin)
    return {"id": cid, "object": "text_completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": fin}], "usage": _usage(st)}


def model_name():
    n = settings["MODEL_NAME"]
    if n:
        return n
    from .hub import model_label
    return model_label(running_model_path()) or "rwkv7"


@app.get("/v1/models")
async def models(req: Request):
    _auth(req)
    return {"object": "list", "data": [{"id": model_name(), "object": "model", "owned_by": "rwkv"}]}


@app.post("/api/tokenize")
async def tokenize(req: Request):
    _auth(req)
    body = await req.json()
    if "messages" in body:
        text = _render(body)[0]
    else:
        text = body.get("text", "")
    out = {"tokens": len(tokenizer.encode(text))}
    if body.get("return_prompt"):
        out["prompt"] = text
    return out


@app.get("/api/status")
async def status():
    return {"state": engine.state, "error": engine.error, "info": engine.info, "stats": engine.stats,
            "model_name": model_name(), "chat_template": _tmpl["t"].origin if _tmpl["t"] else None, "uptime_s": (time.time() - engine.started_at) if engine.started_at else None,
            "log": list(engine.log)[-30:],
            "defaults": {k.lower(): settings[k] for k in ("TEMPERATURE", "TOP_P", "TOP_K", "MIN_P", "PRESENCE_PENALTY",
                                                          "FREQUENCY_PENALTY", "PENALTY_DECAY", "MAX_TOKENS", "THINKING_DEFAULT")},
            "pending_restart": engine.cfg != settings.engine_view() if engine.cfg else False,
            "pending_server_restart": bool(_LAUNCH) and any(settings[k] != v for k, v in _LAUNCH.items())}


@app.get("/api/settings")
async def get_settings(req: Request):
    _auth(req)
    return settings.schema_json()


@app.put("/api/settings")
async def put_settings(req: Request):
    _local(req)
    body = await req.json()
    try:
        r = settings.save(body.get("values") or {})
    except ValueError as e:
        raise HTTPException(400, str(e))
    if body.get("restart") and r["restart"]:
        threading.Thread(target=engine.restart, daemon=True).start()
        r["restarting"] = True
    return r


@app.post("/api/engine/restart")
async def restart(req: Request):
    _local(req)
    threading.Thread(target=engine.restart, daemon=True).start()
    return {"restarting": True}


@app.post("/api/engine/stop")
async def stop(req: Request):
    _local(req)
    threading.Thread(target=engine.stop, daemon=True).start()
    return {"stopping": True}


@app.get("/api/files")
async def files(req: Request, dir: str = ""):
    """List model files (*.pth, *.pt, *.safetensors) and folders under a directory (the settings page's file
    picker). A folder holding safetensors / config.json is marked as a model folder (it can be picked itself)."""
    _local(req)
    d = os.path.expanduser(dir or os.path.dirname(settings["MODEL_PATH"] or os.getcwd()) or "/")
    if not os.path.isdir(d):
        raise HTTPException(404, "no such directory")
    out = []
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n)
        if os.path.isdir(p) and not n.startswith("."):
            try:
                inner = os.listdir(p)
            except OSError:
                inner = []
            is_model = "config.json" in inner and any(x.endswith(".safetensors") for x in inner)
            out.append({"name": n, "path": p, "dir": True, "model": is_model})
        elif n.endswith((".pth", ".pt", ".safetensors")):
            out.append({"name": n, "path": p, "dir": False, "size": os.path.getsize(p)})
    return {"dir": d, "parent": os.path.dirname(d.rstrip("/")) or "/", "entries": out}


@app.get("/")
async def root():
    return RedirectResponse("/ui")


@app.get("/ui")
@app.get("/ui/{path:path}")
async def ui(path: str = ""):
    f = os.path.join(WEBUI, path) if path else os.path.join(WEBUI, "index.html")
    if not os.path.isfile(f) or not os.path.abspath(f).startswith(WEBUI):
        f = os.path.join(WEBUI, "index.html")
    return FileResponse(f)


_LAUNCH = {}        # the server-start-only settings (HOST, PORT) this process was started with


def pid_file():
    return os.environ.get("RWKV_PID_FILE") or os.path.join(os.getcwd(), ".rwkv_next.pid")


def main():
    """python -m rwkv_next            run the server (writes the PID file)
       python -m rwkv_next --stop     stop the server started from this directory (SIGTERM via the PID file:
                                      no process-name matching, which can hit unrelated processes)"""
    import signal
    import sys
    if "--stop" in sys.argv:
        try:
            pid = int(open(pid_file()).read().strip())
        except Exception:
            print(f"no PID file at {pid_file()}"); return
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            print(f"server {pid} is not running"); os.remove(pid_file()); return
        for _ in range(100):
            try:
                os.kill(pid, 0); time.sleep(0.1)
            except ProcessLookupError:
                print(f"stopped server {pid}"); return
        print(f"server {pid} did not exit within 10 s"); return
    import uvicorn
    _LAUNCH.update({k: settings[k] for k in ("HOST", "PORT")})
    with open(pid_file(), "w") as f:
        f.write(str(os.getpid()))
    try:
        uvicorn.run(app, host=settings["HOST"], port=int(settings["PORT"]), log_level="warning")
    finally:
        engine.stop()
        try:
            if open(pid_file()).read().strip() == str(os.getpid()):
                os.remove(pid_file())
        except Exception:
            pass
