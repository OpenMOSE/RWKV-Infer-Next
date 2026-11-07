# -*- coding: utf-8 -*-
"""Persist triton autotune choices for the int8 kernels across processes.

Each `@triton.autotune` kernel benchmarks every config the first time it sees
a new key (M bucket, N, K, group, split) — 8 configs x a compile each, several
seconds per key, and a server start-up touches ~20 keys per engine. The chosen
configs only depend on the GPU, the kernel source and the shape, so they are
saved after warm-up and restored on the next start by pre-filling each
kernel's `cache` dict (the same dict the autotuner consults before
benchmarking). Decode graphs themselves cannot be persisted (they hold device
addresses), but they are captured in ~1 s once the kernels are tuned.

File: <RWKV_CACHE_DIR>/autotune/<gpu>-<triton>-<source hash>.json. Anything
that could change the right answer is in the name or checked in the file:
GPU name, triton version, the hash of int8_gemm.py, torch version.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import threading

import torch

KERNEL_NAMES = ("_w8a16_gemm", "_w8a16_gemv", "_w8a16_wide", "_w8a16_splitk")
INT6_KERNEL_NAMES = ("_w6a16",)                                    # kernels/int6_gemm.py
WKV_KERNEL_NAMES = ("_wkv7_fwd",)                                  # kernels/wkv7.py
_lock = threading.Lock()
_loaded = False
_path = None


def cache_dir():
    return os.path.expanduser(os.environ.get("RWKV_CACHE_DIR", "~/.cache/rwkv-infer-next"))


def _kernels():
    from . import int8_gemm, int6_gemm, wkv7
    from triton.runtime.autotuner import Autotuner
    ks = {n: getattr(int8_gemm, n) for n in KERNEL_NAMES
          if isinstance(getattr(int8_gemm, n, None), Autotuner)}
    ks.update({n: getattr(int6_gemm, n) for n in INT6_KERNEL_NAMES
               if isinstance(getattr(int6_gemm, n, None), Autotuner)})
    ks.update({n: getattr(wkv7, n) for n in WKV_KERNEL_NAMES if isinstance(getattr(wkv7, n, None), Autotuner)})
    return ks


def _file():
    import triton
    from . import int8_gemm, int6_gemm, wkv7
    src = hashlib.sha1(open(int8_gemm.__file__, "rb").read() + open(int6_gemm.__file__, "rb").read()
                       + open(wkv7.__file__, "rb").read()).hexdigest()[:10]
    gpu = "cpu"
    if torch.cuda.is_available():
        gpu = re.sub(r"[^A-Za-z0-9]+", "-", torch.cuda.get_device_name(torch.cuda.current_device())).strip("-")
    return os.path.join(cache_dir(), "autotune", f"{gpu}-triton{triton.__version__}-{src}.json")


def _config_to_dict(c):
    return {"kwargs": dict(c.kwargs), "num_warps": c.num_warps, "num_stages": c.num_stages,
            "num_ctas": getattr(c, "num_ctas", 1)}


def _config_from_dict(d):
    import triton
    kw = {}
    if "num_ctas" in d:
        kw["num_ctas"] = d["num_ctas"]
    return triton.Config(dict(d["kwargs"]), num_warps=d["num_warps"], num_stages=d["num_stages"], **kw)


def load(verbose=False):
    """Pre-fill the kernels' autotune caches from disk. Idempotent."""
    global _loaded, _path
    with _lock:
        if _loaded:
            return 0
        _loaded = True
        try:
            _path = _file()
            if not os.path.exists(_path):
                return 0
            data = json.load(open(_path))
            if data.get("torch") != torch.__version__:
                return 0
            n = 0
            for name, k in _kernels().items():
                for ent in data.get("kernels", {}).get(name, []):
                    key = tuple(ent["key"])
                    if key not in k.cache:
                        k.cache[key] = _config_from_dict(ent["config"])
                        n += 1
            if verbose:
                print(f"[autotune] restored {n} tuned shapes from {_path}", flush=True)
            return n
        except Exception as e:                      # a bad cache must never block start-up
            if verbose:
                print(f"[autotune] cache ignored: {type(e).__name__}: {e}", flush=True)
            return 0


def save(verbose=False):
    """Merge the kernels' current autotune caches into the file."""
    global _path
    with _lock:
        try:
            path = _path or _file()
            data = {"torch": torch.__version__, "kernels": {}}
            if os.path.exists(path):
                try:
                    old = json.load(open(path))
                    if old.get("torch") == torch.__version__:
                        data = old
                except Exception:
                    pass
            n_new = 0
            for name, k in _kernels().items():
                ents = data["kernels"].setdefault(name, [])
                have = {tuple(e["key"]) for e in ents}
                for key, cfg in k.cache.items():
                    if key not in have:
                        ents.append({"key": list(key), "config": _config_to_dict(cfg)})
                        n_new += 1
            if n_new:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + ".tmp"
                json.dump(data, open(tmp, "w"), indent=0)
                os.replace(tmp, path)
            if verbose:
                print(f"[autotune] saved {n_new} new tuned shapes to {path}", flush=True)
            return n_new
        except Exception as e:
            if verbose:
                print(f"[autotune] save failed: {type(e).__name__}: {e}", flush=True)
            return 0


_last_saved = {"n": -1, "t": 0.0}


def save_if_new(min_interval=30.0, verbose=False):
    """Persist when the kernels hold tuned shapes the last save did not, at most once per
    `min_interval` seconds. Cheap when nothing changed (a count). Engines call this when idle,
    so a server that is killed (no atexit) still leaves the shapes it tuned at run time."""
    import time
    now = time.time()
    if now - _last_saved["t"] < min_interval:
        return 0
    n = sum(len(k.cache) for k in _kernels().values())
    if n == _last_saved["n"]:
        _last_saved["t"] = now
        return 0
    _last_saved["n"], _last_saved["t"] = n, now
    return save(verbose=verbose)


atexit.register(save)       # plain processes; engine ranks leave through os._exit and use save_if_new() when idle
