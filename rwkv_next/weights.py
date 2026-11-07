# -*- coding: utf-8 -*-
"""Checkpoint access: a plain RWKV-7 .pth, a single .safetensors file, or a Hugging Face style model folder
(model.safetensors.index.json + shards, config.json, chat_template.jinja, ...).

All of them come back as a lazy mapping with the canonical RWKV key names (`blocks.N.att.*`, `emb.weight`,
`ln_out.*`, `head.weight`): tensors are read one at a time on first access (mmap / safetensors), so loading a
shard never materialises the whole model in host RAM.
"""

from __future__ import annotations

import json
import os

import torch

_PREFIXES = ("rwkv7.", "model.", "llm.", "rwkv.")


def _canon(k):
    for p in _PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k


class LazyWeights:
    """dict-like, read-only; values are CPU tensors (views of mmap'd files where the format allows)."""

    def __init__(self, getters: dict):
        self._get = getters                    # canonical name -> zero-arg loader
        self._cache = {}

    def __contains__(self, k):
        return k in self._get

    def __getitem__(self, k):
        t = self._cache.get(k)
        if t is None:
            t = self._get[k]()
        return t

    def keys(self):
        return self._get.keys()

    def __iter__(self):
        return iter(self._get)

    def __len__(self):
        return len(self._get)

    def items(self):
        for k in self._get:
            yield k, self[k]


def model_dir(path):
    """The folder holding the model's metadata (chat template, config, sidecars)."""
    return path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))


def open_weights(path) -> LazyWeights:
    path = os.path.expanduser(str(path))
    if os.path.isdir(path):
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            wm = json.load(open(idx))["weight_map"]
            files = {f: os.path.join(path, f) for f in set(wm.values())}
        else:
            st = sorted(f for f in os.listdir(path) if f.endswith(".safetensors") and "mtp" not in f)
            if not st:
                pth = sorted(f for f in os.listdir(path) if f.endswith(".pth"))
                if not pth:
                    raise FileNotFoundError(f"no model weights (*.safetensors / *.pth) in {path}")
                return open_weights(os.path.join(path, pth[0]))
            wm, files = {}, {}
            from safetensors import safe_open
            for f in st:
                files[f] = os.path.join(path, f)
                with safe_open(files[f], "pt") as h:
                    for k in h.keys():
                        wm[k] = f
        return _safetensors(wm, files)
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(path, "pt") as h:
            wm = {k: os.path.basename(path) for k in h.keys()}
        return _safetensors(wm, {os.path.basename(path): path})
    sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return LazyWeights({_canon(k): (lambda k=k: sd[k]) for k in sd})


def _safetensors(weight_map, files):
    from safetensors import safe_open
    handles = {}

    def get(k, f):
        h = handles.get(f)
        if h is None:
            h = handles[f] = safe_open(files[f], "pt")
        return h.get_tensor(k)
    return LazyWeights({_canon(k): (lambda k=k, f=f: get(k, f)) for k, f in weight_map.items()})


def model_shape(path):
    """(n_layer, n_embd) of a model without reading its weights: config.json when present, else the tensor names
    and the embedding's header shape."""
    cfg = read_json(path, "config.json") or {}
    L, C = cfg.get("num_hidden_layers") or cfg.get("n_layer"), cfg.get("hidden_size") or cfg.get("n_embd")
    if L and C:
        return int(L), int(C)
    sd = open_weights(path)
    L = 1 + max(int(k.split(".")[1]) for k in sd.keys() if k.startswith("blocks."))
    return L, int(sd["emb.weight"].shape[1])


def read_json(path, name):
    p = os.path.join(model_dir(path), name)
    return json.load(open(p)) if os.path.exists(p) else None
