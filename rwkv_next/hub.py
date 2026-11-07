# -*- coding: utf-8 -*-
"""Hugging Face Hub support: models and MTP sidecars by repository id.

  RWKV_MODEL_PATH=RWKV/RWKV7-G1l-7.2B-20261031          a model repository (downloaded once into the HF cache)
  RWKV_MODEL_PATH=RWKV/RWKV7-G1l-7.2B-20261031@<rev>    a branch, tag or commit
  RWKV_MTP_REPO=OpenMOSE/RWKV7-MTP-Exp                  where `MTP_PATH=auto` looks for a sidecar when the model
                                                        folder has none

A sidecar repository holds one folder per base model plus an `index.json`:

  index.json   {"format": "rwkv-mtp", "version": 1,
                "sidecars": [{"base_model": "RWKV7-G1l-7.2B-20261031", "path": "RWKV7-G1l-7.2B-20261031",
                              "n_layer": 32, "n_embd": 4096, "hf_base": ["RWKV/RWKV7-G1l-7.2B-20261031"]}, ...]}
  <path>/mtp.safetensors, <path>/mtp.json

A sidecar matches when its `base_model` equals the model's folder / repository name, or `hf_base` lists the
model's repository id. With force (MTP_FORCE), a model without a match takes the sidecar of the same width
(n_embd) whose base model has the same layer count, else the nearest one: e.g. a new release of a 7B or 13B
model with the same shape. Drafts only steer speed, so a forced sidecar never changes the output; how much it
speeds up depends on how close the new model is to the one it was trained on. Only the chosen folder is downloaded. Private or gated repositories use the usual
`HF_TOKEN` / `huggingface-cli login`; `HF_HUB_OFFLINE=1` uses only what is cached.
"""

from __future__ import annotations

import json
import os
import re

_REPO_RE = re.compile(r"^(?:hf://)?([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9_.-]+)(?:@([A-Za-z0-9_./-]+))?$")
# a model repository: weights, configs, tokenizer, chat template, a bundled mtp/ sidecar. No *.pth / *.bin / *.py.
MODEL_FILES = ["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "mtp/*"]


def parse_repo(ref):
    """'org/name[@rev]' (optionally 'hf://...') -> (repo_id, revision) when `ref` is not an existing local path."""
    ref = (ref or "").strip()
    if not ref or os.path.exists(os.path.expanduser(ref)) or ref.startswith(("/", ".", "~")):
        return None
    m = _REPO_RE.match(ref)
    return (m.group(1), m.group(2)) if m else None


def model_label(ref):
    """the model's name: folder / file name, or the repository's name part"""
    r = parse_repo(ref)
    if r:
        return r[0].split("/")[-1]
    p = (ref or "").rstrip("/")
    base = os.path.basename(p)
    return base.rsplit(".", 1)[0] if base.endswith((".pth", ".safetensors")) else base


def _hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError as e:
        raise RuntimeError("Hugging Face repository ids need the huggingface_hub package: pip install huggingface_hub") from e


def resolve_model(ref, note=print):
    """MODEL_PATH -> a local path (a repository id is downloaded / revalidated in the HF cache)."""
    r = parse_repo(ref)
    if r is None:
        return os.path.expanduser(ref)
    repo, rev = r
    hub = _hub()
    note(f"model {repo}{'@' + rev if rev else ''}: fetching from the Hugging Face Hub (cached after the first time)")
    path = hub.snapshot_download(repo, revision=rev, allow_patterns=MODEL_FILES)
    note(f"model {repo}: {path}")
    return path


def find_hub_sidecar(model_ref, mtp_repo, note=print, force=False, shape=None):
    """The sidecar for `model_ref` from the sidecar repository `mtp_repo` ('org/name[@rev]'), or None.
    force + shape (n_layer, n_embd): fall back to the closest compatible sidecar (see the module docstring)."""
    r = parse_repo(mtp_repo)
    if r is None:
        return None
    repo, rev = r
    hub = _hub()
    try:
        idx = json.load(open(hub.hf_hub_download(repo, "index.json", revision=rev)))
    except Exception as e:                                   # offline, no such repository, no index: MTP stays off
        note(f"MTP: no sidecar index from {repo} ({type(e).__name__}: {str(e).splitlines()[0][:120]})")
        return None
    name = model_label(model_ref)
    mr = parse_repo(model_ref)
    ents = idx.get("sidecars", [])
    pick = None
    for ent in ents:
        hf_base = ent.get("hf_base") or []
        hf_base = [hf_base] if isinstance(hf_base, str) else hf_base
        if ent.get("base_model") == name or (mr and mr[0] in hf_base):
            pick = ent
            break
    if pick is None and force and shape:
        pick = pick_compatible(ents, shape)
        if pick is not None:
            note(f"MTP: FORCED - no sidecar for {name}; using the one trained for {pick.get('base_model')} "
                 f"(L{pick.get('n_layer')} C{pick.get('n_embd')}, model L{shape[0]} C{shape[1]}). "
                 "Output is unchanged; the speed-up may be smaller")
    if pick is None:
        note(f"MTP: {repo} has no sidecar for {name}" + ("" if force else " (MTP_FORCE=1 would try a compatible one)"))
        return None
    sub = pick.get("path") or pick["base_model"]
    note(f"MTP: sidecar for {name} from {repo}/{sub}")
    root = hub.snapshot_download(repo, revision=rev, allow_patterns=[f"{sub}/*"])
    path = os.path.join(root, sub)
    return path if os.path.isfile(os.path.join(path, "mtp.safetensors")) else None


def pick_compatible(ents, shape):
    """the entry of the same width with the closest layer count (ties: the later entry, i.e. the newer one)"""
    L, C = shape
    same = [e for e in ents if int(e.get("n_embd") or 0) == int(C)]
    if not same:
        return None
    return min(reversed(same), key=lambda e: abs(int(e.get("n_layer") or 0) - int(L)))
