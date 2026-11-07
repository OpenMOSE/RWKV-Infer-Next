# -*- coding: utf-8 -*-
"""MTP drafter on rank 0, with per-slot state matching the model's slot pool.

The drafter math is mtp.MTPDrafter (MTP sidecar format v1, docs/mtp.md: one gated SWA-attention block, depth-unrolled,
fusion of mid-layer residual streams). Its state (ring KV, positions, channel-mix shift) is kept as one
tensor per field over all S slots; a decode batch is the view [:B], exactly like the main state pool.
Also kept per slot: H, the drafter input row of the last committed position (hidden after ln_out + taps).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .mtp import DraftState, MTPDrafter, load_sidecar, parse_k_rules, k_for


def find_sidecar(model_path, setting):
    """MTP_PATH setting -> sidecar path or None. 'auto' looks in the model folder for mtp/ (mtp.safetensors +
    mtp.json), *.mtp.safetensors, *.mtp.pt, in that order."""
    import os
    setting = (setting or "").strip()
    if not setting or setting.lower() in ("off", "none", "0"):
        return None
    if setting.lower() != "auto":
        return os.path.expanduser(setting)
    if not model_path:
        return None
    from .weights import model_dir
    d = model_dir(model_path)
    if os.path.isfile(os.path.join(d, "mtp", "mtp.safetensors")):
        return os.path.join(d, "mtp")
    for suf in (".mtp.safetensors", ".mtp.pt"):
        hits = sorted(f for f in os.listdir(d) if f.endswith(suf))
        if hits:
            return os.path.join(d, hits[0])
    return None


def check_sidecar(meta, model, force=False):
    """Refuse a sidecar trained for another model (same width is not enough: a 7B sidecar would load on the 13B
    and only produce useless drafts). meta['n_layer'] / meta['base_model'] are written by tools/export_mtp.py.
    force (MTP_FORCE): only warn. Drafts never change the output, so a forced sidecar can only cost speed."""
    nl = meta.get("n_layer")
    if nl is not None and int(nl) != model.L:
        msg = f"MTP sidecar was trained for a {nl}-layer model; this model has {model.L} layers"
        if not force:
            raise ValueError(msg + " (RWKV_MTP_FORCE=1 to use it anyway)")
        print(f"[mtp] WARNING (forced): {msg}", flush=True)
    return meta.get("base_model") or meta.get("model")


class DraftPool:
    def __init__(self, path, model, slots, K_max=4, draft_vocab=16384, compile=False, quant="int6", force=False):
        sd, meta, dv = load_sidecar(path)
        self.base_model = check_sidecar(meta, model, force)
        self.d = MTPDrafter(sd, meta, dv, K_max=K_max, device=model.device, draft_vocab_size=draft_vocab)
        nl = meta.get("n_layer")
        if force and nl and int(nl) != model.L:
            # forced onto a model of another depth: tap the layers at the same relative depth
            old = list(self.d.cfg["fuse_layers"])
            self.d.cfg["fuse_layers"] = [min(model.L - 1, max(0, round((l + 1) * model.L / int(nl)) - 1)) for l in old]
            print(f"[mtp] forced: fuse_layers {old} (of {nl} layers) -> {self.d.cfg['fuse_layers']} (of {model.L})", flush=True)
        if self.d.cfg["C"] != model.C:
            raise ValueError(f"MTP sidecar is for n_embd {self.d.cfg['C']}, the model has {model.C}")
        if any(not 0 <= l < model.L for l in self.d.fuse_layers):
            raise ValueError(f"MTP fuse_layers {self.d.fuse_layers} outside a {model.L}-layer model")
        m = model
        ln0w, ln0b = m.layers[0].ln0_w, m.layers[0].ln0_b
        self.d.bind_main(lambda ids: F.layer_norm(m.embed(ids), (m.C,), ln0w, ln0b, 1e-5), lambda ids: head_rows(m, ids))
        self.d.quantize(quant)        # weight-only int6 (drafts only steer speed, never the output): 0.5 GB bf16 -> 0.19 GB
        if compile:
            self.d.compile()          # fuses the block's ~60 small kernels (static shapes inside the spec graphs)
        self.taps = tuple(self.d.fuse_layers)
        self.cin = m.C * (1 + len(self.taps))
        self.st = self.d.new_state(slots)
        self.H = torch.zeros(slots, self.cin, device=m.device, dtype=torch.bfloat16)
        self.window = self.d.cfg["window"]

    # ---- slot pool -----------------------------------------------------------------------------------
    def view(self, idx):
        """DraftState over slots `idx` (a slice -> views, so in-place ring writes land in the pool)."""
        s = self.st
        return DraftState(s.kc[idx], s.vc[idx], s.slot_pos[idx], s.ffn_x[idx], s.pos[idx])

    def store(self, idx, v):
        s = self.st
        s.ffn_x[idx] = v.ffn_x; s.pos[idx] = v.pos           # commit() rebinds these two

    def reset(self, slots):
        s = self.st
        i = torch.as_tensor(slots, device=s.pos.device)
        s.kc[i] = 0; s.vc[i] = 0; s.slot_pos[i] = -1; s.ffn_x[i] = 0; s.pos[i] = 0; self.H[i] = 0

    def move(self, src, dst):
        s = self.st
        a = torch.as_tensor(src, device=s.pos.device); b = torch.as_tensor(dst, device=s.pos.device)
        for t in (s.kc, s.vc, s.slot_pos, s.ffn_x, s.pos, self.H):
            t[b] = t[a]

    # ---- protocol ------------------------------------------------------------------------------------
    @torch.inference_mode()
    def commit_rows(self, slot, rows, toks, start_pos):
        """Prefill: commit (rows[i] = drafter input of absolute position start_pos+i, toks[i] = the token at
        start_pos+i+1) into one slot. Called per prefill chunk, in order."""
        i = torch.tensor([slot], device=rows.device)
        v = DraftState(self.st.kc[i], self.st.vc[i], self.st.slot_pos[i], self.st.ffn_x[i], self.st.pos[i])
        if int(v.pos[0]) == 0 and start_pos > 0:
            v.pos = torch.tensor([start_pos], device=rows.device)
        n = torch.tensor([rows.shape[0]], device=rows.device)
        self.d.commit(v, rows.unsqueeze(0), toks.unsqueeze(0), n)
        s = self.st
        s.kc[i] = v.kc; s.vc[i] = v.vc; s.slot_pos[i] = v.slot_pos; s.ffn_x[i] = v.ffn_x; s.pos[i] = v.pos

    @torch.inference_mode()
    def draft(self, B, t, K):
        """Commit (H[b], t[b]) for slots [0, B) and draft K tokens. t (B,) long. -> (B, K) long."""
        v = self.view(slice(0, B))
        d = self.d.chain(v, self.H[:B], t, K)
        self.store(slice(0, B), v)
        return d

    @torch.inference_mode()
    def accept(self, B, Hs, drafts, j):
        """After verify: commit the j[b] accepted drafts' depth-1 steps; H[b] := Hs[b, j[b]]."""
        K = drafts.shape[1]
        v = self.view(slice(0, B))
        if K and bool((j > 0).any()):
            self.d.commit(v, Hs[:, :K], drafts, j)
            self.store(slice(0, B), v)
        ar = torch.arange(B, device=Hs.device)
        self.H[:B] = Hs[ar, j]

    def accept_graph(self, B, Hs, drafts, j):
        """accept() without host-side branches (inside the spec-step graph): rows with j = 0 commit nothing."""
        v = self.view(slice(0, B))
        self.d.commit(v, Hs[:, :drafts.shape[1]], drafts, j)
        self.store(slice(0, B), v)
        self.H[:B] = Hs[torch.arange(B, device=Hs.device), j]

    def snapshot(self, B):
        s = self.st
        return tuple(t[:B].clone() for t in (s.kc, s.vc, s.slot_pos, s.ffn_x, s.pos, self.H))

    def restore(self, B, snap):
        s = self.st
        for t, v in zip((s.kc, s.vc, s.slot_pos, s.ffn_x, s.pos, self.H), snap):
            t[:B].copy_(v)

    @torch.inference_mode()
    def plain_step(self, B, H_new, t_prev):
        """A row took a plain decode step with input t_prev: commit (H[b], t_prev[b]) and keep H_new."""
        v = self.view(slice(0, B))
        self.d.commit(v, self.H[:B].unsqueeze(1), t_prev.unsqueeze(1), torch.ones(B, dtype=torch.long, device=t_prev.device))
        self.store(slice(0, B), v)
        self.H[:B] = H_new


def head_rows(m, ids):
    """Rows of rank 0's head for the draft vocabulary, dequantised to bf16: (N, C)."""
    h = m.head
    if h.quant == "int8":
        from .kernels.int8_gemm import dequantize_int8
        q, s = (h.q, h.s) if ids is None else (h.q[:, ids], h.s[:, ids])
        return dequantize_int8(q, s, h.gk).t().contiguous().to(torch.bfloat16)
    if h.quant == "int6":
        from .kernels.int6_gemm import dequantize_int6
        w = dequantize_int6(h.lo, h.hi, h.s, h.gk)
        return (w if ids is None else w[:, ids]).t().contiguous().to(torch.bfloat16)
    return (h.w if ids is None else h.w[:, ids]).t().contiguous()
