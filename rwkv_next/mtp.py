# -*- coding: utf-8 -*-
"""MTP drafter (MTP sidecar format v1) for speculative decoding, batched. Spec: docs/mtp.md.

Normative spec: RWKV-LM train_temp/docs/mtp-spec.md, reference mtp_port/mtp_ref.py (NumPy, one row at a time).
This is the same math for a whole decode batch, with a different number of committed positions per row:

  one block (gated SWA attention + RWKV channel-mix) run once per committed position (depth 1) and K-1 more times
  per drafting chain (depth 2..K), fed its own output.  Inputs per position: the main model's hidden after ln_out
  followed by the OUTPUT residual stream of each block in meta.fuse_layers, and ln0(emb(token)).

State per row (DraftState; the engine keeps one over all slots in drafter.DraftPool):
  kc, vc    (B, Hk, S+1, D) ring KV, S = window + K_max + 1; position p lives in slot p % S; slot S is a dummy
                           that absorbs masked writes (static shapes, no host sync: CUDA-graph safe)
  slot_pos  (B, S+1) long  absolute position held by each slot (-1 = empty; always -1 for the dummy)
  ffn_x     (B, C)         channel-mix token shift = ln2 output of the last COMMITTED position
  pos       (B,) long      committed positions (= RoPE position of the next depth-1 step)

Attention rule (spec section 4): a query at absolute position qp sees a key at sp iff 0 <= sp <= qp and qp - sp < W.
The reference writes each row's key BEFORE attending, one row at a time.  Here a call handles T rows at once: the
queries see (a) ring entries with sp < first position written by this call for that row (entries at or beyond it
are stale draft keys or will be replaced) and (b) this call's own new keys, both under the rule; the ring is written
afterwards.  That is exactly the reference's view for any T (a ring entry the reference would have overwritten
first is at least S > W positions old, so it is invisible anyway).
"""
import json
import math
import os

import torch
import torch.nn.functional as F


def load_sidecar(path):
    """-> (state_dict {name: tensor}, meta dict, draft_vocab LongTensor or None).
    path: a *.mtp.pt / mtp-*.pt sidecar, or a directory from mtp_port/export.py (mtp.safetensors + mtp.json)."""
    if os.path.isdir(path):
        from safetensors.torch import load_file
        sd = load_file(os.path.join(path, "mtp.safetensors"))
        meta = json.load(open(os.path.join(path, "mtp.json")))["meta"]
        dv = sd.pop("draft_vocab", None)
        return sd, meta, dv
    if path.endswith(".safetensors"):                         # single file: meta in the safetensors header
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(path, "pt") as f:
            meta = json.loads((f.metadata() or {}).get("mtp_meta", "{}"))
        sd = load_file(path)
        dv = sd.pop("draft_vocab", None)
        return sd, meta, dv
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    return dict(sd), ck.get("meta", {}) or {}, ck.get("draft_vocab")


def config_of(sd, meta):
    D = sd["attn.q_norm.weight"].shape[0]
    a = meta.get("attn") or {}
    return dict(C=sd["lh.weight"].shape[0], H=sd["attn.q.weight"].shape[0] // D, Hk=sd["attn.k.weight"].shape[0] // D,
                D=D, rope_dim=int(a.get("rope_dim", 32)), rope_theta=float(a.get("rope_theta", 1e6)),
                window=int(meta.get("window") or meta.get("ctx") or 512),
                fuse=sum(1 for k in sd if k.startswith("lf.") and k.endswith(".weight")),
                fuse_layers=[int(x) for x in (meta.get("fuse_layers") or [])], dd="dep" in sd,
                ln_eps=float(meta.get("ln_eps", 1e-5)), rms_eps=float(meta.get("rms_eps", 1e-6)))


class DraftState:
    """Batched drafter state; the RWKVState.extra protocol (clone / index_select / static stack)."""

    def __init__(self, kc, vc, slot_pos, ffn_x, pos):
        self.kc, self.vc, self.slot_pos, self.ffn_x, self.pos = kc, vc, slot_pos, ffn_x, pos

    @property
    def batch_size(self):
        return self.pos.shape[0]

    def clone(self):
        return DraftState(self.kc.clone(), self.vc.clone(), self.slot_pos.clone(), self.ffn_x.clone(), self.pos.clone())

    def index_select(self, idx):
        return DraftState(self.kc.index_select(0, idx), self.vc.index_select(0, idx),
                          self.slot_pos.index_select(0, idx), self.ffn_x.index_select(0, idx),
                          self.pos.index_select(0, idx))

    @staticmethod
    def stack(items, batch_sizes):
        tmpl = next((it for it in items if it is not None), None)
        if tmpl is None:
            return None
        parts = []
        for it, b in zip(items, batch_sizes):
            if it is None:                     # a row without a drafter state: empty ring, nothing committed
                parts.append(DraftState(tmpl.kc.new_zeros((b,) + tuple(tmpl.kc.shape[1:])),
                                        tmpl.vc.new_zeros((b,) + tuple(tmpl.vc.shape[1:])),
                                        tmpl.slot_pos.new_full((b, tmpl.slot_pos.shape[1]), -1),
                                        tmpl.ffn_x.new_zeros((b, tmpl.ffn_x.shape[1])),
                                        tmpl.pos.new_zeros(b)))
            else:
                parts.append(it)
        cat = lambda a: torch.cat([getattr(p, a) for p in parts], 0)
        return DraftState(cat("kc"), cat("vc"), cat("slot_pos"), cat("ffn_x"), cat("pos"))


class MTPDrafter(torch.nn.Module):
    """The drafter of one main model.  emb_fn(ids) -> ln0(emb(ids)) and the head rows come from the main model."""

    def __init__(self, sd, meta, draft_vocab=None, K_max=8, device="cuda", dtype=torch.bfloat16,
                 draft_vocab_size=16384):
        super().__init__()
        self.cfg = c = config_of(sd, meta)
        self.meta = meta
        self.K_max = int(K_max)
        self.S = c["window"] + self.K_max + 1
        self.dtype = dtype
        self.w = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype) for k, v in sd.items()}
        # one fused input projection per depth-1 call: fproj (F>0) then proj; kept separate (exact reference order)
        self.x_k = self.w["blk.ffn.x_k"].reshape(-1)
        half = c["rope_dim"] // 2
        self.inv_freq = 1.0 / (c["rope_theta"] ** (torch.arange(half, dtype=torch.float64, device=device) / half))
        self.dv = None if draft_vocab is None else draft_vocab.to(device=device, dtype=torch.long)
        if self.dv is not None and draft_vocab_size:
            self.dv = self.dv[:int(draft_vocab_size)]
        self.head = None                          # (Nd, C) rows of the main head, set by bind_main()
        self.emb_fn = None
        self.device = device
        self._blk = self._block                   # compile() swaps in a torch.compile'd copy (static shapes)

    def compile(self):
        """Fuse the block's ~60 small kernels (used by the speculative CUDA graphs; shapes are static there)."""
        if self._blk is self._block:
            self._blk = torch.compile(self._block, dynamic=False)
        return self

    # ---- main model hookup -------------------------------------------------------------------------------- #
    def bind_main(self, emb_fn, head_rows_fn):
        """emb_fn(ids long) -> ln0(emb) (.., C); head_rows_fn(ids or None) -> (N, C) head rows (all when None)."""
        self.emb_fn = emb_fn
        self.head = head_rows_fn(self.dv).to(self.dtype)
        self.head_t = self.head.t().contiguous()                 # (C, Nd) for the draft logits GEMM

    @property
    def fuse_layers(self):
        return self.cfg["fuse_layers"]

    def new_state(self, B):
        c, dev = self.cfg, self.device
        # S ring slots + 1 dummy slot (index S) that absorbs the writes of invalid rows
        return DraftState(torch.zeros(B, c["Hk"], self.S + 1, c["D"], device=dev, dtype=self.dtype),
                          torch.zeros(B, c["Hk"], self.S + 1, c["D"], device=dev, dtype=self.dtype),
                          torch.full((B, self.S + 1), -1, dtype=torch.long, device=dev),
                          torch.zeros(B, c["C"], device=dev, dtype=self.dtype),
                          torch.zeros(B, dtype=torch.long, device=dev))

    # ---- math ---------------------------------------------------------------------------------------------- #
    def _ln(self, x, name):
        return F.layer_norm(x, (x.shape[-1],), self.w[name + ".weight"], self.w[name + ".bias"], self.cfg["ln_eps"])

    def _rms(self, x, name):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.cfg["rms_eps"])).to(x.dtype) * self.w[name]

    def _rope(self, x, pos):
        """x (B, T, h, D); pos (B, T) long absolute positions. rotate-half on the first rope_dim dims."""
        rd, half = self.cfg["rope_dim"], self.cfg["rope_dim"] // 2
        ang = pos.double().unsqueeze(-1) * self.inv_freq                       # (B, T, half), fp64 angles
        cs, sn = ang.cos().float().unsqueeze(2), ang.sin().float().unsqueeze(2)
        xf = x.float()
        x1, x2 = xf[..., :half], xf[..., half:rd]
        return torch.cat([x1 * cs - x2 * sn, x1 * sn + x2 * cs, xf[..., rd:]], -1).to(x.dtype)

    def _block(self, hin, tok, k, ffn_prev, pos, valid, st):
        """T rows per batch row.  hin (B,T,Cin) main hidden[+taps] (k=1) or previous output (k>1); tok (B,T) long;
        ffn_prev (B,C) token shift entering row 0 (row i>0 uses row i-1's ln2 output); pos (B,T) long; valid (B,T)
        bool (invalid rows: computed, never written, never read).  Writes the ring.  -> (out (B,T,C), x2 (B,T,C))"""
        c, w = self.cfg, self.w
        B, T = tok.shape
        C, H, Hk, D, W = c["C"], c["H"], c["Hk"], c["D"], c["window"]
        e = self.emb_fn(tok).to(self.dtype)
        if k > 1:
            hn = self._ln(hin, "lhd" if c["dd"] else "lh")
        elif c["fuse"]:
            parts = [self._ln(hin[..., :C], "lh")] + [self._ln(hin[..., C * (i + 1):C * (i + 2)], f"lf.{i}")
                                                       for i in range(c["fuse"])]
            hn = self._mm(torch.cat(parts, -1), "fproj.weight")
        else:
            hn = self._ln(hin, "lh")
        x = self._mm(torch.cat([hn, self._ln(e, "le")], -1), "proj.weight")
        if c["dd"]:
            x = x + w["dep"][k - 1]
        x = self._ln(x, "blk.ln0")
        xl = self._ln(x, "blk.ln1")
        q = self._rope(self._rms(self._mm(xl, "attn.q.weight").view(B, T, H, D), "attn.q_norm.weight"), pos)
        kn = self._rope(self._rms(self._mm(xl, "attn.k.weight").view(B, T, Hk, D), "attn.k_norm.weight"), pos)
        vn = self._mm(xl, "attn.v.weight").view(B, T, Hk, D)
        # keys: ring (B,Hk,S,D) + this call's (B,Hk,T,D); visibility per query position
        p0 = torch.where(valid, pos, torch.full_like(pos, 1 << 60)).amin(1, keepdim=True)     # (B,1) first written pos
        sp = st.slot_pos                                                                      # (B,S)
        qp = pos.unsqueeze(-1)                                                                # (B,T,1)
        ring_vis = (sp >= 0).unsqueeze(1) & (sp < p0).unsqueeze(1) & (sp.unsqueeze(1) <= qp) & (qp - sp.unsqueeze(1) < W)
        np_ = torch.where(valid, pos, torch.full_like(pos, -1))                               # (B,T)
        new_vis = (np_ >= 0).unsqueeze(1) & (np_.unsqueeze(1) <= qp) & (qp - np_.unsqueeze(1) < W)
        vis = torch.cat([ring_vis, new_vis], -1)                                              # (B,T,S+T)
        vis[..., self.S + 1] |= ~valid                   # invalid rows see their own key: no all -inf softmax
        Kall = torch.cat([st.kc, kn.transpose(1, 2)], 2)                                      # (B,Hk,S+T,D)
        Vall = torch.cat([st.vc, vn.transpose(1, 2)], 2)
        G = H // Hk
        qg = q.view(B, T, Hk, G, D).permute(0, 2, 3, 1, 4).reshape(B, Hk, G * T, D)           # (B,Hk,G*T,D)
        sc = torch.matmul(qg.float(), Kall.float().transpose(-1, -2)) / math.sqrt(D)          # (B,Hk,G*T,S+T)
        sc = sc.view(B, Hk, G, T, -1).masked_fill(~vis[:, None, None], float("-inf"))
        y = torch.matmul(sc.softmax(-1).to(Vall.dtype).view(B, Hk, G * T, -1), Vall)          # (B,Hk,G*T,D)
        y = y.view(B, Hk, G, T, D).permute(0, 3, 1, 2, 4).reshape(B, T, H * D)
        x = x + self._mm(y * torch.sigmoid(self._mm(xl, "attn.g.weight")), "attn.o.weight")
        x2 = self._ln(x, "blk.ln2")
        prev = torch.cat([ffn_prev.unsqueeze(1).to(x2.dtype), x2[:, :-1]], 1)
        kx = x2 + (prev - x2) * self.x_k
        x = x + self._mm(torch.relu(self._mm(kx, "blk.ffn.key.weight")).pow(2), "blk.ffn.value.weight")
        out = self._ln(x, "lo")
        # ring write, static shapes (no host sync: CUDA-graph safe).  Valid rows go to slot pos % S; of a run longer
        # than S only the last S positions can survive (distinct slots); everything else goes to the dummy slot S,
        # whose slot_pos stays -1 (never visible).
        last = torch.where(valid, pos, torch.full_like(pos, -1)).amax(1, keepdim=True)
        wr = valid & (pos > last - self.S)
        slot = torch.where(wr, pos % self.S, torch.full_like(pos, self.S))                     # (B,T)
        bi = torch.arange(B, device=pos.device).unsqueeze(1).expand(B, T)
        st.kc[bi, :, slot] = kn.to(st.kc.dtype)                                                 # (B,T,Hk,D)
        st.vc[bi, :, slot] = vn.to(st.vc.dtype)
        st.slot_pos[bi, slot] = torch.where(wr, pos, torch.full_like(pos, -1))
        return out, x2

    # ---- protocol ------------------------------------------------------------------------------------------ #
    @torch.inference_mode()
    def commit(self, st, hin, tok, n):
        """depth-1 steps of committed positions: row i = (hidden of position pos+i, token at pos+i+1), i < n[b].
        hin (B,T,Cin), tok (B,T), n (B,) long (0..T). -> out (B,T,C) (row i -> the drafts for position pos+i+2)."""
        B, T = tok.shape
        ar = torch.arange(T, device=tok.device)
        valid = ar.unsqueeze(0) < n.unsqueeze(1)
        pos = st.pos.unsqueeze(1) + ar.unsqueeze(0)
        out, x2 = self._blk(hin.to(self.dtype), tok, 1, st.ffn_x, pos, valid, st)
        has = n > 0
        lastx = x2[torch.arange(B, device=tok.device), (n - 1).clamp_min(0)]
        st.ffn_x = torch.where(has.unsqueeze(1), lastx.to(st.ffn_x.dtype), st.ffn_x)
        st.pos = st.pos + n
        return out

    # ---- weight-only quantisation (inference engines): the drafter is ~0.5 GB of bf16 for C=4096 and is read
    # K+1 times per speculative step, so it is bandwidth bound exactly like the main model ---------------------- #
    def quantize(self, quant="int8", group_k=128, min_dim=512):
        if quant not in ("int8", "int6"):
            return self
        from .model import QW
        self.qw = {}
        for name, t in list(self.w.items()):
            if name.endswith(".weight") and t.dim() == 2 and min(t.shape) >= min_dim and t.shape[1] % 64 == 0:
                self.qw[name] = QW(t.t().contiguous(), quant, group_k)
        if self.head_t is not None and self.head_t.dtype != torch.float32:
            self.qhead = QW(self.head_t, "int8", group_k)            # (C, Nd): the draft logits
            self.head = self.head_t = None
        for name in self.qw:                                         # the bf16 originals are no longer read
            del self.w[name]
        torch.cuda.empty_cache()
        return self

    def _mm(self, x, name):
        q = getattr(self, "qw", {}).get(name)
        if q is None:
            return x @ self.w[name].t()
        sh = x.shape
        return q(x.reshape(-1, sh[-1])).view(*sh[:-1], -1)

    def _pick(self, out):
        qh = getattr(self, "qhead", None)
        if qh is not None:
            lg = qh(out.to(torch.bfloat16)).float()
            i = lg.argmax(-1)
            return (i if self.dv is None else self.dv[i]), lg
        if self.head_t.dtype == torch.float32:
            lg = out.float() @ self.head_t
        else:                                                        # bf16 GEMM, fp32 accumulate AND output (spec 6)
            lg = torch.mm(out.to(self.head_t.dtype), self.head_t, out_dtype=torch.float32)
        i = lg.argmax(-1)
        return (i if self.dv is None else self.dv[i]), lg

    @torch.inference_mode()
    def chain(self, st, h, t, K):
        """Commit (h, t) at position pos, then draft K tokens (fixed K for the whole batch: conf 0, the GPU optimum).
        h (B,Cin), t (B,) long. -> drafts (B,K) long."""
        B = t.shape[0]
        out = self.commit(st, h.unsqueeze(1), t.unsqueeze(1), torch.ones(B, dtype=torch.long, device=t.device))[:, 0]
        d, _ = self._pick(out)
        ds = [d]
        shift, base = st.ffn_x, st.pos - 1
        ones = torch.ones(B, 1, dtype=torch.bool, device=t.device)
        for k in range(2, K + 1):
            o, x2 = self._blk(out.unsqueeze(1), d.unsqueeze(1), k, shift, (base + k - 1).unsqueeze(1), ones, st)
            out, shift = o[:, 0], x2[:, 0]
            d, _ = self._pick(out)
            ds.append(d)
        return torch.stack(ds, 1)


def parse_k_rules(s):
    """"8:4,32:3,0:2" -> [(8, 4), (32, 3), (inf, 2)]: K for a decode batch of B rows = first rule with B <= limit."""
    out = []
    for part in str(s).split(","):
        if not part.strip():
            continue
        b, k = part.split(":")
        out.append((float("inf") if int(b) <= 0 else int(b), int(k)))
    return sorted(out) or [(float("inf"), 4)]


def k_for(rules, B):
    return next((k for b, k in rules if B <= b), rules[-1][1])
