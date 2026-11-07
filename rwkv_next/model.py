# -*- coding: utf-8 -*-
"""RWKV-7 for inference: packed, quantised (bf16 / int8 / int6) and tensor parallel.

Everything is loaded straight into the inference layout (no training modules in between):
every matrix is stored (in, out), the twelve attention matrices are batched into four, and
each rank keeps only its shard.

Tensor parallel split (rank r of tp, H heads, Hl = H/tp, Cl = C/tp, Fl = F/tp)

    time mix   x_mix (6,1,C), W_down (3,C,d), W_g1  replicated  (inputs are the full residual)
               W_rkv (3,C,Cl), W_up (3,d,Cl), W_g2  column shard = this rank's heads
               b_w/b_a/b_v/k_k/k_a/r_k/ln_x       this rank's heads
               wkv state (Hl, N, N), v_first       local
               W_out (Cl, C)                      row shard -> partial sum -> ALL-REDUCE
    chan mix   W_k (C, Fl) column shard, W_v (Fl, C) row shard -> ALL-REDUCE
    emb, ln*, token-shift states                  replicated (identical on every rank)
    head                                          rank 0 only (it samples)

State lives in a SLOT POOL: wkv (L, S, Hl, N, N) fp32 and the two shifts (L, S, C). A batch is
always slots [0, B) (the engine keeps live rows compacted at the front), so the decode graphs
read and write the pool directly: no bind / unbind copies.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels.wkv7 import wkv7 as _wkv7, wkv7_replay
from .tp import TP

GROUP_K = 128


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
class QW(nn.Module):
    """One (in, out) weight, or a (B, in, out) batch of them: bf16, int8 or int6."""

    def __init__(self, w, quant="bf16", group_k=GROUP_K):
        super().__init__()
        self.quant = quant
        self.K, self.N = w.shape[-2], w.shape[-1]
        if quant == "int6":
            from .kernels.int6_gemm import quantize_int6
            lo, hi, s, g = quantize_int6(w, group_k)
            self.register_buffer("lo", lo); self.register_buffer("hi", hi); self.register_buffer("s", s)
            self.gk = g
        elif quant == "int8":
            from .kernels.int8_gemm import quantize_int8
            q, s, g = quantize_int8(w, group_k)
            self.register_buffer("q", q); self.register_buffer("s", s)
            self.gk = g
        else:
            self.register_buffer("w", w.contiguous())

    def forward(self, x):
        if self.quant == "int6":
            from .kernels.int6_gemm import int6_matmul_op, plan_int6
            sk, ud = plan_int6(x.shape[-2], self.N, self.K, x.shape[0] if x.dim() == 3 else 1, self.lo.device)
            return int6_matmul_op(x, self.lo, self.hi, self.s, self.gk, sk, ud)
        if self.quant == "int8":
            from .kernels.int8_gemm import int8_matmul_op, plan_int8
            sk, ud = plan_int8(x.shape[-2], self.N, self.K, x.shape[0] if x.dim() == 3 else 1)
            return int8_matmul_op(x, self.q, self.s, self.gk, sk, ud)
        return torch.bmm(x, self.w) if (x.dim() == 3 and self.w.dim() == 3) else x @ self.w

    def nbytes(self):
        return sum(b.numel() * b.element_size() for b in self.buffers())


def wkv7(r, w, k, v, a, b, state, H, N, cu_seqlens=None, final=True):
    """RWKV-7 recurrence for (B, T, H*N) inputs; `w` is the raw log-decay. Returns (o, new_state fp32 | None).
    Kernel: rwkv_next/kernels/wkv7.py (fused recurrent, Triton)."""
    B, T, C = r.shape
    bf = lambda t: t if t.dtype == torch.bfloat16 else t.bfloat16()
    o, s = _wkv7(bf(r), bf(w), bf(k), bf(v), bf(a), bf(b), state, H, N, cu_seqlens, final)
    return o.view(B, T, C), s


def _shift(x, prev, cu_seqlens=None, starts=None):
    """Token shift. x (B, T, C); prev (B, C) = the last x before this chunk (zeros for a fresh row).
    Packed (cu_seqlens): x is (1, sumT, C) and prev holds one row per sequence."""
    if cu_seqlens is None:
        if x.shape[1] == 1:
            return prev.unsqueeze(1)
        return torch.cat([prev.unsqueeze(1), x[:, :-1]], dim=1)
    sh = torch.empty_like(x)
    sh[:, 1:] = x[:, :-1]
    sh[0, starts] = prev
    return sh


# --------------------------------------------------------------------------- #
# layer
# --------------------------------------------------------------------------- #
class Layer(nn.Module):
    """One RWKV-7 block, this rank's shard. The forward is split at the two all-reduces:
    part_att(x) -> partial time-mix output; part_ffn(x) -> partial channel-mix output."""

    def __init__(self, i, C, H, N, tp: TP):
        super().__init__()
        self.i, self.C, self.H, self.N, self.tp = i, C, H, N, tp
        self.Hl = H // tp.world
        self.Cl = self.Hl * N

    # ---- time mix ------------------------------------------------------- #
    def part_att(self, x, v_first, wkv, att_shift, cu_seqlens=None, starts=None, rec=None, want_rec=False):
        """x: residual (B, T, C) (layer 0: the embedding, ln0 applied here).
        Returns (x, out_partial (B,T,C), v_first (B,T,Cl), new_wkv, new_att_shift, xln)."""
        if self.i == 0:
            x = F.layer_norm(x, (self.C,), self.ln0_w, self.ln0_b, 1e-5)
        B, T, C = x.shape
        Hl, N, Cl = self.Hl, self.N, self.Cl
        M = B * T
        xln = F.layer_norm(x, (C,), self.ln1_w, self.ln1_b, 1e-5)
        if cu_seqlens is None:
            new_shift = xln[:, -1].contiguous()
        else:
            new_shift = xln[0].index_select(0, (cu_seqlens[1:] - 1).long())
        xx = _shift(xln, att_shift, cu_seqlens, starts) - xln
        xall = torch.addcmul(xln.reshape(1, M, C), xx.reshape(1, M, C), self.x_mix)      # (6, M, C)
        rkv = self.W_rkv(xall[0:3])                                                     # (3, M, Cl)
        r, k, v = rkv[0], rkv[1], rkv[2]
        # LoRA branches: v/w/a batched (padded to their widest, 128 on G1 7B), g on its own (480): padding all
        # four to the widest read 2.3x the bytes (1 GB per decode step on the 7B)
        if isinstance(self.W_down, QW):         # quantised LoRA (RWKV_LORA_QUANT): the W8A16 kernels, batched
            h = self.W_down(xall[2:5])
            h = torch.stack([h[0], torch.tanh(h[1]), h[2]])
            o = self.W_up(h)
            g = self.W_g2(torch.sigmoid(self.W_g1(xall[5])))
        else:
            h = torch.bmm(xall[2:5], self.W_down)                                       # (3, M, d3)
            h = torch.stack([h[0], torch.tanh(h[1]), h[2]])
            o = torch.bmm(h, self.W_up)                                                 # (3, M, Cl)
            g = torch.sigmoid(xall[5] @ self.W_g1) @ self.W_g2                          # (M, Cl)
        w = -F.softplus(-(self.b_w + o[1])) - 0.5
        a = torch.sigmoid(self.b_a + o[2])
        if self.i == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.b_v + o[0])
        kk = k * self.kk_mul
        kk = F.normalize(kk.view(M, Hl, N), dim=-1, p=2.0).view(M, Cl)
        k = k * (1 + (a - 1) * self.ka_mul)
        vw = lambda t: t.view(B, T, Cl)
        ins = (vw(r), vw(w), vw(k), vw(v), vw(-kk), vw(kk * a))
        if rec is not None:                     # spec verify: keep what rollback replays (views, no copy)
            rec.append(ins)
        out, new_wkv = wkv7(*ins, wkv, Hl, N, cu_seqlens, final=not want_rec)
        out = F.group_norm(out.view(M, Cl).to(x.dtype), Hl, self.lnx_w, self.lnx_b, self.lnx_eps)
        out = out + ((r.view(M, Hl, N) * k.view(M, Hl, N) * self.r_k).sum(-1, keepdim=True) * v.view(M, Hl, N)).view(M, Cl)
        out = self.W_out(out * g).view(B, T, C)
        if want_rec:                            # returned rather than appended: no side effects under torch.compile
            return x, out, v_first, new_wkv, new_shift, xln, ins
        return x, out, v_first, new_wkv, new_shift, xln

    # ---- channel mix ---------------------------------------------------- #
    def part_ffn(self, x, att_out, ffn_shift, cu_seqlens=None, starts=None):
        """x + reduced att_out -> (x, ffn_partial, new_ffn_shift, xx)."""
        x = x + att_out
        B, T, C = x.shape
        xx = F.layer_norm(x, (C,), self.ln2_w, self.ln2_b, 1e-5)
        if cu_seqlens is None:
            new_shift = xx[:, -1].contiguous()
        else:
            new_shift = xx[0].index_select(0, (cu_seqlens[1:] - 1).long())
        k = torch.lerp(xx, _shift(xx, ffn_shift, cu_seqlens, starts), self.ffn_xk)
        hdn = torch.relu(self.W_k(k.reshape(B * T, C))) ** 2
        return x, self.W_v(hdn).view(B, T, C), new_shift, xx


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class RWKV7TP(nn.Module):
    def __init__(self, cfg: dict, tp: TP, device):
        super().__init__()
        self.cfg, self.tp, self.device = cfg, tp, torch.device(device)
        self.L, self.C, self.V, self.N = cfg["n_layer"], cfg["n_embd"], cfg["vocab_size"], cfg["head_size"]
        self.H = self.C // self.N
        if self.H % tp.world:
            raise ValueError(f"{self.H} heads do not split over tp={tp.world}")
        self.F = cfg["dim_ffn"]
        if self.F % tp.world:
            raise ValueError(f"dim_ffn {self.F} does not split over tp={tp.world}")
        self.Hl = self.H // tp.world
        self.layers = nn.ModuleList([Layer(i, self.C, self.H, self.N, tp) for i in range(self.L)])
        self.has_head = tp.rank == 0
        self.slots = 0
        self.tap_layers = ()            # drafter fusion layers (set before capture when MTP is on)
        self._tap_out = {}
        self._compiled = {}
        self._graphs = {}

    # ---- loading -------------------------------------------------------- #
    @staticmethod
    def config_of(sd):
        L = 1 + max(int(k.split(".")[1]) for k in sd.keys() if k.startswith("blocks."))
        V, C = sd["emb.weight"].shape
        H, N = sd["blocks.0.att.r_k"].shape
        F_ = sd["blocks.0.ffn.key.weight"].shape[0]
        return {"n_layer": L, "n_embd": C, "vocab_size": V, "head_size": N, "dim_ffn": F_, "n_head": H}

    @classmethod
    @torch.no_grad()
    def load(cls, path, tp: TP, device, quant="int6", group_k=GROUP_K, quant_head="int8",
             keep_int8=(), verbose=True):
        """Build this rank's shard from a plain RWKV-7 checkpoint. The checkpoint is mmap'd; each
        tensor is sliced on the CPU (a view), moved to the GPU and quantised there, one layer at a
        time, so neither the host nor the GPU ever holds the bf16 model.
        keep_int8: parts ('att', 'ffn') to keep at int8 when quant is int6."""
        from .weights import open_weights
        sd = open_weights(path)            # .pth, .safetensors, or a Hugging Face model folder (lazy, canonical keys)
        cfg = cls.config_of(sd)
        m = cls(cfg, tp, device)
        dev, bf = m.device, torch.bfloat16
        C, N, Cl = m.C, m.N, m.Hl * m.N
        r, Fl = tp.rank, m.F // tp.world
        cs = slice(r * Cl, (r + 1) * Cl)
        fs = slice(r * Fl, (r + 1) * Fl)
        hs = slice(r * m.Hl, (r + 1) * m.Hl)
        g = lambda k: sd[k]
        to = lambda t: t.to(dev, dtype=bf).contiguous()
        q_att = "int8" if (quant == "int6" and "att" in keep_int8) else quant
        q_ffn = "int8" if (quant == "int6" and "ffn" in keep_int8) else quant
        for i, ly in enumerate(m.layers):
            p = f"blocks.{i}."
            a = p + "att."
            vec = lambda name: to(g(a + name).reshape(-1)[cs]).view(1, Cl)
            for n_ in ("ln1", "ln2") + (("ln0",) if i == 0 else ()):
                ly.register_buffer(n_ + "_w", to(g(p + n_ + ".weight"))); ly.register_buffer(n_ + "_b", to(g(p + n_ + ".bias")))
            ly.register_buffer("x_mix", torch.cat([to(g(a + "x_" + c).reshape(1, C)) for c in "rkvwag"]).view(6, 1, C))
            wrkv = torch.stack([to(g(a + n_ + ".weight")[cs].t()) for n_ in ("receptance", "key", "value")])
            ly.W_rkv = QW(wrkv, q_att, group_k); del wrkv
            ly.W_out = QW(to(g(a + "output.weight")[:, cs].t()), q_att, group_k)
            has_v = (a + "v1") in sd
            dims = [g(a + "v1").shape[1] if has_v else 1, g(a + "w1").shape[1], g(a + "a1").shape[1]]
            dmax = max(dims)
            down = torch.zeros(3, C, dmax, device=dev, dtype=bf)
            up = torch.zeros(3, dmax, Cl, device=dev, dtype=bf)
            for j, nm in enumerate(("v", "w", "a")):
                if nm == "v" and not has_v:
                    continue
                down[j, :, :dims[j]] = to(g(a + nm + "1"))
                up[j, :dims[j]] = to(g(a + nm + "2")[:, cs])
            # LoRA stays bf16: int8 LoRA measured dCE +0.024 / KL x13 on G1 7.2B (the decay and gate LoRAs are
            # sensitive). RWKV_LORA_QUANT=int8 only exists to repeat that measurement (tools/test_quality.py).
            lq = os.environ.get("RWKV_LORA_QUANT", "")
            if lq in ("int8", "int6") and quant != "bf16":
                ly.W_down, ly.W_up = QW(down, "int8", 64), QW(up, "int8", 64)
                ly.W_g1, ly.W_g2 = QW(to(g(a + "g1")), "int8", 64), QW(to(g(a + "g2")[:, cs]), "int8", 0)
            else:
                ly.register_buffer("W_down", down); ly.register_buffer("W_up", up)
                ly.register_buffer("W_g1", to(g(a + "g1"))); ly.register_buffer("W_g2", to(g(a + "g2")[:, cs]))
            ly.register_buffer("b_w", vec("w0")); ly.register_buffer("b_a", vec("a0"))
            ly.register_buffer("b_v", vec("v0") if has_v else torch.zeros(1, Cl, device=dev, dtype=bf))
            ly.register_buffer("kk_mul", vec("k_k")); ly.register_buffer("ka_mul", vec("k_a"))
            ly.register_buffer("r_k", to(g(a + "r_k")[hs]))
            ly.register_buffer("lnx_w", to(g(a + "ln_x.weight")[cs])); ly.register_buffer("lnx_b", to(g(a + "ln_x.bias")[cs]))
            ly.lnx_eps = 1e-5 * (cfg.get("head_size_divisor", 8) ** 2)
            f = p + "ffn."
            ly.register_buffer("ffn_xk", to(g(f + "x_k")).view(1, 1, C))
            ly.W_k = QW(to(g(f + "key.weight")[fs].t()), q_ffn, group_k)
            ly.W_v = QW(to(g(f + "value.weight")[:, fs].t()), q_ffn, group_k)
            torch.cuda.empty_cache()
        # embedding: per-row int8 when quantised (rows are only gathered)
        emb = g("emb.weight")
        if quant in ("int8", "int6"):
            q = torch.empty(emb.shape, dtype=torch.int8, device=dev); s = torch.empty(emb.shape[0], 1, dtype=torch.float16, device=dev)
            for i0 in range(0, emb.shape[0], 8192):
                x = emb[i0:i0 + 8192].to(dev).float()
                sc = (x.abs().amax(-1, keepdim=True) / 127).clamp(min=1e-10)
                q[i0:i0 + 8192] = torch.round(x / sc).clamp(-127, 127).to(torch.int8); s[i0:i0 + 8192] = sc.half()
            m.register_buffer("emb_q", q); m.register_buffer("emb_s", s)
        else:
            m.register_buffer("emb_w", to(emb))
        m.register_buffer("ln_out_w", to(g("ln_out.weight"))); m.register_buffer("ln_out_b", to(g("ln_out.bias")))
        if m.has_head:
            qh = quant_head if quant in ("int8", "int6") else "bf16"
            m.head = QW(to(g("head.weight").t()), qh, group_k)
        del sd
        torch.cuda.empty_cache()
        if verbose:
            gb = sum(b.numel() * b.element_size() for b in m.buffers()) / 1e9
            print(f"[rank {tp.rank}] {os.path.basename(str(path))}: L{m.L} C{m.C} H{m.H} F{m.F} V{m.V}, "
                  f"tp {tp.world} (heads {m.Hl}/rank), quant {quant}"
                  + (f" (int8: {','.join(keep_int8)})" if keep_int8 and quant == 'int6' else "")
                  + f", weights {gb:.2f} GB on {dev}", flush=True)
        return m

    # ---- state pool ----------------------------------------------------- #
    def alloc_slots(self, S):
        """S rows of recurrent state on this rank."""
        dev, bf = self.device, torch.bfloat16
        self.slots = S
        self.wkv = torch.zeros(self.L, S, self.Hl, self.N, self.N, device=dev, dtype=torch.float32)
        self.att_shift = torch.zeros(self.L, S, self.C, device=dev, dtype=bf)
        self.ffn_shift = torch.zeros(self.L, S, self.C, device=dev, dtype=bf)
        return self.state_bytes_per_slot() * S

    def state_bytes_per_slot(self):
        return self.L * (self.Hl * self.N * self.N * 4 + 2 * self.C * 2)

    def reset_slots(self, idx):
        self.wkv[:, idx] = 0; self.att_shift[:, idx] = 0; self.ffn_shift[:, idx] = 0

    def move_slots(self, src, dst):
        """Copy slot rows src -> dst (lists). Used to keep live rows compacted at the front."""
        s = torch.as_tensor(src, device=self.device); d = torch.as_tensor(dst, device=self.device)
        self.wkv[:, d] = self.wkv[:, s]; self.att_shift[:, d] = self.att_shift[:, s]; self.ffn_shift[:, d] = self.ffn_shift[:, s]

    # ---- forward -------------------------------------------------------- #
    def embed(self, ids):
        if hasattr(self, "emb_q"):
            return self.emb_q[ids].to(torch.bfloat16) * self.emb_s[ids].to(torch.bfloat16)
        return self.emb_w[ids]

    def logits(self, x):
        """(M, C) final hidden (after ln_out) -> (M, V) on rank 0, None elsewhere."""
        return self.head(x) if self.has_head else None

    def _layer_fns(self, B):
        c = self._compiled.get(B)
        if c is not None:
            return c
        return [(ly.part_att, ly.part_ffn) for ly in self.layers]

    def compile(self, batch_sizes, mode="default"):
        """torch.compile the two halves of every layer for these decode batch sizes (T=1)."""
        self._recompile_n = 64 * len(self.layers) * max(1, len(batch_sizes))   # decode sizes + verify (B, T) shapes, per layer
        self.ensure_compile_limits()
        fns = [(torch.compile(ly.part_att, dynamic=False, mode=mode), torch.compile(ly.part_ffn, dynamic=False, mode=mode))
               for ly in self.layers]
        for B in batch_sizes:
            self._compiled[B] = fns
        return self

    def ensure_compile_limits(self):
        """torch._dynamo.config is THREAD-LOCAL: a limit raised in the loading thread reads back as the default (8)
        in any other thread, and every layer past the eighth then silently falls back to eager. Call in every
        thread that runs the compiled layers."""
        n = getattr(self, "_recompile_n", 0)
        if not n:
            return
        import torch._dynamo as dynamo
        for attr in ("recompile_limit", "cache_size_limit"):
            if hasattr(dynamo.config, attr):
                setattr(dynamo.config, attr, max(getattr(dynamo.config, attr) or 0, n))
        for attr in ("accumulated_recompile_limit", "accumulated_cache_size_limit"):
            if hasattr(dynamo.config, attr):
                setattr(dynamo.config, attr, max(getattr(dynamo.config, attr) or 0, 8 * n))

    def forward_decode(self, ids, B, rec=None):
        """One token for slots [0, B). ids (B,) long on device. Updates the pool in place.
        Returns final hidden (B, C) after ln_out (replicated on every rank)."""
        x = self.embed(ids).view(B, 1, self.C)
        v_first = None
        fns = self._layer_fns(B)
        for i, (fa, ff) in enumerate(fns):
            x, out, v_first, nw, nsa, xln = fa(x, v_first, self.wkv[i, :B], self.att_shift[i, :B])
            self.tp.all_reduce(out)
            x, fo, nsf, xx = ff(x, out, self.ffn_shift[i, :B])
            self.tp.all_reduce(fo)
            x = x + fo
            self.wkv[i, :B].copy_(nw); self.att_shift[i, :B].copy_(nsa); self.ffn_shift[i, :B].copy_(nsf)
            if i in self.tap_layers:
                self._tap_out[i] = x[:, -1]
        return F.layer_norm(x[:, -1], (self.C,), self.ln_out_w, self.ln_out_b, 1e-5)

    def forward_prefill(self, ids, cu_seqlens, slots, fresh, logits_positions=None, taps=(), h_positions=None):
        """Packed prefill of N sequences (ids (sumT,) long, cu_seqlens (N+1,) int32 on device) into
        `slots` (N,) long; fresh (N,) bool = start from zero state (else continue the slot's state:
        chunked prefill). Returns the final hidden (after ln_out) at each sequence's last token
        (N, C), or at `logits_positions` (flat indices) when given."""
        Nn = slots.numel()
        if fresh.any():
            self.reset_slots(slots[fresh])
        starts = cu_seqlens[:-1].long()
        x = self.embed(ids).view(1, -1, self.C)
        v_first = None
        tapped = {}
        for i, ly in enumerate(self.layers):
            wkv = self.wkv[i].index_select(0, slots)
            x, out, v_first, nw, nsa, _ = ly.part_att(x, v_first, wkv, self.att_shift[i].index_select(0, slots),
                                                       cu_seqlens, starts)
            self.tp.all_reduce(out)
            x, fo, nsf, _ = ly.part_ffn(x, out, self.ffn_shift[i].index_select(0, slots), cu_seqlens, starts)
            self.tp.all_reduce(fo)
            x = x + fo
            self.wkv[i].index_copy_(0, slots, nw); self.att_shift[i].index_copy_(0, slots, nsa)
            self.ffn_shift[i].index_copy_(0, slots, nsf)
            if i in taps and h_positions is not None:
                tapped[i] = x[0].index_select(0, h_positions)
        pos = logits_positions if logits_positions is not None else (cu_seqlens[1:] - 1).long()
        h = F.layer_norm(x[0].index_select(0, pos), (self.C,), self.ln_out_w, self.ln_out_b, 1e-5)
        if h_positions is None:
            return h
        hr = F.layer_norm(x[0].index_select(0, h_positions), (self.C,), self.ln_out_w, self.ln_out_b, 1e-5)
        return h, torch.cat([hr] + [tapped[l] for l in taps], -1)

    # ---- CUDA graphs ---------------------------------------------------- #
    def capture_decode(self, batch_sizes, verbose=True):
        """One graph per batch size: ids buffer -> pool update -> logits buffer (rank 0) / hidden.
        Every rank captures in the same order (the all-reduces are inside the graphs)."""
        import time
        pool = None
        batch_sizes = sorted({b for b in batch_sizes if 0 < b <= self.slots})
        Bmax = max(batch_sizes)
        self._g_ids = torch.zeros(Bmax, dtype=torch.long, device=self.device)
        self._g_h = torch.zeros(Bmax, self.C, dtype=torch.bfloat16, device=self.device)
        self._g_logits = torch.zeros(Bmax, self.V, dtype=torch.bfloat16, device=self.device) if self.has_head else None
        self._g_H = (torch.zeros(Bmax, self.C * (1 + len(self.tap_layers)), dtype=torch.bfloat16, device=self.device)
                     if self.tap_layers else None)
        for B in sorted(batch_sizes):
            t0 = time.time()
            snap = (self.wkv[:, :B].clone(), self.att_shift[:, :B].clone(), self.ffn_shift[:, :B].clone())
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):                    # warm-up: compile / autotune outside the capture
                    self._graph_body(B)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self._graph_body(B)
            pool = g.pool()
            self._graphs[B] = g
            self.wkv[:, :B].copy_(snap[0]); self.att_shift[:, :B].copy_(snap[1]); self.ffn_shift[:, :B].copy_(snap[2])
            torch.cuda.synchronize()
            if verbose and self.tp.rank == 0:
                print(f"[graph] decode B={B} captured in {time.time() - t0:.1f}s", flush=True)
        return self

    def _graph_body(self, B):
        h = self.forward_decode(self._g_ids[:B], B)
        self._g_h[:B].copy_(h)
        if getattr(self, "_g_H", None) is not None:
            self._g_H[:B].copy_(torch.cat([h] + [self._tap_out[l] for l in self.tap_layers], -1))
        if self.has_head:
            self._g_logits[:B].copy_(self.logits(h))

    def graph_sizes(self):
        return sorted(self._graphs)

    def decode(self, ids, B, limit=None):
        """ids (B,) long. Uses the smallest captured graph Bg >= B whose padding rows [B, Bg) stay
        below `limit` (slots there must be free: they get garbage), else eager at exactly B.
        Returns (logits (B, V) | None, hidden (B, C))."""
        limit = self.slots if limit is None else limit
        Bg = next((b for b in sorted(self._graphs) if B <= b <= max(limit, B)), None)
        if Bg is not None and Bg > B and Bg > limit:
            Bg = None
        if Bg is None:
            h = self.forward_decode(ids, B)
            if self.tap_layers:
                return (self.logits(h) if self.has_head else None), torch.cat([h] + [self._tap_out[l] for l in self.tap_layers], -1)
            return (self.logits(h) if self.has_head else None), h
        self._g_ids[:B].copy_(ids)
        if Bg > B:
            self._g_ids[B:Bg].zero_()
        self._graphs[Bg].replay()
        if self._g_H is not None:
            return (self._g_logits[:B] if self.has_head else None), self._g_H[:B]
        return (self._g_logits[:B] if self.has_head else None), self._g_h[:B]


    # ---- speculative decoding: verify T tokens per row, then roll back to n[b] --------------------------
    def forward_verify(self, ids, B, taps=()):
        """ids (B, T) long for slots [0, B). Reads the pool, does NOT write it (rollback does).
        Returns (hidden (B, T, C) after ln_out, H (B, T, C*(1+len(taps))) drafter inputs, rec)."""
        T = ids.shape[1]
        x = self.embed(ids).view(B, T, self.C)
        v_first = None
        rec = {"wkv": [], "att_in": [], "ffn_in": []}
        tapped = {}
        fns = next(iter(self._compiled.values()), None) or [(ly.part_att, ly.part_ffn) for ly in self.layers]
        for i, (fa, ff) in enumerate(fns):
            x, out, v_first, _, _, xln, ins = fa(x, v_first, self.wkv[i, :B], self.att_shift[i, :B], want_rec=True)
            rec["wkv"].append(ins)
            self.tp.all_reduce(out)
            x, fo, _, xx = ff(x, out, self.ffn_shift[i, :B])
            self.tp.all_reduce(fo)
            x = x + fo
            rec["att_in"].append(xln); rec["ffn_in"].append(xx)
            if i in taps:
                tapped[i] = x
        h = F.layer_norm(x, (self.C,), self.ln_out_w, self.ln_out_b, 1e-5)
        H = torch.cat([h] + [tapped[l] for l in taps], -1) if taps else h
        return h, H, rec

    def rollback(self, rec, n, B):
        """Pool slots [0, B) := state after the first n[b] tokens of the verify (1 <= n <= T): per layer, the
        wkv recurrence is replayed from the pool's incoming state over the first n[b] recorded steps only and
        written back in place (kernels/wkv7.wkv7_replay); token shifts are the recorded inputs at n-1."""
        n = n.to(self.device).long()
        Hl, N = self.Hl, self.N
        for i, ins in enumerate(rec["wkv"]):
            wkv7_replay(*(t[:B] for t in ins), self.wkv[i, :B], n, Hl, N)
        idx = (n - 1).view(1, B, 1, 1).expand(self.L, B, 1, self.C)
        self.att_shift[:, :B].copy_(torch.stack([t[:B] for t in rec["att_in"]]).gather(2, idx).squeeze(2))
        self.ffn_shift[:, :B].copy_(torch.stack([t[:B] for t in rec["ffn_in"]]).gather(2, idx).squeeze(2))

    def verify(self, ids, B, limit=None, taps=()):
        """ids (B, T) long, eager (the graphed path is spec.SpecGraphs, which captures the whole speculative step).
        -> (logits (B, T, V) | None, H (B, T, Cin), rec)."""
        T = ids.shape[1]
        h, H, rec = self.forward_verify(ids, B, taps)
        lg = self.head(h.reshape(-1, self.C)).view(B, T, -1) if self.has_head else None
        return lg, H, rec
