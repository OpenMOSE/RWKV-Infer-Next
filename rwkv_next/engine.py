# -*- coding: utf-8 -*-
"""Engine processes: one per GPU (tensor-parallel rank).

Rank 0 runs the scheduler (continuous batching, chunked prefill, sampling, MTP drafting) and
mirrors every model call to the other ranks through the shared-memory channel; ranks > 0 just
execute those calls. All ranks keep identical slot pools (their own head shard of the state),
so a call means the same thing everywhere.

Slot layout (rank 0 decides, every rank applies the same moves)
    [0, nd)        decoding rows    -> one decode step = slots [0, nd), i.e. pool[:nd] for the graphs
    top of pool    rows still in prefill (chunked: a long prompt takes several passes)
    the rest       free (graph padding rows land here)
A finished decoding row is filled by moving the last decoding row into its slot.

Front <-> rank 0: two multiprocessing queues.
    in:  ("add", req_id, prompt_ids, params) | ("abort", req_id) | ("stop",)
    out: ("info", dict) | ("ready", info) | ("tokens", [(req_id, [ids], finish_reason|None, info)]) | ("stats", dict)
         | ("error", text)
"""

from __future__ import annotations

import os, sys
import queue
import time
import traceback
from collections import deque

import torch

END_TOKEN = 0          # RWKV world vocab: 0 = end of text


# --------------------------------------------------------------------------- #
# model calls, identical on every rank
# --------------------------------------------------------------------------- #
class Executor:
    def __init__(self, model, sampler=None, pool=None):
        self.m = model
        self.dev = model.device
        self.sampler = sampler          # rank 0 only
        self.pool = pool                # rank 0 only: MTP drafter state (drafter.DraftPool)
        self.rec = None
        self.specg = None               # spec.SpecGraphs (every rank, when MTP is on)

    def prefill(self, flat, lens, slots, fresh, h_pos=None):
        """h_pos: flat positions whose drafter input rows rank 0 needs (MTP); then returns (h_last, H rows)."""
        dev = self.dev
        ids = torch.tensor(flat, dtype=torch.long, device=dev)
        cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
        cu[1:] = torch.tensor(lens).cumsum(0)
        hp = torch.tensor(h_pos, dtype=torch.long, device=dev) if h_pos else None
        taps = self.pool.taps if self.pool is not None else self.m.tap_layers
        return self.m.forward_prefill(ids, cu.to(dev), torch.tensor(slots, dtype=torch.long, device=dev),
                                      torch.tensor(fresh, dtype=torch.bool, device=dev), taps=taps, h_positions=hp)

    def verify(self, toks, B, limit):
        lg, H, self.rec = self.m.verify(torch.tensor(toks, dtype=torch.long, device=self.dev), B, limit,
                                        taps=self.pool.taps if self.pool is not None else ())
        return lg, H

    def spec(self, Bg, K, B=0, ids=None, params=None):
        """One graphed spec step (rwkv_next/spec.py). Rank 0 returns (picks, n); workers just replay."""
        if self.pool is not None:
            return self.specg.run_rank0(Bg, K, B, ids, params)
        self.specg.run_worker(Bg, K)

    def rollback(self, n, B):
        self.m.rollback(self.rec, torch.tensor(n, dtype=torch.long, device=self.dev), B)
        self.rec = None

    def decode(self, ids, B, limit):
        return self.m.decode(torch.tensor(ids, dtype=torch.long, device=self.dev), B, limit)

    def move(self, src, dst):
        self.m.move_slots(src, dst)
        if self.sampler is not None:
            self.sampler.move(src, dst)
        if self.pool is not None:
            self.pool.move(src, dst)


def worker_loop(ex: Executor, chan, reader):
    ex.m.ensure_compile_limits()
    while True:
        op, args = chan.recv(reader)
        if op == "stop":
            return
        getattr(ex, op)(*args)


# --------------------------------------------------------------------------- #
# scheduler (rank 0)
# --------------------------------------------------------------------------- #
class Row:
    __slots__ = ("rid", "prompt", "pos", "params", "out", "pending", "t0", "t_first", "max_new", "stop_ids")

    def __eq__(self, other):
        return self is other

    __hash__ = object.__hash__

    def __init__(self, rid, prompt, params):
        self.rid, self.prompt, self.params = rid, prompt, params
        self.pos = 0                    # prompt tokens already prefilled
        self.out = 0                    # tokens generated
        self.pending = None             # next input token (sampled, not yet fed)
        self.t0 = time.time()
        self.t_first = None
        self.max_new = int(params.get("max_tokens") or 4096)
        self.stop_ids = set(params.get("stop_token_ids") or [END_TOKEN])


class Scheduler:
    """Slots: decoding rows at [0, nd) (self.dec, slot == index); rows in prefill take slots from the
    TOP of the pool (self.pre: slot -> row). A decode step pads B up to the next captured graph
    size, and the padding rows write garbage into slots [B, Bg) — which are free as long as no
    prefill slot lies below Bg (else that step runs eager at exactly B)."""

    def __init__(self, ex: Executor, chan, cfg, in_q, out_q, n_workers):
        self.ex, self.chan, self.cfg, self.in_q, self.out_q = ex, chan, cfg, in_q, out_q
        self.S = ex.m.slots
        self.dec: list[Row] = []
        self.pre: dict[int, Row] = {}
        self.waiting: deque[Row] = deque()
        self.by_id = {}
        self.chunk = int(cfg.get("PREFILL_CHUNK", 2048))
        self.chunk_busy = int(cfg.get("PREFILL_CHUNK_BUSY", 512))
        self.pool = ex.pool
        from .mtp import parse_k_rules
        self.k_rules = parse_k_rules(cfg.get("MTP_K") or "0:0")
        self.mtp_calls = self.mtp_tokens = 0
        self.mtp_total = [0, 0]                    # (row-steps, tokens) since start
        self.max_prompt = int(cfg.get("MAX_PROMPT_TOKENS", 32768))
        self.stats = {"tokens": 0, "steps": 0, "prefill_tokens": 0, "t": time.time()}
        self._last_stats = time.time()

    # ---- mirrored calls ----------------------------------------------------
    def call(self, op, *args):
        if self.chan is not None:
            self.chan.send((op, args))
        return getattr(self.ex, op)(*args)

    # ---- queue handling ----------------------------------------------------
    def _drain(self, block):
        try:
            msg = self.in_q.get(timeout=0.05) if block else self.in_q.get_nowait()
        except queue.Empty:
            return True
        while True:
            if msg[0] == "stop":
                return False
            if msg[0] == "add":
                _, rid, prompt, params = msg
                if len(prompt) > self.max_prompt:
                    self.out_q.put(("tokens", [(rid, [], "error:prompt_too_long", {})]))
                elif not prompt:
                    self.out_q.put(("tokens", [(rid, [], "error:empty_prompt", {})]))
                else:
                    r = Row(rid, prompt, params)
                    self.waiting.append(r); self.by_id[rid] = r
            elif msg[0] == "abort":
                self._abort(msg[1])
            try:
                msg = self.in_q.get_nowait()
            except queue.Empty:
                return True

    def _abort(self, rid):
        r = self.by_id.get(rid)
        if r is None:
            return
        if r in self.waiting:
            self.waiting.remove(r); self.by_id.pop(rid, None)
        else:
            r.params["_done"] = True
            self._reap()

    # ---- slots -------------------------------------------------------------
    def _free_top(self):
        used = set(self.pre)
        for s_ in range(self.S - 1, len(self.dec) - 1, -1):
            if s_ not in used:
                return s_
        return None

    def _admit(self):
        while self.waiting and len(self.dec) + len(self.pre) < self.S:
            s_ = self._free_top()
            if s_ is None:
                break
            self.pre[s_] = self.waiting.popleft()

    def _reap(self):
        """Drop finished rows. Decode holes are filled from the end of the decode region."""
        for s_ in [s_ for s_, r in self.pre.items() if r.params.get("_done")]:
            self.by_id.pop(self.pre.pop(s_).rid, None)
        i = len(self.dec) - 1
        moves_s, moves_d = [], []
        home = {}             # decode index -> the slot its row's state is in NOW (moves are applied in one batch)
        while i >= 0:
            if self.dec[i].params.get("_done"):
                self.by_id.pop(self.dec[i].rid, None)
                last = len(self.dec) - 1
                if i != last:
                    # the row at `last` may itself have been moved there in this pass: copy from where its state
                    # really is, or a batched t[dst] = t[src] would give slot i a finished row's state
                    src = home.pop(last, last)
                    moves_s.append(src); moves_d.append(i)
                    home[i] = src
                    self.dec[i] = self.dec[last]
                self.dec.pop()
            i -= 1
        if moves_s:
            self.call("move", moves_s, moves_d)

    # ---- steps -------------------------------------------------------------
    def _prefill_step(self, events):
        # prefill is compute bound (7B: ~4.8K tok/s per GPU), so a 2048-token chunk stalls every decoding
        # row for ~0.4 s; while rows are decoding, chunks are capped so the stall stays ~0.1 s
        budget = self.chunk if not self.dec else min(self.chunk, self.chunk_busy)
        flat, lens, slots, fresh, sel = [], [], [], [], []
        for s_, r in sorted(self.pre.items(), key=lambda kv: kv[1].t0):
            if budget <= 0:
                break
            take = min(budget, len(r.prompt) - r.pos)
            flat += r.prompt[r.pos:r.pos + take]; lens.append(take); slots.append(s_); fresh.append(r.pos == 0)
            sel.append((s_, r, take)); budget -= take
        if not sel:
            return
        h_pos, plan = None, []
        if self.pool is not None:
            # drafter inputs: the prompt's last `window` positions before the final one are committed (pairs
            # (H at q, token at q+1)); the final position's H stays pending for the first draft
            h_pos, off = [], 0
            W = self.pool.window
            for s_, r, take in sel:
                L = len(r.prompt)
                lo, hi = max(r.pos, L - 1 - W, 0), min(r.pos + take, L - 1)
                qs = list(range(lo, hi))
                fin = r.pos + take >= L
                plan.append((s_, r, len(h_pos), qs, fin))
                h_pos += [off + q - r.pos for q in qs] + ([off + L - 1 - r.pos] if fin else [])
                off += take
            out = self.call("prefill", flat, lens, slots, fresh, h_pos)
            # no drafter rows in this chunk (an early chunk of a prompt longer than chunk + window): prefill
            # then returns h alone -- unpacking it would split h's rows (or raise with one row)
            h, Hr = out if h_pos else (out, None)
            for s_, r, i0, qs, fin in plan:
                if r.pos == 0:
                    self.pool.reset([s_])
                if qs:
                    toks = torch.tensor(r.prompt[qs[0] + 1: qs[-1] + 2], dtype=torch.long, device=Hr.device)
                    self.pool.commit_rows(s_, Hr[i0:i0 + len(qs)], toks, qs[0])
                if fin:
                    self.pool.H[s_] = Hr[i0 + len(qs)]
        else:
            h = self.call("prefill", flat, lens, slots, fresh)
        self.stats["prefill_tokens"] += len(flat)
        done = []
        for j, (s_, r, take) in enumerate(sel):
            r.pos += take
            if r.pos >= len(r.prompt):
                done.append((j, s_, r))
        if not done:
            return
        logits = self.ex.m.logits(h.index_select(0, torch.tensor([j for j, _, _ in done], device=h.device)))
        self.ex.sampler.reset([s_ for _, s_, _ in done])
        toks = self.ex.sampler(logits, [r.params for _, _, r in done], [s_ for _, s_, _ in done]).tolist()
        for (_, _, r), t in zip(done, toks):
            r.t_first = time.time()
            self._emit(r, t, events)
        for _, _, r in done:
            if r.params.get("_done"):
                continue
            # promote: its prefill slot -> slot len(dec), the end of the decode region (one move each,
            # looked up afresh: a swap below can relocate a row that is promoted next)
            s_ = next(k for k, v in self.pre.items() if v is r)
            del self.pre[s_]
            d = len(self.dec)
            if d in self.pre:                        # the target holds a prefill row: swap it up into s_
                self.pre[s_] = self.pre.pop(d)
                self.call("move", [s_, d], [d, s_])
            elif s_ != d:
                self.call("move", [s_], [d])
            self.dec.append(r)
        self._reap()

    def _emit(self, r, t, events):
        r.out += 1
        self.stats["tokens"] += 1
        fin = None
        if t in r.stop_ids:
            fin = "stop"
        elif r.out >= r.max_new:
            fin = "length"
        events.append((r.rid, [] if fin == "stop" else [t], fin, {}))
        r.pending = None if fin else t
        if fin:
            r.params["_done"] = True

    def _decode_step(self, events):
        B = len(self.dec)
        if B == 0:
            return
        limit = min(self.pre) if self.pre else self.S       # padding must stay below every prefill slot
        ids = [r.pending for r in self.dec]
        params = [r.params for r in self.dec]
        from .mtp import k_for
        K = k_for(self.k_rules, B) if self.pool is not None else 0
        if K > 0:
            pen = any(p["presence_penalty"] or p["frequency_penalty"] for p in params)
            sg = self.ex.specg
            from .spec import PREFILTER
            # the spec graph samples every setting exactly but top-k > PREFILTER (spec.py)
            Bg = sg.pick(B, K, limit) if sg is not None and all((p["top_k"] or 0) <= PREFILTER for p in params) else None
            if Bg is not None or not pen:            # the eager spec step has no penalty state: penalties go plain
                return self._spec_step(events, B, K, ids, params, limit, Bg)
        logits, H = self.call("decode", ids, B, limit)
        toks = self.ex.sampler(logits, params, list(range(B))).tolist()
        if self.pool is not None:
            self.pool.plain_step(B, H, torch.tensor(ids, dtype=torch.long, device=H.device))
        self.stats["steps"] += 1
        for r, t in zip(self.dec, toks):
            self._emit(r, t, events)
        self._reap()

    def _spec_step(self, events, B, K, ids, params, limit, Bg=None):
        """Draft K tokens per row (rank 0), verify K+1 positions on every rank, sample every position, accept the
        matching prefix, roll every rank back to it. The emitted tokens are sampled by the normal sampler at
        each position, so the output distribution is the plain decode's."""
        if Bg is not None:                                                  # the whole step is one graph per rank
            pl, nl = self.call("spec", Bg, K, B, ids, params)
            self.stats["steps"] += 1
            self.mtp_calls += B; self.mtp_tokens += sum(nl)
            self.mtp_total[0] += B; self.mtp_total[1] += sum(nl)
            for r, row, k in zip(self.dec, pl, nl):
                for tk in row[:k]:
                    self._emit(r, tk, events)
                    if r.params.get("_done"):
                        break
            self._reap()
            return
        dev = self.ex.dev
        t = torch.tensor(ids, dtype=torch.long, device=dev)
        drafts = self.pool.draft(B, t, K)                                   # (B, K)
        toks = torch.cat([t.unsqueeze(1), drafts], 1)                       # (B, K+1)
        T = K + 1
        lg, Hs = self.call("verify", toks.tolist(), B, limit)
        picks = self.ex.sampler(lg.reshape(B * T, -1), [p for p in params for _ in range(T)]).view(B, T)
        j = (picks[:, :K] == drafts).long().cumprod(1).sum(1)
        n = j + 1
        nl, pl = n.tolist(), picks.tolist()
        self.call("rollback", nl, B)
        self.pool.accept(B, Hs, drafts, j)
        self.stats["steps"] += 1
        self.mtp_calls += B; self.mtp_tokens += sum(nl)
        self.mtp_total[0] += B; self.mtp_total[1] += sum(nl)
        for r, row, k in zip(self.dec, pl, nl):
            for tk in row[:k]:
                self._emit(r, tk, events)
                if r.params.get("_done"):
                    break
        self._reap()

    def run(self):
        self.ex.m.ensure_compile_limits()          # thread-local dynamo config (see model.ensure_compile_limits)
        self.out_q.put(("ready", {"slots": self.S}))
        while True:
            idle = not self.dec and not self.pre and not self.waiting
            if idle:
                # shapes tuned at run time (new prefill buckets etc.) are persisted while idle: ranks leave through
                # os._exit, so an atexit save would never run
                try:
                    from .kernels import autotune_cache
                    autotune_cache.save_if_new()
                except Exception:
                    pass
            if not self._drain(block=idle):
                break
            self._admit()
            events = []
            if self.pre:
                self._prefill_step(events)
            if self.dec:
                self._decode_step(events)
            if events:
                self.out_q.put(("tokens", events))
            now = time.time()
            if now - self._last_stats > 1.0:
                dt = now - self.stats["t"]
                self.out_q.put(("stats", {"active": len(self.dec) + len(self.pre), "decoding": len(self.dec),
                                          "prefilling": len(self.pre), "waiting": len(self.waiting),
                                          "tok_s": self.stats["tokens"] / dt, "prefill_tok_s": self.stats["prefill_tokens"] / dt,
                                          "steps_s": self.stats["steps"] / dt, "slots": self.S,
                                          "mtp_tok_per_step": (self.mtp_tokens / self.mtp_calls) if self.mtp_calls else None}))
                self.stats = {"tokens": 0, "steps": 0, "prefill_tokens": 0, "t": now}
                self.mtp_calls = self.mtp_tokens = 0
                self._last_stats = now
        if self.chan is not None:
            self.chan.send(("stop", ()))


# --------------------------------------------------------------------------- #
# process entry
# --------------------------------------------------------------------------- #
def rank_main(rank, world, cfg, chan_name, in_q, out_q, port):
    try:
        from . import tp as tpm
        from .channel import Channel
        from .model import RWKV7TP
        from .sampler import Sampler
        devs = [int(x) for x in str(cfg.get("DEVICES") or "").split(",") if x.strip() != ""] or list(range(world))
        T = tpm.init(rank, world, port, devs[rank])
        dev = torch.device("cuda", devs[rank])
        keep8 = tuple(x for x in str(cfg.get("INT6_KEEP_INT8") or "").replace(" ", "").split(",") if x)
        m = RWKV7TP.load(cfg["MODEL_PATH"], T, dev, quant=cfg["QUANT"], group_k=int(cfg["QUANT_GROUP_K"]),
                         keep_int8=keep8, verbose=True)
        S = int(cfg["MAX_BATCH"])
        m.alloc_slots(S)
        sizes = sorted({min(int(x), S) for x in str(cfg["GRAPH_BATCH_SIZES"]).split(",") if x.strip()})
        pool = None
        from .mtp import parse_k_rules, k_for
        rules = parse_k_rules(cfg.get("MTP_K") or "0:0")
        from .drafter import find_sidecar
        mtp_path = find_sidecar(cfg["MODEL_PATH"], cfg.get("MTP_PATH"))
        cfg = dict(cfg, MTP_PATH=mtp_path or "")
        if mtp_path:
            kmax = max([k for _, k in rules] + [1])
            if rank == 0:
                from .drafter import DraftPool
                pool = DraftPool(cfg["MTP_PATH"], m, S, K_max=kmax, draft_vocab=int(cfg.get("MTP_DRAFT_VOCAB") or 16384),
                                 compile=bool(cfg.get("COMPILE")), force=bool(cfg.get("MTP_FORCE")))
                m.tap_layers = pool.taps
                print(f"[mtp] drafter {os.path.basename(cfg['MTP_PATH'])}: fuse {list(pool.taps)}, window {pool.window}, "
                      f"K rules {rules}", flush=True)
        if cfg.get("COMPILE"):
            m.compile(sizes)
        m.capture_decode(sizes, verbose=(rank == 0))
        specg = sampler = None
        if cfg.get("MTP_PATH"):
            # one spec-step graph per (graph size b, every K that some real batch B in (prev, b] maps to);
            # every rank captures the same list in the same order (the collectives pair up)
            from .spec import SpecGraphs
            sampler = Sampler(S, m.V, dev) if rank == 0 else None
            specg = SpecGraphs(m, pool, counts=sampler.counts if sampler is not None else None)
            prev = 0
            for b in sizes:
                for K in sorted({k_for(rules, B) for B in range(prev + 1, b + 1)} - {0}):
                    specg.capture(b, K, verbose=(rank == 0))
                prev = b
        # prefill warm-up: the int6/int8 kernels autotune per M bucket (powers of two up to 128) and the
        # first request would pay for it (~14 s TTFT measured). Lengths cover every bucket. Slot 0 is
        # reset afterwards (all ranks do the same: the collectives pair up).
        t0 = time.time()
        ex0 = Executor(m)
        for n in (1, 2, 4, 8, 16, 32, 64, 128, 300, 1000):
            ex0.prefill([1] * n, [n], [0], [True])
            ex0.prefill([1] * min(n, 3), [min(n, 3)] * 1, [0], [False])
        m.reset_slots(torch.tensor([0], device=dev))
        torch.cuda.synchronize()
        if rank == 0:
            print(f"[warm-up] prefill buckets in {time.time() - t0:.1f}s", flush=True)
            from .kernels import autotune_cache
            try:
                autotune_cache.save(verbose=True)
            except Exception:
                pass
        chan = Channel(chan_name, world - 1) if world > 1 else None
        if rank == 0:
            ex = Executor(m, sampler or Sampler(S, m.V, dev), pool)
            ex.specg = specg
            info = {"model": cfg.get("MODEL_LABEL") or os.path.basename(cfg["MODEL_PATH"]), "tp": world, "quant": cfg["QUANT"],
                    "mtp": ((pool.base_model or os.path.basename(cfg["MTP_PATH"].rstrip("/"))) + f" ({os.path.basename(cfg['MTP_PATH'].rstrip('/'))})"
                            + (" [forced: trained for another model]" if pool.base_model and cfg.get("MODEL_LABEL")
                               and pool.base_model != cfg["MODEL_LABEL"] else "")) if pool is not None else None,
                    "layers": m.L, "n_embd": m.C, "heads": m.H, "vocab": m.V, "graph_sizes": sizes,
                    "vram_gb": torch.cuda.memory_allocated(dev) / 1e9}
            sch = Scheduler(ex, chan, cfg, in_q, out_q, world - 1)
            out_q.put(("info", info))
            sch.run()
        else:
            ex = Executor(m)
            ex.specg = specg
            worker_loop(ex, chan, rank - 1)
    except BaseException:
        tb = f"rank {rank}: " + traceback.format_exc()
        print("[engine error] " + tb, file=sys.stderr, flush=True)       # the console always gets it
        try:
            out_q.put(("error", tb))
            out_q.close(); out_q.join_thread()       # flush the queue's feeder thread: os._exit would drop the message
        except Exception:
            pass
    finally:
        os._exit(0)
