# -*- coding: utf-8 -*-
"""Rank 0 -> worker command broadcast through shared memory.

Every model call on rank 0 (prefill, decode step, slot move, spec step, capture) has to be
mirrored on the other TP ranks in the same order with the same arguments, because the
collectives inside them pair up across ranks. The commands are tiny (an op name and a few
hundred ints), so they go through a shared-memory ring instead of NCCL: no GPU sync, no
extra collective, ~5-20 us to wake a polling worker.

Layout: a header of 8-byte sequence numbers (one written by the writer, one per reader for
acks) followed by `slots` message slots of `slot_bytes` each. The writer waits until every
reader has consumed slot (seq - slots) before overwriting it; readers spin on the writer's
sequence number, then back off to short sleeps when idle so an idle server does not burn
a core per GPU.
"""

from __future__ import annotations

import pickle
import struct
import time
from multiprocessing import shared_memory

_HDR = 64          # bytes per sequence counter (own cache line)


class Channel:
    def __init__(self, name: str | None, n_readers: int, slots: int = 64, slot_bytes: int = 1 << 20, create=False):
        self.n_readers, self.slots, self.slot_bytes = n_readers, slots, slot_bytes
        size = _HDR * (1 + n_readers) + slots * slot_bytes
        if create:
            self.shm = shared_memory.SharedMemory(create=True, size=size)
            self.shm.buf[:_HDR * (1 + n_readers)] = bytes(_HDR * (1 + n_readers))
        else:
            self.shm = shared_memory.SharedMemory(name=name)
        self.name = self.shm.name
        self.buf = self.shm.buf
        self.seq = 0

    # ---- counters ----------------------------------------------------------
    def _get(self, i):
        return struct.unpack_from("<q", self.buf, _HDR * i)[0]

    def _set(self, i, v):
        struct.pack_into("<q", self.buf, _HDR * i, v)

    def _slot(self, seq):
        off = _HDR * (1 + self.n_readers) + (seq % self.slots) * self.slot_bytes
        return off

    # ---- writer (rank 0) ---------------------------------------------------
    def send(self, obj):
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if len(data) + 8 > self.slot_bytes:
            raise ValueError(f"command of {len(data)} bytes exceeds the channel slot ({self.slot_bytes})")
        nxt = self.seq + 1
        while nxt - min(self._get(1 + r) for r in range(self.n_readers)) > self.slots:
            time.sleep(0)                                    # a reader is a full ring behind
        off = self._slot(nxt)
        struct.pack_into("<q", self.buf, off, len(data))
        self.buf[off + 8: off + 8 + len(data)] = data
        self._set(0, nxt)                                     # publish
        self.seq = nxt

    # ---- reader (ranks > 0) -------------------------------------------------
    def recv(self, reader: int):
        """Block until the next command; reader = 0..n_readers-1."""
        nxt = self.seq + 1
        spins = 0
        while self._get(0) < nxt:
            spins += 1
            if spins > 20000:
                time.sleep(0.0005 if spins < 200000 else 0.005)
        off = self._slot(nxt)
        n = struct.unpack_from("<q", self.buf, off)[0]
        obj = pickle.loads(bytes(self.buf[off + 8: off + 8 + n]))
        self.seq = nxt
        self._set(1 + reader, nxt)                            # ack
        return obj

    def close(self, unlink=False):
        try:
            self.buf = None
            self.shm.close()
            if unlink:
                self.shm.unlink()
        except Exception:
            pass
