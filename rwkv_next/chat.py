# -*- coding: utf-8 -*-
"""Incremental detokenisation (prompts are rendered by template.py)."""

from __future__ import annotations

class Detok:
    """Token ids -> text pieces, never splitting a UTF-8 character across pieces."""

    def __init__(self, tok):
        self.tok = tok
        self.buf = b""

    def push(self, ids) -> str:
        self.buf += self.tok.decodeBytes(ids)
        # longest prefix that is complete UTF-8
        for cut in range(len(self.buf), max(len(self.buf) - 4, -1), -1):
            try:
                s = self.buf[:cut].decode("utf-8")
            except UnicodeDecodeError:
                continue
            self.buf = self.buf[cut:]
            return s
        if len(self.buf) > 16:                                # invalid bytes from the model: don't stall
            return self.flush()
        return ""

    def flush(self) -> str:
        s = self.buf.decode("utf-8", errors="replace")
        self.buf = b""
        return s
