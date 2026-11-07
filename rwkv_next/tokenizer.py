# -*- coding: utf-8 -*-
"""Standalone RWKV TRIE tokenizer (vendored from BlinkDL's RWKV-LM).

Kept dependency-free (no training stack needed). The vocab file ships in `rwkv_next/assets/`; a few
fallback locations are searched (see resolve_vocab).
"""

import os

_DEFAULT_VOCAB = "wr_vocab_v20230424.txt"


def resolve_vocab(path=None):
    """Locate the vocab file. Explicit path wins, then packaged asset, then repo."""
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"vocab not found: {path}")
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "assets", _DEFAULT_VOCAB),           # packaged (standalone)
        os.path.join(os.path.dirname(here), "infer", "rwkv", _DEFAULT_VOCAB),  # in-repo
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"could not find {_DEFAULT_VOCAB}; tried:\n  " + "\n  ".join(candidates))


class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to: list
    values: set

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while fr is not None:
            if fr.ch is not None:
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>" % (ret[::-1], self.values)

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            if val is None:
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        u: TRIE = self
        ch: int = key[idx]
        ret = None
        while u.to[ch] is not None:
            u = u.to[ch]
            idx += 1
            if u.values:
                ret = idx, u, u.values
            if idx == len(key):
                break
            ch = key[idx]
        return ret


class RWKVTokenizer:
    """Byte-level TRIE tokenizer. `encode(str) -> list[int]`, `decode(list[int]) -> str`."""

    def __init__(self, vocab_path=None):
        self.vocab_path = resolve_vocab(vocab_path)
        self.idx2token = {}
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" "): l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(" "):])
            self.idx2token[idx] = x

        self.token2idx = {v: int(k) for k, v in self.idx2token.items()}
        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src: bytes):
        idx = 0
        tokens = []
        while idx < len(src):
            _idx = idx
            found = self.root.find_longest(src, idx)
            if found is None:
                raise ValueError(f"un-encodable byte at position {idx}: {src[idx]!r}")
            idx, _, values = found
            assert idx != _idx
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b"".join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        try:
            return self.decodeBytes(tokens).decode("utf-8")
        except Exception:
            return "�"  # bad utf-8
