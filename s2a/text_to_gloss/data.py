"""Parallel English / ASL-gloss data loading and batching.

Expected layout (one sentence per line, line i of each file aligned):
    <data_dir>/train.en  train.gloss
    <data_dir>/dev.en    dev.gloss
    <data_dir>/test.en   test.gloss
`scripts/prepare_aslg.py` produces this from ASLG-PC12. A team-curated dataset that has
been checked by fluent signers can be dropped in with the same layout.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch

from s2a.text_to_gloss.tokenizer import GlossTokenizer


def read_split(data_dir: str | Path, split: str) -> tuple[list[str], list[str]]:
    data_dir = Path(data_dir)
    src = (data_dir / f"{split}.en").read_text(encoding="utf-8-sig").splitlines()
    tgt = (data_dir / f"{split}.gloss").read_text(encoding="utf-8-sig").splitlines()
    if len(src) != len(tgt):
        raise ValueError(f"{split}: {len(src)} English lines but {len(tgt)} gloss lines")
    pairs = [(s.strip(), t.strip()) for s, t in zip(src, tgt) if s.strip() and t.strip()]
    return [s for s, _ in pairs], [t for _, t in pairs]


class Batches:
    """Length-bucketed batches of roughly `max_tokens` tokens each."""

    def __init__(self, tokenizer: GlossTokenizer, src: list[str], tgt: list[str], max_tokens: int = 4096,
                 max_len: int = 128, shuffle: bool = True):
        self.pad = tokenizer.pad_id
        self.examples = []
        for s, t in zip(src, tgt):
            s_ids, t_ids = tokenizer.encode(s)[:max_len], tokenizer.encode(t, add_bos=True)[: max_len + 1]
            self.examples.append((s_ids, t_ids))
        self.max_tokens, self.shuffle = max_tokens, shuffle

    def __iter__(self):
        order = sorted(range(len(self.examples)), key=lambda i: len(self.examples[i][0]))
        batches, batch, longest = [], [], 0
        for i in order:
            longest = max(longest, len(self.examples[i][0]), len(self.examples[i][1]))
            if batch and longest * (len(batch) + 1) > self.max_tokens:
                batches.append(batch)
                batch, longest = [], max(len(self.examples[i][0]), len(self.examples[i][1]))
            batch.append(i)
        if batch:
            batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield self._collate([self.examples[i] for i in batch])

    def _collate(self, examples):
        def pad(seqs):
            out = torch.full((len(seqs), max(map(len, seqs))), self.pad, dtype=torch.long)
            for row, seq in enumerate(seqs):
                out[row, : len(seq)] = torch.tensor(seq)
            return out

        src = pad([s for s, _ in examples])
        tgt = pad([t for _, t in examples])
        return src, tgt[:, :-1], tgt[:, 1:]  # source, decoder input, decoder target
