"""Shared subword (BPE) vocabulary for English and ASL gloss.

Gloss is lower-cased for the model and upper-cased again on output. Sharing one
vocabulary lets source and target embeddings be tied, so the model can easily learn
to "copy" words such as names that pass through to gloss unchanged.
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]


class GlossTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = (tokenizer.token_to_id(t) for t in SPECIALS)

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 8000) -> "GlossTokenizer":
        tok = Tokenizer(models.BPE(unk_token=UNK))
        tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
        tok.pre_tokenizer = pre_tokenizers.Metaspace()
        tok.decoder = decoders.Metaspace()
        tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS))
        return cls(tok)

    @classmethod
    def load(cls, path: str | Path) -> "GlossTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        self.tok.save(str(path))

    def __len__(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        ids = self.tok.encode(text).ids + [self.eos_id]
        return [self.bos_id] + ids if add_bos else ids

    def decode(self, ids: list[int]) -> str:
        ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.tok.decode(ids).strip()

    def decode_gloss(self, ids: list[int]) -> str:
        return self.decode(ids).upper()
