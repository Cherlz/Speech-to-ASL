"""Our English -> ASL gloss model: a Transformer encoder-decoder trained from scratch."""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass

import torch
from torch import nn

# Harmless PyTorch notice about a speed optimization that norm_first models can't use.
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    heads: int = 4
    encoder_layers: int = 3
    decoder_layers: int = 3
    ff_dim: int = 1024
    dropout: float = 0.1
    max_len: int = 256
    pad_id: int = 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class TextToGlossTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # One embedding table for English, gloss and the output projection (weight tying).
        self.embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.pos = PositionalEncoding(config.d_model, config.max_len)
        self.dropout = nn.Dropout(config.dropout)
        self.transformer = nn.Transformer(
            d_model=config.d_model, nhead=config.heads, num_encoder_layers=config.encoder_layers,
            num_decoder_layers=config.decoder_layers, dim_feedforward=config.ff_dim, dropout=config.dropout,
            batch_first=True, norm_first=True,
        )
        self.out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.out.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=config.d_model ** -0.5)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.pos(self.embed(ids) * math.sqrt(self.config.d_model)))

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_pad = src.eq(self.config.pad_id)
        return self.transformer.encoder(self._embed(src), src_key_padding_mask=src_pad), src_pad

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        causal = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device, dtype=torch.bool)
        h = self.transformer.decoder(self._embed(tgt), memory, tgt_mask=causal,
                                     tgt_key_padding_mask=tgt.eq(self.config.pad_id),
                                     memory_key_padding_mask=src_pad)
        return self.out(h)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, src_pad = self.encode(src)
        return self.decode(tgt_in, memory, src_pad)

    # ---------- inference ----------
    @torch.no_grad()
    def greedy(self, src: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 100) -> list[list[int]]:
        memory, src_pad = self.encode(src)
        ys = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            nxt = self.decode(ys, memory, src_pad)[:, -1].argmax(-1)
            nxt = nxt.masked_fill(done, self.config.pad_id)
            ys = torch.cat([ys, nxt[:, None]], dim=1)
            done |= nxt.eq(eos_id)
            if done.all():
                break
        return [_strip(row.tolist(), eos_id) for row in ys[:, 1:]]

    @torch.no_grad()
    def beam_search(self, src: torch.Tensor, bos_id: int, eos_id: int, beam: int = 5, max_len: int = 100,
                    length_penalty: float = 0.6) -> list[int]:
        """Beam search for a single sentence (src has batch size 1)."""
        memory, src_pad = self.encode(src)
        beams = [([bos_id], 0.0)]
        finished = []
        for _ in range(max_len):
            ys = torch.tensor([b[0] for b in beams], device=src.device)
            logp = self.decode(ys, memory.expand(len(beams), -1, -1), src_pad.expand(len(beams), -1))[:, -1]
            logp = logp.log_softmax(-1)
            top, idx = logp.topk(beam, dim=-1)
            candidates = [(seq + [i], score + s) for (seq, score), ss, ii in zip(beams, top.tolist(), idx.tolist())
                          for s, i in zip(ss, ii)]
            candidates.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for seq, score in candidates:
                if seq[-1] == eos_id:
                    finished.append((seq, score / (len(seq) - 1) ** length_penalty))
                else:
                    beams.append((seq, score))
                if len(beams) == beam:
                    break
            if len(finished) >= beam or not beams:
                break
        if not finished:
            finished = [(seq, score / (len(seq) - 1) ** length_penalty) for seq, score in beams]
        best = max(finished, key=lambda c: c[1])[0]
        return _strip(best[1:], eos_id)

    def checkpoint(self) -> dict:
        return {"config": asdict(self.config), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "TextToGlossTransformer":
        model = cls(ModelConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        return model


def _strip(ids: list[int], eos_id: int) -> list[int]:
    return ids[: ids.index(eos_id)] if eos_id in ids else ids
