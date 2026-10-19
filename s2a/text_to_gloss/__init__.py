"""Text -> ASL gloss."""
from __future__ import annotations

from pathlib import Path

import torch

from s2a.text_to_gloss.model import TextToGlossTransformer
from s2a.text_to_gloss.rules import RuleGlosser
from s2a.text_to_gloss.tokenizer import GlossTokenizer

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "text2gloss"


class ModelGlosser:
    """Wraps our trained Transformer (tokenizer.json + model.pt in one directory)."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, beam: int = 5, device: str = "cpu"):
        model_dir = Path(model_dir)
        if not (model_dir / "model.pt").exists():
            raise FileNotFoundError(
                f"no trained model in {model_dir}; train one with `python -m s2a.text_to_gloss.train` "
                "or use --glosser rules")
        self.tokenizer = GlossTokenizer.load(model_dir / "tokenizer.json")
        self.device = torch.device(device)
        ckpt = torch.load(model_dir / "model.pt", map_location=self.device)
        self.model = TextToGlossTransformer.from_checkpoint(ckpt).to(self.device).eval()
        self.beam = beam

    def translate(self, text: str) -> str:
        src = torch.tensor([self.tokenizer.encode(text)[: self.model.config.max_len]], device=self.device)
        tok = self.tokenizer
        if self.beam > 1:
            ids = self.model.beam_search(src, tok.bos_id, tok.eos_id, beam=self.beam)
        else:
            ids = self.model.greedy(src, tok.bos_id, tok.eos_id)[0]
        return tok.decode_gloss(ids)

    def translate_many(self, texts: list[str]) -> list[str]:
        if self.beam > 1:
            return [self.translate(t) for t in texts]
        from s2a.text_to_gloss.train import translate_batch

        return translate_batch(self.model, self.tokenizer, texts, self.device)


def load_glosser(name: str = "model", model_dir: str | Path | None = None, beam: int = 5):
    if name == "rules":
        return RuleGlosser()
    return ModelGlosser(model_dir or DEFAULT_MODEL_DIR, beam=beam)
