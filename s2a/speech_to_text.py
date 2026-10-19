"""Speech -> English text using an off-the-shelf recognizer (OpenAI Whisper).

Speech recognition is not where our project adds value, so we reuse an existing model.
Requires `pip install openai-whisper` and ffmpeg.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def _load(model_size: str):
    import whisper

    return whisper.load_model(model_size)


def transcribe(audio_path: str, model_size: str = "base") -> str:
    result = _load(model_size).transcribe(str(audio_path), language="en", fp16=False)
    return result["text"].strip()
