"""Local, offline transcription with faster-whisper, in Scribe's JSON shape.

The rest of video-use reads ElevenLabs Scribe output: a `words` list of
`word` and `spacing` entries with start/end seconds and a `speaker_id`.
This module produces the same shape from a local Whisper model, so
pack_transcripts.py, timeline_view.py and render.py work unchanged.

What Scribe has that this does not:
  - Speaker diarization. Every word gets `speaker_0`.
  - Audio events such as (laughter) or (applause).

Filler words: Whisper tends to tidy speech and drop "um" and "uh", which
video-use exists to cut. A verbatim-style initial prompt keeps most of them.

Install: pip install -e ".[local]"   (adds faster-whisper)
Model: --model, or VIDEO_USE_WHISPER_MODEL (default large-v3-turbo).
Vocabulary: --vocab "Beachread, Dima", or VIDEO_USE_VOCAB, biases the model
toward names and brand terms it would otherwise mishear.
The model downloads once (~1.6 GB for large-v3-turbo) and is then cached.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

DEFAULT_MODEL = "large-v3-turbo"

# Nudges Whisper toward verbatim output so disfluencies survive.
VERBATIM_PROMPT = (
    "Umm, so, uh, I was, like, you know, thinking... Hmm. Uh-huh. "
    "Okay, um, let me, uh, start again."
)

_models: dict[str, object] = {}
_lock = threading.Lock()


def model_name(explicit: str | None = None) -> str:
    return explicit or os.environ.get("VIDEO_USE_WHISPER_MODEL") or DEFAULT_MODEL


def _load(name: str):
    """Load a model once per process; batch workers share it."""
    with _lock:
        if name not in _models:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise RuntimeError(
                    "Local transcription needs faster-whisper. "
                    'Install it with: pip install -e ".[local]"'
                ) from e
            _models[name] = WhisperModel(name, device="auto", compute_type="int8")
        return _models[name]


def _read_wav(path: Path):
    """16-bit mono wav (what transcribe.py extracts) to float32 in [-1, 1].

    Passing samples skips faster-whisper's own PyAV decoding, which breaks
    across PyAV versions and is redundant after ffmpeg has already decoded.
    """
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getframerate() != 16000:
            raise RuntimeError(f"{path.name}: expected 16 kHz mono 16-bit wav")
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def vocab_terms(explicit: str | None = None) -> str | None:
    raw = explicit if explicit is not None else os.environ.get("VIDEO_USE_VOCAB", "")
    terms = [t.strip() for t in raw.split(",") if t.strip()]
    return ", ".join(terms) or None


def transcribe_local(
    audio_path: Path,
    language: str | None = None,
    model: str | None = None,
    vocab: str | None = None,
) -> dict:
    """Transcribe a 16 kHz mono wav. Returns a Scribe-shaped dict."""
    name = model_name(model)
    whisper = _load(name)
    samples = _read_wav(audio_path)
    terms = vocab_terms(vocab)
    prompt = f"{VERBATIM_PROMPT} Names: {terms}." if terms else VERBATIM_PROMPT
    with _lock:  # one decode at a time; the model is not thread-safe
        segments, info = whisper.transcribe(
            samples,
            language=language,
            word_timestamps=True,
            initial_prompt=prompt,
            hotwords=terms,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        raw = [w for seg in segments for w in (seg.words or [])]

    words: list[dict] = []
    prev_end: float | None = None
    for w in raw:
        text = w.word.strip()
        if not text:
            continue
        start, end = round(float(w.start), 3), round(float(w.end), 3)
        if end < start:
            end = start
        if prev_end is not None and start > prev_end:
            words.append({
                "text": " ", "start": prev_end, "end": start,
                "type": "spacing", "speaker_id": "speaker_0",
            })
        words.append({
            "text": text, "start": start, "end": end,
            "type": "word", "speaker_id": "speaker_0",
            "probability": round(float(w.probability), 4),
        })
        prev_end = end

    return {
        "language_code": info.language,
        "language_probability": round(float(info.language_probability), 4),
        "text": " ".join(w["text"] for w in words if w["type"] == "word"),
        "words": words,
        "backend": "local",
        "model": name,
    }
