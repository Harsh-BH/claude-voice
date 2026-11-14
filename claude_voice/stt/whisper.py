"""Speech recognition with faster-whisper (CTranslate2): CUDA when it actually works, else CPU.

Measured on an RTX 3050 6 GB with large-v3-turbo int8_float16: ~430 ms per utterance of up to
30 s (the encoder always sees a padded 30 s window, so cost is flat), 1.25 GB VRAM. Language
detection costs a second encoder pass, so the engine detects once per utterance during the
partials and reuses the result.
"""
from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import re
import time
from dataclasses import dataclass

import numpy as np

from claude_voice.config import SttConfig

log = logging.getLogger(__name__)
TURBO = "dropbox-dash/faster-whisper-large-v3-turbo"  # the old mobiuslabsgmbh repo now redirects here
_HALLUCINATIONS = {
    "thank you", "thanks", "thank you very much", "thanks for watching", "thank you for watching",
    "you", "bye", "bye bye", "please subscribe", "subscribe", "subtitles by the amaraorg community",
}
_NORM = re.compile(r"[^\wऀ-ॿ ]+")
_URLISH = re.compile(r"(?:www\.|https?://)\S+|[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|io)(?:\.[a-z]{2})?\.?")


def preload_cublas() -> bool:
    """ctranslate2 dlopens libcublas.so.12; the nvidia-cublas-cu12 wheel keeps it off the loader path."""
    spec = importlib.util.find_spec("nvidia.cublas")
    if not spec or not spec.submodule_search_locations:
        return False
    libdir = os.path.join(list(spec.submodule_search_locations)[0], "lib")
    try:
        for name in ("libcublasLt.so.12", "libcublas.so.12"):
            ctypes.CDLL(os.path.join(libdir, name), mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        log.warning("cuBLAS present but not loadable: %s", e)
        return False
    return True


def resolve(cfg: SttConfig) -> tuple[str, str, str]:
    import ctranslate2

    device = cfg.device
    if device == "auto":
        device = "cuda" if preload_cublas() and ctranslate2.get_cuda_device_count() > 0 else "cpu"
    elif device == "cuda":
        preload_cublas()
    model = cfg.model
    if model in ("auto", "turbo", "large-v3-turbo"):
        model = TURBO if (device == "cuda" or model != "auto") else "small"
    compute = cfg.compute_type
    if compute == "auto":
        compute = "int8_float16" if device == "cuda" else "int8"
    return model, device, compute


@dataclass
class Transcript:
    text: str
    language: str
    seconds: float      # processing time
    no_speech: float
    logprob: float


class Transcriber:
    def __init__(self, cfg: SttConfig):
        from faster_whisper import WhisperModel

        self.cfg = cfg
        self.model_name, self.device, self.compute = resolve(cfg)
        started = time.monotonic()
        try:
            self.model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute, cpu_threads=4)
        except Exception as e:
            if self.device != "cuda":
                raise
            log.warning("CUDA speech recognition failed (%s); using the CPU", e)
            self.device, self.compute = "cpu", "int8"
            self.model_name = "small" if cfg.model == "auto" else self.model_name
            self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8", cpu_threads=4)
        self.hotwords = ", ".join(cfg.hotwords) or None
        self.last_language = cfg.languages[0]
        self.transcribe(np.zeros(16000, np.float32), language=self.last_language)  # CUDA kernel warm-up
        self.load_seconds = time.monotonic() - started

    def detect_language(self, audio: np.ndarray) -> str:
        langs = self.cfg.languages
        if len(langs) == 1:
            return langs[0]
        if len(audio) < 16000:  # under a second, detection is a coin toss: keep the last language
            return self.last_language
        _, _, probs = self.model.detect_language(audio=audio)
        p = dict(probs)
        if "hi" in langs:  # Whisper often labels spoken Hindi as Urdu (same speech, other script)
            p["hi"] = p.get("hi", 0.0) + p.get("ur", 0.0)
        self.last_language = max(langs, key=lambda lang: p.get(lang, 0.0))
        return self.last_language

    def transcribe(self, audio: np.ndarray, language: str | None = None, partial: bool = False) -> Transcript:
        started = time.monotonic()
        if float(np.sqrt(np.mean(np.square(audio)))) < 1e-3:  # below -60 dBFS: Whisper would only hallucinate
            return Transcript("", language or self.last_language, 0.0, 1.0, 0.0)
        language = language or self.detect_language(audio)
        segments, _ = self.model.transcribe(
            audio, language=language, beam_size=1 if partial else self.cfg.beam_size,
            without_timestamps=True, condition_on_previous_text=False, vad_filter=False,
            # English-only: on Hindi the English prompt pushes Whisper into wrong romanization
            # ("कृपया टेस्ट" -> "Kripya Taste"); measured, see tests/test_models.py
            hotwords=self.hotwords if language == "en" else None,
            temperature=0.0 if partial else [0.0, 0.2, 0.4, 0.6],
        )
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        no_speech = max((s.no_speech_prob for s in segs), default=1.0)
        logprob = min((s.avg_logprob for s in segs), default=-10.0)
        if self._hallucination(text, len(audio) / 16000, no_speech, logprob):
            log.debug("dropped likely hallucination %r (no_speech=%.2f logprob=%.2f)", text, no_speech, logprob)
            text = ""
        return Transcript(text, language, time.monotonic() - started, no_speech, logprob)

    @staticmethod
    def _hallucination(text: str, seconds: float, no_speech: float, logprob: float) -> bool:
        if not text:
            return False
        if no_speech > 0.8 and logprob < -0.7:
            return True
        if _URLISH.fullmatch(text.strip().lower()) and seconds < 3:
            return True  # "www.youtube.com.au" on near-silence
        norm = _NORM.sub("", text.lower()).strip()
        return norm in _HALLUCINATIONS and (seconds < 2.5 or no_speech > 0.4)
