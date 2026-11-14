"""Streaming speech: Claude's text deltas -> sentences -> synthesis -> the player.

A single worker thread synthesizes one sentence ahead of playback, so after the first sentence
the speaker never waits. Every reply gets an epoch; cancel() bumps it, so a sentence still being
synthesized when the user barges in is thrown away instead of played.

Measured on an i7-12650H (CPU, 6 threads): Kokoro v1.0 fp32 renders ~5x faster than real time.
A short phrase takes ~0.2 s, a 20-word sentence ~0.6 s.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from claude_voice.audio.pulse import Clip, Player
from claude_voice.config import DATA_DIR, TtsConfig
from claude_voice.tts.text import SpeechText, is_hindi, script_runs

log = logging.getLogger(__name__)
_LANG = {"a": "en-us", "b": "en-gb", "h": "hi", "e": "es", "f": "fr-fr", "i": "it", "j": "ja", "p": "pt-br", "z": "cmn"}


class KokoroTTS:
    rate = 24000

    def __init__(self, cfg: TtsConfig):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        model = Path(cfg.kokoro_model or DATA_DIR / "models/kokoro/kokoro-v1.0.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = cfg.threads
        opts.log_severity_level = 3
        session = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
        self.kokoro = Kokoro.from_session(session, str(model.parent / "voices-v1.0.bin"))
        voices = set(self.kokoro.get_voices())
        for v in (cfg.voice, cfg.hindi_voice):
            if v not in voices:
                raise ValueError(f"unknown Kokoro voice {v!r}; available: {', '.join(sorted(voices))}")
        self.cfg = cfg
        self.hinglish = False  # Hinglish mode: every sentence in the Hindi voice, so it never switches mid-reply
        self.synth("Hi.")  # the first inference is slow (graph warm-up); keep it out of the first reply

    def synth(self, text: str) -> np.ndarray:
        if self.hinglish or is_hindi(text):
            # One voice for the whole sentence, but Hindi words get Hindi phonemes and English
            # words English ones, so "मैंने code push कर दिया" sounds like a person saying it.
            phonemes = " ".join(self.kokoro.tokenizer.phonemize(run, lang="hi" if hindi else "en-us")
                                for hindi, run in script_runs(text))
            samples, _ = self.kokoro.create(phonemes, voice=self.cfg.hindi_voice, speed=self.cfg.speed, is_phonemes=True)
        else:
            voice = self.cfg.voice
            samples, _ = self.kokoro.create(text, voice=voice, speed=self.cfg.speed, lang=_LANG.get(voice[0], "en-us"))
        return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)


class PiperTTS:
    """Lightweight fallback (English voice only)."""

    def __init__(self, cfg: TtsConfig):
        from piper import PiperVoice

        self.voice = PiperVoice.load(cfg.piper_voice or str(DATA_DIR / "models/piper/en_US-lessac-medium.onnx"))
        self.rate = self.voice.config.sample_rate
        self.cfg = cfg

    def synth(self, text: str) -> np.ndarray:
        from piper import SynthesisConfig

        chunks = [c.audio_int16_array for c in self.voice.synthesize(text, SynthesisConfig(length_scale=1 / self.cfg.speed))]
        return np.concatenate(chunks) if chunks else np.zeros(0, np.int16)


def load_tts(cfg: TtsConfig):
    if cfg.engine == "kokoro":
        try:
            return KokoroTTS(cfg)
        except Exception as e:
            log.warning("Kokoro unavailable (%s); falling back to Piper", e)
    return PiperTTS(cfg)


class Speech:
    def __init__(self, tts, player: Player, on_event: Callable[[str, object], None]):
        self.tts, self.player, self.on_event = tts, player, on_event
        self.muted = False             # F9: no speech at all
        self.turn_muted = False        # "stop speaking": rest of this reply stays silent
        self.epoch = 0
        self.pending = 0               # sentences queued or synthesizing in the current epoch
        self.turn_sentences: list[str] = []
        self.last_sentences: list[str] = []
        self._seq = 0
        self._text = SpeechText()
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="tts", daemon=True)
        self._thread.start()

    def begin_turn(self) -> None:
        with self._lock:
            if self.turn_sentences:
                self.last_sentences = self.turn_sentences
            self.epoch += 1
            self.pending = self._seq = 0
            self._text = SpeechText()
            self.turn_sentences, self.turn_muted = [], False

    def feed(self, delta: str) -> None:
        for s in self._text.feed(delta):
            self._enqueue(s)

    def end_block(self) -> None:
        """A text block ended (Claude moved on to a tool call or finished)."""
        for s in self._text.flush():
            self._enqueue(s)

    def _enqueue(self, sentence: str) -> None:
        with self._lock:
            self.turn_sentences.append(sentence)
            if self.muted or self.turn_muted:
                return
            self.pending += 1
            self._seq += 1
            self._q.put((self.epoch, self._seq, sentence))
        self.on_event("queued", {"text": sentence})

    def cancel(self, mute_turn: bool = False) -> None:
        """Drop everything queued or playing for this reply."""
        with self._lock:
            self.epoch += 1
            self.pending = 0
            self.turn_muted = self.turn_muted or mute_turn
            self.player.stop()

    def say(self, sentences: list[str]) -> None:
        """Speak fixed text (repeat that, permission prompts) as its own epoch."""
        with self._lock:
            self.epoch += 1
            self.pending = self._seq = 0
            self.player.stop()
            if self.muted:
                return
            for s in sentences:
                self.pending += 1
                self._seq += 1
                self._q.put((self.epoch, self._seq, s))

    def repeat_text(self) -> list[str]:
        return list(self.turn_sentences or self.last_sentences)

    def shutdown(self) -> None:
        self._q.put(None)

    def _worker(self) -> None:
        while (job := self._q.get()) is not None:
            epoch, seq, text = job
            if epoch != self.epoch:
                continue
            started = time.monotonic()
            try:
                pcm = self.tts.synth(text)
            except Exception:
                log.exception("synthesis failed for %r", text)
                pcm = None
            with self._lock:
                if epoch != self.epoch:
                    continue  # cancelled while synthesizing
                self.pending -= 1
                if pcm is not None and len(pcm):
                    self.player.enqueue(Clip(epoch, seq, text, pcm))
            self.on_event("synthesized", {"text": text, "seconds": time.monotonic() - started, "seq": seq})
