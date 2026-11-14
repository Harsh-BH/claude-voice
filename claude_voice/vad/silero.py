"""Streaming Silero VAD and the utterance segmenter.

Silero v6 ships inside faster-whisper as ONNX, so no torch is needed: one 32 ms frame
(512 samples at 16 kHz plus 64 samples of context) per call, LSTM state carried between calls.

The segmenter runs inside the capture thread and turns per-frame probabilities into events.
Hysteresis plus a minimum voiced duration keep keyboard clicks and coughs out.

While Claude is audible a frame only counts toward barge-in if it is speech-like AND louder than
the room's ambient level by a margin. Measured through this laptop's speaker and mic at 80%
volume: residual echo after AEC reaches p≈0.67 but sits around -69 dBFS (median), well below the
-47 dBFS ambient. A person talking to the laptop is above ambient. In double talk the canceller
also dents the user's speech, so frames are counted over a sliding window rather than requiring
an unbroken run.
"""
from __future__ import annotations

import importlib.util
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from claude_voice.config import BargeInConfig, VadConfig

RATE = 16000
FRAME = 512                 # samples per VAD frame
FRAME_MS = FRAME * 1000 / RATE
_CONTEXT = 64
MODEL = Path(importlib.util.find_spec("faster_whisper").origin).parent / "assets" / "silero_vad_v6.onnx"


class SileroVAD:
    def __init__(self, path: Path = MODEL):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"], sess_options=opts)
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), np.float32)
        self._c = np.zeros((1, 1, 128), np.float32)
        self._context = np.zeros(_CONTEXT, np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability of one float32 frame of FRAME samples in [-1, 1]."""
        x = np.concatenate([self._context, frame])[None, :]
        prob, self._h, self._c = self._session.run(None, {"input": x, "h": self._h, "c": self._c})
        self._context = frame[-_CONTEXT:]
        return float(prob[0])


@dataclass
class SegEvent:
    kind: str                      # start | partial | pause | resume | end
    audio: np.ndarray | None = field(default=None, repr=False)
    barge: bool = False            # start: Claude was audible, playback has been paused
    forced: bool = False           # end: hit max_utterance_s rather than silence
    onset: float = 0.0             # start: time of the first voiced frame of the run


def _frames(ms: float) -> int:
    return max(1, round(ms / FRAME_MS))


class Segmenter:
    """Per-frame state machine: SILENCE -> SPEECH -> (pause/resume)* -> end."""

    PAUSE_MS = 256        # silence that triggers a speculative final transcription
    KEEP_TAIL_MS = 192    # trailing silence kept on the final audio
    FIRST_PARTIAL_MS = 320

    def __init__(self, vad: VadConfig, barge: BargeInConfig, partial_ms: int | None):
        self.vad, self.barge = vad, barge
        self._need = _frames(vad.min_speech_ms)
        self._need_barge = _frames(barge.min_speech_ms)
        self._silence_end = _frames(vad.min_silence_ms)
        self._pause = min(_frames(self.PAUSE_MS), self._silence_end - 1)
        self._keep = _frames(self.KEEP_TAIL_MS)
        self._max = _frames(vad.max_utterance_s * 1000)
        self._partial = _frames(partial_ms) if partial_ms else 0
        self._first_partial = _frames(self.FIRST_PARTIAL_MS)
        self._preroll: deque[np.ndarray] = deque(maxlen=_frames(vad.pre_roll_ms) + 2 * max(self._need, self._need_barge))
        self._hits: deque[bool] = deque(maxlen=2 * self._need_barge)
        self._pre = _frames(vad.pre_roll_ms)
        self._n = 0              # frames seen
        self._onset_n = 0        # frame index of the current onset
        self.ambient_db = -50.0  # slow average of non-speech frames while nothing plays
        self._reset()

    @property
    def active(self) -> bool:
        return self._active

    def _reset(self) -> None:
        self._active = False
        self._run = 0
        self._onset: float | None = None
        self._hits.clear()
        self._utt: list[np.ndarray] = []
        self._silence = 0
        self._paused = False
        self._since_partial = 0

    def snapshot(self) -> np.ndarray | None:
        """Audio of the utterance so far (safe to call from another thread)."""
        utt = list(self._utt)
        return np.concatenate(utt) if utt else None

    def _audio(self, trim: bool) -> np.ndarray:
        utt = self._utt
        if trim and self._silence > self._keep:
            utt = utt[: len(utt) - (self._silence - self._keep)]
        return np.concatenate(utt)

    def feed(self, frame: np.ndarray, prob: float, playing: bool, now: float) -> list[SegEvent]:
        self._n += 1
        level = 20 * math.log10(float(np.sqrt(np.mean(frame * frame))) + 1e-9)
        if prob < 0.3 and not playing:
            self.ambient_db += 0.02 * (max(level, -70.0) - self.ambient_db)
        if not self._active:
            self._preroll.append(frame)
            barge = playing and self.barge.enabled
            if barge:
                hit = prob >= self.barge.threshold and level >= self.ambient_db + self.barge.level_margin_db
                self._hits.append(hit)
                self._run = 0
                if not any(self._hits):
                    self._onset = None
                elif hit and self._onset is None:
                    self._onset, self._onset_n = now, self._n
                start = sum(self._hits) >= self._need_barge
            else:
                self._hits.clear()
                if prob >= self.vad.threshold:
                    if self._run == 0:
                        self._onset, self._onset_n = now, self._n
                    self._run += 1
                else:
                    self._run, self._onset = 0, None
                start = self._run >= self._need
            if not start:
                return []
            onset = self._onset if self._onset is not None else now
            keep = self._n - self._onset_n + 1 + self._pre  # speech so far plus pre-roll before the onset
            utt = list(self._preroll)[-keep:]
            self._reset()
            self._active = True
            self._utt = utt
            self._preroll.clear()
            self._since_partial = self._partial - self._first_partial if self._partial else 0
            return [SegEvent("start", barge=barge, onset=onset)]

        self._utt.append(frame)
        events: list[SegEvent] = []
        if prob >= self.vad.end_threshold:
            if self._paused:
                events.append(SegEvent("resume"))
                self._paused = False
            self._silence = 0
        else:
            self._silence += 1
            if self._silence == self._pause:
                events.append(SegEvent("pause", audio=self._audio(trim=True)))
                self._paused = True
        if self._silence >= self._silence_end or len(self._utt) >= self._max:
            events.append(SegEvent("end", audio=self._audio(trim=True), forced=self._silence < self._silence_end))
            self._reset()
        elif self._partial and prob >= self.vad.threshold:
            self._since_partial += 1
            if self._since_partial >= self._partial:
                self._since_partial = 0
                events.append(SegEvent("partial", audio=self._audio(trim=False)))
        return events
