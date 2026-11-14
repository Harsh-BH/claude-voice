"""PipeWire audio I/O through its PulseAudio protocol (libpulse-simple via ctypes).

PipeWire's native API has no Python binding on Arch; the pulse protocol (pipewire-pulse) lets us
name the exact source/sink, ask for small buffers, read the playback latency, and flush
buffered playback. Flush is what makes a sub-100 ms stop possible.

Capture: one long-lived record stream, read in 32 ms frames in a dedicated thread; reconnects
with backoff when the device or the AEC process disappears.
Player: one long-lived playback stream fed in 20 ms chunks from a dedicated thread; pause keeps
the unplayed remainder (for false-barge-in resume), stop drops everything.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

_S16LE = 3
_RECORD, _PLAYBACK = 2, 1
_NONE = 0xFFFFFFFF  # (uint32_t) -1: let the server pick


class _SampleSpec(ctypes.Structure):
    _fields_ = [("format", ctypes.c_int), ("rate", ctypes.c_uint32), ("channels", ctypes.c_uint8)]


class _BufferAttr(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in ("maxlength", "tlength", "prebuf", "minreq", "fragsize")]


def _load():
    simple = ctypes.CDLL(ctypes.util.find_library("pulse-simple") or "libpulse-simple.so.0")
    pulse = ctypes.CDLL(ctypes.util.find_library("pulse") or "libpulse.so.0")
    simple.pa_simple_new.restype = ctypes.c_void_p
    simple.pa_simple_new.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
                                     ctypes.POINTER(_SampleSpec), ctypes.c_void_p, ctypes.POINTER(_BufferAttr),
                                     ctypes.POINTER(ctypes.c_int)]
    for name in ("pa_simple_read", "pa_simple_write"):
        getattr(simple, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
    for name in ("pa_simple_flush", "pa_simple_drain"):
        getattr(simple, name).argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    simple.pa_simple_get_latency.restype = ctypes.c_uint64
    simple.pa_simple_get_latency.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    simple.pa_simple_free.argtypes = [ctypes.c_void_p]
    pulse.pa_strerror.restype = ctypes.c_char_p
    pulse.pa_strerror.argtypes = [ctypes.c_int]
    return simple, pulse


_simple, _pulse = _load()


class PulseError(OSError):
    pass


class PulseStream:
    """One mono s16 pa_simple stream. Not thread-safe; owned by a single thread."""

    def __init__(self, record: bool, device: str | None, rate: int, latency_ms: float, name: str):
        self.rate = rate
        err = ctypes.c_int(0)
        spec = _SampleSpec(_S16LE, rate, 1)
        nbytes = int(rate * latency_ms / 1000) * 2
        attr = _BufferAttr(_NONE, _NONE, _NONE, _NONE, nbytes) if record else _BufferAttr(_NONE, nbytes, _NONE, _NONE, _NONE)
        self._s = _simple.pa_simple_new(None, b"claude-voice", _RECORD if record else _PLAYBACK,
                                        device.encode() if device else None, name.encode(),
                                        ctypes.byref(spec), None, ctypes.byref(attr), ctypes.byref(err))
        if not self._s:
            raise PulseError(f"cannot open {'source' if record else 'sink'} {device or '(default)'}: {self._strerror(err)}")

    @staticmethod
    def _strerror(err: ctypes.c_int) -> str:
        return _pulse.pa_strerror(err.value).decode()

    def read(self, samples: int) -> np.ndarray:
        buf = np.empty(samples, np.int16)
        err = ctypes.c_int(0)
        if _simple.pa_simple_read(self._s, buf.ctypes.data, buf.nbytes, ctypes.byref(err)) < 0:
            raise PulseError(f"read failed: {self._strerror(err)}")
        return buf

    def write(self, pcm: np.ndarray) -> None:
        err = ctypes.c_int(0)
        if _simple.pa_simple_write(self._s, pcm.ctypes.data, pcm.nbytes, ctypes.byref(err)) < 0:
            raise PulseError(f"write failed: {self._strerror(err)}")

    def flush(self) -> None:
        err = ctypes.c_int(0)
        if _simple.pa_simple_flush(self._s, ctypes.byref(err)) < 0:
            raise PulseError(f"flush failed: {self._strerror(err)}")

    def drain(self) -> None:
        err = ctypes.c_int(0)
        _simple.pa_simple_drain(self._s, ctypes.byref(err))

    def latency(self) -> float:
        """Seconds of written audio not yet played."""
        err = ctypes.c_int(0)
        usec = _simple.pa_simple_get_latency(self._s, ctypes.byref(err))
        return 0.0 if usec == 2**64 - 1 else usec / 1e6

    def close(self) -> None:
        if self._s:
            _simple.pa_simple_free(self._s)
            self._s = None


class Capture(threading.Thread):
    """Reads 16 kHz mono frames forever; `on_frame(int16 frame)` runs in this thread."""

    def __init__(self, device: str | None, frame: int, on_frame: Callable[[np.ndarray], None],
                 on_status: Callable[[str, str], None], rate: int = 16000):
        super().__init__(name="capture", daemon=True)
        self.device, self.frame, self.rate = device, frame, rate
        self.on_frame, self.on_status = on_frame, on_status
        self._stop = threading.Event()
        self._resampler = None
        if rate != 16000:
            import av  # PyAV ships with faster-whisper; libswresample does the resampling
            self._resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        self._carry = np.empty(0, np.int16)

    def stop(self) -> None:
        self._stop.set()

    def _to_16k(self, pcm: np.ndarray) -> np.ndarray:
        import av
        f = av.AudioFrame.from_ndarray(pcm[None, :], format="s16", layout="mono")
        f.sample_rate = self.rate
        out = [fr.to_ndarray()[0] for fr in self._resampler.resample(f)]
        return np.concatenate([self._carry, *out]) if out else self._carry

    def run(self) -> None:
        backoff = 0.25
        read_size = self.frame * self.rate // 16000
        while not self._stop.is_set():
            try:
                stream = PulseStream(True, self.device, self.rate, latency_ms=20, name="claude-voice mic")
            except PulseError as e:
                self.on_status("error", str(e))
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 4.0)
                continue
            self.on_status("ok", self.device or "default source")
            backoff = 0.25
            try:
                while not self._stop.is_set():
                    pcm = stream.read(read_size)
                    if self._resampler is None:
                        self.on_frame(pcm)
                        continue
                    pcm = self._to_16k(pcm)
                    while len(pcm) >= self.frame:
                        self.on_frame(pcm[: self.frame])
                        pcm = pcm[self.frame:]
                    self._carry = pcm
            except PulseError as e:
                self.on_status("error", f"{e}; reconnecting")
            except Exception:
                log.exception("frame handler failed")
                self.on_status("error", "frame handler crashed; restarting capture")
            finally:
                stream.close()


@dataclass
class Clip:
    epoch: int
    seq: int
    text: str
    pcm: np.ndarray = field(repr=False)  # int16 mono at Player.rate
    pos: int = 0                         # samples already played


class Player(threading.Thread):
    """Plays clips in order; pause/stop take effect within one 20 ms chunk plus a flush."""

    CHUNK_MS = 20

    def __init__(self, device: str | None, on_event: Callable[[str, Clip | None], None],
                 rate: int = 24000, tail_ms: int = 250):
        super().__init__(name="player", daemon=True)
        self.device, self.rate, self.on_event = device, rate, on_event
        self.tail = tail_ms / 1000
        self.volume = 1.0
        self._clips: deque[Clip] = deque()
        self._cond = threading.Condition()
        self._paused = False
        self._stop_gen = 0
        self._quit = False
        self._current: Clip | None = None
        self._last_audio = 0.0     # monotonic time the last written sample is expected to finish
        self._heard: list[str] = []  # texts of clips fully played in the current epoch
        self._epoch = -1
        self._stop_requested = 0.0
        self.stop_latency: float | None = None  # seconds from stop()/pause() to a flushed stream

    # ----- called from any thread -----
    def enqueue(self, clip: Clip) -> None:
        with self._cond:
            self._clips.append(clip)
            self._cond.notify_all()

    def pause(self) -> bool:
        """Barge-in fast path. Returns True if something was audible."""
        with self._cond:
            was = self.audible
            self._paused = True
            self._stop_requested = time.monotonic()
            self._cond.notify_all()
            return was

    def resume(self) -> None:
        with self._cond:
            self._paused = False
            self._cond.notify_all()

    def stop(self) -> None:
        with self._cond:
            self._stop_requested = time.monotonic()
            self._stop_gen += 1
            self._clips.clear()
            if self._paused:  # the thread is parked in wait(); drop the paused clip here
                self._current = None
            self._paused = False
            self._last_audio = 0.0
            self._cond.notify_all()

    def shutdown(self) -> None:
        with self._cond:
            self._quit = True
            self._clips.clear()
            self._cond.notify_all()

    @property
    def audible(self) -> bool:
        """Claude's voice is (or just was) coming out of the speaker."""
        return (self._current is not None and not self._paused) or time.monotonic() < self._last_audio + self.tail

    def audible_until(self) -> float:
        """Monotonic time at which the echo tail of what was written ends."""
        return self._last_audio + self.tail

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def busy(self) -> bool:
        return self._current is not None or bool(self._clips)

    def playing_text(self) -> str:
        """Text of the clip playing now plus the one before it (echo check window)."""
        cur = self._current
        prev = self._heard[-1] if self._heard else ""
        return f"{prev} {cur.text}" if cur else prev

    def heard_text(self) -> str:
        """Roughly what the listener has heard of the current reply."""
        cur = self._current
        parts = list(self._heard)
        if cur is not None and len(cur.pcm):
            parts.append(cur.text[: int(len(cur.text) * min(1.0, cur.pos / len(cur.pcm)))])
        return " ".join(parts)

    # ----- player thread -----
    def run(self) -> None:
        chunk = self.rate * self.CHUNK_MS // 1000
        stream = None
        while True:
            with self._cond:
                while not self._quit and (self._paused or (not self._clips and self._current is None)):
                    self._cond.wait()
                if self._quit:
                    break
                if self._current is None:
                    self._current = self._clips.popleft()
                    if self._current.epoch != self._epoch:
                        self._epoch, self._heard = self._current.epoch, []
                clip, gen = self._current, self._stop_gen
            try:
                if stream is None:
                    stream = PulseStream(False, self.device, self.rate, latency_ms=60, name="Claude")
                if clip.pos == 0:
                    self.on_event("clip_start", clip)
                interrupted = self._play(stream, clip, chunk, gen)
            except Exception as e:
                log.error("playback failed: %s", e)
                self.on_event("error", None)
                if stream is not None:
                    stream.close()
                stream = None
                with self._cond:
                    self._current = None
                time.sleep(0.5)
                continue
            with self._cond:
                if interrupted == "stop" or self._stop_gen != gen:
                    self._current = None
                    self._last_audio = 0.0
                    self.on_event("stopped", clip)
                elif interrupted == "pause":
                    self.on_event("paused", clip)
                else:
                    self._heard.append(clip.text)
                    self._current = None
                    self.on_event("clip_end", clip)
                    if not self._clips:
                        self.on_event("drained", clip)
        if stream is not None:
            stream.close()

    def _play(self, stream: PulseStream, clip: Clip, chunk: int, gen: int) -> str:
        while clip.pos < len(clip.pcm):
            if self._stop_gen != gen or self._paused:
                written_until = clip.pos
                latency = stream.latency()
                stream.flush()
                self.stop_latency = time.monotonic() - self._stop_requested
                self._last_audio = 0.0
                if self._stop_gen != gen:
                    return "stop"
                # rewind to what was actually heard, minus a little for context on resume
                clip.pos = max(0, written_until - int((latency + 0.25) * self.rate))
                return "pause"
            part = clip.pcm[clip.pos: clip.pos + chunk]
            if self.volume != 1.0:
                part = np.clip(part.astype(np.float32) * self.volume, -32768, 32767).astype(np.int16)
            stream.write(np.ascontiguousarray(part))
            clip.pos += len(part)
            self._last_audio = time.monotonic() + stream.latency()
        return ""
