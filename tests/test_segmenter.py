"""Segmenter timing with synthetic VAD probabilities (32 ms frames).

Frame amplitude sets the level (voiced -20 dBFS, silence -60 dBFS); sample 0 encodes the frame
index so the utterance audio bounds can be checked.
"""
import numpy as np

from claude_voice.config import BargeInConfig, VadConfig
from claude_voice.vad.silero import FRAME, Segmenter


def frame(i: int, amp: float) -> np.ndarray:
    f = np.full(FRAME, amp, np.float32)
    f[0] = i * 1e-4
    return f


def run(probs, playing=False, partial_ms=None, vad=None, amps=None, seg=None):
    seg = seg or Segmenter(vad or VadConfig(), BargeInConfig(), partial_ms)
    out = []
    for i, p in enumerate(probs):
        amp = amps[i] if amps is not None else (0.1 if p >= 0.35 else 0.001)
        is_playing = playing(i) if callable(playing) else playing
        for ev in seg.feed(frame(i, amp), p, is_playing, i * 0.032):
            out.append((i, ev))
    return out


def test_start_pause_end_and_audio_bounds():
    events = run([0.1] * 20 + [0.9] * 30 + [0.1] * 40)
    assert [(i, e.kind) for i, e in events] == [(24, "start"), (57, "pause"), (71, "end")]  # 160 / 256 / 700 ms
    start, end = events[0][1], events[-1][1]
    assert not start.barge and start.onset == 20 * 0.032
    first, last = (round(x / 1e-4) for x in end.audio[::FRAME][[0, -1]])
    assert (first, last) == (10, 55)  # 320 ms pre-roll before the onset ... speech + 192 ms tail
    assert not end.forced


def test_short_blips_are_ignored():
    assert run([0.1] * 5 + [0.95] * 4 + [0.1] * 30) == []  # 128 ms click


def test_barge_in_rule():
    assert run([0.45] * 30, playing=True) == []                        # not speech-like enough
    assert run([0.9] * 4 + [0.1] * 30, playing=True) == []             # 128 ms < 160 ms
    events = run([0.9] * 5 + [0.1] * 30, playing=True)
    assert events[0][0] == 4 and events[0][1].barge and events[0][1].onset == 0.0


def test_barge_in_counts_dented_speech_in_a_window():
    """Double talk dents the user's speech: bursts that are each under 160 ms still add up."""
    events = run([0.9] * 3 + [0.2] * 2 + [0.9] * 2 + [0.1] * 30, playing=True)
    assert events[0][0] == 6 and events[0][1].barge


def test_quiet_residual_echo_cannot_barge_in():
    quiet = 10 ** (-48 / 20)  # speech-like but only ~2 dB above the -50 dBFS ambient
    assert run([0.95] * 40, playing=True, amps=[quiet] * 40) == []


def test_ambient_level_adapts():
    seg = Segmenter(VadConfig(), BargeInConfig(), None)
    noisy = 10 ** (-30 / 20)
    run([0.1] * 300, amps=[noisy] * 300, seg=seg)                     # a fan at -30 dBFS
    assert -32 < seg.ambient_db < -28
    soft = 10 ** (-26 / 20)
    assert run([0.95] * 30, playing=True, amps=[soft] * 30, seg=seg) == []   # only 4 dB above
    assert run([0.95] * 30, playing=True, amps=[0.1] * 30, seg=seg)[0][0] == 4


def test_pause_then_resume_keeps_one_utterance():
    events = run([0.9] * 10 + [0.1] * 12 + [0.9] * 10 + [0.1] * 30)
    assert [e.kind for _, e in events] == ["start", "pause", "resume", "pause", "end"]


def test_hysteresis_keeps_quiet_speech_alive():
    events = run([0.9] * 10 + [0.4] * 40 + [0.1] * 30)
    assert [e.kind for _, e in events] == ["start", "pause", "end"]
    assert events[-1][0] == 10 + 40 + 21


def test_partials_and_forced_end():
    events = run([0.9] * 80, partial_ms=700, vad=VadConfig(max_utterance_s=2.0))
    kinds = [(i, e.kind) for i, e in events]
    assert kinds[0] == (4, "start")
    assert kinds[1] == (14, "partial")          # first partial ~320 ms after start
    assert kinds[2] == (36, "partial")          # then every ~700 ms
    assert [e for _, e in events if e.kind == "end"][0].forced
