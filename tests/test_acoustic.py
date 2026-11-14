"""Real speaker -> air -> real mic. Plays ~15 s of speech (45% volume, or VOICE_ACOUSTIC_VOLUME); restores mute/volume after.

    VOICE_ACOUSTIC=1 uv run pytest -s tests/test_acoustic.py
"""
import os
import subprocess
import threading
import time

import numpy as np
import pytest
from conftest import record, rms_db

from claude_voice.aec.pipewire import SOURCE, EchoCanceller
from claude_voice.audio.pulse import Clip, Player
from claude_voice.config import AecConfig, BargeInConfig, VadConfig
from claude_voice.vad.silero import FRAME, Segmenter, SileroVAD

pytestmark = [pytest.mark.acoustic, pytest.mark.audio, pytest.mark.models]
TEXT = ("Here is the plan. I'll add a Redis cache in front of the PostgreSQL queries, give every key a "
        "time to live of sixty seconds, and invalidate it whenever the row changes. Then I'll run the tests "
        "and tell you what broke, if anything. That should cut the dashboard load time roughly in half.")


@pytest.fixture
def speaker():
    sink = subprocess.check_output(["pactl", "get-default-sink"], text=True).strip()
    mute = "yes" in subprocess.check_output(["pactl", "get-sink-mute", sink], text=True)
    volume = subprocess.check_output(["pactl", "get-sink-volume", sink], text=True).split("/")[1].strip()
    subprocess.run(["pactl", "set-sink-volume", sink, os.environ.get("VOICE_ACOUSTIC_VOLUME", "45%")], check=True)
    subprocess.run(["pactl", "set-sink-mute", sink, "0"], check=True)
    try:
        yield sink
    finally:
        subprocess.run(["pactl", "set-sink-volume", sink, volume])
        subprocess.run(["pactl", "set-sink-mute", sink, "1" if mute else "0"])


def probs(audio: np.ndarray) -> np.ndarray:
    vad, x = SileroVAD(), audio.astype(np.float32) / 32768
    return np.array([vad(x[i:i + FRAME]) for i in range(0, len(x) - FRAME + 1, FRAME)])


def test_real_room_echo_cancellation(speaker, tts):
    mic = subprocess.check_output(["pactl", "get-default-source"], text=True).strip()
    aec = EchoCanceller(AecConfig(mode="monitor"))
    aec.start()
    player = Player(None, lambda k, c: None, rate=tts.rate)
    player.start()
    try:
        time.sleep(0.5)
        pcm = tts.synth(TEXT)
        seconds = len(pcm) / tts.rate
        raw, out = {}, {}
        t1 = threading.Thread(target=lambda: raw.update(zip(("a", "ts"), record(mic, seconds + 1.5))))
        t2 = threading.Thread(target=lambda: out.update(zip(("a", "ts"), record(SOURCE, seconds + 1.5))))
        t1.start()
        t2.start()
        time.sleep(0.5)
        player.enqueue(Clip(1, 1, "test", pcm))
        time.sleep(seconds - 2.0)
        t_stop = time.monotonic()
        player.stop()
        t1.join()
        t2.join()
    finally:
        player.shutdown()
        aec.stop()

    def window(rec, a, b):
        return np.concatenate([f for f, ts in zip(rec["a"].reshape(-1, 512), rec["ts"]) if a <= ts <= b])

    play_from = raw["ts"][0] + 0.5
    ambient_before = rms_db(window(raw, raw["ts"][0], play_from))
    echo_raw = window(raw, play_from + 3.5, t_stop)
    echo_out = window(out, play_from + 3.5, t_stop)
    p_raw, p_out = probs(echo_raw), probs(echo_out)
    # false barge-ins the default rules would raise on Claude's own voice
    seg = Segmenter(VadConfig(), BargeInConfig(), None)
    x = echo_out.astype(np.float32) / 32768
    vad = SileroVAD()
    false_barges = sum(ev.kind == "start" for i in range(0, len(x) - FRAME + 1, FRAME)
                       for ev in seg.feed(x[i:i + FRAME], vad(x[i:i + FRAME]), True, i / 16000))
    # acoustic stop latency: when does the raw mic fall back to ambient after stop()?
    frames = raw["a"].reshape(-1, 512)
    ambient = np.median([rms_db(f) for f, ts in zip(frames, raw["ts"]) if ts > t_stop + 0.6])
    quiet_at = next((ts for f, ts in zip(frames, raw["ts"]) if ts > t_stop and rms_db(f) < ambient + 6), None)
    print(f"\nambient before playback {ambient_before:.1f} dBFS; raw mic during playback {rms_db(echo_raw):.1f} dBFS, after AEC {rms_db(echo_out):.1f} dBFS "
          f"-> {rms_db(echo_raw) - rms_db(echo_out):.1f} dB echo reduction")
    print(f"VAD on Claude's echo: raw mic p>0.5 {np.mean(p_raw > 0.5):.0%}, p>0.7 {np.mean(p_raw > 0.7):.0%}; "
          f"after AEC p>0.5 {np.mean(p_out > 0.5):.1%}, p>0.7 {np.mean(p_out > 0.7):.1%}, max {p_out.max():.2f}, "
          f"p99 {np.quantile(p_out, 0.99):.2f}")
    print(f"false barge-ins with default rules over {len(x) / 16000:.1f}s of echo: {false_barges}")
    if quiet_at:
        print(f"stop() -> mic back at ambient ({ambient:.0f} dBFS): {(quiet_at - t_stop) * 1000:.0f} ms "
              f"(player flush {player.stop_latency * 1000:.1f} ms)")
    assert rms_db(echo_raw) - rms_db(echo_out) > 10
    assert false_barges == 0
