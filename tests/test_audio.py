"""Capture and playback against silent PipeWire null sinks."""
import subprocess
import threading
import time

import numpy as np
import pytest
from conftest import play_async, record, rms_db

from claude_voice.audio.pulse import Capture, Clip, Player

pytestmark = pytest.mark.audio


def tone(freq: float, seconds: float, rate: int, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * freq * t) * amp * 32767).astype(np.int16)


def test_capture_reads_frames_from_a_virtual_mic(null_sink):
    mic = null_sink("cvmic")
    frames, statuses = [], []
    cap = Capture(f"{mic}.monitor", 512, frames.append, lambda s, d: statuses.append(s))
    cap.start()
    time.sleep(0.3)
    play_async(mic, tone(440, 1.0, 48000), 48000).join()
    time.sleep(0.2)
    cap.stop()
    cap.join(2)
    assert statuses[0] == "ok"
    assert frames and all(f.shape == (512,) and f.dtype == np.int16 for f in frames)
    audio = np.concatenate(frames).astype(np.float32)
    freqs = np.fft.rfftfreq(len(audio), 1 / 16000)
    assert abs(freqs[np.abs(np.fft.rfft(audio)).argmax()] - 440) < 5


def test_capture_resamples_a_configured_input_rate(null_sink):
    mic = null_sink("cvmic48")
    frames = []
    cap = Capture(f"{mic}.monitor", 512, frames.append, lambda s, d: None, rate=48000)
    cap.start()
    time.sleep(0.3)
    play_async(mic, tone(1000, 0.8, 48000), 48000).join()
    cap.stop()
    cap.join(2)
    audio = np.concatenate(frames).astype(np.float32)
    assert all(len(f) == 512 for f in frames)
    freqs = np.fft.rfftfreq(len(audio), 1 / 16000)
    assert abs(freqs[np.abs(np.fft.rfft(audio)).argmax()] - 1000) < 8


def test_stop_silences_the_speaker_within_100ms(null_sink):
    spk = null_sink("cvspk")
    player = Player(spk, lambda k, c: None, rate=24000)
    player.start()
    rec: dict = {}
    t = threading.Thread(target=lambda: rec.update(zip(("audio", "stamps"), record(f"{spk}.monitor", 2.4))))
    t.start()
    time.sleep(0.3)
    player.enqueue(Clip(1, 1, "tone", tone(300, 3.0, 24000)))
    time.sleep(1.0)
    t_stop = time.monotonic()
    player.stop()
    t.join()
    player.shutdown()
    frames = rec["audio"].reshape(-1, 512)
    loud_before = [rms_db(f) for f, ts in zip(frames, rec["stamps"]) if t_stop - 0.5 < ts < t_stop]
    assert min(loud_before) > -20, "tone was not playing before stop"
    silent_at = next(ts for f, ts in zip(frames, rec["stamps"]) if ts > t_stop and rms_db(f) < -60)
    latency = silent_at - t_stop
    print(f"\nstop() -> silence read back from the sink monitor: {latency * 1000:.0f} ms "
          f"(player flush {player.stop_latency * 1000:.1f} ms; includes up to ~32 ms recorder framing)")
    assert player.stop_latency < 0.05
    assert latency < 0.100


def test_pause_keeps_the_rest_and_resume_finishes_it(null_sink):
    spk = null_sink("cvspk")
    events = []
    player = Player(spk, lambda k, c: events.append(k), rate=24000)
    player.start()
    clip = Clip(1, 1, "two seconds", tone(300, 2.0, 24000))
    player.enqueue(clip)
    time.sleep(1.0)
    assert player.audible and player.pause()
    time.sleep(0.2)
    assert not player.audible and "paused" in events
    heard = clip.pos / 24000
    assert 0.4 < heard < 1.0, heard          # rewound to what was audible minus ~0.25 s of context
    player.resume()
    deadline = time.monotonic() + 3
    while "clip_end" not in events and time.monotonic() < deadline:
        time.sleep(0.05)
    player.shutdown()
    assert "clip_end" in events and "stopped" not in events


def test_capture_survives_the_device_disappearing_and_returning():
    """PipeWire parks an orphaned stream on a fallback and WirePlumber re-links it when the
    device returns; the capture thread must keep running throughout (and reconnect itself if
    the stream errors instead)."""
    name = "cv_flaky"
    load = ["pactl", "load-module", "module-null-sink", f"sink_name={name}", "rate=48000", "channels=1",
            f"sink_properties=device.description={name} priority.session=0 priority.driver=0"]
    mod = subprocess.check_output(load, text=True).strip()
    frames = []
    cap = Capture(f"{name}.monitor", 512, frames.append, lambda s, d: None)
    try:
        cap.start()
        time.sleep(0.4)
        subprocess.run(["pactl", "unload-module", mod], check=True)
        time.sleep(1.0)
        mod = subprocess.check_output(load, text=True).strip()
        time.sleep(1.5)
        assert cap.is_alive()
        n = len(frames)
        play_async(name, tone(1000, 0.8, 48000), 48000).join()
        time.sleep(0.2)
        audio = np.concatenate(frames[n:]).astype(np.float32)
        freqs = np.fft.rfftfreq(len(audio), 1 / 16000)
        assert abs(freqs[np.abs(np.fft.rfft(audio)).argmax()] - 1000) < 8  # hearing the returned device
    finally:
        cap.stop()
        subprocess.run(["pactl", "unload-module", mod], capture_output=True)
