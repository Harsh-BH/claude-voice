"""Fixtures: silent virtual audio devices (null sinks + pw-loopback), shared models, helpers.

Virtual devices get priority 0, so they never become the default sink, and they are always
unloaded afterwards. Nothing in the `audio` tests is audible.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

# Tests get their own runtime dir: the live app's canceller config, lock and status stay untouched.
_LIVE_LOCK = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/claude-voice-{os.getuid()}") / "claude-voice" / "voice.lock"
os.environ["CLAUDE_VOICE_RUNTIME_DIR"] = f"/tmp/claude-voice-tests-{os.getuid()}"
os.environ["CLAUDE_VOICE_STATE_DIR"] = f"/tmp/claude-voice-tests-{os.getuid()}/state"  # test logs/metrics

from claude_voice.config import DATA_DIR  # noqa: E402

KOKORO = DATA_DIR / "models/kokoro/kokoro-v1.0.onnx"


def _voice_running() -> bool:
    import fcntl
    if not _LIVE_LOCK.exists():
        return False
    fd = os.open(_LIVE_LOCK, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def pytest_collection_modifyitems(config, items):
    if _voice_running():  # its canceller owns the claude_voice_aec_source name
        for item in items:
            if "audio" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="voice mode is running; stop it first (`voice stop`)"))
    gates = {
        "acoustic": (os.environ.get("VOICE_ACOUSTIC") == "1", "set VOICE_ACOUSTIC=1 (plays sound on the real speaker)"),
        "claude": (os.environ.get("VOICE_CLAUDE") == "1", "set VOICE_CLAUDE=1 (uses your Claude subscription)"),
        "audio": (shutil.which("pactl") is not None and "PipeWire" in subprocess.run(
            ["pactl", "info"], capture_output=True, text=True).stdout, "needs a running PipeWire server"),
        "models": (KOKORO.exists(), "run scripts/install.sh to download the models"),
    }
    for item in items:
        for mark, (ok, why) in gates.items():
            if mark in item.keywords and not ok:
                item.add_marker(pytest.mark.skip(reason=why))


@pytest.fixture(autouse=True)
def _forget_test_sessions(request, tmp_path):
    """Claude tests run in tmp_path; drop the session/memory dirs they leave in ~/.claude/projects."""
    yield
    if request.node.get_closest_marker("claude"):
        import re
        slug = re.sub(r"[^A-Za-z0-9]", "-", str(tmp_path))
        shutil.rmtree(Path.home() / ".claude" / "projects" / slug, ignore_errors=True)


@pytest.fixture
def null_sink():
    modules = []

    def make(prefix: str = "cv") -> str:
        name = f"{prefix}_{uuid.uuid4().hex[:6]}"
        mod = subprocess.check_output(["pactl", "load-module", "module-null-sink", f"sink_name={name}", "rate=48000",
                                       "channels=1", f"sink_properties=device.description={name} priority.session=0 priority.driver=0"],
                                      text=True).strip()
        modules.append(mod)
        deadline = time.monotonic() + 3
        while name not in subprocess.run(["pactl", "list", "short", "sinks"], capture_output=True, text=True).stdout:
            assert time.monotonic() < deadline, f"null sink {name} did not appear"
            time.sleep(0.05)
        return name

    yield make
    for mod in modules:
        subprocess.run(["pactl", "unload-module", mod], capture_output=True)


@pytest.fixture
def loopback():
    procs = []

    def make(from_sink: str, to_sink: str, delay: float = 0.08) -> subprocess.Popen:
        """An 'acoustic path': everything played on from_sink reaches to_sink `delay` seconds later."""
        p = subprocess.Popen(["pw-loopback", "-c", "1", "-m", "[ MONO ]", "-d", str(delay),
                              "-C", from_sink, "-i", "{ stream.capture.sink = true node.dont-move = true node.passive = false }",
                              "-P", to_sink, "-o", "{ node.dont-move = true }"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
        time.sleep(0.4)
        return p

    yield make
    for p in procs:
        p.terminate()
        p.wait(3)


@pytest.fixture(scope="session")
def tts():
    from claude_voice.config import TtsConfig
    from claude_voice.tts.speech import KokoroTTS
    return KokoroTTS(TtsConfig())


@pytest.fixture(scope="session")
def stt():
    from claude_voice.config import SttConfig
    from claude_voice.stt.whisper import Transcriber
    return Transcriber(SttConfig())


def resample(pcm: np.ndarray, src: int, dst: int) -> np.ndarray:
    x = pcm.astype(np.float32) / (32768.0 if pcm.dtype == np.int16 else 1.0)
    n = int(len(x) * dst / src)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def play_async(sink: str, pcm: np.ndarray, rate: int, gain: float = 1.0) -> threading.Thread:
    """Play int16 audio into a sink from a background thread; returns the thread."""
    from claude_voice.audio.pulse import PulseStream

    def run():
        s = PulseStream(False, sink, rate, 60, "cv test source")
        data = np.clip(pcm.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
        s.write(np.ascontiguousarray(data))
        s.drain()
        s.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def record(source: str, seconds: float, frame: int = 512) -> tuple[np.ndarray, list[float]]:
    """Record 16 kHz mono; returns samples and the monotonic time each frame was read."""
    from claude_voice.audio.pulse import PulseStream

    s = PulseStream(True, source, 16000, 20, "cv test recorder")
    frames, stamps = [], []
    for _ in range(int(seconds * 16000 / frame)):
        frames.append(s.read(frame))
        stamps.append(time.monotonic())
    s.close()
    return np.concatenate(frames), stamps


def rms_db(x: np.ndarray) -> float:
    x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    return 20 * np.log10(max(float(np.sqrt(np.mean(x * x))), 1e-9))
