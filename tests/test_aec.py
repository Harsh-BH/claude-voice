"""Echo cancellation in a silent virtual room.

speaker sink --pw-loopback (60 ms "acoustic" delay)--> room sink; the mic is the room's monitor.
The canceller (monitor mode, the default) takes the speaker's monitor as its reference, exactly
as with the real speaker. The simulated user talks straight into the room, so the user's voice is
not in the reference.
"""
import threading
import time

import numpy as np
import pytest
from conftest import play_async, record, rms_db

from claude_voice.aec.pipewire import CAPTURE, REFERENCE, SOURCE, EchoCanceller, links
from claude_voice.audio.pulse import Clip, Player
from claude_voice.config import AecConfig, BargeInConfig, TtsConfig, VadConfig
from claude_voice.tts.speech import KokoroTTS
from claude_voice.vad.silero import FRAME, Segmenter, SileroVAD

pytestmark = [pytest.mark.audio, pytest.mark.models]

CLAUDE_TEXT = ("A Redis cache keeps hot query results in memory, so the database does less work. "
               "Each entry gets a time to live, and stale values simply expire. "
               "For writes, you can invalidate the key, or update it right after the database commit.")
USER_TEXT = "Stop. Use Postgres instead of MySQL for the users table."


@pytest.fixture(scope="module")
def user_voice():
    return KokoroTTS(TtsConfig(voice="am_michael"))


@pytest.fixture
def room(null_sink, loopback):
    spk, rm = null_sink("cvspk"), null_sink("cvroom")
    loopback(spk, rm, delay=0.06)
    aec = EchoCanceller(AecConfig(mode="monitor"), mic=f"{rm}.monitor", speaker=spk)
    aec.start()
    time.sleep(0.5)
    yield spk, rm
    aec.stop()


def vad_probs(audio: np.ndarray) -> np.ndarray:
    vad = SileroVAD()
    x = audio.astype(np.float32) / 32768
    return np.array([vad(x[i:i + FRAME]) for i in range(0, len(x) - FRAME + 1, FRAME)])


def test_canceller_is_wired_to_mic_and_speaker(room):
    spk, rm = room
    lk = links()
    cap = [p for port, peers in lk.items() if port.startswith(CAPTURE) for p in peers]
    ref = [p for port, peers in lk.items() if port.startswith(REFERENCE) for p in peers]
    assert any(p.startswith(f"{rm}:monitor") for p in cap), cap
    assert any(p.startswith(f"{spk}:monitor") for p in ref), ref


def test_claudes_voice_is_cancelled(room, tts):
    spk, rm = room
    claude = tts.synth(CLAUDE_TEXT)
    seconds = len(claude) / tts.rate + 0.5
    out, raw = {}, {}
    t1 = threading.Thread(target=lambda: out.update(a=record(SOURCE, seconds)[0]))
    t2 = threading.Thread(target=lambda: raw.update(a=record(f"{rm}.monitor", seconds)[0]))
    t1.start()
    t2.start()
    time.sleep(0.2)
    play_async(spk, claude, tts.rate).join()
    t1.join()
    t2.join()
    skip = 3 * 16000  # AEC3 convergence
    a_raw, a_out = raw["a"][skip:], out["a"][skip:]
    erle = rms_db(a_raw) - rms_db(a_out)
    p_raw, p_out = vad_probs(a_raw), vad_probs(a_out)
    print(f"\nERLE {erle:.1f} dB; frames above the barge-in threshold: mic {np.mean(p_raw > 0.7):.0%}, "
          f"after AEC {np.mean(p_out > 0.7):.1%}")
    assert np.mean(p_raw > 0.7) > 0.4  # the echo really is speech-like before cancellation
    assert erle > 20
    assert np.mean(p_out > 0.7) < 0.03


def test_barge_in_is_detectable_through_double_talk(room, tts, user_voice, stt):
    """The user cuts in while Claude talks; as the engine would, Claude's playback stops 300 ms
    later. The onset (overlapping Claude) must satisfy the barge-in rule, and the whole
    instruction must transcribe."""
    spk, rm = room
    claude, user = tts.synth(CLAUDE_TEXT), user_voice.synth(USER_TEXT)
    player = Player(spk, lambda k, c: None, rate=tts.rate)
    player.start()
    out: dict = {}
    t = threading.Thread(target=lambda: out.update(zip(("a", "ts"), record(SOURCE, 9.0))))
    t.start()
    time.sleep(0.2)
    player.enqueue(Clip(1, 1, "claude", claude))
    time.sleep(4.0)
    user_start = time.monotonic()
    speaking = play_async(rm, user, user_voice.rate)
    time.sleep(0.3)
    player.stop()
    speaking.join()
    t.join()
    player.shutdown()
    a, ts = out["a"], out["ts"]
    # run the canceller's real output through the real segmenter, "playing" until stop + tail
    seg, vad = Segmenter(VadConfig(), BargeInConfig(), None), SileroVAD()
    barge_at = None
    for f, t_read in zip(a.reshape(-1, 512), ts):
        x = f.astype(np.float32) / 32768
        playing = t_read < user_start + 0.3 + 0.25
        for ev in seg.feed(x, vad(x), playing, t_read):
            if ev.kind == "start" and barge_at is None and t_read >= user_start:
                barge_at = t_read
    i0 = next(i for i, s in enumerate(ts) if s >= user_start) * 512
    seg_audio = a[max(0, i0 - 3200):i0 + int((len(user) / user_voice.rate + 0.5) * 16000)].astype(np.float32) / 32768
    text = stt.transcribe(seg_audio, language="en").text
    delay = None if barge_at is None else round((barge_at - user_start) * 1000)
    print(f"\nbarge-in fired {delay} ms after the user started (includes ~100 ms playback+capture latency); heard: {text!r}")
    assert delay is not None and delay < 800
    assert "postgres" in text.lower() and "user" in text.lower()
