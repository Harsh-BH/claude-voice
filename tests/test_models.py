"""VAD, STT and TTS on the real models (no audio devices involved)."""
import time

import numpy as np
import pytest
from conftest import resample

from claude_voice.config import BargeInConfig, VadConfig
from claude_voice.vad.silero import FRAME, Segmenter, SileroVAD

pytestmark = pytest.mark.models
EN = "Deploy the Redis cache to Kubernetes and enable OAuth on the WebSocket server."
HI = "कृपया टेस्ट चलाओ और मुझे नतीजे बताओ।"


def probs(vad: SileroVAD, audio: np.ndarray) -> np.ndarray:
    vad.reset()
    return np.array([vad(audio[i:i + FRAME]) for i in range(0, len(audio) - FRAME + 1, FRAME)])


def test_tts_english_and_hindi(tts):
    started = time.monotonic()
    short = tts.synth("Sure.")
    short_s = time.monotonic() - started
    assert 0.2 < len(short) / tts.rate < 2.0 and short_s < 2.0, short_s  # ~0.3 s idle; generous under load
    assert 3.0 < len(tts.synth(EN)) / tts.rate < 10.0
    assert len(tts.synth(HI)) / tts.rate > 1.5


def test_tts_hinglish_mode(tts):
    mixed = "हाँ, मैंने code GitHub पे push कर दिया है।"
    try:
        tts.hinglish = True
        assert 1.5 < len(tts.synth(mixed)) / tts.rate < 8.0
        assert 0.2 < len(tts.synth("Done.")) / tts.rate < 2.0  # English-only sentences stay in the same voice
    finally:
        tts.hinglish = False


def test_vad_separates_speech_from_silence_noise_and_clicks(tts):
    vad = SileroVAD()
    speech = probs(vad, resample(tts.synth(EN), tts.rate, 16000))
    assert speech.max() > 0.9 and (speech > 0.5).mean() > 0.5
    assert probs(vad, np.zeros(16000 * 2, np.float32)).max() < 0.1
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(16000 * 3) * 0.03).astype(np.float32)   # about -30 dBFS hiss
    assert (probs(vad, noise) > 0.5).mean() < 0.05
    clicks = np.zeros(16000 * 3, np.float32)
    for start in range(0, len(clicks) - 80, 1920):                         # a key every 120 ms
        clicks[start:start + 80] = rng.uniform(-0.6, 0.6, 80)
    seg = Segmenter(VadConfig(), BargeInConfig(), None)
    vad.reset()
    starts = [ev for i in range(0, len(clicks) - FRAME + 1, FRAME)
              for ev in seg.feed(clicks[i:i + FRAME], vad(clicks[i:i + FRAME]), False, i / 16000) if ev.kind == "start"]
    assert starts == []


def test_stt_english_with_technical_terms(tts, stt):
    audio = resample(tts.synth(EN), tts.rate, 16000)
    assert stt.detect_language(audio) == "en"
    text = stt.transcribe(audio, language="en").text.lower()
    for word in ("redis", "kubernetes", "oauth", "websocket"):
        assert word in text, text


def test_stt_hindi(tts, stt):
    audio = resample(tts.synth(HI), tts.rate, 16000)
    assert stt.detect_language(audio) == "hi"
    text = stt.transcribe(audio, language="hi").text
    assert "टेस्ट" in text and "नतीजे" in text, text  # Devanagari, not "Kripya Taste"


def test_stt_silence_gives_nothing(stt):
    assert stt.transcribe(np.zeros(32000, np.float32), language="en").text == ""


def test_piper_fallback_when_kokoro_is_missing():
    from claude_voice.config import DATA_DIR, TtsConfig
    from claude_voice.tts.speech import PiperTTS, load_tts
    if not (DATA_DIR / "models/piper/en_US-lessac-medium.onnx").exists():
        pytest.skip("Piper voice not downloaded (scripts/install.sh --with-piper)")
    tts = load_tts(TtsConfig(kokoro_model="/nonexistent/kokoro.onnx"))
    assert isinstance(tts, PiperTTS)
    assert len(tts.synth("Fallback voice.")) / tts.rate > 0.5
