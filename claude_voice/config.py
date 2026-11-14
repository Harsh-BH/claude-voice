"""Configuration: defaults live here, ~/.config/claude-voice/config.toml overrides them.

Every key is documented in config.example.toml. Unknown keys and wrong types are rejected
so a typo never silently falls back to a default.
"""
from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

HOME = Path.home()
CONFIG_PATH = Path(
    os.environ.get("CLAUDE_VOICE_CONFIG")
    or Path(os.environ.get("XDG_CONFIG_HOME") or HOME / ".config") / "claude-voice" / "config.toml"
)
DATA_DIR = Path(os.environ.get("XDG_DATA_HOME") or HOME / ".local/share") / "claude-voice"
STATE_DIR = Path(os.environ.get("CLAUDE_VOICE_STATE_DIR")
                 or Path(os.environ.get("XDG_STATE_HOME") or HOME / ".local/state") / "claude-voice")
RUNTIME_DIR = Path(os.environ.get("CLAUDE_VOICE_RUNTIME_DIR")
                   or Path(os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/claude-voice-{os.getuid()}") / "claude-voice")

VOICE_PROMPT = """\
You are in a live, full-duplex voice conversation (claude-voice). The user's messages are \
transcribed speech from a local Whisper model: expect recognition errors, missing punctuation, \
and English, Hindi or Hinglish. Infer intent from context; if something important is ambiguous, \
ask one short question. Everything you write as plain text is read aloud as you stream it, so:
- Talk like a person on a call: lead with the answer, short natural sentences, usually one to \
four of them unless the user asks for detail.
- Never put code, diffs, tables, long lists or full file paths in your spoken text. Make the \
change with your tools, then say briefly what you did and where, by file name.
- Don't use markdown formatting; it is not rendered.
- Before a long-running tool call, say one short sentence about what you are about to do.
- A message may start with a note saying the user interrupted you and what they heard. Drop \
the old plan and follow the new instruction.
- Reply in the language the user speaks."""

HINGLISH_PROMPT = """\
Hinglish mode is on: reply in natural Hinglish, mixing Hindi and English the way the user does. \
Write Hindi words in Devanagari and English words (technical terms, names, anything people \
usually say in English) in Latin script, for example: "हाँ, मैंने code GitHub पे push कर दिया है।" \
The speech engine pronounces each script with its own rules, so never write Hindi words in Latin \
letters."""

DEFAULT_HOTWORDS = [
    "Claude", "Claude Code", "Kubernetes", "PostgreSQL", "WebSockets", "OAuth", "Redis", "CUDA",
    "TypeScript", "JavaScript", "Python", "React", "Next.js", "Node.js", "Docker", "GitHub",
    "npm", "pytest", "JSON", "YAML", "API", "PipeWire", "Arch Linux",
]


@dataclass
class AudioConfig:
    input_device: str = ""   # microphone (source name); "" = follow the default source
    output_device: str = ""  # speaker (sink name); "" = follow the default sink
    input_rate: int = 16000  # capture rate; other rates are resampled to 16 kHz for VAD/STT


@dataclass
class AecConfig:
    enabled: bool = True
    mode: str = "monitor"   # monitor: cancel everything the speaker plays; sink: only Claude's voice
    noise_suppression: bool = True
    gain_control: bool = False         # off by default: WebRTC AGC disturbs AEC delay estimation
    transient_suppression: bool = True  # keyboard clicks
    high_pass_filter: bool = True


@dataclass
class VadConfig:
    threshold: float = 0.5       # speech-start probability
    end_threshold: float = 0.35  # below this counts as silence
    min_speech_ms: int = 160     # voiced time needed to start an utterance
    min_silence_ms: int = 700    # silence that ends an utterance
    pre_roll_ms: int = 320       # audio kept from before the detected onset
    max_utterance_s: float = 30.0


@dataclass
class BargeInConfig:
    enabled: bool = True
    # Calibrated on this machine: 0 false barge-ins over 32 s of real 80%-volume echo, and
    # 5/5 double-talk phrases caught when the user is 10 dB above the echo at the mic.
    threshold: float = 0.5          # speech probability a frame needs while Claude is audible
    level_margin_db: float = 6.0    # ...and this much louder than the room's ambient level
    min_speech_ms: int = 160        # qualifying speech within a 2x window before playback pauses
    confirm_timeout_s: float = 2.0  # paused audio resumes if nothing intelligible arrives in time
    echo_similarity: float = 0.8    # share of transcript words found in what was playing => echo
    tail_ms: int = 250              # playback counts as audible this long after its last sample


@dataclass
class SttConfig:
    model: str = "auto"         # auto = large-v3-turbo on CUDA, small on CPU; or a size / HF repo / path
    device: str = "auto"        # auto | cuda | cpu
    compute_type: str = "auto"  # auto = int8_float16 on CUDA, int8 on CPU
    languages: list[str] = field(default_factory=lambda: ["en", "hi"])
    beam_size: int = 5
    partials: bool = True
    partial_interval_ms: int = 700
    hotwords: list[str] = field(default_factory=lambda: list(DEFAULT_HOTWORDS))


@dataclass
class TtsConfig:
    engine: str = "kokoro"   # kokoro | piper
    voice: str = "af_heart"
    hindi_voice: str = "hf_alpha"
    speed: float = 1.0
    volume: float = 1.0
    threads: int = 6
    kokoro_model: str = ""   # "" = <data dir>/models/kokoro/kokoro-v1.0.onnx
    piper_voice: str = ""    # "" = <data dir>/models/piper/en_US-lessac-medium.onnx


@dataclass
class ClaudeConfig:
    model: str = ""            # "" = your Claude Code default model
    effort: str = ""           # "" = default; low | medium | high | xhigh | max
    permission_mode: str = ""  # "" = from your Claude Code settings
    session: str = "continue"  # continue = latest conversation in the project; new = fresh one
    voice_prompt: str = VOICE_PROMPT
    kill_tools_on_interrupt: bool = True
    cli_path: str = ""         # "" = `claude` on PATH, falling back to the SDK's bundled CLI


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    aec: AecConfig = field(default_factory=AecConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)


_CHOICES = {
    ("aec", "mode"): {"monitor", "sink"},
    ("stt", "device"): {"auto", "cuda", "cpu"},
    ("tts", "engine"): {"kokoro", "piper"},
    ("claude", "session"): {"continue", "new"},
    ("claude", "effort"): {"", "low", "medium", "high", "xhigh", "max"},
    ("claude", "permission_mode"): {"", "default", "acceptEdits", "plan", "dontAsk", "bypassPermissions", "auto"},
}


def load(path: Path = CONFIG_PATH) -> Config:
    cfg = Config()
    if path.exists():
        try:
            data = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"{path}: {e}") from None
        _merge(cfg, data, "")
    validate(cfg)
    return cfg


def _merge(obj, data: dict, prefix: str) -> None:
    names = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise ValueError(f"unknown config key: {prefix}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key} must be a table")
            _merge(current, value, f"{prefix}{key}.")
            continue
        expected = type(current)
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            raise ValueError(f"{prefix}{key} must be {expected.__name__}, got {type(value).__name__}")
        if expected is list and not all(isinstance(v, str) for v in value):
            raise ValueError(f"{prefix}{key} must be a list of strings")
        setattr(obj, key, value)


def validate(cfg: Config) -> None:
    for (section, key), allowed in _CHOICES.items():
        value = getattr(getattr(cfg, section), key)
        if value not in allowed:
            raise ValueError(f"{section}.{key} = {value!r}; expected one of {sorted(allowed)}")
    if not 8000 <= cfg.audio.input_rate <= 192000:
        raise ValueError("audio.input_rate must be between 8000 and 192000")
    for key in ("input_device", "output_device"):
        name = getattr(cfg.audio, key)
        if name and not re.fullmatch(r"[A-Za-z0-9._:-]+", name):
            raise ValueError(f"audio.{key} is not a valid PipeWire node name: {name!r}")
    for name in ("threshold", "end_threshold"):
        if not 0.0 < getattr(cfg.vad, name) < 1.0:
            raise ValueError(f"vad.{name} must be between 0 and 1")
    if cfg.vad.end_threshold > cfg.vad.threshold:
        raise ValueError("vad.end_threshold must not exceed vad.threshold")
    if not 0.0 < cfg.barge_in.threshold < 1.0 or not 0.0 < cfg.barge_in.echo_similarity <= 1.0:
        raise ValueError("barge_in.threshold and barge_in.echo_similarity must be in (0, 1]")
    if not 0.5 <= cfg.tts.speed <= 2.0 or not 0.0 <= cfg.tts.volume <= 2.0:
        raise ValueError("tts.speed must be 0.5-2.0 and tts.volume 0.0-2.0")
    if not cfg.stt.languages:
        raise ValueError("stt.languages must list at least one language")
