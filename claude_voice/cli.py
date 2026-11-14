"""`voice` command: start | stop | status | doctor | say."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from claude_voice import config as config_mod
from claude_voice.config import RUNTIME_DIR, STATE_DIR, Config

LOCK = RUNTIME_DIR / "voice.lock"
COMMANDS = {"start", "stop", "status", "doctor", "say"}


def _setup_logging() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logfile = STATE_DIR / "voice.log"
    if logfile.exists() and logfile.stat().st_size > 5_000_000:
        logfile.replace(logfile.with_suffix(".log.1"))
    level = logging.DEBUG if os.environ.get("CLAUDE_VOICE_DEBUG") else logging.INFO
    logging.basicConfig(filename=logfile, level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "faster_whisper", "phonemizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _lock():
    """Hold an exclusive lock for the process lifetime; None if another instance has it."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    return fd


def _running_pid() -> int | None:
    if not LOCK.exists():
        return None
    fd = os.open(LOCK, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None  # lock is free: nothing running
    except BlockingIOError:
        try:
            return int(LOCK.read_text().strip() or 0) or None
        except ValueError:
            return None
    finally:
        os.close(fd)


def _load_config(args) -> Config:
    cfg = config_mod.load(Path(args.config)) if getattr(args, "config", None) else config_mod.load()
    if getattr(args, "model", None):
        cfg.claude.model = args.model
    if getattr(args, "effort", None):
        cfg.claude.effort = args.effort
    if getattr(args, "no_aec", False):
        cfg.aec.enabled = False
    if getattr(args, "aec_mode", None):
        cfg.aec.mode = args.aec_mode
    if getattr(args, "input", None):
        cfg.audio.input_device = args.input
    if getattr(args, "output", None):
        cfg.audio.output_device = args.output
    config_mod.validate(cfg)
    return cfg


def cmd_start(args) -> int:
    project = Path(args.project).expanduser().resolve()
    if not project.is_dir():
        print(f"voice: {project} is not a directory", file=sys.stderr)
        return 2
    try:
        cfg = _load_config(args)
    except ValueError as e:
        print(f"voice: config error: {e}", file=sys.stderr)
        return 2
    fd = _lock()
    if fd is None:
        print(f"voice: already running (pid {_running_pid()}); `voice stop` ends it", file=sys.stderr)
        return 1
    _setup_logging()
    os.chdir(project)
    new = args.new or cfg.claude.session == "new"
    if args.headless:
        from claude_voice.ui import console
        asyncio.run(console.run(cfg, str(project), args.resume, new))
    else:
        from claude_voice.ui.tui import VoiceApp
        VoiceApp(cfg, str(project), args.resume, new).run()
    return 0


def cmd_stop(_args) -> int:
    pid = _running_pid()
    if not pid:
        print("voice mode is not running")
        return 0
    os.kill(pid, signal.SIGTERM)
    for _ in range(150):
        if _running_pid() is None:
            print(f"stopped (pid {pid})")
            return 0
        time.sleep(0.1)
    print(f"pid {pid} did not exit within 15 s", file=sys.stderr)
    return 1


def cmd_status(_args) -> int:
    pid = _running_pid()
    if not pid:
        print("voice mode: not running")
        return 3
    status_file = RUNTIME_DIR / "status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    age = time.time() - status.get("updated", 0)
    print(f"voice mode: running (pid {pid})")
    for key in ("cwd", "state", "mic", "aec", "session_id", "model", "muted"):
        print(f"  {key:10s} {status.get(key)}")
    if age > 5:
        print(f"  ! status is {age:.0f}s old; the engine may be stuck")
    return 0


# ----------------------------------------------------------------------------- doctor
class _Report:
    def __init__(self):
        self.failed = 0

    def __call__(self, ok: bool | None, what: str, hint: str = "") -> bool:
        mark = {True: "\033[32m✓\033[0m", False: "\033[31m✗\033[0m", None: "\033[33m!\033[0m"}[ok]
        print(f"  {mark} {what}" + (f"\n      → {hint}" if hint and ok is not True else ""))
        self.failed += ok is False
        return bool(ok)


def _run(cmd: list[str], timeout: float = 10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def cmd_doctor(args) -> int:
    r = _Report()
    try:
        cfg = _load_config(args)
        r(True, f"config {config_mod.CONFIG_PATH if config_mod.CONFIG_PATH.exists() else '(defaults; no config file)'}")
    except ValueError as e:
        r(False, f"config: {e}")
        return 1

    print("Audio")
    info = _run(["pactl", "info"])
    r("PipeWire" in info, "PipeWire with pipewire-pulse running", "install/start pipewire, pipewire-pulse and wireplumber")
    r(Path("/usr/lib/spa-0.2/aec/libspa-aec-webrtc.so").exists(), "WebRTC echo-cancel plugin (libspa-aec-webrtc)",
      "it ships with the pipewire package; reinstall pipewire")
    r(bool(shutil.which("setpriv")), "setpriv (ties the canceller's life to ours)", "install util-linux")
    sink = next((line.split(": ", 1)[1] for line in info.splitlines() if line.startswith("Default Sink")), "?")
    source = next((line.split(": ", 1)[1] for line in info.splitlines() if line.startswith("Default Source")), "?")
    r(True, f"default speaker: {sink}")
    r(True, f"default mic: {source}")
    if "yes" in _run(["pactl", "get-sink-mute", "@DEFAULT_SINK@"]):
        r(None, "the default speaker is MUTED: you will not hear Claude", "unmute it (e.g. wpctl set-mute @DEFAULT_AUDIO_SINK@ 0)")
    if "yes" in _run(["pactl", "get-source-mute", "@DEFAULT_SOURCE@"]):
        r(False, "the default mic is MUTED", "unmute it (e.g. wpctl set-mute @DEFAULT_AUDIO_SOURCE@ 0)")
    if "easyeffects" in _run(["pactl", "list", "short", "sinks"]):
        r(True, "EasyEffects detected: the canceller pins its streams (node.dont-move) so EasyEffects cannot re-route them")

    print("Speech models")
    from claude_voice.stt.whisper import preload_cublas, resolve
    import ctranslate2
    from huggingface_hub import try_to_load_from_cache
    gpu = preload_cublas() and ctranslate2.get_cuda_device_count() > 0
    r(True if gpu else None, f"CUDA for speech recognition: {'yes' if gpu else 'no (CPU fallback)'}",
      "for GPU: NVIDIA driver + `uv sync --extra gpu` (installs only cuBLAS 12)")
    model, device, compute = resolve(cfg.stt)
    cached = Path(model).exists() or isinstance(try_to_load_from_cache(model, "model.bin"), str)
    r(cached, f"Whisper model {model} ({device}, {compute}) {'downloaded' if cached else 'NOT downloaded'}",
      "scripts/install.sh downloads it (or it downloads on first start)")
    from claude_voice.vad.silero import MODEL as VAD_MODEL
    r(VAD_MODEL.exists(), "Silero VAD v6 (bundled with faster-whisper)")
    kokoro = Path(cfg.tts.kokoro_model or config_mod.DATA_DIR / "models/kokoro/kokoro-v1.0.onnx")
    r(kokoro.exists() and (kokoro.parent / "voices-v1.0.bin").exists() or cfg.tts.engine != "kokoro",
      f"Kokoro model {kokoro}", "scripts/install.sh downloads it")

    print("Claude Code")
    cli = cfg.claude.cli_path or shutil.which("claude")
    r(bool(cli), f"claude CLI: {cli} {_run([cli, '--version']).strip() if cli else ''}", "install Claude Code")
    try:
        auth = json.loads(_run([cli, "auth", "status"]) or "{}") if cli else {}
    except json.JSONDecodeError:
        auth = {}
    r(bool(auth.get("loggedIn")), f"logged in via {auth.get('authMethod')} ({auth.get('subscriptionType')} plan)", "run `claude` and /login")
    if os.environ.get("ANTHROPIC_API_KEY"):
        r(None, "ANTHROPIC_API_KEY is set: the CLI will bill that API key instead of your subscription",
          "unset it to use your Claude plan")

    if args.quick:
        return 1 if r.failed else 0

    print("Live checks")
    _live_checks(cfg, r)
    print(f"\n{'all good' if not r.failed else f'{r.failed} problem(s) found'}")
    return 1 if r.failed else 0


def _live_checks(cfg: Config, r: _Report) -> None:
    import numpy as np

    from claude_voice.aec.pipewire import CAPTURE, REFERENCE, SOURCE, EchoCanceller, links
    from claude_voice.audio.pulse import PulseStream

    if cfg.aec.enabled and _running_pid() is None:
        aec = EchoCanceller(cfg.aec, cfg.audio.input_device, cfg.audio.output_device)
        try:
            aec.start()
            r(True, f"echo canceller starts ({cfg.aec.mode} mode)")
            mic = PulseStream(True, SOURCE, 16000, 20, "claude-voice doctor")
            pcm = np.concatenate([mic.read(512) for _ in range(47)])  # 1.5 s
            mic.close()
            lk = links()
            cap_peers = [p for port, peers in lk.items() if port.startswith(CAPTURE) for p in peers]
            ref_peers = [p for port, peers in lk.items() if port.startswith(REFERENCE) for p in peers]
            r(bool(cap_peers) and not any("easyeffects" in p for p in cap_peers),
              f"canceller mic input ← {cap_peers[0].split(':')[0] if cap_peers else 'NOTHING'}")
            if cfg.aec.mode == "monitor":
                r(any(":monitor_" in p for p in ref_peers), f"echo reference ← {ref_peers[0].split(':')[0] if ref_peers else 'NOTHING'}")
            r(len(pcm) == 47 * 512, "the canceller delivers audio")
            raw = PulseStream(True, cfg.audio.input_device or None, 16000, 20, "claude-voice doctor raw")
            pcm = np.concatenate([raw.read(512) for _ in range(31)])  # 1 s straight from the mic
            raw.close()
            db = 20 * np.log10(max(float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768) ** 2))), 1e-7))
            r(True if db > -90 else None, f"raw mic level (quiet room): {db:.0f} dBFS",
              "the mic looks muted or at zero volume: wpctl set-mute @DEFAULT_AUDIO_SOURCE@ 0")
        except Exception as e:
            r(False, f"echo canceller: {e}", f"see {STATE_DIR / 'aec.log'}")
        finally:
            aec.stop()
    else:
        r(None, "skipped echo canceller check (voice mode running or AEC disabled)")

    try:
        from claude_voice.stt.whisper import Transcriber
        from claude_voice.tts.speech import load_tts
        from claude_voice.vad.silero import FRAME, SileroVAD

        t = time.monotonic()
        tts = load_tts(cfg.tts)
        pcm = tts.synth("Please add a Redis cache in front of the PostgreSQL queries.")
        r(len(pcm) > 0, f"TTS ({type(tts).__name__}) rendered {len(pcm) / tts.rate:.1f}s of speech in {time.monotonic() - t:.2f}s")
        audio = pcm.astype(np.float32) / 32768
        audio16 = np.interp(np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / tts.rate)), np.arange(len(audio)), audio).astype(np.float32)
        vad = SileroVAD()
        probs = [vad(audio16[i:i + FRAME]) for i in range(0, len(audio16) - FRAME, FRAME)]
        r(max(probs) > 0.8, f"VAD hears the synthesized speech (peak p={max(probs):.2f})")
        stt = Transcriber(cfg.stt)
        res = stt.transcribe(audio16, language="en")
        ok = "redis" in res.text.lower() and "postgres" in res.text.lower()
        r(ok, f"STT ({stt.device}) in {res.seconds * 1000:.0f} ms: “{res.text}”")
    except Exception as e:
        r(False, f"speech pipeline: {e}")


def cmd_say(args) -> int:
    import numpy as np

    from claude_voice.audio.pulse import PulseStream
    from claude_voice.tts.speech import load_tts

    cfg = _load_config(args)
    tts = load_tts(cfg.tts)
    pcm = tts.synth(" ".join(args.text))
    out = PulseStream(False, cfg.audio.output_device or None, tts.rate, 60, "claude-voice say")
    out.write(np.ascontiguousarray((pcm.astype(np.float32) * cfg.tts.volume).clip(-32768, 32767).astype(np.int16)))
    out.drain()
    out.close()
    return 0


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help")):
        argv.insert(0, "start")
    p = argparse.ArgumentParser(prog="voice", description="Full-duplex voice mode for Claude Code.")
    p.add_argument("--config", help="config file (default ~/.config/claude-voice/config.toml)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start", help="start voice mode in a project (the default command)")
    s.add_argument("project", nargs="?", default=".")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--new", action="store_true", help="start a fresh conversation")
    g.add_argument("--resume", metavar="SESSION_ID", help="continue a specific Claude Code session")
    s.add_argument("--model", help="Claude model for this run (default: your Claude Code default)")
    s.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    s.add_argument("--headless", action="store_true", help="plain event log instead of the TUI")
    s.add_argument("--no-aec", action="store_true", help="disable echo cancellation (headphones only)")
    s.add_argument("--aec-mode", choices=["monitor", "sink"])
    s.add_argument("--input", help="microphone node name")
    s.add_argument("--output", help="speaker node name")
    sub.add_parser("stop", help="stop the running voice mode")
    sub.add_parser("status", help="show the running voice mode's state")
    d = sub.add_parser("doctor", help="diagnose audio, models and Claude Code setup")
    d.add_argument("--quick", action="store_true", help="skip the live audio/model checks")
    say = sub.add_parser("say", help="speak text with the configured voice")
    say.add_argument("text", nargs="+")
    args = p.parse_args(argv)
    handlers = {"start": cmd_start, "stop": cmd_stop, "status": cmd_status, "doctor": cmd_doctor, "say": cmd_say}
    sys.exit(handlers[args.cmd](args))
