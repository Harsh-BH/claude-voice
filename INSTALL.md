# Installing claude-voice

## Requirements

| | |
|---|---|
| OS | Arch Linux (any distro with PipeWire ≥ 1.0 works) |
| Audio | `pipewire`, `pipewire-pulse`, `wireplumber`, `libpulse`. The WebRTC echo-cancel plugin ships with `pipewire` |
| Tools | `uv`, `curl`, `util-linux` (`setpriv`), Claude Code logged in (`claude`, then `/login`) |
| GPU (optional) | NVIDIA driver with CUDA ≥ 12. No CUDA toolkit is needed: the installer adds the cuBLAS 12 wheel (1.1 GB installed, with its nvrtc dependency), the only CUDA library ctranslate2 4.9 loads. Without a GPU, Whisper `small` runs on the CPU |
| Disk | ~3.9 GB with GPU: venv 2.0 GB (1.1 GB of it cuBLAS) + Whisper turbo 1.6 GB + Kokoro 0.34 GB. ~1.7 GB CPU-only (venv 0.9 GB + Whisper small 0.5 GB + Kokoro). Piper adds 61 MB |
| RAM / VRAM | ~1.5 GB RAM for the voice process plus the Claude CLI. ~1.3 GB VRAM for Whisper turbo |

Nothing here needs root. If a package is missing, the installer prints the `pacman` command and
stops.

## Install

```
git clone <this repo> ~/claude-voice      # or use the existing ~/claude-voice
~/claude-voice/scripts/install.sh          # add --with-piper for the Piper fallback voice (63 MB)
```

The script:

1. Checks the prerequisites. It never installs system packages and never changes PipeWire,
   WirePlumber, EasyEffects or Claude Code configuration.
2. Creates `.venv` with Python 3.13 via `uv sync`, adding `--extra gpu` when `nvidia-smi` works.
3. Downloads models:
   - Kokoro v1.0 (and Piper, if asked) → `~/.local/share/claude-voice/models/`, each file
     verified against a pinned SHA-256 hash.
   - Whisper large-v3-turbo (GPU) or small (CPU) → the Hugging Face cache.
4. Links `~/.local/bin/voice` → `.venv/bin/voice`.
5. Copies `config.example.toml` to `~/.config/claude-voice/config.toml` if that file doesn't
   exist yet.
6. Runs `voice doctor --quick`.

Then:

```
voice doctor          # live check: canceller wiring, mic level, TTS, STT
cd ~/some-project && voice
```

## Manual install

```
cd ~/claude-voice
uv sync --python 3.13 --extra gpu          # drop --extra gpu without NVIDIA
ln -s "$PWD/.venv/bin/voice" ~/.local/bin/voice
voice doctor                               # Whisper downloads itself on first start; Kokoro must be fetched:
# https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0 -> kokoro-v1.0.onnx, voices-v1.0.bin
```

## Configuration

`~/.config/claude-voice/config.toml`. Every key is documented in
[config.example.toml](config.example.toml). Unknown keys and wrong types are refused at startup.

Environment variables:

| Variable | Meaning |
|---|---|
| `CLAUDE_VOICE_CONFIG` | config file path (default `~/.config/claude-voice/config.toml`) |
| `CLAUDE_VOICE_DEBUG=1` | debug logging to `~/.local/state/claude-voice/voice.log` |
| `CLAUDE_VOICE_RUNTIME_DIR` | where the lock, status file and generated `aec.conf` live (default `$XDG_RUNTIME_DIR/claude-voice`) |
| `CLAUDE_VOICE_STATE_DIR` | where logs and `metrics.jsonl` go (default `$XDG_STATE_HOME/claude-voice`) |
| `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_STATE_HOME`, `XDG_RUNTIME_DIR` | standard locations for config, models, logs/metrics, runtime files |
| `HF_HOME` | Hugging Face cache (Whisper model) |
| `ANTHROPIC_API_KEY` | **leave unset**: if set, Claude Code bills this API key instead of your plan |
| `VOICE_CLAUDE=1` | tests: enable tests that talk to Claude (Haiku, small usage) |
| `VOICE_ACOUSTIC=1`, `VOICE_ACOUSTIC_VOLUME=45%` | tests: enable the real speaker→mic test, and its volume (restored afterwards) |

Files written at run time:

| Path | What |
|---|---|
| `~/.local/state/claude-voice/voice.log`, `aec.log` | logs (rotated at 5 MB). No transcripts, no credentials |
| `~/.local/state/claude-voice/metrics.jsonl` | per-turn and per-barge-in latency records |
| `$XDG_RUNTIME_DIR/claude-voice/` | `voice.lock`, `status.json`, generated `aec.conf` |

## A launcher key (optional)

`voice` needs a terminal. With Hyprland and kitty, add a bind to your Hyprland config (not done for
you):

```
bind = SUPER, V, exec, kitty --directory ~/your-project voice
```

## Uninstall

```
~/claude-voice/scripts/uninstall.sh          # command, venv, runtime files
~/claude-voice/scripts/uninstall.sh --purge  # also models, config, logs, Whisper cache
```

Sessions created in voice mode are ordinary Claude Code sessions and stay in `~/.claude/projects`.
