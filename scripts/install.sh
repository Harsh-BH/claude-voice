#!/usr/bin/env bash
# Install claude-voice for the current user. Safe to re-run. Never uses sudo and never changes
# system, PipeWire or Claude Code configuration; it only reports missing Arch packages.
#
#   scripts/install.sh [--with-piper] [--skip-models]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}/claude-voice"
CONF="${XDG_CONFIG_HOME:-$HOME/.config}/claude-voice"
BIN="$HOME/.local/bin"
WITH_PIPER=0
SKIP_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --with-piper) WITH_PIPER=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
missing=0
need() { command -v "$1" >/dev/null 2>&1 || { echo "  missing: $1  ->  $2"; missing=1; }; }

say "checking prerequisites"
need pactl    "sudo pacman -S libpulse"
need pipewire "sudo pacman -S pipewire pipewire-pulse wireplumber"
need pw-link  "sudo pacman -S pipewire"
need setpriv  "sudo pacman -S util-linux"
need uv       "sudo pacman -S uv"
need curl     "sudo pacman -S curl"
need claude   "install Claude Code (https://code.claude.com) and log in"
[ -e /usr/lib/spa-0.2/aec/libspa-aec-webrtc.so ] || { echo "  missing: WebRTC echo-cancel plugin  ->  sudo pacman -S pipewire"; missing=1; }
if command -v pactl >/dev/null 2>&1 && ! pactl info 2>/dev/null | grep -q PipeWire; then
  echo "  pipewire-pulse is not running  ->  systemctl --user enable --now pipewire-pulse"; missing=1
fi
[ "$missing" = 0 ] || { echo "Install the packages above, then run this script again."; exit 1; }

gpu=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then gpu=1; fi
extras=()
[ "$gpu" = 1 ] && extras+=(--extra gpu)
[ "$WITH_PIPER" = 1 ] && extras+=(--extra piper)
say "creating the Python environment (uv, Python 3.13)${extras[*]:+ with ${extras[*]}}"
[ "$gpu" = 1 ] && echo "    NVIDIA GPU found: adding cuBLAS 12 (1.1 GB installed), the only CUDA library speech recognition needs"
(cd "$ROOT" && uv sync --python 3.13 "${extras[@]}")

fetch() {  # url dest sha256
  local url=$1 dest=$2 sum=$3
  if [ -f "$dest" ] && echo "$sum  $dest" | sha256sum -c --quiet 2>/dev/null; then return; fi
  echo "    downloading $(basename "$dest")"
  curl -fL --retry 3 --progress-bar -o "$dest.part" "$url"
  echo "$sum  $dest.part" | sha256sum -c --quiet || { rm -f "$dest.part"; echo "checksum mismatch for $url" >&2; exit 1; }
  mv "$dest.part" "$dest"
}

if [ "$SKIP_MODELS" = 0 ]; then
  say "downloading speech models into $DATA/models"
  mkdir -p "$DATA/models/kokoro"
  K=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
  fetch "$K/kokoro-v1.0.onnx" "$DATA/models/kokoro/kokoro-v1.0.onnx" 7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5
  fetch "$K/voices-v1.0.bin" "$DATA/models/kokoro/voices-v1.0.bin" bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d
  if [ "$WITH_PIPER" = 1 ]; then
    mkdir -p "$DATA/models/piper"
    P=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium
    fetch "$P/en_US-lessac-medium.onnx" "$DATA/models/piper/en_US-lessac-medium.onnx" 5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f
    fetch "$P/en_US-lessac-medium.onnx.json" "$DATA/models/piper/en_US-lessac-medium.onnx.json" efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0
  fi
  if [ "$gpu" = 1 ]; then repo=dropbox-dash/faster-whisper-large-v3-turbo; size="1.6 GB"; else repo=Systran/faster-whisper-small; size="0.5 GB"; fi
  echo "    Whisper: $repo ($size, Hugging Face cache)"
  "$ROOT/.venv/bin/python" -c "from huggingface_hub import snapshot_download as s; s('$repo', allow_patterns=['*.json', '*.bin', '*.txt'])"
fi

say "installing the 'voice' command"
mkdir -p "$BIN"
ln -sfn "$ROOT/.venv/bin/voice" "$BIN/voice"
case ":$PATH:" in *":$BIN:"*) ;; *) echo "    add $BIN to your PATH" ;; esac

mkdir -p "$CONF"
if [ ! -e "$CONF/config.toml" ]; then
  cp "$ROOT/config.example.toml" "$CONF/config.toml"
  echo "    wrote $CONF/config.toml (all defaults, documented)"
else
  echo "    kept your existing $CONF/config.toml"
fi

say "checking the setup"
"$BIN/voice" doctor --quick || true
cat <<EOF

Done. Start voice mode in a project:   cd ~/your-project && voice
Full live check (mic, canceller, STT): voice doctor
EOF
