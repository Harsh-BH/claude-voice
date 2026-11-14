#!/usr/bin/env bash
# Remove what scripts/install.sh added. The project directory itself is left for you to delete.
#
#   scripts/uninstall.sh            remove the command, the Python environment and runtime files
#   scripts/uninstall.sh --purge    also delete downloaded models, config, logs and the Whisper cache
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}/claude-voice"
CONF="${XDG_CONFIG_HOME:-$HOME/.config}/claude-voice"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/claude-voice"
RUN="${XDG_RUNTIME_DIR:-/tmp/claude-voice-$(id -u)}/claude-voice"
HF="${HF_HOME:-$HOME/.cache/huggingface}/hub"
PURGE=0
[ "${1:-}" = "--purge" ] && PURGE=1

say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }

if [ -x "$ROOT/.venv/bin/voice" ]; then
  "$ROOT/.venv/bin/voice" stop >/dev/null 2>&1 || true
fi

link="$HOME/.local/bin/voice"
if [ -L "$link" ] && [ "$(readlink "$link")" = "$ROOT/.venv/bin/voice" ]; then
  rm -f "$link"; say "removed $link"
elif [ -e "$link" ]; then
  echo "    left $link alone: it is not this install's symlink"
fi

rm -rf "$ROOT/.venv" "$RUN"
say "removed the Python environment and runtime files"

if [ "$PURGE" = 1 ]; then
  for d in "$DATA" "$STATE" "$CONF"; do
    [ -e "$d" ] && { du -sh "$d" | sed 's/^/    deleting /'; rm -rf "$d"; }
  done
  for m in models--dropbox-dash--faster-whisper-large-v3-turbo models--Systran--faster-whisper-small; do
    [ -e "$HF/$m" ] && { du -sh "$HF/$m" | sed 's/^/    deleting /'; rm -rf "$HF/$m"; }
  done
  say "purged models, config, logs and the Whisper cache"
else
  echo "    kept models ($DATA), config ($CONF) and logs ($STATE); --purge deletes them"
fi
echo "    Claude Code sessions created in voice mode stay in ~/.claude/projects (they are ordinary sessions)."
echo "    Delete $ROOT yourself if you no longer want the source."
