# claude-voice

Full-duplex voice mode for Claude Code on Arch Linux: talk, get streamed speech back, interrupt any time.

## Stack

Python 3.13 (uv), Claude Agent SDK driving the user's `claude` CLI, faster-whisper on CUDA (cuBLAS 12 wheel only), Kokoro ONNX on CPU, Silero VAD ONNX, libpulse-simple via ctypes, PipeWire `module-echo-cancel` (WebRTC AEC3) as a child `pipewire -c` process, Textual TUI.

## Commands

| | |
|---|---|
| install | `scripts/install.sh` (no sudo; reports missing pacman packages) |
| run | `voice` in a project dir (`--headless`, `--new`, `--resume ID`) |
| test | `uv run pytest` (unit + models + silent null-sink audio) |
| test, live | `VOICE_CLAUDE=1 uv run pytest tests/test_claude.py tests/test_e2e.py` · `VOICE_ACOUSTIC=1 ... tests/test_acoustic.py` (real speaker) |
| diagnose | `voice doctor` |

## Gotchas

- `system_prompt=None` in the SDK passes `--system-prompt ""` and wipes Claude Code's prompt; always send the preset.
- `interrupt()` ends the turn but leaves a running Bash command alive; `kill_tool_processes` handles it. Record the Bash start at `content_block_start`, not at `AssistantMessage`.
- PipeWire moves an orphaned capture stream to a fallback (raw mic / speaker monitor); the engine drops frames while the canceller is down.
- EasyEffects hijacks unpinned streams: every canceller stream sets `node.dont-move`.
- Textual's MessagePump owns `self._closing`; never reuse Textual private attribute names on the App (test_tui guards this).
- Hotwords only for English; on Hindi they cause wrong romanization.
- `pgrep -f`/`pkill -f` in tests also match the shell running the test if its command line contains the pattern.

## Don't

- Don't run audio tests while `voice` is running (conftest skips them: the canceller source name is global).
- Don't make the canceller always-on: its streams share one graph driver, so the mic runs whenever anything plays.
