# Troubleshooting

Start with `voice doctor`. It checks:
- PipeWire and the echo-cancel plugin;
- the default mic and speaker, and whether either is muted;
- EasyEffects;
- CUDA;
- the models;
- the Claude login and a stray `ANTHROPIC_API_KEY`.

Then it starts the canceller, verifies its wiring, measures the mic level through it, and runs a
TTS → VAD → STT round trip. Logs: `~/.local/state/claude-voice/voice.log` (`CLAUDE_VOICE_DEBUG=1`
for more) and `aec.log`. Latency records: `metrics.jsonl`.

| Symptom | Likely cause | Fix |
|---|---|---|
| You hear nothing | Default speaker muted. On this laptop it often is, and the configured default is Bluetooth headphones | `wpctl set-mute @DEFAULT_AUDIO_SINK@ 0`; check `voice doctor`; press F9 in case speech is muted |
| Claude interrupts itself, or "hears" its own words | Echo reaching the VAD: canceller off (`--no-aec`), very loud speakers, or a speaker that isn't the default sink in monitor mode | Keep AEC on; set `audio.output_device` to the speaker you actually use; lower the volume; raise `barge_in.level_margin_db` to 9–12 |
| Barge-in doesn't trigger while Claude talks | Your voice reaches the mic no louder than the echo, so AEC3 suppresses it (double talk) | Speak up or closer; lower the volume; use headphones; or lower `barge_in.level_margin_db` (3) and `barge_in.threshold` (0.4) |
| It cuts you off mid-sentence | 700 ms of silence ends a turn | Raise `vad.min_silence_ms` (900–1200) |
| Long delay before Claude speaks | Model latency; Opus' first token is ~0.8 s warm, ~2.2 s cold. Tool use before the first sentence adds more | `voice --effort low`, or `--model sonnet`/`haiku`; check the `⏱` line in the log for where the time goes |
| Technical words misheard | Whisper vocabulary | Add them to `stt.hotwords` (English only) |
| Hindi comes out romanized or as Urdu script | Language detection on very short or mixed speech | Keep `languages = ["en", "hi"]`; speak a full sentence; on Hindi, hotwords are not applied by design |
| `STT … on cpu` though you have NVIDIA | cuBLAS 12 wheel missing or driver problem | `uv sync --extra gpu`; `nvidia-smi` must work; `voice doctor` shows the CUDA check |
| `echo canceller exited` / `did not create its source` | Old PipeWire without the WebRTC plugin, or a broken generated config | `voice doctor`; read `~/.local/state/claude-voice/aec.log`; `pacman -S pipewire` |
| Mic indicator stays on after quitting | Should not happen: the canceller dies with the app | `pgrep -a pipewire` should list only the system one; file a bug with `aec.log` |
| EasyEffects effects missing on Claude's voice in sink mode | Sink mode pins Claude's stream to the canceller sink | Use monitor mode (default), which cancels after EasyEffects |
| Bluetooth headset mic sounds bad | HFP narrowband profile | Use the laptop mic with headphones (`audio.input_device`); headphones need no AEC (`--no-aec`) |
| "already running" | Another `voice` holds the lock | `voice status`, `voice stop` |
| Claude says it can't do something it can in the TUI | Permission mode or settings differ | `voice` uses your settings; a prompt shows as "⚠ … allow?". Answer `y`/`n` or say yes/no |
| `Claude usage limit: allowed_warning` | Near your plan's limit | Voice turns count like TUI turns |
| Bills an API key instead of the plan | `ANTHROPIC_API_KEY` is set | Unset it; `voice doctor` warns |
| Want the conversation in the normal TUI | | Quit voice mode, then `claude -c` in the same directory |

## Tests

- Run `uv run pytest` while `voice` is **not** running. The audio tests skip themselves otherwise,
  because the canceller's source name is global.
- A test that hangs in Textual is almost always an App attribute shadowing a Textual private name;
  `tests/test_tui.py` guards against it.
- `pgrep -f` / `pkill -f` patterns also match your own shell if its command line contains the
  pattern. Keep patterns out of the command that runs the test.
