# claude-voice

Talk to Claude Code like a phone call. The microphone stays open, Claude's reply is spoken while
it is still being written, and you can cut in at any moment. Playback stops, Claude is
interrupted, and your correction goes into the same conversation.

Everything except Claude itself runs locally on your machine:
- **Echo cancellation:** PipeWire's WebRTC AEC3.
- **Voice activity detection:** Silero VAD.
- **Speech recognition:** Whisper large-v3-turbo on the GPU.
- **Speech synthesis:** Kokoro, with Piper as fallback.

Only the recognized text goes to Claude.

```
cd ~/your-project
voice
```

## How it relates to your Claude Code session

`voice` drives the Claude Code CLI you already have, through the official Claude Agent SDK. That's
the same harness as the TUI:
- the same login;
- the same model and permission settings;
- `CLAUDE.md`, skills, plugins, hooks and MCP servers (verified on this machine: 13 plugins,
  103 skills, 42 MCP servers loaded).

The conversation is an ordinary Claude Code session:

- `voice` continues the latest conversation in the current directory (`--new` for a fresh one,
  `--resume <id>` for a specific one).
- If that conversation is still open in a `claude` TUI, voice mode forks it, so the two never
  write to the same transcript.
- When you are done talking, `claude -c` in the same directory picks the conversation up in the
  TUI, voice turns included.

Why not inject speech into the open TUI? Claude Code's supported injection paths (channels, hooks)
queue messages until the current turn ends and cannot cancel it, so real barge-in is impossible
that way. Details in [ARCHITECTURE.md](ARCHITECTURE.md#claude-code-integration).

## Controls

| Key | Action |
|---|---|
| **F8** | Voice mode on/off (off releases the microphone entirely) |
| **F9** | Mute/unmute spoken replies (text keeps streaming) |
| **Esc** | Stop speaking and cancel what Claude is doing (including a running command) |
| **Ctrl+C** | Quit cleanly |
| typing + Enter | Send text to the same session (`y`/`n` answers a permission prompt) |

| Say | Effect |
|---|---|
| anything, while Claude talks | barge-in: playback stops within ~0.1 s; Claude is interrupted and gets your words plus a note of what you heard |
| "stop" / "wait" / "cancel that" / "interrupt" / "रुको" | stop speaking **and** cancel Claude's turn |
| "stop speaking" / "be quiet" / "बस" | stop speaking; Claude keeps working silently |
| "repeat that" | say the last reply again |
| "voice off" / "voice on" | pause listening (only "voice on" is recognized while paused) |
| "yes" / "no" / "हाँ" / "नहीं" | answer a permission prompt. A longer reply ("no, write it to staging instead") declines and hands your words to Claude; an unclear one gets "yes or no?" again |

Commands only count as the whole utterance: "stop using Redis and switch to Postgres" is an
instruction, not a command.

## Commands

```
voice [start] [DIR] [--new | --resume ID] [--model M] [--effort E] [--headless]
                    [--no-aec] [--aec-mode monitor|sink] [--input SOURCE] [--output SINK]
voice stop        # end the running voice mode (from any terminal)
voice status      # state, session, mic and canceller of the running instance
voice doctor      # check PipeWire, the canceller, GPU, models, Claude login; live mic/STT/TTS test
voice say TEXT    # speak with the configured voice
```

## Billing

With a claude.ai login and no `ANTHROPIC_API_KEY`, voice mode uses your **plan's normal usage
limits**, just like the TUI. This machine is on a Team plan, and the SDK reported
`apiKeySource: none`. Anthropic's support article ["Use the Claude Agent SDK with your Claude
plan"](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan)
says: *"when you're signed in with your Claude plan, … the Claude Agent SDK and `claude -p` still
draw from your subscription limits"*. The separate SDK credit announced for June 2026 was paused.
If `ANTHROPIC_API_KEY` is set, the CLI bills that key instead; `voice doctor` warns about it.

## Measured on this machine

i7-12650H, RTX 3050 6 GB, PipeWire 1.6.8, Claude Code 2.1.323.

| What | Result |
|---|---|
| Speech recognition (turbo, CUDA int8) | ~430 ms per utterance, 1.25 GB VRAM. The speculative final makes the transcript ready **0–96 ms** after end-of-speech is detected (700 ms of silence) |
| First audio after you stop talking | 1.46 s with Haiku (end-to-end test). Opus 5.5 first token: 0.8 s warm, 2.2 s cold |
| Kokoro synthesis (CPU) | ~5× faster than real time; "Sure." in 0.2 s |
| `stop()` → silence | flush 5–16 ms; ≤ 85 ms on a null sink; 35–101 ms through the real speaker and mic (upper bounds, incl. 32 ms measurement framing) |
| Barge-in detection | detector fires 122–139 ms after the first qualifying frame. 263–393 ms from voice onset in the calibration room (incl. ~100 ms virtual-device path) |
| Echo cancellation, real speaker at 80% | 11–17 dB echo reduction. Echo frames VAD calls speech: 35–83% raw → 0–16% after AEC. **0 false barge-ins over 56 s of echo** |
| `interrupt()` → turn cancelled | 7–20 ms; a running Bash command is killed too |

## Known limitations

- **Talking over the speaker at the same loudness.** If your voice reaches the mic no louder than
  Claude's echo, WebRTC AEC3 suppresses the onset of your words and barge-in can be missed (2 of 5
  phrases in the equal-level stress test). Speak up, lower the volume, or use headphones. With
  headphones you can also run `--no-aec`.
- **Synthetic-voice recognition errors.** One synthetic test voice's "Redis" was heard as
  "Ariti's" despite the hotword list. Real voices are untested here beyond the live acceptance
  test you run yourself.
- **Not the TUI.** Voice mode is not the TUI: `/config`, `/mcp` and other TUI-only screens need
  `claude`. Slash commands that work in `claude -p` work when typed into voice mode.
- **Fast models finish first.** A fast model (Haiku, Opus at low effort) often finishes generating
  long before it finishes speaking. Barge-in then only drops queued speech; there is nothing left
  to cancel.

More: [INSTALL.md](INSTALL.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [TROUBLESHOOTING.md](TROUBLESHOOTING.md) · [config.example.toml](config.example.toml)
