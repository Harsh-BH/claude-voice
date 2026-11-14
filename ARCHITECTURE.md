# Architecture

```
            ┌──────────────────────── separate process: pipewire -c aec.conf (setpriv --pdeathsig) ───┐
 mic ──────►│ capture ─┐                                                                              │
            │          ├─ WebRTC AEC3 + NS + transient suppression ──► claude_voice_aec_source ───────┼──┐
 speaker ◄──│ monitor ─┘   (reference = monitor of the speaker: everything you hear)                  │  │
   ▲        └─────────────────────────────────────────────────────────────────────────────────────────┘  │
   │                                                                                                      ▼
   │  Player thread ◄── Clip queue ◄── TTS thread (Kokoro/Piper) ◄── sentences ◄─┐       Capture thread (32 ms)
   │  (20 ms writes, flush on stop)                                              │       Silero VAD → Segmenter
   │        ▲  pause() fast path ◄───────────────────────────────────────────────┼────── barge-in rule
   │        │                                                                    │            │ events
   │   asyncio loop: Engine ── state machine ── InterruptionController ──────────┼────────────┘
   │        │                         │                                          │
   │        │                    STT thread (faster-whisper, CUDA) ◄── utterances│
   │        ▼                                                                    │
   │   ClaudeSession ──(Agent SDK, stream-json)──► claude CLI ──► Claude ────► text deltas
   │        (Textual TUI or headless console renders every engine event)
```

## Threads and ownership

| Thread | Does | Talks to the loop by |
|---|---|---|
| capture | reads 512-sample frames (16 kHz) from the canceller's source, runs VAD + segmenter, measures level | `call_soon_threadsafe`; on barge-in it also calls `player.pause()` directly |
| stt | one Whisper job at a time (partials, speculative finals, finals) | `run_in_executor` futures |
| tts | synthesizes queued sentences one ahead of playback; drops jobs from old epochs | events (`synthesized`) |
| player | writes 20 ms chunks to one long-lived pulse stream; pause/stop flush the server buffer | events (`clip_start`, `clip_end`, `drained`) |
| asyncio loop | every decision: state transitions, interruption verdicts, Claude I/O, UI | n/a |

Only the loop changes state; `StateMachine.to()` refuses calls from any other thread. The single
cross-thread shortcut is the barge-in pause: it acts on the player, not on the state, so playback
stops without waiting for the loop.

## State machine

`claude_voice/state/machine.py` defines the states and the full table of legal transitions:

`IDLE` (F8 off) · `PAUSED` ("voice off") · `LISTENING` · `USER_SPEAKING` · `TRANSCRIBING` ·
`CLAUDE_THINKING` · `CLAUDE_STREAMING` · `TTS_GENERATING` · `SPEAKING` · `INTERRUPTING` · `ERROR`

Full duplex means a turn's sub-activities overlap: Claude can be generating while sentence 1
plays and sentence 2 synthesizes. The *turn* states are therefore derived by a single policy,
`Engine._settle()`, from what the components report, in priority order:

1. audible → `SPEAKING`
2. synthesizing → `TTS_GENERATING`
3. text streaming → `CLAUDE_STREAMING`
4. turn running → `CLAUDE_THINKING`
5. otherwise `LISTENING`

The user-driven states (`USER_SPEAKING`, `TRANSCRIBING`, `INTERRUPTING`, `PAUSED`, `IDLE`) are only
entered and left by explicit events. Every transition goes through the legality table; the
tests check that every state is reachable and that `IDLE`/`ERROR` are reachable from everywhere.

## Barge-in

Barge-in runs in two stages.

**Stage 1: acoustic.** This runs in the capture thread, under ~300 ms. While Claude is audible
(including a 250 ms echo tail), a frame qualifies only if both hold:
- Silero p ≥ 0.5;
- level ≥ ambient + 6 dB, where ambient is a slow average of non-speech frames taken while nothing
  plays.

When 160 ms of qualifying frames land within a 320 ms window, the capture thread calls
`player.pause()`. The player flushes within one 20 ms chunk.

**Stage 2: semantic.** This runs on the loop. The first partial transcript, about 320 ms in, is
judged:

| Verdict | Condition | Action |
|---|---|---|
| command | the whole utterance is a known phrase | run it (stop speaking / interrupt / repeat / voice off/on / yes / no) |
| real | anything else intelligible | confirm: drop queued speech, `interrupt()` Claude, kill an in-flight foreground Bash command, then submit the full utterance with a note of what you heard |
| echo | ≥ 80% of its words were in what was playing | stay paused, keep listening |
| noise | empty or only fillers ("um", "hmm") | stay paused, keep listening |

If nothing intelligible arrives within 2 s, the paused clip resumes from what you actually heard
(rewound by 250 ms). Speech while Claude works silently goes through the same judge without a
pause, so "cancel that" stops a running tool call.

### How the thresholds were chosen

The thresholds come from measurements on this laptop, not guesses.

**Real echo.** Echo after AEC, through the laptop speaker at 80% volume into the digital mic,
sits around −69 dBFS (median); ambient is −47 dBFS. Its speech probability still reaches 0.77,
so probability alone cannot tell it from you. Level does. Result: 0 false barge-ins over 56 s of
real echo.

**Double talk.** Calibrated in a silent virtual room: speaker → pw-loopback 60 ms → room ← user +
noise, mic = room monitor. With the user 10 dB above the echo, all five test phrases were caught
in 263–393 ms from voice onset, and that figure includes ~100 ms of virtual-device latency.

**The limit.** At *equal* levels AEC3 chops the user's onset, which is its design trade-off
(clipped double talk over leaked echo), and 2 of 5 phrases were missed. See the README's
limitations.

The calibration scripts live in the test suite: `tests/test_aec.py` and `tests/test_acoustic.py`.

## Echo cancellation

`claude_voice/aec/pipewire.py` writes `aec.conf` into the runtime dir and runs
`setpriv --pdeathsig TERM -- pipewire -c aec.conf`. That process loads
`libpipewire-module-echo-cancel` with `aec/libspa-aec-webrtc`, i.e. webrtc-audio-processing 1.3,
which is AEC3.

- **Monitor mode** (default): the reference is the monitor of the speaker sink, i.e. exactly what
  reaches the speaker after EasyEffects. Music and videos are cancelled too.
- **Sink mode**: Claude plays into a dedicated sink, and only its voice is cancelled.
- **Never your config:** the user's PipeWire/WirePlumber configuration is not modified. The nodes
  exist only while voice mode runs, and `pdeathsig` removes them if the app dies.
- **`node.dont-move` on every stream.** Without it, EasyEffects' "process all inputs" re-routed
  the canceller's capture through its own chain (observed here). It also kept the mic open.
- **Not always-on.** The canceller's capture and reference streams share one graph driver, so an
  idle canceller still runs the mic whenever anything plays. It starts with voice mode and F8
  stops it.
- **Supervision.** A crashed canceller is restarted by the engine's supervisor. Meanwhile PipeWire
  parks the app's capture stream on the raw default mic, so frames are dropped while the
  canceller process is down. Otherwise Claude's voice would reach the VAD.

## Claude Code integration

| Option | Pushes text into the open TUI | Streams tokens | Cancels a turn | Verdict |
|---|---|---|---|---|
| Channels (`--channels`, MCP `notifications/claude/channel`) | yes, but queued until the turn ends | no (only reply tools / transcript) | no | no barge-in; also off by default for Team orgs |
| Hooks (Stop `decision: block`, `asyncRewake`) | only at turn boundaries | no | no | half-duplex |
| Peer messaging / `claude --resume <id> "msg"` | queued | no | no | same limits |
| PTY / tmux keystrokes | yes | screen-scraping | Esc, blindly | not a supported interface |
| **Agent SDK `ClaudeSDKClient`** | its own session (continue/resume/fork) | **yes** (`include_partial_messages`) | **yes** (`interrupt()`, 7–20 ms) | **chosen** |

`claude_voice/claude/session.py` configures the SDK as follows:

- `cli_path`: the `claude` on your PATH, so it auto-updates with Claude Code.
- `setting_sources=["user","project","local"]`.
- `system_prompt={"type":"preset","preset":"claude_code","append": VOICE_PROMPT}`. Passing `None`
  would make the SDK send `--system-prompt ""` and wipe Claude Code's own prompt.
- `continue_conversation`, `resume` and `fork_session`.
- A `can_use_tool` callback that turns permission prompts into a spoken question plus `y`/`n`.

**Where the session resumes from.** `continue` resumes the project's latest conversation. If
`~/.claude/sessions/*.json` shows a live interactive session in the same directory, the
conversation is forked instead of being written to concurrently. When that registry is
unreadable, it forks to stay safe.

**Interrupts.** `interrupt()` ends the turn (`terminal_reason: aborted_streaming|aborted_tools`)
but, measured, leaves a running Bash command alive until the CLI exits. The session therefore:
1. records each foreground Bash `tool_use` at `content_block_start`. That moment precedes the
   process start; the `AssistantMessage` arrives ~0.5 s too late.
2. On interrupt, SIGTERMs Claude Code's Bash-tool shells under our process that started after
   that time, with their children, then SIGKILLs any survivors after 1.5 s.

Background shells (`run_in_background`) are left alone.

**Context.** Context is preserved: after an interrupt mid-count, Claude correctly recalled the
last number it had streamed. The note sent with an interruption carries what you heard, so Claude
knows where you cut in.

## Speech recognition

- **Model.** faster-whisper large-v3-turbo, int8_float16 on CUDA. CTranslate2 links cudart
  statically and only dlopens `libcublas.so.12`, which is preloaded from the pip wheel. On CPU it
  falls back to `small` int8.
- **Cost.** About 430 ms per utterance regardless of length, because the encoder always sees a
  padded 30 s window. Detecting the language costs another encoder pass, so it happens once per
  utterance in the first partial and is reused.
- **Speculative final.** After 256 ms of silence the utterance is transcribed. If no speech
  resumes before the 700 ms end-of-turn, that result is used: the transcript is ready 0–16 ms
  after end-of-turn.
- **Languages.** Detection is restricted to `stt.languages`, and spoken Hindi labelled Urdu counts
  as Hindi.
- **Hotwords.** They apply to English only. On Hindi they caused wrong romanization ("कृपया
  टेस्ट" → "Kripya Taste").
- **Hallucinations.** Audio below −60 dBFS never reaches Whisper. Known silence hallucinations
  ("Thank you.", "www.youtube.com.au") are dropped.

## Speech synthesis

- **Markdown to sentences.** `tts/text.py` turns streamed markdown into sentences line by line,
  releasing a sentence from the still-open line as soon as it ends. Cuts are never made inside
  inline code or links.
- **What gets dropped.** Code fences, tables and rules are skipped. Inline code, emphasis, links
  and paths are reduced to their words (`/a/b/cache_layer.py` → "cache layer.py"). The first
  chunk may end at a comma, so the first audio comes sooner.
- **Epochs.** `tts/speech.py` keeps an epoch per reply. Cancel bumps it, so a sentence still being
  synthesized when you barge in is discarded, never played.
- **Voices.** Kokoro v1.0 fp32 runs on CPU at about 5× real time. Devanagari sentences switch to
  the Hindi voice. Piper is the fallback if Kokoro fails to load.

## Failure handling

| Failure | Handling |
|---|---|
| mic/source disappears | PipeWire re-links when it returns. If the stream errors instead, the capture thread reconnects with backoff (0.25 → 4 s) |
| canceller crashes | frames are dropped while it is down; the supervisor restarts it within 1 s |
| player or capture thread dies | the supervisor restarts it |
| Claude CLI dies / network error | error shown, state `ERROR`; reconnects with `resume=<session id>` at 1, 2, 5, 10, 30 s, so the conversation is kept |
| a turn hangs after interrupt | after 15 s the CLI is restarted on the same session |
| second `voice` instance | refused (`flock` on the runtime lock) |
| SIGTERM (`voice stop`) | graceful: capture, player, canceller and CLI are stopped in order |

## Tests

`uv run pytest` covers:
- state machine, markdown and sentence splitting, segmenter timing, interruption policy, config
  validation, TUI keys and rendering;
- Silero/Whisper/Kokoro/Piper on real models;
- capture, `stop()` latency, pause/resume and device loss on silent null sinks;
- AEC wiring, echo suppression and double-talk barge-in in the virtual room.

Gated tests:
- `VOICE_CLAUDE=1`: the real session (streaming, context, interrupt, tool kill, resume) and three
  end-to-end runs of the whole engine. These are the acceptance sequence, "cancel that" during a
  running script, and the spoken commands.
- `VOICE_ACOUSTIC=1`: the real speaker→mic run.
