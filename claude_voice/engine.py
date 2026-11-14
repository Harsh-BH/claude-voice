"""The voice engine: capture -> VAD -> STT -> Claude -> TTS -> player.

It owns the state machine and carries out the interruption controller's verdicts.

Threads: capture (mic, VAD, segmenter), stt (one job at a time), tts (synthesis), player, and
the asyncio loop, which makes every decision. Threads only post events to the loop. The one
exception is the barge-in fast path: the capture thread pauses the player directly, so playback
stops without waiting for the loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from claude_voice.aec.pipewire import SOURCE as AEC_SOURCE
from claude_voice.aec.pipewire import EchoCanceller
from claude_voice.audio.pulse import Capture, Player
from claude_voice.claude.session import ClaudeSession
from claude_voice.config import RUNTIME_DIR, STATE_DIR, Config
from claude_voice.interruption.controller import Command, InterruptionController, Verdict, is_noise, parse_command, words
from claude_voice.state.machine import State, StateMachine
from claude_voice.stt.whisper import Transcriber, Transcript
from claude_voice.tts.speech import Speech, load_tts
from claude_voice.vad.silero import FRAME, SegEvent, Segmenter, SileroVAD

S = State
log = logging.getLogger(__name__)
UiFn = Callable[[str, dict], None]
_QUIET = {S.IDLE, S.PAUSED}
_HEDGES = {"no", "not", "dont", "instead", "but", "wait", "stop", "nahi", "mat", "नहीं", "मत"}
_DRIVEN = {S.IDLE, S.PAUSED, S.USER_SPEAKING, S.TRANSCRIBING, S.INTERRUPTING, S.ERROR}


@dataclass
class Utterance:
    id: int
    onset: float = 0.0
    barge: bool = False
    language: str | None = None
    candidate: asyncio.Future | None = None  # speculative final started at the first pause


@dataclass
class Permission:
    title: str
    detail: str
    future: asyncio.Future = field(repr=False)


def summarize_tool(name: str, tool_input: dict) -> str:
    for key in ("command", "file_path", "path", "pattern", "url", "query", "description", "prompt"):
        if isinstance(tool_input.get(key), str):
            value = tool_input[key].strip().splitlines()[0] if tool_input[key].strip() else ""
            return value[:160]
    return ""


class Engine:
    def __init__(self, cfg: Config, cwd: str, ui: UiFn, resume: str | None = None, new: bool = False):
        self.cfg, self.cwd, self.ui = cfg, cwd, ui
        self.sm = StateMachine(on_change=lambda old, new, why: self.ui("state", {"state": new.value, "from": old.value, "reason": why}))
        self.ctl = InterruptionController(cfg.barge_in)
        mic, speaker = cfg.audio.input_device, cfg.audio.output_device
        self.aec = EchoCanceller(cfg.aec, mic, speaker) if cfg.aec.enabled else None
        self.input_device = AEC_SOURCE if self.aec else (mic or None)
        self.output_device = self.aec.playback_sink if self.aec else (speaker or None)
        self.session = ClaudeSession(cfg.claude, cwd, self._on_claude, self._ask_permission, resume=resume, new=new)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stt: Transcriber | None = None
        self.vad: SileroVAD | None = None
        self.seg: Segmenter | None = None
        self.capture: Capture | None = None
        self.player: Player | None = None
        self.speech: Speech | None = None
        self.permission: Permission | None = None
        self.mic_status = "off"
        self._stt_pool = ThreadPoolExecutor(1, thread_name_prefix="stt")
        self._stt_busy = False
        self._utt = Utterance(0)
        self._utt_seq = 0
        self._level_frames = 0
        self._level = 0.0
        self._barge: dict[str, Any] = {}
        self._turn: dict[str, Any] = {}
        self._settle_timer: asyncio.TimerHandle | None = None
        self._tasks: set[asyncio.Task] = set()
        self._supervisor: asyncio.Task | None = None
        self.metrics_path = STATE_DIR / "metrics.jsonl"

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.ui("info", {"text": "loading speech models"})
        self.stt = await self.loop.run_in_executor(self._stt_pool, Transcriber, self.cfg.stt)
        tts = await self.loop.run_in_executor(None, load_tts, self.cfg.tts)
        self.ui("info", {"text": f"STT {self.stt.model_name.split('/')[-1]} on {self.stt.device} ({self.stt.load_seconds:.1f}s), "
                                 f"TTS {type(tts).__name__.removesuffix('TTS')}"})
        self.vad = SileroVAD()
        self.seg = Segmenter(self.cfg.vad, self.cfg.barge_in, self.cfg.stt.partial_interval_ms if self.cfg.stt.partials else None)
        self.player = Player(self.output_device, self._player_event, rate=tts.rate, tail_ms=self.cfg.barge_in.tail_ms)
        self.player.volume = self.cfg.tts.volume
        self.player.start()
        self.speech = Speech(tts, self.player, self._speech_event)
        self.ui("info", {"text": "connecting to Claude Code"})
        await self.session.start()
        await self.voice_on()
        self._supervisor = asyncio.create_task(self._supervise())

    async def stop(self) -> None:
        if self._supervisor:
            self._supervisor.cancel()
        self._stop_listening()
        if self.speech:
            self.speech.cancel()
            self.speech.shutdown()
        if self.player:
            self.player.shutdown()
        await self.session.close()
        self._stt_pool.shutdown(wait=False, cancel_futures=True)
        (RUNTIME_DIR / "status.json").unlink(missing_ok=True)

    def _start_listening(self) -> None:
        if self.aec and not self.aec.alive():
            self.aec.start()  # on the loop thread: pdeathsig is tied to the spawning thread
        self.vad.reset()
        self.capture = Capture(self.input_device, FRAME, self._on_frame, self._capture_status, rate=self.cfg.audio.input_rate)
        self.capture.start()

    def _stop_listening(self) -> None:
        if self.capture:
            self.capture.stop()
            self.capture = None
        if self.aec:
            self.aec.stop()
        self.mic_status = "off"

    async def voice_on(self) -> None:
        if self.sm.state in (S.IDLE, S.ERROR):
            self._start_listening()
        self.sm.to(S.LISTENING, "voice on")
        self.ui("voice", {"on": True})

    async def voice_off(self) -> None:
        """F8: release the mic entirely; Claude keeps working silently."""
        self.ctl.clear()
        self.speech.cancel(mute_turn=True)
        self._stop_listening()
        self.sm.to(S.IDLE, "voice off (F8)")
        self.ui("voice", {"on": False})

    async def toggle_voice(self) -> None:
        await (self.voice_on() if self.sm.state == S.IDLE else self.voice_off())

    def toggle_mute(self) -> None:
        self.speech.muted = not self.speech.muted
        if self.speech.muted:
            self.speech.cancel(mute_turn=True)
        self.ui("muted", {"muted": self.speech.muted})
        self._settle()

    async def escape(self) -> None:
        """Esc: stop speech and cancel whatever Claude is doing."""
        self._deny_permission()
        self.ctl.clear()
        self.speech.cancel(mute_turn=True)
        if await self.session.interrupt():
            self.ui("barge", {"stage": "cancelled", "by": "Esc", "tools_stopped": self.session.tools_stopped})
        if self.sm.state == S.INTERRUPTING:
            self.sm.to(S.LISTENING, "Esc")
        self._settle()

    async def submit_text(self, text: str) -> None:
        """Typed input goes to the same session."""
        text = text.strip()
        if not text:
            return
        if self.permission and not self.permission.future.done():
            result = {"y": True, "n": False}.get(text.lower(), self._permission_answer(text, typed=True))
            if result is not None:
                self.ui("user", {"text": text, "typed": True})
                self.permission.future.set_result(result)
                return
        note = ""
        if self.session.busy or self.player.busy:
            self.ctl.begin(barge=self.player.audible, spoken="", heard=self.player.heard_text(), onset=0, now=time.monotonic())
            self.speech.cancel()
            if await self.session.interrupt():
                note = self.ctl.note(self.session.tools_stopped)
            self.ctl.clear()
        self.ui("user", {"text": text, "typed": True})
        await self._submit(text, note)

    # ------------------------------------------------------------------ capture thread
    def _capture_status(self, status: str, detail: str) -> None:
        self.mic_status = status
        self._post(self.ui, "mic", {"status": status, "detail": detail})

    def _post(self, fn, *args) -> None:
        try:
            self.loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:
            pass  # loop closed during shutdown

    def _on_frame(self, pcm: np.ndarray) -> None:
        if self.aec and not self.aec.alive():
            # PipeWire moves an orphaned capture stream to the raw default mic until the
            # restarted canceller's source reappears; that audio still contains Claude's voice.
            return
        x = pcm.astype(np.float32) / 32768.0
        now = time.monotonic()
        prob = self.vad(x)
        state = self.sm.state
        playing = self.player.audible and state not in _QUIET  # no barge-in while voice is off
        for ev in self.seg.feed(x, prob, playing, now):
            if ev.kind == "start" and ev.barge:
                audible = self.player.pause()  # fast path: playback stops without the event loop
                self._barge = {"onset": ev.onset, "detected": now, "audible": audible}
            self._post(self._on_seg, ev)
        self._level = max(float(np.sqrt(np.mean(x * x))), self._level * 0.7)
        self._level_frames += 1
        if self._level_frames % 3 == 0:
            self._post(self.ui, "level", {"rms": self._level, "vad": prob})
            self._level *= 0.5

    # ------------------------------------------------------------------ speech events (loop)
    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception():
            log.error("engine task failed", exc_info=task.exception())
            self._fail(f"internal error: {task.exception()}")

    def _on_seg(self, ev: SegEvent) -> None:
        if self.sm.state == S.IDLE:
            return
        if ev.kind == "start":
            self._speech_started(ev)
        elif ev.kind == "partial" and ev.audio is not None:
            self._spawn(self._partial(ev.audio, self._utt))
        elif ev.kind == "pause" and ev.audio is not None:
            self._utt.candidate = asyncio.ensure_future(self._final_transcript(ev.audio, self._utt))
        elif ev.kind == "resume":
            self._utt.candidate = None
        elif ev.kind == "end":
            self._spawn(self._speech_ended(ev, self._utt))

    def _busy(self) -> bool:
        return self.session.busy or self.player.busy or bool(self.speech.pending)

    def _speech_started(self, ev: SegEvent) -> None:
        self._utt_seq += 1
        self._utt = Utterance(self._utt_seq, ev.onset, ev.barge)
        state = self.sm.state
        if state == S.PAUSED or state == S.ERROR:
            return
        if self.ctl.pending:  # the user paused mid-interruption and carried on
            return
        self._settle()  # make the state reflect what the components are doing right now
        if (ev.barge or self._busy()) and self.sm.can(S.INTERRUPTING):
            self.ctl.begin(barge=ev.barge, spoken=self.player.playing_text(), heard=self.player.heard_text(),
                           onset=ev.onset, now=time.monotonic())
            self.sm.to(S.INTERRUPTING, "barge-in" if ev.barge else "speech while Claude works")
            if ev.barge:
                self.loop.call_later(0.08, self._report_pause)
            self.loop.call_later(self.cfg.barge_in.confirm_timeout_s, self._barge_timeout, self._utt.id)
            return
        if ev.barge:
            self.player.resume()  # paused by the fast path, but there is nothing to interrupt
        if self.sm.can(S.USER_SPEAKING):
            self.sm.to(S.USER_SPEAKING, "speech")

    def _report_pause(self) -> None:
        b = self._barge
        if not b:
            return
        stop = self.player.stop_latency if b.get("audible") else None
        b.update(detect_ms=round((b["detected"] - b["onset"]) * 1000), stop_ms=None if stop is None else round(stop * 1000, 1))
        self.ui("barge", {"stage": "paused", "detect_ms": b["detect_ms"], "stop_ms": b["stop_ms"]})

    async def _stt_call(self, fn, *args):
        return await self.loop.run_in_executor(self._stt_pool, fn, *args)

    async def _partial(self, audio: np.ndarray, utt: Utterance) -> None:
        if self._stt_busy:
            return  # skip: the next partial or the final covers it
        self._stt_busy = True
        try:
            if utt.language is None and len(audio) >= 16000:
                utt.language = await self._stt_call(self.stt.detect_language, audio)
            r: Transcript = await self._stt_call(self.stt.transcribe, audio, utt.language or self.stt.last_language, True)
        finally:
            self._stt_busy = False
        if utt is not self._utt or not r.text:
            return
        self.ui("partial", {"text": r.text})
        if self.ctl.pending and self.sm.state == S.INTERRUPTING:
            await self._judge(r.text, final=False)

    async def _final_transcript(self, audio: np.ndarray, utt: Utterance) -> Transcript:
        lang = utt.language
        if lang is None:
            lang = utt.language = await self._stt_call(self.stt.detect_language, audio)
        return await self._stt_call(self.stt.transcribe, audio, lang, False)

    async def _speech_ended(self, ev: SegEvent, utt: Utterance) -> None:
        eou = time.monotonic()
        if self.sm.state == S.USER_SPEAKING:
            self.sm.to(S.TRANSCRIBING, "end of speech")
        speculative = utt.candidate is not None
        r = await (utt.candidate if speculative else self._final_transcript(ev.audio, utt))
        self._turn = {"eou": eou, "text_ready": time.monotonic(), "stt_s": r.seconds, "speculative": speculative}
        await self._handle_final(r)

    async def _handle_final(self, r: Transcript) -> None:
        text, state = r.text, self.sm.state
        if state == S.IDLE:
            return
        if text:
            self.ui("user", {"text": text, "language": r.language,
                             "latency_ms": round((self._turn["text_ready"] - self._turn["eou"]) * 1000)})
        if state == S.PAUSED:
            if parse_command(text) == Command.VOICE_ON:
                self.sm.to(S.LISTENING, "voice on (spoken)")
                self.ui("voice", {"on": True})
            return
        if self.permission and not self.permission.future.done():
            result = self._permission_answer(text)
            if result is None:  # a garbled short answer, not an instruction: ask again
                self._false_interruption("unclear answer")
                return
            self.ctl.clear()
            self.permission.future.set_result(result)
            self._back_from_interrupting()
            return
        if not self.ctl.pending and self._busy() and self.sm.can(S.INTERRUPTING):
            # speech that began before Claude got busy (e.g. during transcription) and ended after
            self.ctl.begin(barge=False, spoken=self.player.playing_text(), heard=self.player.heard_text(),
                           onset=self._utt.onset, now=time.monotonic())
            self.sm.to(S.INTERRUPTING, "speech while Claude works")
        if self.ctl.pending:
            await self._judge(text, final=True)
            return
        cmd = parse_command(text)
        if cmd:
            await self._run_command(cmd)
        elif not text or is_noise(text):
            self._back_from_interrupting()
        else:
            await self._submit(text)

    # ------------------------------------------------------------------ interruption
    async def _judge(self, text: str, final: bool) -> None:
        p = self.ctl.pending
        if p is None:
            return
        verdict, cmd = self.ctl.judge(text, expecting_answer=self.permission is not None)
        if not final:
            # a prefix of the utterance: act only on what needs speed (never on a pending
            # permission question, where the final transcript decides answer vs instruction)
            if self.permission and not self.permission.future.done():
                return
            if verdict is Verdict.REAL or cmd is Command.INTERRUPT:
                await self._confirm(p, "barge-in confirmed")
            elif cmd is Command.STOP_SPEAKING and not p.confirmed:
                self.speech.cancel(mute_turn=True)
                p.confirmed = p.muted = True
            return
        self._barge_metric(verdict, cmd)
        if verdict is Verdict.COMMAND:
            self.ctl.clear()
            await self._run_command(cmd)
        elif verdict is Verdict.REAL:
            if not p.confirmed or p.muted:
                await self._confirm(p, "barge-in confirmed")
            note = self.ctl.note(self.session.tools_stopped) if p.confirmed and not p.muted else ""
            self.ctl.clear()
            await self._submit(text, note)
        elif p.confirmed:  # interrupted, but nothing intelligible followed
            self.ctl.clear()
            self._back_from_interrupting()
        else:
            self._false_interruption(verdict.value)

    async def _confirm(self, p, why: str) -> None:
        if p.confirmed and not p.muted:
            return
        p.confirmed, p.muted = True, False
        self.speech.cancel()
        self._deny_permission()
        t = time.monotonic()
        stopped = await self.session.interrupt()
        self.ui("barge", {"stage": "confirmed", "claude_interrupted": stopped, "tools_stopped": self.session.tools_stopped,
                          "interrupt_ms": round((time.monotonic() - t) * 1000)})
        if self.sm.state == S.INTERRUPTING:
            self.sm.to(S.USER_SPEAKING if self.seg.active else S.TRANSCRIBING, why)

    def _false_interruption(self, why: str) -> None:
        p = self.ctl.pending
        self.ctl.clear()
        if self.permission and not self.permission.future.done():
            # someone answered over the question but it didn't come through: ask again, briefly
            self.speech.say(["Sorry, I didn't catch that. Yes or no?"])
            self.ui("barge", {"stage": "reprompt", "why": why})
        elif p and p.barge and self.player.paused:
            self.player.resume()
            self.ui("barge", {"stage": "resumed", "why": why})
        self._back_from_interrupting()

    def _barge_timeout(self, utt_id: int) -> None:
        p = self.ctl.pending
        if p and not p.confirmed and self._utt.id == utt_id and self.ctl.timed_out(time.monotonic()):
            self._false_interruption("nothing intelligible in time")

    def _back_from_interrupting(self) -> None:
        state = self.sm.state
        if state in (S.INTERRUPTING, S.TRANSCRIBING, S.USER_SPEAKING):
            self.sm.to(S.SPEAKING if self.player.audible else S.LISTENING, "resolved")
        self._settle()

    def _barge_metric(self, verdict: Verdict, cmd: Command | None) -> None:
        p = self.ctl.pending
        record = {"type": "barge", "verdict": verdict.value, "command": cmd.value if cmd else None,
                  "was_audible": bool(p and p.barge), **{k: v for k, v in self._barge.items() if k.endswith("_ms")}}
        self._metric(record)
        self._barge = {}

    @staticmethod
    def _permission_answer(text: str, typed: bool = False) -> bool | str | None:
        """While Claude waits on a permission question, everything said is taken as the answer.

        True = allow. A str = deny, passing the user's words to Claude as the reason (so "no, write
        it to staging instead" redirects the turn without interrupting it). None = unclear, ask again.
        """
        w = words(text)
        first = parse_command(w[0], expecting_answer=True) if w else None
        if len(w) <= 3:
            answer = parse_command(text, expecting_answer=True) or first
            if answer is Command.YES:
                return True
            if answer is Command.NO:
                return f"The user said no: {text}"
            return f"The user answered: {text}" if typed and w else None
        if first is Command.YES and not set(w[1:]) & _HEDGES:  # "yes, please go ahead"
            return True
        return f"The user answered: {text}"

    def _deny_permission(self) -> None:
        """Claude is blocked in can_use_tool; resolve it so an interrupt is not stuck behind it."""
        if self.permission and not self.permission.future.done():
            self.permission.future.set_result(False)

    async def _run_command(self, cmd: Command) -> None:
        self.ui("command", {"command": cmd.value})
        if cmd is Command.STOP_SPEAKING:
            self.speech.cancel(mute_turn=True)
        elif cmd is Command.INTERRUPT:
            self.speech.cancel(mute_turn=True)
            self._deny_permission()
            await self.session.interrupt()
        elif cmd is Command.REPEAT:
            sentences = self.speech.repeat_text()
            if sentences:
                self.speech.say(sentences)
        elif cmd is Command.VOICE_OFF:
            self.speech.cancel(mute_turn=True)
            self.sm.to(S.PAUSED, "voice off (spoken)")
            self.ui("voice", {"on": False, "paused": True})
            return
        elif cmd is Command.VOICE_ON:
            pass
        elif cmd in (Command.YES, Command.NO) and self.permission and not self.permission.future.done():
            self.permission.future.set_result(cmd is Command.YES)
        self._back_from_interrupting()

    # ------------------------------------------------------------------ Claude
    async def _submit(self, text: str, note: str = "") -> None:
        if self.sm.state not in _QUIET:
            self.sm.to(S.CLAUDE_THINKING, "submitted")
        self.speech.begin_turn()
        self._turn.setdefault("eou", time.monotonic())
        self._turn.update(submitted=time.monotonic(), first_token=None, first_audio=None, text=text[:80])
        try:
            await self.session.submit(f"{note}\n{text}" if note else text)
        except Exception as e:
            log.exception("submit failed")
            self._fail(f"could not reach Claude: {e}")
            self._spawn(self._recover_claude())

    def _on_claude(self, kind: str, data: Any) -> None:
        voice = self.sm.state not in _QUIET
        if kind == "text":
            if self._turn.get("first_token") is None:
                self._turn["first_token"] = time.monotonic()
            if voice:
                self.speech.feed(data)
            self.ui("claude_delta", {"text": data})
            self._settle()
        elif kind == "text_end":
            if voice:
                self.speech.end_block()
            self._settle()
        elif kind == "tool":
            self.ui("tool", {"name": data["name"], "summary": summarize_tool(data["name"], data["input"])})
            self._settle()
        elif kind == "turn_end":
            if voice:
                self.speech.end_block()
            self.ui("claude_done", {"subtype": data.subtype, "reason": data.terminal_reason, "error": data.is_error,
                                    "duration_s": round((data.duration_ms or 0) / 1000, 1)})
            self._turn["ended"] = time.monotonic()
            self._turn_metric(final=self.speech.pending == 0 and not self.player.busy)
            self._settle()
        elif kind == "init":
            d = data
            self.ui("session", {"id": d.get("session_id"), "model": d.get("model"), "forked": self.session.forked,
                                "permission_mode": d.get("permissionMode"), "auth": d.get("apiKeySource"),
                                "mcp": len(d.get("mcp_servers") or []), "plugins": len(d.get("plugins") or []),
                                "skills": len(d.get("skills") or []), "version": d.get("claude_code_version")})
        elif kind == "rate_limit" and data is not None and data.status != "allowed":
            self.ui("warning", {"text": f"Claude usage limit: {data.status} ({data.rate_limit_type})"})
        elif kind == "error":
            self._fail(str(data))
            self._spawn(self._recover_claude())

    async def _recover_claude(self) -> None:
        for delay in (1, 2, 5, 10, 30):
            await asyncio.sleep(delay)
            try:
                await self.session.restart()
            except Exception as e:
                self.ui("error", {"text": f"reconnect failed: {e}; retrying in {delay}s"})
                continue
            self.ui("info", {"text": "reconnected to Claude Code (same conversation)"})
            if self.sm.state == S.ERROR:
                self.sm.to(S.LISTENING if self.capture else S.IDLE, "recovered")
            return

    async def _ask_permission(self, tool: str, tool_input: dict, ctx: Any) -> bool | str:
        title = getattr(ctx, "title", None) or f"Claude wants to use {tool}"
        detail = summarize_tool(tool, tool_input)
        self.permission = Permission(title, detail, self.loop.create_future())
        self.ui("permission", {"title": title, "detail": detail, "tool": tool})
        if self.sm.state not in _QUIET:
            self.speech.say([f"{title.rstrip('.')}. Should I allow it? Say yes or no."])
        try:
            return await self.permission.future
        finally:
            self.permission = None
            self.ui("permission_done", {})

    # ------------------------------------------------------------------ player / tts events
    def _player_event(self, kind: str, clip) -> None:
        self._post(self._on_player, kind, clip)

    def _speech_event(self, kind: str, data) -> None:
        self._post(self._on_speech, kind, data)

    def _on_player(self, kind: str, clip) -> None:
        if kind == "clip_start":
            if self._turn.get("submitted") and self._turn.get("first_audio") is None and clip.seq == 1:
                self._turn["first_audio"] = time.monotonic()
            self.ui("speaking", {"text": clip.text})
            if self.sm.state not in _DRIVEN:
                self.sm.to(S.SPEAKING, "audio")
        elif kind == "error":
            self.ui("error", {"text": "audio output failed; retrying"})
        self._settle()

    def _on_speech(self, kind: str, data) -> None:
        if kind == "synthesized":
            self.ui("synth", data)
        elif kind == "queued":
            self.ui("queued", data)
        self._settle()

    def _settle(self) -> None:
        """Move between the turn states from what the components are actually doing."""
        state = self.sm.state
        if state in _DRIVEN or self.speech is None:
            return
        if self.player.audible or (self.player.busy and not self.player.paused):
            target = S.SPEAKING
        elif self.speech.pending:
            target = S.TTS_GENERATING
        elif self.session.busy:
            target = S.CLAUDE_STREAMING if self.session.streaming_text else S.CLAUDE_THINKING
        else:
            target = S.LISTENING
        if target != state:
            self.sm.to(target, "settle")
        if target == S.SPEAKING and not self.player.busy:
            if self._settle_timer:
                self._settle_timer.cancel()
            self._settle_timer = self.loop.call_at(self.player.audible_until() + 0.02, self._settle)
        if target == S.LISTENING and self._turn.get("ended") and not self._turn.get("reported"):
            self._turn_metric(final=True)

    # ------------------------------------------------------------------ metrics / health
    def _turn_metric(self, final: bool) -> None:
        t = self._turn
        if not final or t.get("reported") or not t.get("submitted"):
            return
        t["reported"] = True
        ms = lambda a, b: round((t[b] - t[a]) * 1000) if t.get(a) and t.get(b) else None  # noqa: E731
        record = {"type": "turn", "eou_to_text_ms": ms("eou", "text_ready"), "submit_to_first_token_ms": ms("submitted", "first_token"),
                  "first_token_to_first_audio_ms": ms("first_token", "first_audio"), "eou_to_first_audio_ms": ms("eou", "first_audio"),
                  "speculative_stt": t.get("speculative"), "stt_s": round(t.get("stt_s") or 0, 3)}
        self._metric(record)
        self.ui("latency", record)

    def _metric(self, record: dict) -> None:
        record["t"] = time.time()
        try:
            with open(self.metrics_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            log.debug("metrics: %s", e)

    def _fail(self, message: str) -> None:
        self.ui("error", {"text": message})
        if self.sm.can(S.ERROR) and self.sm.state not in _QUIET:
            self.sm.to(S.ERROR, message[:60])

    async def _supervise(self) -> None:
        """Restart crashed audio pieces and keep `voice status` up to date."""
        while True:
            await asyncio.sleep(1.0)
            try:
                if self.capture and self.aec and self.aec.ensure():
                    self.ui("warning", {"text": "echo canceller crashed and was restarted"})
                if self.capture and not self.capture.is_alive():
                    self.ui("warning", {"text": "capture thread died; restarting"})
                    self.capture = None
                    self._start_listening()
                if self.player and not self.player.is_alive():
                    self.ui("warning", {"text": "player thread died; restarting"})
                    self.player = Player(self.output_device, self._player_event, rate=self.player.rate, tail_ms=self.cfg.barge_in.tail_ms)
                    self.speech.player = self.player
                    self.player.start()
                if self.sm.state == S.ERROR and not self.session.busy and self.capture and self.mic_status == "ok":
                    self.sm.to(S.LISTENING, "recovered")
                self.write_status()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("supervisor")
                self.ui("error", {"text": f"supervisor: {e}"})

    def write_status(self) -> None:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        status = {"pid": os.getpid(), "cwd": self.cwd, "state": self.sm.state.value, "mic": self.mic_status,
                  "aec": (self.aec.cfg.mode if self.aec.alive() else "down") if self.aec else "disabled",
                  "session_id": self.session.session_id, "model": self.session.model,
                  "muted": bool(self.speech and self.speech.muted), "updated": time.time()}
        tmp = RUNTIME_DIR / "status.json.tmp"
        tmp.write_text(json.dumps(status))
        tmp.replace(RUNTIME_DIR / "status.json")
