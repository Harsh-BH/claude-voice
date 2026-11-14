"""End to end, silently: the whole engine and a real Claude Code session in a virtual room.

    speaker sink --pw-loopback (60 ms)--> room sink <-- simulated user voice + -50 dBFS noise
    mic = room monitor; canceller in monitor mode (reference = speaker monitor)

    VOICE_CLAUDE=1 uv run pytest -s tests/test_e2e.py      (Haiku, a few cents of usage)

The user (Kokoro "am_michael") is 10 dB louder than Claude's echo at the mic: realistic for this
laptop, where 80%-volume echo measured only ~5 dB above the fan noise.
"""
import asyncio
import os
import time

import numpy as np
import pytest
from conftest import play_async

from claude_voice.config import Config, TtsConfig
from claude_voice.engine import Engine
from claude_voice.tts.speech import KokoroTTS
from claude_voice.ui.console import ConsoleUI

pytestmark = [pytest.mark.audio, pytest.mark.models, pytest.mark.claude]


@pytest.fixture
def room(null_sink, loopback):
    spk, rm = null_sink("cvspk"), null_sink("cvroom")
    loopback(spk, rm, 0.06)
    noise = (np.random.default_rng(2).standard_normal(16000 * 150) * 32767 * 10 ** (-50 / 20)).astype(np.int16)
    play_async(rm, noise, 16000)
    return spk, rm


class Watch:
    def __init__(self, ui: ConsoleUI):
        self.ui = ui

    async def until(self, pred, timeout: float, what: str, since: int = 0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for i, (t, kind, data) in enumerate(self.ui.events[since:], since):
                if pred(kind, data):
                    return i, t, data
            await asyncio.sleep(0.02)
        raise AssertionError(f"timed out waiting for {what}")


def test_full_duplex_conversation(room, tmp_path):
    spk, rm = room
    cfg = Config()
    cfg.audio.input_device, cfg.audio.output_device = f"{rm}.monitor", spk
    cfg.claude.model = "haiku"
    cfg.tts.volume = 0.32
    user = KokoroTTS(TtsConfig(voice="am_michael"))
    ui = ConsoleUI()
    w = Watch(ui)

    def say(text: str) -> float:
        pcm = user.synth(text)
        lead = int(np.argmax(np.abs(pcm) > 0.02 * 32767)) / user.rate
        t = time.monotonic()
        play_async(rm, pcm, user.rate)
        return t + lead  # when the voice itself starts

    async def main():
        engine = Engine(cfg, str(tmp_path), ui, new=True)
        await engine.start()
        try:
            await w.until(lambda k, d: k == "state" and d["state"] == "LISTENING", 90, "LISTENING")
            await asyncio.sleep(1.5)  # let the canceller and the ambient estimate settle

            # 1-4: a programming instruction; the reply starts playing while it is still being generated
            say("My project codename is papaya. Remember it. Then, without using any tools, explain in about six sentences how a Redis cache speeds up a web app.")
            i_user, _, d = await w.until(lambda k, d: k == "user" and "papaya" in d["text"].lower(), 40, "first transcript")
            print(f"\n[1] heard: {d['text']!r} ({d['latency_ms']} ms after end of speech)")
            i_q, t_q, _ = await w.until(lambda k, d: k == "queued", 60, "first sentence queued for speech", i_user)
            i_spk, t_spk, _ = await w.until(lambda k, d: k == "speaking", 60, "first audio", i_user)
            i_end, t_end, _ = await w.until(lambda k, d: k == "claude_done", 60, "generation end", i_user)
            # streaming: sentence 1 goes to TTS while Claude is still generating. Whether its audio also
            # starts before generation ends is a race between model speed and synthesis (Haiku often wins).
            assert t_q < t_end, "speech was only prepared after Claude finished generating"
            print(f"[2-4] first sentence queued {t_end - t_q:.2f}s before generation ended; "
                  f"audio started {'before' if t_spk < t_end else 'after'} it ended ({t_spk - t_end:+.2f}s)")

            # 5-6: interrupt halfway through a sentence
            await asyncio.sleep(1.5)
            assert engine.sm.state.value in ("SPEAKING", "TTS_GENERATING"), engine.sm.state
            onset = say("Stop. What is my project codename?")
            _, t_pause, d = await w.until(lambda k, d: k == "barge" and d["stage"] == "paused", 5, "barge-in pause", i_spk)
            barge_ms = (t_pause - onset) * 1000
            print(f"[5-6] barge-in: playback paused {barge_ms:.0f} ms after the user's voice started "
                  f"(detector {d['detect_ms']} ms after onset, flush {d['stop_ms']} ms)")
            _, _, d = await w.until(lambda k, d: k == "barge" and d["stage"] == "confirmed", 8, "confirmation", i_spk)
            print(f"     Claude {'was still generating and got interrupted' if d['claude_interrupted'] else 'had finished generating; queued speech was dropped'}")
            await w.until(lambda k, d: k == "claude_done", 15, "first turn to end", i_spk)

            # 7-10: the correction is recognized, answered with the old context, and spoken
            i2, _, d = await w.until(lambda k, d: k == "user" and "codename" in d["text"].lower(), 20, "correction transcript", i_spk)
            print(f"[7] heard: {d['text']!r}")
            i_done, _, _ = await w.until(lambda k, d: k == "claude_done", 60, "second turn", i2 + 1)
            reply = "".join(d["text"] for _, k, d in ui.events[i2:i_done] if k == "claude_delta")
            print(f"[8-9] Claude: {reply.strip()!r}")
            assert "papaya" in reply.lower()
            _, _, d = await w.until(lambda k, d: k == "speaking", 30, "spoken answer", i2)
            print(f"[10] spoken: {d['text']!r}")
            lat = [d for _, k, d in ui.events if k == "latency"]
            for record in lat:
                print("     latency:", {k: v for k, v in record.items() if v is not None})
            assert barge_ms < 600
        finally:
            await engine.stop()

    asyncio.run(main())


def test_voice_cancels_a_running_command(room, tmp_path):
    """Claude is running a 30-second test script; saying "cancel that" aborts the turn and kills it."""
    spk, rm = room
    (tmp_path / "run_tests.sh").write_text("#!/bin/sh\necho running\nsleep 30\necho all passed\n")
    (tmp_path / "run_tests.sh").chmod(0o755)
    (tmp_path / "CLAUDE.md").write_text("To run the tests, run ./run_tests.sh in the foreground (never in the background). It takes 30 seconds.\n")
    cfg = Config()
    cfg.audio.input_device, cfg.audio.output_device = f"{rm}.monitor", spk
    cfg.claude.model = "haiku"
    cfg.tts.volume = 0.32
    user = KokoroTTS(TtsConfig(voice="am_michael"))
    ui = ConsoleUI()
    w = Watch(ui)
    script = str(tmp_path / "run_tests.sh")

    def running() -> bool:
        """The script is alive somewhere under this test process (never matches other shells)."""
        from claude_voice.claude.session import _proc_table
        table = _proc_table()
        kids: dict[int, list[int]] = {}
        for pid, (ppid, _) in table.items():
            kids.setdefault(ppid, []).append(pid)
        stack, found = [os.getpid()], False
        while stack and not found:
            for c in kids.get(stack.pop(), []):
                stack.append(c)
                try:
                    found = found or script.encode() in open(f"/proc/{c}/cmdline", "rb").read() \
                        or b"/run_tests.sh" in open(f"/proc/{c}/cmdline", "rb").read() \
                        or b"./run_tests.sh" in open(f"/proc/{c}/cmdline", "rb").read()
                except OSError:
                    pass
        return found

    async def main():
        engine = Engine(cfg, str(tmp_path), ui, new=True)
        await engine.start()
        try:
            await w.until(lambda k, d: k == "state" and d["state"] == "LISTENING", 90, "LISTENING")
            await asyncio.sleep(1.5)
            play_async(rm, user.synth("Please run the tests."), user.rate)
            await w.until(lambda k, d: k == "tool" and "run_tests" in d["summary"], 60, "Claude running the tests")
            for _ in range(100):
                if running():
                    break
                await asyncio.sleep(0.1)
            assert running(), "test script did not start"
            await asyncio.sleep(3.0)  # let any spoken preface finish; Claude is now silently running the script
            n = len(ui.events)
            play_async(rm, user.synth("Cancel that."), user.rate)
            await w.until(lambda k, d: k == "command" and d["command"] == "interrupt", 15, "interrupt command", n)
            _, _, d = await w.until(lambda k, d: k == "claude_done", 15, "turn aborted", n)
            print(f"\nturn ended: {d['reason']}")
            assert d["reason"] in ("aborted_tools", "aborted_streaming")
            await asyncio.sleep(1.0)
            assert not running(), "the test script survived the interruption"
            assert engine.sm.state.value == "LISTENING"
        finally:
            await engine.stop()

    asyncio.run(main())


def test_spoken_commands(room, tmp_path):
    """stop speaking (Claude keeps going, silently) -> repeat that -> voice off (ignored speech) -> voice on."""
    spk, rm = room
    cfg = Config()
    cfg.audio.input_device, cfg.audio.output_device = f"{rm}.monitor", spk
    cfg.claude.model = "haiku"
    cfg.tts.volume = 0.32
    user = KokoroTTS(TtsConfig(voice="am_michael"))
    ui = ConsoleUI()
    w = Watch(ui)

    async def main():
        engine = Engine(cfg, str(tmp_path), ui, new=True)
        await engine.start()
        try:
            await w.until(lambda k, d: k == "state" and d["state"] == "LISTENING", 90, "LISTENING")
            await asyncio.sleep(1.5)
            play_async(rm, user.synth("Without using any tools, name six planets, one short sentence each."), user.rate)
            i, _, _ = await w.until(lambda k, d: k == "speaking", 60, "speech")
            await asyncio.sleep(1.0)
            n = len(ui.events)
            play_async(rm, user.synth("Stop speaking."), user.rate)
            await w.until(lambda k, d: k == "command" and d["command"] == "stop speaking", 10, "stop speaking", n)
            await w.until(lambda k, d: k == "state" and d["state"] == "LISTENING", 60, "back to listening", n)
            assert not [d for _, k, d in ui.events[n:] if k == "user" and "planet" in d["text"].lower()]
            await asyncio.sleep(0.8)
            assert not engine.player.busy, "speech kept playing after 'stop speaking'"

            n = len(ui.events)
            play_async(rm, user.synth("Repeat that."), user.rate)
            await w.until(lambda k, d: k == "command" and d["command"] == "repeat that", 10, "repeat", n)
            await w.until(lambda k, d: k == "speaking", 10, "repeated speech", n)
            await asyncio.sleep(1.0)
            await engine.escape()

            n = len(ui.events)
            play_async(rm, user.synth("Voice off."), user.rate)
            await w.until(lambda k, d: k == "state" and d["state"] == "PAUSED", 10, "PAUSED", n)
            play_async(rm, user.synth("What is two plus two?"), user.rate)
            await asyncio.sleep(5.0)
            assert engine.sm.state.value == "PAUSED"
            assert not [d for _, k, d in ui.events[n:] if k == "claude_delta"], "speech was sent to Claude while voice was off"
            play_async(rm, user.synth("Voice on."), user.rate)
            await w.until(lambda k, d: k == "state" and d["state"] == "LISTENING", 10, "LISTENING again", n)
            print("\nstop speaking / repeat that / voice off / voice on all behaved")
        finally:
            await engine.stop()

    asyncio.run(main())


def test_permission_prompt_answered_by_voice(room, tmp_path):
    """In 'default' permission mode, Claude asks before writing; saying "yes" lets it."""
    spk, rm = room
    cfg = Config()
    cfg.audio.input_device, cfg.audio.output_device = f"{rm}.monitor", spk
    cfg.claude.model = "haiku"
    cfg.claude.permission_mode = "default"
    cfg.tts.volume = 0.32
    user = KokoroTTS(TtsConfig(voice="am_michael"))
    ui = ConsoleUI()
    w = Watch(ui)

    async def main():
        engine = Engine(cfg, str(tmp_path), ui, new=True)
        await engine.start()
        try:
            await w.until(lambda k, d: k == "state" and d["state"] == "LISTENING", 90, "LISTENING")
            await asyncio.sleep(1.5)
            play_async(rm, user.synth("Create a file called hello.txt that contains the word papaya."), user.rate)
            _, _, d = await w.until(lambda k, d: k == "permission", 60, "permission prompt")
            print(f"\nprompt: {d['title']}")
            await w.until(lambda k, d: k == "speaking" and "allow" in d["text"].lower(), 15, "spoken prompt")
            await asyncio.sleep(1.0)
            n = len(ui.events)
            play_async(rm, user.synth("Yes."), user.rate)  # answering over the question
            try:
                await w.until(lambda k, d: k == "permission_done", 6, "answer accepted", n)
            except AssertionError:
                # a short word in double talk can be lost: the engine must re-ask, and a repeat must work
                await w.until(lambda k, d: k == "barge" and d["stage"] == "reprompt", 5, "re-prompt", n)
                await w.until(lambda k, d: k == "speaking" and "yes or no" in d["text"].lower(), 10, "re-prompt spoken", n)
                await asyncio.sleep(2.0)
                play_async(rm, user.synth("Yes."), user.rate)
                await w.until(lambda k, d: k == "permission_done", 15, "answer accepted after re-prompt", n)
                print("first answer was lost in double talk; re-prompt and repeat worked")
            await w.until(lambda k, d: k == "claude_done", 60, "turn end")
            assert (tmp_path / "hello.txt").exists() and "papaya" in (tmp_path / "hello.txt").read_text().lower()
            print("hello.txt written after a spoken yes")
        finally:
            await engine.stop()

    asyncio.run(main())
