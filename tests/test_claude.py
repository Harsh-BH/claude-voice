"""The real Claude Code session through the Agent SDK (uses your subscription; Haiku to keep it cheap).

    VOICE_CLAUDE=1 uv run pytest -s tests/test_claude.py
"""
import asyncio
import re
import subprocess
import time

import pytest

from claude_voice.claude.session import ClaudeSession
from claude_voice.config import ClaudeConfig

pytestmark = pytest.mark.claude


class Recorder:
    def __init__(self):
        self.events: list[tuple[float, str, object]] = []
        self.done = asyncio.Event()

    def __call__(self, kind, data):
        self.events.append((time.monotonic(), kind, data))
        if kind == "turn_end":
            self.done.set()

    def text(self, since: int = 0) -> str:
        return "".join(d for _, k, d in self.events[since:] if k == "text")

    async def turn(self, session: ClaudeSession, prompt: str, timeout: float = 90) -> int:
        start = len(self.events)
        self.done.clear()
        await session.submit(prompt)
        await asyncio.wait_for(self.done.wait(), timeout)
        return start


async def deny(*_):
    return False


def session(tmp_path, rec, **kw) -> ClaudeSession:
    return ClaudeSession(ClaudeConfig(model="haiku"), str(tmp_path), rec, deny, new=kw.pop("new", True), **kw)


def test_streaming_and_context(tmp_path):
    async def main():
        rec = Recorder()
        s = session(tmp_path, rec)
        await s.start()
        try:
            await rec.turn(s, "Remember this codeword: PAPAYA. Reply with just: OK.")
            start = await rec.turn(s, "What was the codeword? Answer in one word.")
            assert "PAPAYA" in rec.text(start).upper()
            init = [d for _, k, d in rec.events if k == "init"][0]
            assert init["apiKeySource"] == "none"  # your claude.ai login, not an API key
            first_text = next(t for t, k, _ in rec.events[start:] if k == "text")
            end = next(t for t, k, _ in rec.events[start:] if k == "turn_end")
            assert first_text <= end
        finally:
            await s.close()
    asyncio.run(main())


def test_interrupt_mid_stream_keeps_context(tmp_path):
    async def main():
        rec = Recorder()
        s = session(tmp_path, rec)
        await s.start()
        try:
            rec.done.clear()
            await s.submit("Count from 1 to 120, one number per line, nothing else.")
            while len(rec.text()) < 60:
                await asyncio.sleep(0.02)
            t0 = time.monotonic()
            assert await s.interrupt()
            took = time.monotonic() - t0
            result = [d for _, k, d in rec.events if k == "turn_end"][-1]
            assert result.terminal_reason in ("aborted_streaming", "aborted_tools"), result.terminal_reason
            heard = [int(n) for n in re.findall(r"\d+", rec.text())]
            start = await rec.turn(s, "What was the last number you wrote before I stopped you? Just the number.")
            answer = int(re.findall(r"\d+", rec.text(start))[0])
            print(f"\ninterrupt() returned in {took * 1000:.0f} ms; streamed up to {heard[-1]}, Claude says {answer}")
            assert took < 2.0 and abs(answer - heard[-1]) <= 3
        finally:
            await s.close()
    asyncio.run(main())


def test_interrupt_stops_a_running_command(tmp_path):
    async def main():
        rec = Recorder()
        s = session(tmp_path, rec)
        await s.start()
        try:
            rec.done.clear()
            await s.submit('Run exactly this bash command in the foreground (not in the background) and nothing else: python3 -c "import time; time.sleep(47.731)"')
            for _ in range(600):
                if subprocess.run(["pgrep", "-f", "time.sleep.47.731"], capture_output=True).returncode == 0:
                    break
                await asyncio.sleep(0.1)
            else:
                pytest.skip(f"Claude did not run the command: {[d for _, k, d in rec.events if k == 'tool']}")
            assert await s.interrupt()
            await asyncio.sleep(0.5)
            alive = subprocess.run(["pgrep", "-f", "time.sleep.47.731"], capture_output=True).returncode == 0
            assert s.tools_stopped and not alive
        finally:
            subprocess.run(["pkill", "-f", "time.sleep.47.731"])
            await s.close()
    asyncio.run(main())


def test_resume_continues_the_same_conversation(tmp_path):
    async def main():
        rec = Recorder()
        a = session(tmp_path, rec)
        await a.start()
        await rec.turn(a, "Remember: the release name is MANGO-7. Reply with just: OK.")
        sid = a.session_id
        await a.close()
        rec2 = Recorder()
        b = session(tmp_path, rec2, new=False, resume=sid)
        await b.start()
        try:
            await rec2.turn(b, "What is the release name? Answer with just the name.")
            assert "MANGO-7" in rec2.text().upper()
        finally:
            await b.close()
    asyncio.run(main())
