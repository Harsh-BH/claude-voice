"""Headless mode: events as timestamped lines on stdout, typed prompts from stdin.

Used by `voice start --headless`, the end-to-end tests, and anyone who prefers a plain log.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import stat
import sys
import time

from claude_voice.config import Config
from claude_voice.engine import Engine

_SKIP = {"level", "claude_delta", "synth"}


class ConsoleUI:
    def __init__(self, stream=sys.stdout, show_deltas: bool = False):
        self.stream, self.show_deltas = stream, show_deltas
        self.events: list[tuple[float, str, dict]] = []
        self.t0 = time.monotonic()

    def __call__(self, kind: str, data: dict) -> None:
        now = time.monotonic()
        self.events.append((now, kind, data))
        if kind in _SKIP and not (kind == "claude_delta" and self.show_deltas):
            return
        print(f"{now - self.t0:8.3f} {kind:14s} {json.dumps(data, ensure_ascii=False, default=str)}", file=self.stream, flush=True)


async def run(cfg: Config, cwd: str, resume: str | None, new: bool) -> None:
    ui = ConsoleUI()
    engine = Engine(cfg, cwd, ui, resume=resume, new=new)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await engine.start()
    reader = None
    try:
        mode = os.fstat(0).st_mode
        # epoll only takes ttys, pipes and sockets; /dev/null or a file would fail later, in a callback
        if os.isatty(0) or stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode):
            reader = asyncio.StreamReader()
            await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    except (OSError, ValueError):  # stdin closed: no typed input
        reader = None

    async def read_lines():
        while line := await reader.readline():
            await engine.submit_text(line.decode())

    if reader:
        loop.create_task(read_lines())
    await stop.wait()
    await engine.stop()
