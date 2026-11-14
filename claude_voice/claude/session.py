"""The Claude Code session, driven through the Claude Agent SDK.

The SDK runs the Claude Code CLI on your PATH in streaming-JSON mode. That is the same harness
as the interactive TUI: tools, CLAUDE.md, skills, plugins, hooks, MCP servers, permission
settings and your claude.ai login. The conversation is an ordinary Claude Code session, so
`claude --resume` / `claude -c` opens it later in the TUI.

Why not drive the open TUI itself: channels and hooks can push text into it, but they queue
until the current turn ends and cannot cancel it, so there would be no real barge-in.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    ToolUseBlock,
    UserMessage,
)

from claude_voice.config import ClaudeConfig

log = logging.getLogger(__name__)
SESSIONS_DIR = Path.home() / ".claude" / "sessions"
_SNAPSHOT_MARK = b"shell-snapshots/snapshot-"


def live_sessions() -> list[dict] | None:
    """Interactive Claude Code sessions running right now (None if the registry is unreadable).

    The registry (~/.claude/sessions/<pid>.json) is internal to Claude Code; any surprise
    means "unknown" and the caller forks to stay safe.
    """
    if not SESSIONS_DIR.is_dir():
        return None
    out = []
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            info = json.loads(f.read_text())
            pid = int(info["pid"])
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                if b"claude" not in fh.read():
                    continue  # pid reused by something else
        except (OSError, ValueError, KeyError, TypeError):
            continue
        out.append(info)
    return out


def _proc_table() -> dict[int, tuple[int, float]]:
    """pid -> (ppid, start time as epoch seconds)."""
    hz = os.sysconf("SC_CLK_TCK")
    with open("/proc/stat") as f:
        btime = next(int(line.split()[1]) for line in f if line.startswith("btime"))
    table = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            table[int(d)] = (int(fields[1]), btime + int(fields[19]) / hz)
        except (OSError, IndexError, ValueError):
            continue
    return table


def kill_tool_processes(since: float, root: int | None = None) -> list[int]:
    """SIGTERM Claude Code Bash-tool shells under `root` started after `since`, with their children.

    Only commands launched by the interrupted tool call match; background shells started
    earlier (dev servers etc.) are left alone.
    """
    table = _proc_table()
    kids: dict[int, list[int]] = {}
    for pid, (ppid, _) in table.items():
        kids.setdefault(ppid, []).append(pid)

    def subtree(pid: int) -> list[int]:
        out, stack = [], [pid]
        while stack:
            for c in kids.get(stack.pop(), []):
                out.append(c)
                stack.append(c)
        return out

    victims: list[int] = []
    for pid in subtree(root or os.getpid()):
        if table[pid][1] < since:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if _SNAPSHOT_MARK not in f.read():
                    continue
        except OSError:
            continue
        victims += [pid, *subtree(pid)]
    for pid in victims:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    return victims


def _sigkill_survivors(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


EventFn = Callable[[str, Any], None]
PermissionFn = Callable[[str, dict, Any], Awaitable[bool | str]]  # True, False, or a denial reason


class ClaudeSession:
    def __init__(self, cfg: ClaudeConfig, cwd: str, on_event: EventFn, ask_permission: PermissionFn,
                 resume: str | None = None, new: bool = False):
        self.cfg, self.cwd = cfg, cwd
        self.on_event, self.ask_permission = on_event, ask_permission
        self.resume_id, self.new = resume, new
        self.forked = False
        self.session_id: str | None = None
        self.model: str | None = None
        self.busy = False             # a turn is running
        self.streaming_text = False   # the current assistant block is text
        self.turn_started = 0.0
        self.first_token: float | None = None
        self.tools_stopped = False
        self._fg_bash: dict[str, float] = {}  # in-flight foreground Bash tool_use id -> start
        self._client: ClaudeSDKClient | None = None
        self._reader: asyncio.Task | None = None

    def _options(self) -> ClaudeAgentOptions:
        cont, resume, fork = False, self.resume_id, False
        live = live_sessions()
        if self.new:
            resume = None
        elif resume:
            fork = live is None or any(s.get("sessionId") == resume for s in live)
        else:
            cont = True
            fork = live is None or any(s.get("cwd") == self.cwd for s in live)
        self.forked = fork and (cont or bool(resume))
        prompt: dict = {"type": "preset", "preset": "claude_code"}  # None would *replace* Claude Code's prompt
        if self.cfg.voice_prompt:
            prompt["append"] = self.cfg.voice_prompt
        return ClaudeAgentOptions(
            cwd=self.cwd,
            cli_path=self.cfg.cli_path or shutil.which("claude"),
            setting_sources=["user", "project", "local"],
            system_prompt=prompt,
            include_partial_messages=True,
            continue_conversation=cont,
            resume=resume,
            fork_session=self.forked,
            model=self.cfg.model or None,
            effort=self.cfg.effort or None,
            permission_mode=self.cfg.permission_mode or None,
            can_use_tool=self._can_use_tool,
            stderr=lambda line: log.debug("claude: %s", line.rstrip()),
            client_app="claude-voice",
        )

    async def start(self) -> None:
        self._client = ClaudeSDKClient(options=self._options())
        await self._client.connect()

    async def restart(self) -> None:
        """Reconnect after the CLI died, continuing the same conversation."""
        await self.close()
        if self.session_id:
            self.resume_id, self.new = self.session_id, False
        await self.start()

    async def close(self) -> None:
        client, self._client = self._client, None
        if self._reader and not self._reader.done():
            self._reader.cancel()
        if client:
            try:
                await asyncio.wait_for(client.disconnect(), 10)
            except Exception as e:  # shutting down anyway
                log.debug("disconnect: %s", e)
        self.busy = False

    async def submit(self, text: str) -> None:
        if self._client is None:
            await self.start()
        if self.busy:
            await self.interrupt()
        self.busy, self.streaming_text, self.first_token = True, False, None
        self.turn_started = time.monotonic()
        self._fg_bash.clear()
        await self._client.query(text)
        self._reader = asyncio.create_task(self._read_turn())

    async def interrupt(self) -> bool:
        """Cancel the running turn. Returns False if nothing was running."""
        if not self.busy or self._client is None:
            return False
        self.tools_stopped = False
        await self._client.interrupt()
        # interrupt() ends the turn but leaves an already-running Bash command alive (measured:
        # still running 12 s later). Esc in the TUI stops it, so we do too.
        if self.cfg.kill_tools_on_interrupt and self._fg_bash:
            victims = kill_tool_processes(min(self._fg_bash.values()) - 1.0)
            self.tools_stopped = bool(victims)
            if victims:
                asyncio.get_running_loop().call_later(1.5, _sigkill_survivors, victims)
        if self._reader:
            try:
                await asyncio.wait_for(asyncio.shield(self._reader), 15)
            except TimeoutError:
                log.warning("turn did not finish after interrupt; restarting the CLI")
                await self.restart()
        return True

    async def _read_turn(self) -> None:
        try:
            async for msg in self._client.receive_response():
                self._dispatch(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("Claude stream failed")
            self.busy = False
            self.on_event("error", f"Claude connection lost: {e}")
            return

    def _dispatch(self, msg: Any) -> None:
        if isinstance(msg, StreamEvent):
            if msg.parent_tool_use_id:  # subagent output: shown by its tool call, never spoken
                return
            ev = msg.event
            kind = ev.get("type")
            if kind == "content_block_start":
                block = ev.get("content_block", {})
                self.streaming_text = block.get("type") == "text"
                if block.get("type") == "tool_use" and block.get("name") == "Bash":
                    # the CLI starts the command as soon as this block finishes streaming,
                    # ~0.5 s before the AssistantMessage arrives; record the earlier time
                    self._fg_bash[block.get("id", "")] = time.time()
            elif kind == "content_block_delta" and ev.get("delta", {}).get("type") == "text_delta":
                if self.first_token is None:
                    self.first_token = time.monotonic()
                self.on_event("text", ev["delta"]["text"])
            elif kind == "content_block_stop" and self.streaming_text:
                self.streaming_text = False
                self.on_event("text_end", None)
        elif isinstance(msg, AssistantMessage) and msg.parent_tool_use_id is None:
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    if block.name == "Bash":
                        if block.input.get("run_in_background"):
                            self._fg_bash.pop(block.id, None)  # meant to outlive the turn
                        else:
                            self._fg_bash.setdefault(block.id, time.time() - 2.0)
                    self.on_event("tool", {"name": block.name, "input": block.input})
        elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
            for block in msg.content:
                tool_use_id = getattr(block, "tool_use_id", None)
                if tool_use_id:
                    self._fg_bash.pop(tool_use_id, None)
        elif isinstance(msg, SystemMessage) and msg.subtype == "init":
            self.session_id = msg.data.get("session_id") or self.session_id
            self.model = msg.data.get("model")
            self.on_event("init", msg.data)
        elif isinstance(msg, ResultMessage):
            self.busy, self.streaming_text = False, False
            self.session_id = msg.session_id or self.session_id
            self._fg_bash.clear()
            self.on_event("turn_end", msg)
        elif type(msg).__name__ == "RateLimitEvent":
            self.on_event("rate_limit", getattr(msg, "rate_limit_info", None))

    async def _can_use_tool(self, tool: str, tool_input: dict, ctx: Any):
        answer = await self.ask_permission(tool, tool_input, ctx)
        if answer is True:
            return PermissionResultAllow()
        return PermissionResultDeny(message=answer if isinstance(answer, str) else "The user declined this action.")
