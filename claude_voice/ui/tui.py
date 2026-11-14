"""Textual terminal UI: conversation log, live line, status bar, and a text input to the same session."""
from __future__ import annotations

import asyncio
import logging
import math
import signal

from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Input, RichLog, Static

from claude_voice.config import Config
from claude_voice.engine import Engine

log = logging.getLogger(__name__)
_BARS = " ▁▂▃▄▅▆▇█"


def meter(rms: float) -> str:
    db = 20 * math.log10(max(rms, 1e-6))
    level = int(max(0.0, min(1.0, (db + 60) / 50)) * (len(_BARS) - 1))
    return (_BARS[level] * 6).ljust(6)


class VoiceApp(App):
    TITLE = "Claude Voice"
    AUTO_FOCUS = "#prompt"  # typing goes straight to Claude; F8/F9/Esc/Ctrl+C are priority bindings
    CSS = """
    #log { height: 1fr; border: round $primary-darken-2; padding: 0 1; }
    #live { height: auto; max-height: 7; padding: 0 1; }
    #status { height: 1; background: $boost; padding: 0 1; }
    """
    BINDINGS = [
        Binding("f8", "toggle_voice", "Voice on/off", priority=True),
        Binding("f9", "toggle_mute", "Mute speech", priority=True),
        Binding("escape", "escape", "Stop / cancel", priority=True),
        Binding("ctrl+c", "shutdown", "Quit", priority=True),
    ]

    def __init__(self, cfg: Config, cwd: str, resume: str | None, new: bool):
        super().__init__()
        self.engine = Engine(cfg, cwd, self.on_engine, resume=resume, new=new)
        self.cwd = cwd
        self._claude = ""
        self._partial = ""
        self._prompt = ""
        self._status = {"state": "STARTING", "mic": "off", "level": "", "muted": False, "session": "", "model": "",
                        "speaking": False, "aec": "on" if cfg.aec.enabled else "off"}
        self._quitting = False

    def compose(self) -> ComposeResult:
        yield RichLog(id="log", wrap=True, markup=False, highlight=False)
        yield Static(id="live")
        yield Static(id="status")
        yield Input(placeholder="Type to Claude and press Enter (y/n answers a permission prompt)", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#log", RichLog).write(Text(f"Claude Voice — {self.cwd}", style="bold"))
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, lambda: asyncio.ensure_future(self.action_shutdown()))
        self._render_status()
        self.run_worker(self._start(), exclusive=True)

    async def _start(self) -> None:
        try:
            await self.engine.start()
        except Exception as e:
            log.exception("start failed")
            self._log(Text(f"✗ could not start voice mode: {e}", style="bold red"))
            self._log(Text("  run `voice doctor` for a diagnosis", style="red"))

    # ----- actions
    async def action_toggle_voice(self) -> None:
        await self.engine.toggle_voice()

    def action_toggle_mute(self) -> None:
        self.engine.toggle_mute()

    async def action_escape(self) -> None:
        await self.engine.escape()

    async def action_shutdown(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        self._log(Text("shutting down…", style="dim"))
        try:
            await asyncio.wait_for(self.engine.stop(), 15)
        except Exception as e:
            log.warning("shutdown: %s", e)
        self.exit()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value
        event.input.value = ""
        await self.engine.submit_text(text)

    # ----- engine events
    def _log(self, renderable) -> None:
        self.query_one("#log", RichLog).write(renderable)

    def on_engine(self, kind: str, d: dict) -> None:
        if kind == "state":
            self._status["state"] = d["state"]
        elif kind == "level":
            self._status["level"] = meter(d["rms"])
        elif kind == "mic":
            self._status["mic"] = d["status"]
            if d["status"] != "ok":
                self._log(Text(f"🎤 {d['detail']}", style="yellow"))
        elif kind == "partial":
            self._partial = d["text"]
        elif kind == "user":
            self._partial = ""
            tag = " (typed)" if d.get("typed") else (f" [{d['language']}]" if d.get("language") not in (None, "en") else "")
            self._log(Text.assemble(("You", "bold cyan"), (f"{tag} ▸ ", "cyan"), d["text"]))
        elif kind == "claude_delta":
            self._claude += d["text"]
        elif kind == "tool":
            self._flush_claude()
            self._log(Text(f"  ⚙ {d['name']}  {d['summary']}", style="dim yellow"))
        elif kind == "claude_done":
            self._flush_claude()
            if d["reason"] and d["reason"].startswith("aborted"):
                self._log(Text("  ⏹ turn cancelled", style="magenta"))
            elif d["error"]:
                self._log(Text(f"  ✗ turn ended with {d['subtype']}", style="red"))
        elif kind == "speaking":
            self._status["speaking"] = True
        elif kind == "barge":
            self._log(Text(self._barge_text(d), style="bold magenta"))
        elif kind == "command":
            self._log(Text(f"🗣 command: {d['command']}", style="magenta"))
        elif kind == "latency":
            parts = [f"{k.removesuffix('_ms').replace('_', ' ')} {v} ms" for k, v in d.items() if k.endswith("_ms") and v is not None]
            self._log(Text("  ⏱ " + " · ".join(parts), style="dim"))
        elif kind == "permission":
            self._prompt = f"⚠ {d['title']}  {d['detail']}\n   allow? type y/n or say yes/no"
        elif kind == "permission_done":
            self._prompt = ""
        elif kind == "session":
            if (d.get("id") or "")[:8] == self._status["session"]:
                return  # the CLI re-sends its init frame every turn
            self._status["session"] = (d.get("id") or "")[:8]
            self._status["model"] = d.get("model") or ""
            self._log(Text(f"session {d.get('id')}{' (forked: the TUI has it open)' if d.get('forked') else ''} · "
                           f"{d.get('model')} · Claude Code {d.get('version')} · auth {d.get('auth')} · "
                           f"{d.get('plugins')} plugins, {d.get('skills')} skills, {d.get('mcp')} MCP servers · "
                           f"permissions {d.get('permission_mode')}", style="dim"))
        elif kind == "muted":
            self._status["muted"] = d["muted"]
        elif kind == "voice":
            self._log(Text("🎤 voice " + ("on" if d["on"] else ("paused — say 'voice on'" if d.get("paused") else "off (F8)")), style="green"))
        elif kind in ("info", "warning", "error"):
            style = {"info": "dim", "warning": "yellow", "error": "bold red"}[kind]
            self._log(Text(("" if kind == "info" else "! ") + d["text"], style=style))
        self._render_status()
        self._render_live()

    @staticmethod
    def _barge_text(d: dict) -> str:
        stage = d["stage"]
        if stage == "paused":
            stop = f", playback stopped in {d['stop_ms']} ms" if d.get("stop_ms") is not None else ""
            return f"⚡ barge-in detected {d['detect_ms']} ms after onset{stop}"
        if stage == "confirmed":
            extra = " (running command stopped)" if d.get("tools_stopped") else ""
            return f"⚡ interruption confirmed — Claude {'interrupted' if d.get('claude_interrupted') else 'idle'}{extra}"
        if stage == "resumed":
            return f"↺ false barge-in ({d['why']}): speech resumed"
        if stage == "reprompt":
            return "↺ didn't catch the answer: asking again"
        return f"⏹ cancelled by {d.get('by', '?')}"

    def _flush_claude(self) -> None:
        if self._claude.strip():
            self._log(Text("Claude ▸", style="bold green"))
            self._log(Markdown(self._claude.strip()))
        self._claude = ""

    def _render_live(self) -> None:
        lines = []
        if self._prompt:
            lines.append(Text(self._prompt, style="bold yellow"))
        if self._partial:
            lines.append(Text(f"… {self._partial}", style="italic cyan"))
        if self._claude:
            tail = self._claude[-400:].replace("\n", " ")
            lines.append(Text.assemble(("Claude ▸ ", "green"), tail))
        self.query_one("#live", Static).update(Text("\n").join(lines) if lines else "")

    def _render_status(self) -> None:
        s = self._status
        mic = {"ok": "🎤 live", "off": "🎤 off", "error": "🎤 ERROR"}.get(s["mic"], s["mic"])
        speech = "🔇 muted" if s["muted"] else "🔊 on"
        text = f"{mic} {s['level']} │ {s['state']} │ {speech} │ AEC {s['aec']} │ {s['model']} {s['session']}"
        self.query_one("#status", Static).update(Text(text))
