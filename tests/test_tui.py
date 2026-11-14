"""The Textual UI against a fake engine: keys reach the engine, engine events render."""
import asyncio

from textual.widgets import Input, RichLog, Static

from claude_voice.config import Config
from claude_voice.ui import tui


class FakeEngine:
    def __init__(self, cfg, cwd, ui, resume=None, new=False):
        self.ui, self.calls = ui, []

    async def start(self):
        self.ui("state", {"state": "LISTENING", "from": "IDLE", "reason": "voice on"})
        self.ui("mic", {"status": "ok", "detail": "claude_voice_aec_source"})

    async def stop(self):
        self.calls.append("stop")

    async def toggle_voice(self):
        self.calls.append("voice")

    def toggle_mute(self):
        self.calls.append("mute")
        self.ui("muted", {"muted": True})

    async def escape(self):
        self.calls.append("escape")

    async def submit_text(self, text):
        self.calls.append(("text", text))


def test_keys_and_rendering(monkeypatch):
    monkeypatch.setattr(tui, "Engine", FakeEngine)

    async def main():
        app = tui.VoiceApp(Config(), "/tmp/project", None, False)
        async with app.run_test() as pilot:
            await pilot.pause()
            engine = app.engine
            for key in ("f8", "f9", "escape"):
                await pilot.press(key)
            app.query_one(Input).value = "run the tests"
            await pilot.press("enter")
            await pilot.pause()
            assert engine.calls[:4] == ["voice", "mute", "escape", ("text", "run the tests")]
            status = str(app.query_one("#status", Static).render())
            assert "LISTENING" in status and "muted" in status and "live" in status
            engine.ui("user", {"text": "add a Redis cache", "language": "en"})
            engine.ui("claude_delta", {"text": "Sure, adding it now."})
            assert "adding it" in str(app.query_one("#live", Static).render())
            engine.ui("tool", {"name": "Edit", "summary": "cache.py"})
            engine.ui("barge", {"stage": "paused", "detect_ms": 180, "stop_ms": 9.5})
            engine.ui("permission", {"title": "Claude wants to run rm -rf build", "detail": "", "tool": "Bash"})
            assert "allow?" in str(app.query_one("#live", Static).render())
            await pilot.pause()
            log_text = "\n".join(str(line.text) for line in app.query_one("#log", RichLog).lines)
            assert "add a Redis cache" in log_text and "Edit" in log_text and "barge-in detected 180 ms" in log_text
            await pilot.press("ctrl+c")
        assert "stop" in engine.calls

    asyncio.run(main())


def test_no_attribute_shadows_textual_internals():
    """VoiceApp once set `self._closing`, Textual's own MessagePump flag, and Ctrl+C stopped exiting."""
    import ast
    import inspect

    from textual.app import App
    tree = ast.parse(inspect.getsource(tui.VoiceApp))
    assigned = {t.attr for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"}
    textual = set(vars(App())) | {name for cls in App.__mro__ for name in vars(cls)}
    assert assigned and not (assigned & textual), assigned & textual
