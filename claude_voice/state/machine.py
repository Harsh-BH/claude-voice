"""Explicit interaction states and the only transitions allowed between them.

All transitions happen on the asyncio event-loop thread. Audio, STT and TTS threads never touch
the machine; they post events to the loop, which decides the next state. The one fast path that
bypasses the loop (pausing playback on barge-in) acts on the player, not on the state.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum


class State(StrEnum):
    IDLE = "IDLE"                          # voice mode off (F8): no listening, no speaking
    PAUSED = "PAUSED"                      # "voice off": only listening for "voice on"
    LISTENING = "LISTENING"                # waiting for speech, Claude idle
    USER_SPEAKING = "USER_SPEAKING"
    TRANSCRIBING = "TRANSCRIBING"
    CLAUDE_THINKING = "CLAUDE_THINKING"    # turn running, nothing to say yet (thinking or tools)
    CLAUDE_STREAMING = "CLAUDE_STREAMING"  # text arriving, no complete sentence yet
    TTS_GENERATING = "TTS_GENERATING"      # synthesizing, nothing audible yet
    SPEAKING = "SPEAKING"
    INTERRUPTING = "INTERRUPTING"          # barge-in: playback paused while we decide if it is real
    ERROR = "ERROR"


S = State
_ANY_TURN = {S.CLAUDE_THINKING, S.CLAUDE_STREAMING, S.TTS_GENERATING, S.SPEAKING}
_EXITS = {S.IDLE, S.PAUSED, S.ERROR}  # always reachable (F8, "voice off", component failure)

TRANSITIONS: dict[State, frozenset[State]] = {
    S.IDLE: frozenset({S.LISTENING, S.ERROR}),
    S.PAUSED: frozenset({S.LISTENING, S.IDLE, S.ERROR}),
    S.ERROR: frozenset({S.LISTENING, S.IDLE, S.PAUSED, S.CLAUDE_THINKING}),
    S.LISTENING: frozenset({S.USER_SPEAKING, S.CLAUDE_THINKING, S.TTS_GENERATING, S.SPEAKING} | _EXITS),
    # full duplex: the user may start or stop talking while any part of a turn is happening
    S.USER_SPEAKING: frozenset(_ANY_TURN | {S.TRANSCRIBING, S.LISTENING, S.INTERRUPTING} | _EXITS),
    S.TRANSCRIBING: frozenset(_ANY_TURN | {S.LISTENING, S.USER_SPEAKING, S.INTERRUPTING} | _EXITS),
    S.CLAUDE_THINKING: frozenset((_ANY_TURN - {S.CLAUDE_THINKING}) | {S.LISTENING, S.INTERRUPTING} | _EXITS),
    S.CLAUDE_STREAMING: frozenset((_ANY_TURN - {S.CLAUDE_STREAMING}) | {S.LISTENING, S.INTERRUPTING} | _EXITS),
    S.TTS_GENERATING: frozenset((_ANY_TURN - {S.TTS_GENERATING}) | {S.LISTENING, S.INTERRUPTING} | _EXITS),
    S.SPEAKING: frozenset((_ANY_TURN - {S.SPEAKING}) | {S.LISTENING, S.INTERRUPTING} | _EXITS),
    S.INTERRUPTING: frozenset(_ANY_TURN | {S.USER_SPEAKING, S.TRANSCRIBING, S.LISTENING} | _EXITS),
}


class IllegalTransition(RuntimeError):
    pass


class StateMachine:
    def __init__(self, on_change: Callable[[State, State, str], None] | None = None):
        self.state = S.IDLE
        self.since = time.monotonic()
        self.history: deque[tuple[float, State, State, str]] = deque(maxlen=500)
        self._on_change = on_change
        self._owner: int | None = None

    def to(self, new: State, reason: str = "") -> bool:
        """Move to `new`; returns False for a no-op, raises IllegalTransition for a forbidden move."""
        owner = threading.get_ident()
        if self._owner is None:
            self._owner = owner
        elif owner != self._owner:
            raise RuntimeError("state transitions must happen on the event-loop thread")
        if new == self.state:
            return False
        if new not in TRANSITIONS[self.state]:
            raise IllegalTransition(f"{self.state.value} -> {new.value} ({reason})")
        old, self.state, self.since = self.state, new, time.monotonic()
        self.history.append((self.since, old, new, reason))
        if self._on_change:
            self._on_change(old, new, reason)
        return True

    def can(self, new: State) -> bool:
        return new == self.state or new in TRANSITIONS[self.state]
