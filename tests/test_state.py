import threading
from collections import deque

import pytest

from claude_voice.state.machine import TRANSITIONS, IllegalTransition, StateMachine
from claude_voice.state.machine import State as S


def machine(at: S) -> StateMachine:
    sm = StateMachine()
    sm.state = at
    return sm


def test_full_turn():
    sm = StateMachine()
    path = [S.LISTENING, S.USER_SPEAKING, S.TRANSCRIBING, S.CLAUDE_THINKING, S.CLAUDE_STREAMING,
            S.TTS_GENERATING, S.SPEAKING, S.CLAUDE_THINKING, S.SPEAKING, S.LISTENING]
    for s in path:
        assert sm.to(s, "test")
    assert [h[2] for h in sm.history] == path


def test_barge_in_confirmed_and_false():
    sm = machine(S.SPEAKING)
    for s in (S.INTERRUPTING, S.USER_SPEAKING, S.TRANSCRIBING, S.CLAUDE_THINKING):
        sm.to(s)
    sm = machine(S.SPEAKING)
    sm.to(S.INTERRUPTING)
    sm.to(S.SPEAKING)  # false barge-in: playback resumes


@pytest.mark.parametrize("a,b", [
    (S.IDLE, S.SPEAKING), (S.IDLE, S.USER_SPEAKING), (S.PAUSED, S.SPEAKING),
    (S.LISTENING, S.INTERRUPTING), (S.ERROR, S.SPEAKING), (S.PAUSED, S.CLAUDE_THINKING),
])
def test_illegal(a, b):
    with pytest.raises(IllegalTransition):
        machine(a).to(b)


def test_exits_always_available():
    for s in S:
        if s not in (S.IDLE, S.ERROR):
            assert {S.IDLE, S.ERROR} <= TRANSITIONS[s], s


def test_every_state_reachable_from_idle():
    seen, queue = {S.IDLE}, deque([S.IDLE])
    while queue:
        for nxt in TRANSITIONS[queue.popleft()]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    assert seen == set(S)


def test_noop_and_thread_guard():
    sm = StateMachine()
    sm.to(S.LISTENING)
    assert sm.to(S.LISTENING) is False
    errors = []
    t = threading.Thread(target=lambda: errors.append(pytest.raises(RuntimeError, sm.to, S.IDLE)))
    t.start()
    t.join()
    assert errors and sm.state == S.LISTENING
