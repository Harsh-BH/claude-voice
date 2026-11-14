from claude_voice.config import BargeInConfig
from claude_voice.interruption.controller import Command, InterruptionController, Verdict, is_echo, is_noise, parse_command


def test_commands():
    assert parse_command("Stop speaking.") is Command.STOP_SPEAKING
    assert parse_command("Hey Claude, interrupt!") is Command.INTERRUPT
    assert parse_command("Stop!") is Command.INTERRUPT
    assert parse_command("repeat that please") is Command.REPEAT
    assert parse_command("Voice off.") is Command.VOICE_OFF
    assert parse_command("voice on") is Command.VOICE_ON
    assert parse_command("रुको") is Command.INTERRUPT
    assert parse_command("बस करो") is Command.STOP_SPEAKING
    assert parse_command("VoiceOff.") is Command.VOICE_OFF          # as Whisper actually wrote it
    assert parse_command("Voice-on") is Command.VOICE_ON
    assert parse_command("Stop-speaking!") is Command.STOP_SPEAKING
    assert parse_command("stop using Redis and switch to Postgres") is None
    assert parse_command("yes") is None
    assert parse_command("yes", expecting_answer=True) is Command.YES
    assert parse_command("Okay.", expecting_answer=True) is Command.YES
    assert parse_command("No, don't.", expecting_answer=True) is None
    assert parse_command("nahi", expecting_answer=True) is Command.NO
    assert parse_command("yet.", expecting_answer=True) is Command.YES   # Whisper on a lone "yes" over the prompt
    assert parse_command("yet", expecting_answer=False) is None


def test_echo_and_noise():
    spoken = "I will put a Redis cache in front of the PostgreSQL queries."
    assert is_echo("a Redis cache in front of", spoken, 0.8)
    assert not is_echo("no, use memcached instead", spoken, 0.8)
    assert not is_echo("yes, run the tests", "Should I run the tests?", 0.8)  # 3 of 4 words: a real reply
    assert is_noise("Um.") and is_noise("") and not is_noise("um, use Redis")


def test_controller_verdicts_and_note():
    ctl = InterruptionController(BargeInConfig())
    assert ctl.note() == ""
    ctl.begin(barge=True, spoken="Then I will update the migration and run the tests.",
              heard="I found the bug. Then I will update the", onset=0.0, now=10.0)
    assert ctl.judge("update the migration and") == (Verdict.ECHO, None)
    assert ctl.judge("Hmm.") == (Verdict.NOISE, None)
    assert ctl.judge("stop speaking") == (Verdict.COMMAND, Command.STOP_SPEAKING)
    assert ctl.judge("no wait, don't touch the migration") == (Verdict.REAL, None)
    assert not ctl.timed_out(11.0) and ctl.timed_out(12.5)
    note = ctl.note(tools_stopped=True)
    assert "interrupted you while you were speaking" in note and "Then I will update the" in note
    assert note.endswith("The command you were running was stopped.]")
    ctl.begin(barge=False, spoken="", heard="", onset=0.0, now=0.0)
    assert ctl.note() == "[Voice note: I interrupted you while you were working.]"


def test_permission_answers():
    from claude_voice.engine import Engine
    answer = Engine._permission_answer
    assert answer("Yes.") is True and answer("yet.") is True and answer("yes, do it") is True
    assert answer("yes please go ahead") is True
    assert answer("No, don't.") == "The user said no: No, don't."
    assert answer("hmm") is None and answer("the") is None                  # unclear: ask again
    assert answer("yes but use the staging config") == "The user answered: yes but use the staging config"
    assert answer("write it to staging instead please") == "The user answered: write it to staging instead please"
    assert answer("hmm", typed=True) == "The user answered: hmm"            # typed text is never garbled
