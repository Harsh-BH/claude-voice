"""Barge-in policy: is speech heard while Claude is busy a real interruption, residual echo,
noise, or a voice command?

Barge-in runs in two stages:
1. Acoustic (capture thread, no event loop involved): sustained speech above the stricter
   barge-in threshold pauses playback at once. That is the "stop within 100 ms" path.
2. Semantic (here): the first transcript of that speech is judged. REAL confirms the
   interruption (drop queued speech, interrupt Claude). ECHO/NOISE leaves it pending, and if
   nothing intelligible arrives within `confirm_timeout_s` the paused audio resumes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from claude_voice.config import BargeInConfig


class Command(Enum):
    STOP_SPEAKING = "stop speaking"
    INTERRUPT = "interrupt"
    REPEAT = "repeat that"
    VOICE_OFF = "voice off"
    VOICE_ON = "voice on"
    HINGLISH = "hinglish mode"
    ENGLISH = "english mode"
    YES = "yes"
    NO = "no"


_PHRASES: dict[Command, set[str]] = {
    Command.STOP_SPEAKING: {
        "stop speaking", "stop talking", "shut up", "be quiet", "quiet", "silence", "enough",
        "thats enough", "chup", "chup raho", "bas", "bas karo", "चुप", "चुप रहो", "बस", "बस करो",
    },
    Command.INTERRUPT: {
        "stop", "stop it", "stop that", "stop stop", "interrupt", "cancel", "cancel that", "abort",
        "wait", "wait wait", "hold on", "hang on", "never mind", "nevermind", "ruko", "ruk jao",
        "रुको", "रुको रुको", "रुक जाओ",
    },
    Command.REPEAT: {
        "repeat", "repeat that", "say that again", "say again", "come again", "pardon",
        "what did you say", "can you repeat that", "phir se bolo", "dobara bolo",
        "फिर से बोलो", "दोबारा बोलो",
    },
    Command.VOICE_OFF: {"voice off", "stop listening", "go to sleep"},
    Command.VOICE_ON: {"voice on", "start listening", "wake up", "resume listening"},
    Command.HINGLISH: {
        "hinglish mode", "hindi mode", "hinglish", "switch to hinglish", "hinglish mode on",
        "हिंग्लिश मोड", "हिंगलिश मोड", "हिंदी मोड", "हिंग्लिश",
    },
    Command.ENGLISH: {"english mode", "switch to english", "english mode on", "इंग्लिश मोड", "english only"},
    # answers are only parsed while a permission prompt is pending; the last line of each set is
    # how Whisper actually wrote a lone "yes"/"no" said over the prompt
    Command.YES: {
        "yes", "yeah", "yep", "yup", "sure", "allow", "allow it", "approve", "go ahead", "do it",
        "okay", "ok", "yes please", "haan", "han", "theek hai", "हाँ", "हां", "ठीक है",
        "yet", "yea", "ya", "yah", "yess", "yas",
    },
    Command.NO: {
        "no", "nope", "deny", "dont", "do not", "reject", "no thanks", "nahi", "nahin",
        "mat karo", "नहीं", "मत करो",
        "nah", "know", "noo",
    },
}
_COMPACT = {cmd: {p.replace(" ", "") for p in phrases} for cmd, phrases in _PHRASES.items()}
_LEAD = {"hey", "ok", "okay", "claude", "please", "so", "um", "uh", "oh"}
_TRAIL = {"please", "claude", "now", "thanks"}
_FILLERS = {"um", "umm", "uh", "uhh", "hmm", "hm", "mm", "mhm", "ah", "oh", "huh", "er", "erm"}
_NON_WORD = re.compile(r"[^\wऀ-ॿ ]+")


def words(text: str) -> list[str]:
    text = text.lower().replace("'", "").replace("’", "")
    return _NON_WORD.sub(" ", text).split()


def parse_command(text: str, expecting_answer: bool = False) -> Command | None:
    """A whole-utterance command; "stop using Redis" is an instruction, not a command."""
    w = words(text)
    core = list(w)
    while core and core[0] in _LEAD:
        core.pop(0)
    while core and core[-1] in _TRAIL:
        core.pop()
    # Whisper sometimes runs short commands together ("VoiceOff."), so also compare without spaces
    candidates = [" ".join(core), " ".join(w), "".join(core)]
    for candidate in candidates:
        for cmd, phrases in _PHRASES.items():
            if cmd in (Command.YES, Command.NO) and not expecting_answer:
                continue
            if candidate in phrases or candidate in _COMPACT[cmd]:
                return cmd
    return None


def is_noise(text: str) -> bool:
    return not [x for x in words(text) if x not in _FILLERS]


def is_echo(text: str, spoken: str, threshold: float) -> bool:
    """Most of the transcript's words were in what Claude was saying at that moment."""
    heard, said = words(text), set(words(spoken))
    if not heard or not said:
        return False
    return sum(x in said for x in heard) / len(heard) >= threshold


class Verdict(Enum):
    REAL = "real"
    ECHO = "echo"
    NOISE = "noise"
    COMMAND = "command"


@dataclass
class Pending:
    barge: bool     # Claude was audible (playback paused) rather than silently working
    spoken: str     # text playing when the user cut in, for the echo check
    heard: str      # what the user had heard of this reply
    onset: float
    started: float
    confirmed: bool = False  # acted on: speech dropped (and Claude interrupted unless muted)
    muted: bool = False      # "stop speaking": speech dropped, Claude left running


class InterruptionController:
    def __init__(self, cfg: BargeInConfig):
        self.cfg = cfg
        self.pending: Pending | None = None

    def begin(self, *, barge: bool, spoken: str, heard: str, onset: float, now: float) -> Pending:
        self.pending = Pending(barge, spoken, heard, onset, now)
        return self.pending

    def judge(self, text: str, expecting_answer: bool = False) -> tuple[Verdict, Command | None]:
        cmd = parse_command(text, expecting_answer)
        if cmd:
            return Verdict.COMMAND, cmd
        if is_noise(text):
            return Verdict.NOISE, None
        if self.pending and is_echo(text, self.pending.spoken, self.cfg.echo_similarity):
            return Verdict.ECHO, None
        return Verdict.REAL, None

    def timed_out(self, now: float) -> bool:
        p = self.pending
        return bool(p and not p.confirmed and now - p.started >= self.cfg.confirm_timeout_s)

    def note(self, tools_stopped: bool = False) -> str:
        """Context for Claude, prepended to the instruction that interrupted it."""
        p = self.pending
        if not p:
            return ""
        if p.barge and p.heard.strip():
            tail = " ".join(p.heard.split()[-25:])
            note = f'[Voice note: I interrupted you while you were speaking. The last thing I heard was: "…{tail}"'
        else:
            note = "[Voice note: I interrupted you while you were working"
        if tools_stopped:
            note += ". The command you were running was stopped"
        return note + ".]"

    def clear(self) -> None:
        self.pending = None
