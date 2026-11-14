"""Turn Claude's streamed markdown into short, speakable sentences.

Deltas split anywhere, so text is handled per line. Complete lines are classified (code fence,
table, heading, bullet...) and cleaned. Inside the still-open last line, text is released only up
to a sentence boundary that is not inside inline code or a link, so speech can start before the
line (often a whole paragraph) is finished.
"""
from __future__ import annotations

import re

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+")
_QUOTE = re.compile(r"^\s*>+\s?")
_RULE = re.compile(r"^\s*(?:[-*_=]\s*){3,}$")
_TABLE = re.compile(r"^\s*\|")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"\bhttps?://([^/\s)>\]]+)(?:[^\s)>\]]*[^\s)>\].,;:!?])?")
_CODE = re.compile(r"`+([^`]*)`+")
_PATH = re.compile(r"(?<![\w/.])(?:~/|/|\.{1,2}/)?(?:[\w.@+-]+/){2,}[\w.@+-]*")
_EMPH = re.compile(r"\*\*|__|~~|(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)")
_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_SNAKE = re.compile(r"(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])")
_SYMBOLS = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")
_WS = re.compile(r"\s+")
_BOUNDARY = re.compile(r"[.!?…।][\"')\]]*(?=\s)")
_SOFT = re.compile(r"[,;:—–](?=\s)")
_SPEAKABLE = re.compile(r"[A-Za-z0-9ऀ-ॿ]")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_ABBREV = {"e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "inc", "approx", "fig", "no", "st", "eg", "ie"}

MAX_CHUNK = 280


def clean_inline(text: str) -> str:
    """Strip markdown syntax that should never be read aloud, keeping the words."""
    text = _LINK.sub(r"\1", text)
    text = _URL.sub(lambda m: m.group(1), text)
    text = _CODE.sub(r"\1", text)
    text = _PATH.sub(lambda m: m.group(0).rstrip("/").rsplit("/", 1)[-1], text)
    text = _TAG.sub("", text)
    text = _EMPH.sub("", text)
    text = _SNAKE.sub(" ", text)
    text = _SYMBOLS.sub("", text)
    return _WS.sub(" ", text).strip()


def is_hindi(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))


def _is_abbrev(text: str, period: int) -> bool:
    k = period
    while k > 0 and not text[k - 1].isspace():
        k -= 1
    word = text[k:period].lower().lstrip("([\"'")
    return word in _ABBREV or (len(word) == 1 and word.isalpha())


def split_sentences(raw: str) -> list[str]:
    """Split markdown text at sentence ends that are not inside inline code or a link."""
    pieces, start = [], 0
    for m in _BOUNDARY.finditer(raw):
        if _is_abbrev(raw, m.start()) or not _safe(raw[: m.end()]):
            continue
        pieces.append(raw[start:m.end()].strip())
        start = m.end()
    pieces.append(raw[start:].strip())
    return [p for p in pieces if p]


def _chunks(text: str) -> list[str]:
    """Cap synthesis units at MAX_CHUNK characters, preferring clause boundaries."""
    out = []
    while len(text) > MAX_CHUNK:
        cut = max((m.end() for m in _SOFT.finditer(text, 0, MAX_CHUNK)), default=-1)
        if cut < MAX_CHUNK // 3:
            cut = text.rfind(" ", 0, MAX_CHUNK)
        if cut <= 0:
            cut = MAX_CHUNK
        out.append(text[:cut].strip())
        text = text[cut:].strip()
    out.append(text)
    return [s for s in out if _SPEAKABLE.search(s)]


def _classify(line: str) -> str:
    if _FENCE.match(line):
        return "fence"
    if _TABLE.match(line):
        return "table"
    if _RULE.match(line):
        return "rule"
    if _HEADING.match(line):
        return "heading"
    if _BULLET.match(line):
        return "bullet"
    if _QUOTE.match(line):
        return "quote"
    return "text"


def _strip_prefix(line: str, kind: str) -> str:
    rx = {"heading": _HEADING, "bullet": _BULLET, "quote": _QUOTE}.get(kind)
    return rx.sub("", line, count=1) if rx else line


def _safe(prefix: str) -> bool:
    """A cut here would not split inline code or a markdown link."""
    if prefix.count("`") % 2 or prefix.count("[") > prefix.count("]"):
        return False
    i = prefix.rfind("](")
    return i == -1 or prefix.find(")", i) != -1


class SpeechText:
    """Incremental markdown -> sentences for one streamed reply."""

    def __init__(self, first_chunk_chars: int = 24):
        self.first_chunk_chars = first_chunk_chars  # 0 disables the early comma split
        self._line = ""     # raw text of the open (incomplete) line
        self._done = 0      # chars of the open line already released
        self._in_fence = False
        self._ready: list[str] = []
        self._emitted = False

    def feed(self, delta: str) -> list[str]:
        self._line += delta
        while "\n" in self._line:
            line, self._line = self._line.split("\n", 1)
            self._close_line(line)
        self._release_open_line()
        return self._take()

    def flush(self) -> list[str]:
        if self._line:
            line, self._line = self._line, ""
            self._close_line(line)
        self._in_fence = False
        return self._take()

    def _take(self) -> list[str]:
        out, self._ready = self._ready, []
        return out

    def _emit(self, raw: str, terminate: bool = False) -> None:
        pieces = split_sentences(raw)
        for i, piece in enumerate(pieces):
            text = clean_inline(piece)
            if terminate and i == len(pieces) - 1 and text and text[-1] not in ".!?…:।":
                text += "."
            for s in _chunks(text):
                self._ready.append(s)
                self._emitted = True

    def _close_line(self, line: str) -> None:
        done, self._done = self._done, 0
        kind = _classify(line)
        if kind == "fence":
            self._in_fence = not self._in_fence
            return
        if self._in_fence or kind in ("table", "rule"):
            return
        rest = line[done:] if done else _strip_prefix(line, kind)
        self._emit(rest, terminate=kind in ("heading", "bullet"))

    def _release_open_line(self) -> None:
        line = self._line
        if self._in_fence or len(line.strip()) < 4:
            return  # too short to tell a fence/table/rule from text
        kind = _classify(line)
        if kind in ("fence", "table", "rule"):
            return
        start = self._done or len(line) - len(_strip_prefix(line, kind))
        segment = line[start:]
        cut = None
        for m in _BOUNDARY.finditer(segment):
            if not _is_abbrev(segment, m.start()) and _safe(line[: start + m.end()]):
                cut = m.end()
        if cut is None and not self._emitted and self.first_chunk_chars:
            for m in _SOFT.finditer(segment):
                if m.end() >= self.first_chunk_chars and _safe(line[: start + m.end()]):
                    cut = m.end()
                    break
        if cut is not None:
            self._emit(segment[:cut])
            self._done = start + cut
