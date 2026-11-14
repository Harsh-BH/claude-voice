from claude_voice.tts.text import SpeechText, clean_inline, split_sentences

REPLY = """I'll add the cache now.

## Plan
- Put **Redis** in front of `PostgreSQL`.
- Add a TTL of 60 seconds to `src/db/cache_layer.py`.

```python
cache = Redis(host="localhost")
print("this must never be spoken.")
```

| step | status |
|------|--------|
| one  | done   |

---
Done. Python 3.12 handles it, e.g. via asyncio. See https://redis.io/docs for details."""

EXPECTED = [
    "I'll add the cache now.", "Plan.", "Put Redis in front of PostgreSQL.",
    "Add a TTL of 60 seconds to cache layer.py.", "Done.", "Python 3.12 handles it, e.g. via asyncio.",
    "See redis.io for details.",
]


def stream(text: str, chunk: int, first: int = 0) -> list[str]:
    st = SpeechText(first_chunk_chars=first)
    out: list[str] = []
    for i in range(0, len(text), chunk):
        out += st.feed(text[i:i + chunk])
    return out + st.flush()


def test_clean_inline():
    assert clean_inline("Run `npm test` in **/home/me/proj/src/main.py**") == "Run npm test in main.py"
    assert clean_inline("See [the docs](https://x.example/a) or https://github.com/a/b.") == "See the docs or github.com."
    assert clean_inline("rename user_id to *account_id*") == "rename user id to account id"
    assert clean_inline("✅ Done 🎉") == "Done"
    assert clean_inline("TCP/IP and/or I/O") == "TCP/IP and/or I/O"


def test_split_sentences():
    assert split_sentences("Python 3.12 is out. It is fast, e.g. for loops. Done!") == [
        "Python 3.12 is out.", "It is fast, e.g. for loops.", "Done!"]
    st = SpeechText()
    chunks = st.feed("word " * 120) + st.flush()
    assert len(chunks) == 3 and all(len(s) <= 280 for s in chunks)


def test_markdown_reply_one_shot():
    assert stream(REPLY, len(REPLY)) == EXPECTED


def test_any_chunking_gives_the_same_sentences():
    for chunk in (1, 2, 3, 5, 7, 13, 64):
        assert stream(REPLY, chunk) == EXPECTED, chunk


def test_sentences_released_before_the_line_ends():
    st = SpeechText(first_chunk_chars=0)
    assert st.feed("First sentence here. Second one") == ["First sentence here."]
    assert st.feed(" continues") == []
    assert st.flush() == ["Second one continues"]


def test_first_chunk_splits_at_a_comma():
    st = SpeechText(first_chunk_chars=10)
    assert st.feed("Sure, I can refactor the parser module, and then") == ["Sure, I can refactor the parser module,"]
    assert st.feed(" run the tests, and report back. Next") == ["and then run the tests, and report back."]


def test_no_cut_inside_inline_code_or_links():
    st = SpeechText(first_chunk_chars=0)
    assert st.feed("Call `os.path.join(a, b). ") == []          # unclosed backtick: wait
    assert st.feed("` first. Then") == ["Call os.path.join(a, b). first."]
    st = SpeechText(first_chunk_chars=0)
    assert st.feed("Read [the guide. Really](http://x.io/a). Then") == ["Read the guide. Really."]


def test_hindi_danda():
    st = SpeechText(first_chunk_chars=0)
    assert st.feed("ठीक है। मैं टेस्ट चलाता हूँ।\n") == ["ठीक है।", "मैं टेस्ट चलाता हूँ।"]
