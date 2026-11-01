"""Long gateway replies stay valid MarkdownV2 at every chunk boundary.

``send()`` used to format the reply and split the FORMATTED text with ``truncate_message()``, escaping only the
`` (i/n)`` marker. A split could still land between an escape's backslash and its character, or inside a
``*bold*`` span, a link or inline code; Telegram refused that chunk and it went out as plain text. The final
overflow edit split the RAW text at the limit and formatted each piece, so a piece with enough reserved characters
formatted past 4096 units and went out plain too (as did a reply that fits raw but not formatted).

Both now split the way the standalone sender does (``send_message_senders._mdv2_split``): raw text split, each
chunk formatted on its own, the raw budget shrinking until every formatted chunk fits. The fake bot here refuses
whatever a MarkdownV2 scanner rejects (the scanner of ``tests/tools/test_telegram_send_chunk_markers.py``) or
anything over the limit, the way Telegram does.
"""
import logging
import random
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import utf16_len
from plugins.platforms.telegram import adapter as adapter_mod
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.tools.test_telegram_send_chunk_markers import _mdv2_problems, _random_markdown

LIMIT = 4096


class _Bot:
    """Telegram's answer to each text: refused when the scanner finds a fault in MarkdownV2 text, or when any text
    is over the limit; otherwise accepted with a fresh message id."""

    def __init__(self):
        self.calls, self._next = [], 500

    def _judge(self, kind, kw):
        text, mode = kw["text"], kw.get("parse_mode")
        mode = None if mode is None else str(getattr(mode, "value", mode))
        problems = _mdv2_problems(text) if mode == "MarkdownV2" else []
        if utf16_len(text) > LIMIT:
            problems.append("message is too long")
        self.calls.append({"kind": kind, "mode": mode, "text": text, "refused": problems})
        if problems:
            if problems[-1] == "message is too long":
                raise Exception("Bad Request: message is too long")
            raise Exception("Bad Request: can't parse entities: " + problems[0])
        self._next += 1
        return MagicMock(message_id=self._next)

    async def send_message(self, **kw):
        return self._judge("send", kw)

    async def edit_message_text(self, **kw):
        return self._judge("edit", kw)

    def delivered(self):
        return [c for c in self.calls if not c["refused"]]


def _adapter(bot):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._rich_send_disabled = True
    adapter._bot = bot
    return adapter


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock())


def _assert_all_formatted(bot, caplog, *, at_least):
    delivered = bot.delivered()
    assert len(delivered) >= at_least
    assert [c for c in bot.calls if c["refused"]] == [], [c["refused"][:1] for c in bot.calls if c["refused"]]
    assert all(c["mode"] == "MarkdownV2" for c in delivered)
    assert "falling back to plain text" not in caplog.text
    assert "escaped, unformatted" not in caplog.text


def _old_send_split(adapter, content):
    """What the live ``send()`` did before this change (for the bites-on-old-code checks)."""
    import re
    chunks = adapter.truncate_message(adapter.format_message(content), LIMIT, len_fn=utf16_len)
    if len(chunks) > 1:
        chunks = [adapter_mod._separate_chunk_indicator_from_fence(re.sub(r" \((\d+)/(\d+)\)$", r" \\(\1/\2\\)", c))
                  for c in chunks]
    return chunks


# The first three are refused under the old formatted split (asserted in test_cases_bite_old_split); inline code
# is kept whole by truncate_message, so it is a validity case only.
_ESCAPE_PAIR = "a" * 4081 + "." + "b" * 100          # the cut lands between "\" and "."
_BOLD_SPAN = "x " * 3900 + "**" + "bold words " * 40 + "** tail"
_LINK = "a" * 4054 + " [link text](https://example.com/path) tail"
_INLINE_CODE = "x " * 2000 + "`" + "code words " * 30 + "` tail"
_FITS_RAW_NOT_FORMATTED = "Note. " * 680   # 4080 raw units, each "." escaped: formats past the limit
_CASES = {"escape_pair": _ESCAPE_PAIR, "bold_span": _BOLD_SPAN, "link": _LINK, "inline_code": _INLINE_CODE,
          "fits_raw_not_formatted": _FITS_RAW_NOT_FORMATTED}


def test_cases_bite_old_split():
    """The boundary cases are real faults for the old send(): some chunk is refused. The raw-fits case is one
    raw edit that formats past the limit (the final-edit fault)."""
    adapter = _adapter(_Bot())
    for name in ("escape_pair", "bold_span", "link"):
        old = _old_send_split(adapter, _CASES[name])
        assert [p for c in old for p in _mdv2_problems(c)], name
    assert utf16_len(_FITS_RAW_NOT_FORMATTED) <= LIMIT < utf16_len(adapter.format_message(_FITS_RAW_NOT_FORMATTED))


@pytest.mark.asyncio
async def test_short_reply_is_one_unchanged_message(caplog):
    bot = _Bot()
    adapter = _adapter(bot)
    msg = "Short **bold** reply with a [link](https://example.com) and `code`."
    with caplog.at_level(logging.WARNING):
        result = await adapter.send("4242", msg)
    assert result.success
    assert [c["text"] for c in bot.calls] == [adapter.format_message(msg)]
    _assert_all_formatted(bot, caplog, at_least=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(_CASES))
async def test_long_send_is_valid_markdownv2_in_every_chunk(caplog, name):
    bot = _Bot()
    adapter = _adapter(bot)
    with caplog.at_level(logging.WARNING):
        result = await adapter.send("4242", _CASES[name])
    assert result.success
    _assert_all_formatted(bot, caplog, at_least=2)
    texts = [c["text"] for c in bot.delivered()]
    n = len(texts)
    assert all(t.endswith(f"\\({i}/{n}\\)") for i, t in enumerate(texts, 1))
    assert result.raw_response["message_ids"] == [str(501 + i) for i in range(n)]


@pytest.mark.asyncio
async def test_bold_span_across_the_split_stays_bold_on_both_sides(caplog):
    bot = _Bot()
    with caplog.at_level(logging.WARNING):
        await _adapter(bot).send("4242", _BOLD_SPAN)
    texts = [c["text"] for c in bot.delivered()]
    seams = [i for i in range(len(texts) - 1)
             if texts[i].rstrip().rsplit(" ", 1)[0].endswith("*") and texts[i + 1].startswith("*")]
    assert seams, "the bold span the split cut is closed and reopened"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["send", "final_edit"])
async def test_random_mixed_markdown_never_breaks_on_a_split(caplog, path):
    """Mixed messages of 4.5-16k characters (the standalone sender's generator): whenever the formatter's output
    for the whole message parses, every chunk parses too (the split introduces nothing), fits, and goes out
    formatted, through send() and through the final overflow edit."""
    rng, checked, old_faulty = random.Random(1776), 0, 0
    fmt = TelegramAdapter.__new__(TelegramAdapter).format_message
    for _ in range(40):
        msg = _random_markdown(rng)
        if _mdv2_problems(fmt(msg)):
            continue  # a formatter fault on the whole text is not the splitter's (short messages share it)
        bot = _Bot()
        adapter = _adapter(bot)
        old_faulty += bool([p for c in _old_send_split(adapter, msg) for p in _mdv2_problems(c)])
        with caplog.at_level(logging.WARNING):
            if path == "send":
                result = await adapter.send("4242", msg)
            else:
                result = await adapter.edit_message("4242", "400", msg, finalize=True)
        assert result.success
        _assert_all_formatted(bot, caplog, at_least=2)
        caplog.clear()
        checked += 1
    assert checked >= 30
    assert old_faulty >= 1  # the generator reaches the old formatted split's faults


@pytest.mark.asyncio
async def test_unfittable_formatting_goes_escaped_never_split(caplog, monkeypatch):
    """A formatter that never fits (each chunk inflates past the limit): the text goes escaped and unformatted,
    every chunk valid and within the limit, with the warning."""
    bot = _Bot()
    adapter = _adapter(bot)
    monkeypatch.setattr(adapter, "format_message", lambda text: "x" * (LIMIT + 1))
    msg = "words and. dots- " * 600
    with caplog.at_level(logging.WARNING):
        result = await adapter.send("4242", msg)
    assert result.success
    assert [c for c in bot.calls if c["refused"]] == []
    assert all(c["mode"] == "MarkdownV2" for c in bot.calls)
    assert "escaped, unformatted" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(_CASES))
async def test_final_overflow_edit_keeps_every_chunk_formatted(caplog, name):
    """The stream consumer's final edit of an over-limit reply (or one that fits raw but formats past the limit):
    the preview is edited with chunk 1 and the rest sent as continuations, every one accepted as MarkdownV2."""
    bot = _Bot()
    adapter = _adapter(bot)
    msg = _CASES[name]
    with caplog.at_level(logging.WARNING):
        result = await adapter.edit_message("4242", "400", msg, finalize=True)
    assert result.success
    _assert_all_formatted(bot, caplog, at_least=2)
    assert bot.calls[0]["kind"] == "edit"
    assert result.message_id == str(bot._next)
    assert len(result.continuation_message_ids) == len(bot.delivered()) - 1


@pytest.mark.asyncio
async def test_final_edit_that_fits_formatted_is_one_edit(caplog):
    bot = _Bot()
    adapter = _adapter(bot)
    msg = "A reply **that fits** with some. punctuation-"
    with caplog.at_level(logging.WARNING):
        result = await adapter.edit_message("4242", "400", msg, finalize=True)
    assert result.success and result.message_id == "400"
    assert [(c["kind"], c["text"]) for c in bot.calls] == [("edit", adapter.format_message(msg))]


@pytest.mark.asyncio
async def test_streaming_overflow_edit_is_unchanged():
    """Mid-stream (finalize=False) previews stay raw and truncated, as before."""
    bot = _Bot()
    adapter = _adapter(bot)
    msg = "word " * 1200
    result = await adapter.edit_message("4242", "400", msg, finalize=False)
    assert result.success
    assert [c["kind"] for c in bot.calls] == ["edit"]
    assert bot.calls[0]["mode"] is None
    assert bot.calls[0]["text"] == adapter.truncate_message(msg, LIMIT, len_fn=utf16_len)[0]


@pytest.mark.asyncio
async def test_continuation_refused_as_markdown_is_logged_then_sent_plain(caplog, monkeypatch):
    """A continuation Telegram still refuses as MarkdownV2 falls back to plain text, and now says so."""
    bot = _Bot()
    adapter = _adapter(bot)
    real = bot.send_message

    async def refuse_markdown(**kw):
        if kw.get("parse_mode") is not None:
            bot.calls.append({"kind": "send", "mode": "MarkdownV2", "text": kw["text"], "refused": ["forced"]})
            raise Exception("Bad Request: can't parse entities: forced")
        return await real(**kw)

    monkeypatch.setattr(bot, "send_message", refuse_markdown)
    with caplog.at_level(logging.WARNING):
        result = await adapter.edit_message("4242", "400", _ESCAPE_PAIR, finalize=True)
    assert result.success
    assert "Overflow continuation: MarkdownV2 send failed, falling back to plain text" in caplog.text
