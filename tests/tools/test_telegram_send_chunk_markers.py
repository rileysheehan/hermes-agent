"""Long standalone Telegram sends keep valid MarkdownV2 (``hermes send`` / ``send_message``'s ``_send_telegram``).

``truncate_message()`` appends a raw `` (1/2)`` marker AFTER ``format_message()`` has escaped the text. ``(`` and
``)`` are reserved in MarkdownV2, so before this fix Telegram refused every chunk of a message over 4096 UTF-16 units
("character '(' is reserved and must be escaped") and the sender fell back to plain text: the whole message lost its
formatting and its blockquote/emphasis markers showed literally. The gateway adapter's ``send()`` already escapes the
marker; the standalone sender now does the same. ``telegram`` is stubbed: no network.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Imported at module level: the first import of the gateway runs Hermes' early recovery, which probes the checkout,
# and the home-I/O guard fails that inside a test body.
import gateway.run
import tools.send_message_tool

# A reserved character MarkdownV2 requires escaped, not preceded by a backslash. Text sent by this sender is
# format_message() output: every literal "(" or ")" in it is escaped, so any bare one is an appended marker.
_BARE_PAREN = re.compile(r"(?<!\\)[()]")


@pytest.fixture(autouse=True)
def _forget_modules_imported_under_the_stub():
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if "telegram" in name:
            sys.modules.pop(name, None)


def _install_telegram_mock(monkeypatch, bot):
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    telegram_mod = SimpleNamespace(Bot=MagicMock(return_value=bot), MessageEntity=lambda **kw: SimpleNamespace(**kw),
                                   constants=constants_mod)
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)
    for var in ("TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None, raising=False)
    monkeypatch.setattr("gateway.platforms.base._detect_macos_system_proxy", lambda: None)


def _bot(fail_markdown=False):
    """A bot that records every send. ``fail_markdown``: refuse any MarkdownV2 text holding a bare parenthesis,
    the way Telegram does, so the test exercises the real fallback branch."""
    bot = MagicMock()

    async def send_message(**kw):
        if fail_markdown and kw.get("parse_mode") == "MarkdownV2" and _BARE_PAREN.search(kw["text"]):
            raise Exception("Can't parse entities: character '(' is reserved and must be escaped with the "
                            "preceding '\\'")
        return SimpleNamespace(message_id=len(bot.send_message.await_args_list))

    bot.send_message = AsyncMock(side_effect=send_message)
    return bot


def _send(monkeypatch, message, bot=None):
    from tools.send_message_tool import _send_telegram
    bot = bot or _bot()
    _install_telegram_mock(monkeypatch, bot)
    res = asyncio.run(_send_telegram("tok", "123", message))
    return res, [c.kwargs for c in bot.send_message.await_args_list]


# Shape of the message that failed live: prose paragraphs, numbered headings, blockquotes, no parentheses at all.
_LONG = "\n\n".join(
    f"{n}. A SONG TITLE NUMBER {n}\n> A quoted line about the record, with a wink.\n"
    "Listen for: the horns coming in late. " + "Words about the recording and why it matters. " * 12
    for n in range(1, 12))


def test_long_markdown_send_escapes_every_chunk_marker(monkeypatch):
    res, sends = _send(monkeypatch, _LONG)
    assert res["success"] is True
    assert len(sends) >= 2
    for i, kw in enumerate(sends, 1):
        assert kw["parse_mode"] == "MarkdownV2"
        assert not _BARE_PAREN.findall(kw["text"]), kw["text"][-40:]
        assert kw["text"].endswith(f" \\({i}/{len(sends)}\\)")


def test_long_send_is_not_refused_and_never_falls_back(monkeypatch, caplog):
    """Against a bot that refuses bare parentheses the way Telegram did: every chunk goes out formatted."""
    bot = _bot(fail_markdown=True)
    with caplog.at_level(logging.INFO, logger="tools.send_message_tool"):
        res, sends = _send(monkeypatch, _LONG, bot)
    assert res["success"] is True
    assert all(kw["parse_mode"] == "MarkdownV2" for kw in sends)
    assert "Parse mode" not in caplog.text
    assert re.search(rf"_send_telegram split the message into {len(sends)} chunks", caplog.text)


def test_short_message_control_is_one_chunk_without_marker(monkeypatch, caplog):
    from plugins.platforms.telegram.adapter import TelegramAdapter
    msg = "A short reply (no split)."
    with caplog.at_level(logging.INFO, logger="tools.send_message_tool"):
        res, sends = _send(monkeypatch, msg)
    assert res["success"] is True
    assert len(sends) == 1
    assert sends[0]["text"] == TelegramAdapter.__new__(TelegramAdapter).format_message(msg)
    assert not _BARE_PAREN.findall(sends[0]["text"])
    assert "split the message" not in caplog.text


@pytest.mark.parametrize("extra, chunks", [(0, 1), (1, 2)])
def test_exact_limit_boundary(monkeypatch, extra, chunks):
    """4096 formatted UTF-16 units is one chunk with no marker; one unit more is two escaped markers."""
    msg = "a" * (4096 + extra)  # nothing to escape: formatted length == raw length
    res, sends = _send(monkeypatch, msg)
    assert res["success"] is True
    assert len(sends) == chunks
    for kw in sends:
        assert not _BARE_PAREN.findall(kw["text"])
    if chunks == 1:
        assert sends[0]["text"] == msg


def test_split_inside_a_code_fence_keeps_marker_off_the_fence(monkeypatch):
    """A chunk that ends mid-fence is closed with a synthesized fence; the marker goes on its own line after it
    (Telegram rejects ``` \\(1/2\\) as a fence) and stays escaped."""
    msg = "Here is the log:\n```\n" + "\n".join(f"line {n} of the output" for n in range(400)) + "\n```\nDone."
    res, sends = _send(monkeypatch, msg)
    assert res["success"] is True
    assert len(sends) >= 2
    first = sends[0]["text"]
    assert first.endswith("```\n\\(1/" + str(len(sends)) + "\\)")
    for kw in sends:
        assert not _BARE_PAREN.findall(kw["text"])
        assert not re.search(r"(?m)^``` \\?\(", kw["text"])


def test_literal_parentheses_in_the_body_are_escaped_once(monkeypatch):
    """Body text ending in its own "(3/4)" stays escaped exactly once; only the appended marker is rewritten."""
    msg = ("Part of a series (3/4) " * 300).strip()
    _, sends = _send(monkeypatch, msg)
    assert len(sends) >= 2
    for kw in sends:
        assert "\\\\(" not in kw["text"]
        assert not _BARE_PAREN.findall(kw["text"])


def test_html_message_keeps_plain_marker(monkeypatch):
    """HTML mode: parentheses are not reserved, so the marker is left as it is."""
    msg = "<b>Report</b>\n" + "word " * 1500
    res, sends = _send(monkeypatch, msg)
    assert res["success"] is True
    assert len(sends) >= 2
    for i, kw in enumerate(sends, 1):
        assert kw["parse_mode"] == "HTML"
        assert kw["text"].endswith(f" ({i}/{len(sends)})")


def test_fallback_warning_names_the_chunk(monkeypatch, caplog):
    """When Telegram still refuses a chunk (some other formatting fault), the warning says which chunk of how many,
    so a log reader counts refused chunks, not messages; the plain retry carries no MarkdownV2 escapes."""
    from tools import send_message_senders as senders
    bot = MagicMock()
    calls = []

    async def send_message(**kw):
        calls.append(kw)
        if kw.get("parse_mode") == "MarkdownV2" and len(calls) == 1:
            raise Exception("Can't parse entities: can't find end of the entity starting at byte offset 7")
        return SimpleNamespace(message_id=len(calls))

    bot.send_message = AsyncMock(side_effect=send_message)
    _install_telegram_mock(monkeypatch, bot)
    with caplog.at_level(logging.WARNING, logger="tools.send_message_tool"):
        res = asyncio.run(senders._send_telegram("tok", "123", _LONG))
    assert res["success"] is True
    n = len([c for c in calls if c.get("parse_mode") == "MarkdownV2"])
    assert f"failed in _send_telegram (chunk 1/{n}), falling back to plain text" in caplog.text
    plain = calls[1]
    assert plain["parse_mode"] is None
    assert plain["text"].endswith(f" (1/{n})")
    assert "\\(" not in plain["text"]
