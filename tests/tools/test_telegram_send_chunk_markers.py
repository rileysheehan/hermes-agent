"""Long standalone Telegram sends keep valid MarkdownV2 (``hermes send`` / ``send_message``'s ``_send_telegram``).

Before this fix the sender formatted the whole message, split the FORMATTED text and appended a raw `` (1/2)``
marker. ``(`` is reserved in MarkdownV2, and a split can also land between an escape's backslash and its character
or inside an entity (``*bold*``, a link), so Telegram refused chunks ("Can't parse entities") and the sender fell
back to plain text. Now a long message is split as Markdown and each chunk is formatted on its own, with an escaped
`` \\(i/n\\)`` marker. ``telegram`` is stubbed: no network.

The fake bot refuses any MarkdownV2 text that ``_mdv2_problems`` (a scanner of Telegram's MarkdownV2 rules: escapes,
reserved characters, entity pairing and nesting, code and pre bodies, link URLs, quotes) finds fault with, so a
regression of any of those kinds shows up as a plain-text fallback, not only a bare parenthesis.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Imported at module level: the first import of the gateway runs Hermes' early recovery, which probes the checkout,
# and the home-I/O guard fails that inside a test body.
import gateway.run
import tools.send_message_tool
from gateway.platforms.base import utf16_len

_RESERVED = set("_*[]()~`>#+-=|{}.!")
_TOGGLES = (("||", "spoiler"), ("__", "underline"), ("*", "bold"), ("_", "italic"), ("~", "strike"))
_MARKER = re.compile(r"\s*\\\(\d+/\d+\\\)$")


def _code_end(text, j, delim, problems):
    """Index just past the closing ``delim`` of a code/pre body starting at ``j`` (escapes skipped), or None."""
    n = len(text)
    while j < n:
        if text[j] == "\\":
            j += 2
            continue
        if text.startswith(delim, j):
            return j + len(delim)
        if text[j] == "`":
            problems.append(f"unescaped ` inside code at {j}")
        j += 1
    return None


def _toggle(stack, name, i, problems):
    """Open or close an entity; closing anything but the innermost open one is refused."""
    if stack and stack[-1] == name:
        stack.pop()
    elif name in stack:
        problems.append(f"{name} closed out of order at {i}")
    else:
        stack.append(name)


def _close_link(text, i, stack, problems):
    """At a ``]``: the index after the link's ``(url)``, or None to stop scanning (unclosed URL)."""
    if not stack or stack[-1] != "link":
        problems.append(f"stray ] at {i}")
        return i + 1
    stack.pop()
    if not text.startswith("(", i + 1):
        problems.append(f"link text without a URL at {i}")
        return i + 1
    j = i + 2
    while j < len(text) and text[j] != ")":
        j += 2 if text[j] == "\\" else 1
    if j >= len(text):
        problems.append(f"unclosed link URL at {i}")
        return None
    return j + 1


def _quote_end(text, i):
    """Index after a line-start quote marker (``>``, ``>>``, ``**>``)."""
    i += 3 if text[i] == "*" else 1
    while i < len(text) and text[i] == ">":
        i += 1
    return i


def _mdv2_problems(text):
    """Everything Telegram's MarkdownV2 parser would refuse in ``text`` (empty list when it parses)."""
    problems, stack, i, n, line_start, expandable = [], [], 0, len(text), True, False
    while i is not None and i < n:
        c, at_line_start, line_start = text[i], line_start, False
        if c == "\\":
            if i + 1 >= n:
                problems.append("lone backslash at the end")
            i += 2
        elif c == "\n":
            i, line_start = i + 1, True
        elif c == "`":
            delim = "```" if text.startswith("```", i) else "`"
            end = _code_end(text, i + len(delim), delim, problems)
            if end is None:
                problems.append(f"unclosed {'pre' if len(delim) == 3 else 'code'} at {i}")
            i = end
        elif at_line_start and (c == ">" or text.startswith("**>", i)):
            expandable = expandable or c == "*"
            i = _quote_end(text, i)
        elif expandable and text.startswith("||", i) and (i + 2 == n or text[i + 2] == "\n") and "spoiler" not in stack:
            i, expandable = i + 2, False
        elif toggled := next(((d, name) for d, name in _TOGGLES if text.startswith(d, i)), None):
            _toggle(stack, toggled[1], i, problems)
            i += len(toggled[0])
        elif c == "[":
            stack.append("link")
            i += 1
        elif c == "]":
            i = _close_link(text, i, stack, problems)
        else:
            if c in _RESERVED:
                problems.append(f"unescaped {c!r} at {i}")
            i += 1
    if stack:
        problems.append(f"unclosed {stack}")
    return problems


def test_the_validator_bites():
    """The scanner refuses the faults the old split produced and accepts well-formed MarkdownV2."""
    assert _mdv2_problems("text \\(1/2\\)") == []
    assert _mdv2_problems("text (1/2)")                     # bare marker
    assert _mdv2_problems("aaaa\\")                         # escape pair cut: lone backslash
    assert _mdv2_problems(".bbbb")                          # ... and its character, unescaped
    assert _mdv2_problems("*aaaa")                          # bold span cut
    assert _mdv2_problems("[text](https://exa")             # link cut
    assert _mdv2_problems("```\ncode")                      # pre cut
    assert _mdv2_problems("*bold _both* italic_")           # crossed entities
    assert _mdv2_problems("> quote\n*b* _i_ `c\\`` [l](https://x.y/a\\)) ||s|| ~t~ __u__") == []


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


def _bot(refuse_invalid=True, transport_error=None):
    """A bot that records every send and, like Telegram, refuses MarkdownV2 it cannot parse or text over 4096
    UTF-16 units. ``transport_error``: every send raises it (a network/permission failure, not a parse one)."""
    bot = MagicMock()

    async def send_message(**kw):
        if transport_error:
            raise Exception(transport_error)
        if utf16_len(kw["text"]) > 4096:
            raise Exception("Bad Request: message is too long")
        if refuse_invalid and kw.get("parse_mode") == "MarkdownV2" and _mdv2_problems(kw["text"]):
            raise Exception("Can't parse entities: " + _mdv2_problems(kw["text"])[0])
        return SimpleNamespace(message_id=100 + len(bot.send_message.await_args_list))

    bot.send_message = AsyncMock(side_effect=send_message)
    return bot


def _send(monkeypatch, message, bot=None):
    from tools.send_message_tool import _send_telegram
    bot = bot or _bot()
    _install_telegram_mock(monkeypatch, bot)
    res = asyncio.run(_send_telegram("tok", "123", message))
    return res, [c.kwargs for c in bot.send_message.await_args_list]


def _assert_all_formatted(res, sends, caplog=None):
    """Every chunk went out once, as valid MarkdownV2 within the limit, with its escaped marker."""
    assert res["success"] is True
    n = len(sends)
    for i, kw in enumerate(sends, 1):
        assert kw["parse_mode"] == "MarkdownV2", kw["text"][:60]
        assert _mdv2_problems(kw["text"]) == [], (i, _mdv2_problems(kw["text"])[:3])
        assert utf16_len(kw["text"]) <= 4096
        if n > 1:
            assert kw["text"].endswith(f"\\({i}/{n}\\)")
    if caplog is not None:
        assert "falling back to plain text" not in caplog.text


# Shape of the message that failed live: prose paragraphs, numbered headings, blockquotes, no parentheses at all.
_LONG = "\n\n".join(
    f"{n}. A SONG TITLE NUMBER {n}\n> A quoted line about the record, with a wink.\n"
    "Listen for: the horns coming in late. " + "Words about the recording and why it matters. " * 12
    for n in range(1, 12))


def test_long_markdown_send_is_valid_in_every_chunk(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="tools.send_message_tool"):
        res, sends = _send(monkeypatch, _LONG)
    assert len(sends) >= 2
    _assert_all_formatted(res, sends, caplog)
    ids = ",".join(str(101 + k) for k in range(len(sends)))
    assert (f"_send_telegram delivered all {len(sends)} chunks in parse mode MarkdownV2 (message ids {ids})"
            in caplog.text)


def test_short_message_control_is_one_chunk_without_marker(monkeypatch, caplog):
    from plugins.platforms.telegram.adapter import TelegramAdapter
    msg = "A short reply (no split)."
    with caplog.at_level(logging.INFO, logger="tools.send_message_tool"):
        res, sends = _send(monkeypatch, msg)
    _assert_all_formatted(res, sends, caplog)
    assert len(sends) == 1
    assert sends[0]["text"] == TelegramAdapter.__new__(TelegramAdapter).format_message(msg)
    assert "delivered all" not in caplog.text  # the completion record is for multi-chunk sends


@pytest.mark.parametrize("extra, chunks", [(0, 1), (1, 2)])
def test_exact_limit_boundary(monkeypatch, extra, chunks):
    """4096 formatted UTF-16 units is one chunk with no marker; one unit more is two valid chunks."""
    msg = "a" * (4096 + extra)  # nothing to escape: formatted length == raw length
    res, sends = _send(monkeypatch, msg)
    _assert_all_formatted(res, sends)
    assert len(sends) == chunks
    if chunks == 1:
        assert sends[0]["text"] == msg


def test_escape_pair_is_never_cut(monkeypatch, caplog):
    """Marin's case: the formatted dot (``\\.``) sat across the old split, leaving ``.`` unescaped in chunk 2."""
    msg = "a" * 4081 + "." + "b" * 100
    with caplog.at_level(logging.INFO, logger="tools.send_message_tool"):
        res, sends = _send(monkeypatch, msg)
    _assert_all_formatted(res, sends, caplog)
    joined = "".join(_MARKER.sub("", kw["text"]) for kw in sends)
    assert joined.replace("\\.", ".") == msg


def test_a_bold_span_across_the_split_stays_bold_in_both_chunks(monkeypatch, caplog):
    """Marin's case: one ``*bold*`` entity cut in two left an unmatched ``*`` in each chunk. It is closed at the end
    of the first and reopened at the start of the second."""
    msg = "**" + "a" * 5000 + "**"
    with caplog.at_level(logging.INFO, logger="tools.send_message_tool"):
        res, sends = _send(monkeypatch, msg)
    _assert_all_formatted(res, sends, caplog)
    assert len(sends) == 2
    for kw in sends:
        body = _MARKER.sub("", kw["text"])
        assert body.startswith("*") and body.endswith("*") and body.count("*") == 2
    assert sum(kw["text"].count("a") for kw in sends) == 5000


def _unescaped_text(sends):
    """The chunks' visible characters, markers and MarkdownV2 escapes removed, whitespace ignored."""
    joined = "".join(_MARKER.sub("", kw["text"]) for kw in sends)
    return re.sub(r"\s", "", re.sub(r"\\(.)", r"\1", joined))


@pytest.mark.parametrize("name, msg", [
    ("Marin r2: raw fits, formatted doubles", "a" + "." * 3000),
    ("raw exactly at the limit", "." * 4096),
    ("raw one under the limit", "a" + "." * 4095),
    ("astral characters, raw fits", "😀" * 1000 + "." * 2000),
    ("astral at the cut", ("😀." * 1365) + "!"),
])
def test_a_message_that_fits_raw_but_not_formatted_is_split_safely(monkeypatch, caplog, name, msg):
    """Marin's round-2 case: the raw text fits 4096 units, escaping takes it past the limit, and the raw splitter
    returns one piece. The budget keeps shrinking until the formatted pieces fit; the formatted text is never cut."""
    assert utf16_len(msg) <= 4096
    with caplog.at_level(logging.INFO, logger="tools.send_message_tool"):
        res, sends = _send(monkeypatch, msg)
    _assert_all_formatted(res, sends, caplog)
    assert len(sends) >= 2, name
    assert _unescaped_text(sends) == re.sub(r"\s", "", msg)
    assert "could not fit" not in caplog.text


@pytest.mark.parametrize("formatter", ["never fits", "missing"])
def test_when_formatted_chunks_cannot_fit_the_text_goes_escaped_never_cut(monkeypatch, caplog, formatter):
    """The last resort keeps the guarantee: every chunk is escaped literal text (valid, within the limit, nothing
    lost), with a warning; it never falls back to splitting formatted text."""
    from tools import send_message_senders as senders
    monkeypatch.setattr(senders, "_mdv2_formatter",
                        (lambda: (lambda text: "x" * 5000)) if formatter == "never fits" else (lambda: None))
    msg = "**Bold** and [a link](https://example.com/x_(y)). `code` > quote! " * 120
    with caplog.at_level(logging.INFO, logger="tools.send_message_tool"):
        res, sends = _send(monkeypatch, msg)
    _assert_all_formatted(res, sends, caplog)
    assert len(sends) >= 2
    assert _unescaped_text(sends) == re.sub(r"\s", "", msg)
    assert "could not fit the formatted chunks; sending the text escaped" in caplog.text


def test_bold_prose_across_the_split(monkeypatch, caplog):
    msg = "Intro line.\n\n**" + "a bold sentence that keeps going, " * 160 + "end.** After."
    with caplog.at_level(logging.INFO, logger="tools.send_message_tool"):
        res, sends = _send(monkeypatch, msg)
    _assert_all_formatted(res, sends, caplog)
    assert len(sends) >= 2


def test_split_inside_a_code_fence(monkeypatch, caplog):
    """A chunk that ends mid-fence is closed with a synthesized fence and the next reopens it; the marker sits on
    its own line after the fence (Telegram rejects ``` \\(1/2\\) as a fence)."""
    msg = ("Here is the log:\n```python\n"
           + "\n".join(f"x_{n} = call(n, '{n}') * 2  # [{n}]" for n in range(300)) + "\n```\nDone.")
    with caplog.at_level(logging.INFO, logger="tools.send_message_tool"):
        res, sends = _send(monkeypatch, msg)
    _assert_all_formatted(res, sends, caplog)
    assert len(sends) >= 2
    assert sends[0]["text"].endswith("```\n\\(1/" + str(len(sends)) + "\\)")
    assert sends[1]["text"].startswith("```python\n")


def test_literal_parentheses_in_the_body_are_escaped_once(monkeypatch):
    """Body text holding its own "(3/4)" stays escaped exactly once; only the appended marker is added."""
    msg = ("Part of a series (3/4) " * 300).strip()
    res, sends = _send(monkeypatch, msg)
    _assert_all_formatted(res, sends)
    for kw in sends:
        assert "\\\\(" not in kw["text"]


@pytest.mark.parametrize("name, msg", [
    ("links", " ".join(f"see [doc {n}](https://example.com/a_(b)/{n}?q=1.2) now." for n in range(150))),
    ("inline code", " ".join(f"run `cmd --flag={n} (x)` then." for n in range(300))),
    ("quotes and headers", "\n".join(f"## Part {n}\n> quoted {n}. (aside)\n- item {n}!" for n in range(220))),
    ("astral and CJK", ("😀 漢字 测试 " * 600) + "end."),
    ("every reserved character", "_*[]()~`>#+-=|{}.!\\ " * 400),
])
def test_boundary_kinds_all_parse(monkeypatch, caplog, name, msg):
    with caplog.at_level(logging.INFO, logger="tools.send_message_tool"):
        res, sends = _send(monkeypatch, msg)
    _assert_all_formatted(res, sends, caplog)
    assert len(sends) >= 2, name


def _random_markdown(rng):
    blocks = [
        lambda: f"Plain words, with punctuation: a-b = c + d! #{rng.randint(1, 99)} (note) {{x}} | ok.",
        lambda: f"**bold {rng.randint(1, 9)} words** and *italic* and ~~gone~~ and ||hidden||.",
        lambda: f"`code_{rng.randint(1, 9)}(x) * 2` inline.",
        lambda: f"[a link {rng.randint(1, 9)}](https://example.com/p_{rng.randint(1, 9)}.html?a=(1))",
        lambda: "```bash\n" + "\n".join(f"echo 'line {k}' | grep -v \"#\"" for k in range(rng.randint(2, 40))) + "\n```",
        lambda: f"> quoted thought {rng.randint(1, 9)}. With (parens).",
        lambda: f"## Heading {rng.randint(1, 9)}",
        lambda: f"- list item {rng.randint(1, 9)}: done.",
        lambda: "Emoji 😀🎉 and 漢字. " * rng.randint(1, 5),
    ]
    parts, size = [], rng.randint(4500, 16000)
    while sum(len(p) for p in parts) < size:
        parts.append(rng.choice(blocks)())
    return "\n".join(parts) if rng.random() < 0.5 else " ".join(parts)


def test_random_mixed_markdown_never_breaks_on_a_split(monkeypatch):
    """Mixed messages of 4.5-16k characters: whenever the formatter's output for the whole message parses, every
    chunk parses too (the split introduces nothing), fits, and goes out formatted."""
    from plugins.platforms.telegram.adapter import TelegramAdapter
    fmt = TelegramAdapter.__new__(TelegramAdapter).format_message
    rng, checked = random.Random(1752), 0
    for _ in range(40):
        msg = _random_markdown(rng)
        if _mdv2_problems(fmt(msg)):
            continue  # a formatter fault on the whole text is not the splitter's (short messages share it)
        res, sends = _send(monkeypatch, msg)
        _assert_all_formatted(res, sends)
        checked += 1
    assert checked >= 30


def test_html_message_keeps_plain_marker(monkeypatch):
    """HTML mode: parentheses are not reserved, so the marker is left as it is."""
    msg = "<b>Report</b>\n" + "word " * 1500
    res, sends = _send(monkeypatch, msg)
    assert res["success"] is True
    assert len(sends) >= 2
    for i, kw in enumerate(sends, 1):
        assert kw["parse_mode"] == "HTML"
        assert kw["text"].endswith(f" ({i}/{len(sends)})")


def test_a_refused_chunk_names_itself_and_writes_no_completion_record(monkeypatch, caplog):
    """When Telegram still refuses a chunk (some other formatting fault), the warning says which chunk of how many
    and the plain retry carries no escapes; the success record is not written."""
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
    with caplog.at_level(logging.INFO, logger="tools.send_message_tool"):
        res = asyncio.run(senders._send_telegram("tok", "123", _LONG))
    assert res["success"] is True
    n = len([c for c in calls if c.get("parse_mode") == "MarkdownV2"])
    assert f"failed in _send_telegram (chunk 1/{n}), falling back to plain text" in caplog.text
    assert calls[1]["parse_mode"] is None
    assert calls[1]["text"].endswith(f" (1/{n})") and "\\(" not in calls[1]["text"]
    assert "delivered all" not in caplog.text


def test_a_failed_transport_writes_no_completion_record(monkeypatch, caplog):
    """Marin's negative control: a send that fails (not a parse refusal) reports an error and leaves neither a
    fallback warning nor the success record, so the record can never read as delivery when nothing went."""
    with caplog.at_level(logging.INFO, logger="tools.send_message_tool"):
        res, _ = _send(monkeypatch, _LONG, bot=_bot(transport_error="Forbidden: offline transport failure"))
    assert "error" in res and not res.get("success")
    assert "delivered all" not in caplog.text
    assert "falling back to plain text" not in caplog.text


def test_a_failure_on_a_later_chunk_writes_no_completion_record(monkeypatch, caplog):
    bot = _bot()
    inner = bot.send_message.side_effect

    async def second_fails(**kw):
        if len(bot.send_message.await_args_list) == 2:
            raise Exception("Forbidden: bot was blocked by the user")
        return await inner(**kw)

    bot.send_message.side_effect = second_fails
    with caplog.at_level(logging.INFO, logger="tools.send_message_tool"):
        res, sends = _send(monkeypatch, _LONG, bot=bot)
    assert not res.get("success") and len(sends) == 2
    assert "delivered all" not in caplog.text
