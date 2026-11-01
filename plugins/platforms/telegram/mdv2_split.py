"""MarkdownV2 splitting that never leaves a chunk Telegram refuses (Mercury MER-1752).

Split the RAW Markdown and format each chunk on its own: a chunk formatted on its own is exactly as valid as a
short message. Splitting the FORMATTED text instead (upstream's ``split_formatted_mdv2``, still what
``TelegramAdapter.send`` and the standalone sender use on v0.23.2) can cut a fence or an escape pair on mixed
Markdown and Telegram refuses that chunk. The raw split is upstream's own ``split_raw_carrying_emphasis`` (the
overflow-edit path's): it closes and reopens bold, italic, strike and spoiler spans cut by a split, and shrinks the
raw budget until every formatted chunk fits. When it still cannot fit, the text goes escaped and unformatted
(``mdv2_literal_split``), never as a split of formatted text. Used by ``standalone._send_telegram`` and
``TelegramAdapter.send``.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

TELEGRAM_TEXT_LIMIT = 4096  # UTF-16 units per message
_MARKER_RE = re.compile(r" \(\d+/\d+\)$")


def _mdv2_formatter():
    """The adapter's Markdown -> MarkdownV2 converter, or None when the adapter is unavailable."""
    try:
        from plugins.platforms.telegram.adapter import TelegramAdapter
    except ImportError:
        return None
    return TelegramAdapter.__new__(TelegramAdapter).format_message


def _split_raw(message, budget):
    """``truncate_message`` pieces of the RAW text with its `` (i/n)`` markers taken off (they are re-added escaped)."""
    from gateway.platforms.base import BasePlatformAdapter, utf16_len
    pieces = BasePlatformAdapter.truncate_message(message, budget, len_fn=utf16_len)
    n = len(pieces)
    return [p.removesuffix(f" ({i}/{n})") for i, p in enumerate(pieces, 1)] if n > 1 else pieces


def _with_marker(text, i, n):
    """``text`` with its escaped `` \\\\(i/n\\\\)`` marker (on its own line after a closing fence); none for one chunk."""
    if n == 1:
        return text
    text = text.rstrip()
    return f"{text}{chr(10) if text.endswith('```') else ' '}\\({i}/{n}\\)"


def mdv2_split(message, fmt, limit=TELEGRAM_TEXT_LIMIT):
    """``(raw bodies, formatted chunks)`` of ``message``, or None when the formatted chunks cannot be made to fit.
    The raw bodies (markers off, before any emphasis carry) are what each formatted chunk shows, for a caller that
    tracks the visible text or needs a plain fallback per chunk."""
    from gateway.platforms.base import BasePlatformAdapter, utf16_len
    from plugins.platforms.telegram.telegram_chunking import split_raw_carrying_emphasis
    budget = limit
    for _ in range(16):
        raw, carried = split_raw_carrying_emphasis(message, budget, BasePlatformAdapter.truncate_message, utf16_len,
                                                   formatted_len=lambda chunk: utf16_len(fmt(chunk)))
        n = len(carried)
        # The raw " (i/n)" marker comes off before formatting (after a fence the splitter closed it would keep the
        # fence line from closing) and goes back escaped, on its own line after a fence.
        out = [_with_marker(fmt(_MARKER_RE.sub("", chunk.rstrip())), i, n) for i, chunk in enumerate(carried, 1)]
        longest = max(utf16_len(c) for c in out)
        if longest <= limit:
            return [_MARKER_RE.sub("", r) for r in raw], out
        if budget <= 64:
            break
        budget = max(64, min(budget - 1, budget * limit // longest - 32))
    return None


_MDV2_LITERAL_RE = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")


def mdv2_literal_split(message, limit=TELEGRAM_TEXT_LIMIT):
    """``(raw bodies, chunks)``: the text split as raw pieces and sent with every reserved character escaped: no
    formatting, but every chunk is valid MarkdownV2 and fits (an escape at most doubles a piece of
    (limit - 16) // 2 units)."""
    bodies = _split_raw(message, (limit - 16) // 2)
    n = len(bodies)
    return bodies, [_with_marker(_MDV2_LITERAL_RE.sub(r"\\\1", body), i, n) for i, body in enumerate(bodies, 1)]


def reply_split(adapter, content):
    """``(raw bodies, MarkdownV2 chunks)`` for ``TelegramAdapter.send``: one chunk, unchanged, when the formatted
    reply fits; otherwise :func:`mdv2_split` with the adapter's formatter and limit, else the escaped last resort."""
    from gateway.platforms.base import utf16_len
    formatted = adapter.format_message(content)
    if utf16_len(formatted) <= adapter.MAX_MESSAGE_LENGTH:
        return [content], [formatted]
    split = mdv2_split(content, adapter.format_message, adapter.MAX_MESSAGE_LENGTH)
    if split is None:
        logging.getLogger(type(adapter).__module__).warning(   # the adapter's own logger, as its records use
            "[%s] could not fit the formatted reply chunks; sending the text escaped, unformatted",
            getattr(adapter, "name", "Telegram"))
        split = mdv2_literal_split(content, adapter.MAX_MESSAGE_LENGTH)
    return split


def telegram_chunks(message, formatted, parse_mode, has_html):
    """Telegram-sized text chunks (UTF-16 units) for the standalone sender.

    MarkdownV2 (the default): a message whose formatted text fits is one chunk, unchanged. A longer one is split
    as Markdown and each chunk formatted on its own (:func:`mdv2_split`). If that cannot fit (or there is no
    formatter) the text goes escaped instead, never as a split of formatted text. HTML or plain: split as is."""
    from gateway.platforms.base import BasePlatformAdapter, utf16_len
    if not formatted.strip():
        return []
    if utf16_len(formatted) <= TELEGRAM_TEXT_LIMIT:
        return [formatted]
    if has_html or parse_mode is None:
        return BasePlatformAdapter.truncate_message(formatted, TELEGRAM_TEXT_LIMIT, len_fn=utf16_len)
    fmt = _mdv2_formatter()
    split = mdv2_split(message, fmt) if fmt else None
    if split is None:
        logger.warning("_send_telegram could not fit the formatted chunks; sending the text escaped, unformatted")
        split = mdv2_literal_split(message)
    return split[1]
