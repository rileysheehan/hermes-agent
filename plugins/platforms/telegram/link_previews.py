"""Which link in a Telegram message earns the preview card.

Telegram previews the first link in a message. When that link is a routine pointer (a ticket on a dashboard behind a
sign-in page, whose card only says "Sign in") the card competes with the answer and says nothing. The
``link_preview_skip_hosts`` setting lists such hosts: a message whose links all go there gets no card, and a message
that also links elsewhere previews that other link, so a link that IS the point keeps its card.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple, Union
from urllib.parse import urlsplit

# Markdown ``[label](url)`` targets and bare URLs, in order of appearance.
_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"'`]+", re.I)


def parse_skip_hosts(value) -> Tuple[str, ...]:
    """The configured host list: a YAML list or a comma/space separated string; lower-case, no scheme, no port."""
    if not value:
        return ()
    items = value if isinstance(value, (list, tuple, set)) else re.split(r"[,\s]+", str(value))
    out = []
    for item in items:
        host = str(item or "").strip().lower()
        if "://" in host:
            host = urlsplit(host).hostname or ""
        host = host.split("/", 1)[0].split(":", 1)[0].lstrip(".")
        if host and host not in out:
            out.append(host)
    return tuple(out)


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _skipped(host: str, skip: Iterable[str]) -> bool:
    return any(host == s or host.endswith("." + s) for s in skip)


def link_preview_choice(text: Optional[str], skip_hosts: Iterable[str]) -> Union[None, bool, str]:
    """None: leave Telegram's default. False: no card (every link goes to a skipped host). A URL string: preview
    that link (the first link is skipped but a later one is not)."""
    skip = tuple(skip_hosts or ())
    if not skip or not text:
        return None
    # MarkdownV2 escapes punctuation inside link targets (``MER\\-100``): the backslashes are not part of the URL.
    urls = [u.replace("\\", "").rstrip(".,;:!?") for u in _URL_RE.findall(text)]
    if not urls:
        return None
    if not _skipped(_host(urls[0]), skip):
        return None
    other = next((u for u in urls[1:] if not _skipped(_host(u), skip)), None)
    return other if other else False
