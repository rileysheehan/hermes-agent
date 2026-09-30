"""Per-message Telegram notification choice: `hermes send --silent/--notify` -> send_message ``notify`` -> the plugin's
standalone sender, and the adapter's explicit-``notify=False`` rule.

Contract (on upstream's notifications mode, ``display.platforms.telegram.notifications``): ``notify=True`` rings,
``notify=False`` is silent in EVERY mode, ``None`` follows the mode (``important``: silent; ``all``: rings).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import plugins.platforms.telegram.standalone as standalone
import plugins.platforms.telegram.telegram_outbound as outbound


def _bot():
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=11))
    bot.send_document = AsyncMock(return_value=SimpleNamespace(message_id=12))
    return bot


@pytest.fixture
def bot(monkeypatch):
    b = _bot()
    monkeypatch.setattr(standalone, "_telegram_bot", lambda token, extra=None: b)
    monkeypatch.setattr(outbound, "send_rich_standalone", AsyncMock(return_value=None))  # rich path not eligible
    for var in ("TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    return b


@pytest.mark.parametrize("mode", ["important", "all"])
@pytest.mark.parametrize("notify, silent", [(True, False), (False, True)])
def test_every_text_chunk_carries_the_explicit_choice(monkeypatch, bot, mode, notify, silent):
    monkeypatch.setenv("HERMES_TELEGRAM_NOTIFICATIONS", mode)
    res = asyncio.run(standalone._send_telegram("tok", "123", "word " * 1500, notify=notify))  # > 4096: 2 chunks
    assert res["success"] is True
    assert bot.send_message.await_count >= 2
    for call in bot.send_message.await_args_list:
        assert call.kwargs.get("disable_notification", False) is silent


@pytest.mark.parametrize("mode, silent", [("important", True), ("all", False)])
def test_no_choice_follows_the_mode(monkeypatch, bot, mode, silent):
    monkeypatch.setenv("HERMES_TELEGRAM_NOTIFICATIONS", mode)
    asyncio.run(standalone._send_telegram("tok", "123", "hello"))
    assert bot.send_message.await_args.kwargs.get("disable_notification", False) is silent


def test_media_upload_carries_the_choice(monkeypatch, bot):
    monkeypatch.setenv("HERMES_TELEGRAM_NOTIFICATIONS", "all")
    f = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    f.write(b"x"); f.close()
    res = asyncio.run(standalone._send_telegram("tok", "123", "cap", media_files=[(f.name, False)], notify=False))
    assert res["success"] is True
    assert bot.send_document.await_args.kwargs["disable_notification"] is True


@pytest.mark.parametrize("notify", [False, True, None])
def test_send_to_platform_passes_notify_to_the_plugin_sender(monkeypatch, notify):
    from gateway.config import Platform
    import gateway.outbound.send as smt
    seen = {}

    async def fake_sender(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False,
                          notify=None, reply_to=None, link_preview=None):
        seen.update(notify=notify)
        return {"success": True}
    monkeypatch.setattr(smt, "_plugin_standalone_sender", lambda name, **kw: (fake_sender, None))
    pconfig = SimpleNamespace(token="tok", extra={})
    kw = {} if notify is None else {"notify": notify}
    asyncio.run(smt._send_to_platform(Platform.TELEGRAM, pconfig, "123", "hi", **kw))
    assert seen["notify"] is notify


@pytest.mark.parametrize("argv, expected", [(["--silent"], False), (["--notify"], True), ([], None)])
def test_cli_flags_reach_the_tool(monkeypatch, argv, expected):
    import gateway.outbound.send as smt
    from hermes_cli import send_cmd
    calls = []
    monkeypatch.setattr(smt, "send_message_tool", lambda args: calls.append(args) or json.dumps({"success": True}))
    monkeypatch.setattr(send_cmd, "_load_hermes_env", lambda: None)
    parser = argparse.ArgumentParser()
    send_cmd.register_send_subparser(parser.add_subparsers())
    args = parser.parse_args(["send", "-t", "telegram:123", *argv, "hi"])
    with pytest.raises(SystemExit) as exit_:
        send_cmd.cmd_send(args)
    assert exit_.value.code == 0
    assert calls[0].get("notify") is expected
    assert ("notify" in calls[0]) is (expected is not None)


def test_cli_rejects_nothing_else_when_both_flags_given(monkeypatch):
    """--silent then --notify: the last one wins (argparse store_const on one dest); no crash."""
    from hermes_cli import send_cmd
    parser = argparse.ArgumentParser()
    send_cmd.register_send_subparser(parser.add_subparsers())
    assert parser.parse_args(["send", "--silent", "--notify", "x"]).notify is True


@pytest.mark.parametrize("mode, notify, expected", [
    ("important", None, {"disable_notification": True}),
    ("important", True, {}),
    ("important", False, {"disable_notification": True}),
    ("all", None, {}),
    ("all", True, {}),
    ("all", False, {"disable_notification": True}),   # explicit silent beats the mode
])
def test_standalone_notification_kwargs(mode, notify, expected):
    assert outbound.notification_kwargs(mode, notify) == expected
    meta = outbound._metadata(None, notify)
    if notify is None:
        assert "notify" not in meta
    else:
        assert meta["notify"] is notify


@pytest.mark.parametrize("mode, metadata, expected", [
    ("important", None, {"disable_notification": True}),
    ("important", {"notify": True}, {}),
    ("important", {"notify": False}, {"disable_notification": True}),
    ("all", None, {}),
    ("all", {"notify": True}, {}),
    ("all", {"notify": False}, {"disable_notification": True}),   # explicit silent beats the mode
])
def test_adapter_notification_kwargs(mode, metadata, expected):
    from plugins.platforms.telegram.adapter import TelegramAdapter
    a = TelegramAdapter.__new__(TelegramAdapter)
    a._notifications_mode = mode
    assert a._notification_kwargs(metadata) == expected
