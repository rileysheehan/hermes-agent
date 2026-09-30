"""Per-message notification choice on the standalone Telegram path (``hermes send --silent/--notify``).

Bot API: ``disable_notification`` — "Sends the message silently. Users will receive a notification with no
sound." The standalone sender must put the caller's choice, explicitly, on every text chunk and every media
upload, and must send nothing extra when no choice was made (the default path is unchanged). The live
adapter honours an explicit ``metadata["notify"] is False`` in every notification mode. ``telegram`` is stubbed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _forget_modules_imported_under_the_stub():
    """Modules first imported while ``telegram`` is stubbed bind the stub (ChatType=None) and would leak into
    later test files; drop them afterwards so this file leaves ``sys.modules`` as it found it."""
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


def _bot():
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=11))
    bot.send_document = AsyncMock(return_value=SimpleNamespace(message_id=12))
    return bot


@pytest.mark.parametrize("choice, expected", [(True, True), (False, False)])
def test_every_text_chunk_carries_the_explicit_choice(monkeypatch, choice, expected):
    from tools.send_message_tool import _send_telegram
    bot = _bot()
    _install_telegram_mock(monkeypatch, bot)
    res = asyncio.run(_send_telegram("tok", "123", "word " * 1500, disable_notification=choice))  # > 4096: 2 chunks
    assert res["success"] is True
    assert bot.send_message.await_count >= 2
    for call in bot.send_message.await_args_list:
        assert call.kwargs["disable_notification"] is expected


def test_no_choice_sends_no_notification_key(monkeypatch):
    from tools.send_message_tool import _send_telegram
    bot = _bot()
    _install_telegram_mock(monkeypatch, bot)
    asyncio.run(_send_telegram("tok", "123", "hello"))
    assert "disable_notification" not in bot.send_message.await_args.kwargs


def test_media_upload_carries_the_choice(monkeypatch):
    from tools.send_message_tool import _send_telegram
    bot = _bot()
    _install_telegram_mock(monkeypatch, bot)
    f = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    f.write(b"x"); f.close()
    res = asyncio.run(_send_telegram("tok", "123", "cap", media_files=[(f.name, False)], disable_notification=True))
    assert res["success"] is True
    assert bot.send_document.await_args.kwargs["disable_notification"] is True


@pytest.mark.parametrize("notify, expected", [(False, True), (True, False), (None, None)])
def test_send_to_platform_maps_notify_to_disable_notification(monkeypatch, notify, expected):
    from gateway.config import Platform
    import tools.send_message_tool as smt
    seen = {}

    async def fake_send_telegram(*a, **kw):
        seen.update(kw)
        return {"success": True}
    monkeypatch.setattr(smt, "_send_telegram", fake_send_telegram)
    pconfig = SimpleNamespace(token="tok", extra={})
    kw = {} if notify is None else {"notify": notify}
    asyncio.run(smt._send_to_platform(Platform.TELEGRAM, pconfig, "123", "hi", **kw))
    assert seen["disable_notification"] is expected


@pytest.mark.parametrize("argv, expected", [(["--silent"], False), (["--notify"], True), ([], None)])
def test_cli_flags_reach_the_tool(monkeypatch, argv, expected):
    import types
    from hermes_cli import send_cmd
    calls = []
    mod = types.ModuleType("tools.send_message_tool")
    mod.send_message_tool = lambda args: calls.append(args) or json.dumps({"success": True})
    monkeypatch.setitem(sys.modules, "tools.send_message_tool", mod)
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
