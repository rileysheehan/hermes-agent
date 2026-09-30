"""Status-message plumbing on the standalone Telegram path (MER-100 part 2): ``hermes send --edit / --reply-to /
--no-preview / --preview``.

Bot API: ``editMessageText`` replaces a sent message's text (an edit never notifies); ``reply_parameters`` with
``allow_sending_without_reply`` anchors a message to an earlier one without failing when that one is gone;
``disable_web_page_preview`` / ``link_preview_options.is_disabled`` drops the link card. ``telegram`` is stubbed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _forget_modules_imported_under_the_stub():
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if "telegram" in name:
            sys.modules.pop(name, None)


def _install(monkeypatch, bot):
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    telegram_mod = SimpleNamespace(Bot=MagicMock(return_value=bot), MessageEntity=lambda **kw: SimpleNamespace(**kw),
                                   constants=constants_mod,
                                   ReplyParameters=lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)
    for var in ("TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None, raising=False)
    monkeypatch.setattr("gateway.platforms.base._detect_macos_system_proxy", lambda: None)


def _bot():
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=11))
    bot.edit_message_text = AsyncMock(return_value=SimpleNamespace(message_id=7))
    return bot


# --- reply anchor ---------------------------------------------------------------------------------------------

def test_reply_to_anchors_only_the_first_chunk_and_survives_a_deleted_original(monkeypatch):
    from tools.send_message_tool import _send_telegram
    bot = _bot(); _install(monkeypatch, bot)
    res = asyncio.run(_send_telegram("tok", "123", "word " * 1500, reply_to="42", disable_notification=False))
    assert res["success"] is True
    calls = bot.send_message.await_args_list
    assert len(calls) >= 2
    rp = calls[0].kwargs["reply_parameters"]
    assert (rp.message_id, rp.allow_sending_without_reply) == (42, True)
    assert all("reply_parameters" not in c.kwargs for c in calls[1:])
    assert all(c.kwargs["disable_notification"] is False for c in calls)   # the choice still rides every chunk


def test_no_reply_to_sends_no_reply_key(monkeypatch):
    from tools.send_message_tool import _send_telegram
    bot = _bot(); _install(monkeypatch, bot)
    asyncio.run(_send_telegram("tok", "123", "hello"))
    kw = bot.send_message.await_args.kwargs
    assert "reply_parameters" not in kw and "reply_to_message_id" not in kw


# --- edit -------------------------------------------------------------------------------------------------------

def test_edit_replaces_the_text_of_that_message(monkeypatch):
    from tools.send_message_senders import _edit_telegram
    bot = _bot(); _install(monkeypatch, bot)
    res = asyncio.run(_edit_telegram("tok", "123", "7", "**Working:** step 2", disable_link_previews=True))
    assert res == {"success": True, "platform": "telegram", "chat_id": "123", "message_id": "7", "edited": True}
    kw = bot.edit_message_text.await_args.kwargs
    assert kw["message_id"] == 7 and kw["chat_id"] == 123 and kw["disable_web_page_preview"] is True
    assert "disable_notification" not in kw          # editMessageText has no such parameter: an edit never rings
    bot.send_message.assert_not_awaited()


def test_edit_not_modified_is_success(monkeypatch):
    from tools.send_message_senders import _edit_telegram
    bot = _bot(); _install(monkeypatch, bot)
    bot.edit_message_text = AsyncMock(side_effect=Exception("Bad Request: message is not modified"))
    assert asyncio.run(_edit_telegram("tok", "123", "7", "same"))["success"] is True


def test_edit_gone_message_is_an_error_so_the_caller_can_send_fresh(monkeypatch):
    from tools.send_message_senders import _edit_telegram
    bot = _bot(); _install(monkeypatch, bot)
    bot.edit_message_text = AsyncMock(side_effect=Exception("Bad Request: message to edit not found"))
    res = asyncio.run(_edit_telegram("tok", "123", "7", "x"))
    assert "error" in res and "not found" in res["error"]
    bot.send_message.assert_not_awaited()


def test_edit_too_long_is_refused_without_calling_telegram(monkeypatch):
    from tools.send_message_senders import _edit_telegram
    bot = _bot(); _install(monkeypatch, bot)
    res = asyncio.run(_edit_telegram("tok", "123", "7", "x" * 5000))
    assert "too long" in res["error"]
    bot.edit_message_text.assert_not_awaited()


def test_edit_parse_failure_falls_back_to_plain(monkeypatch):
    from tools.send_message_senders import _edit_telegram
    bot = _bot(); _install(monkeypatch, bot)
    bot.edit_message_text = AsyncMock(side_effect=[Exception("Can't parse entities"), SimpleNamespace(message_id=7)])
    assert asyncio.run(_edit_telegram("tok", "123", "7", "**x**"))["success"] is True
    assert bot.edit_message_text.await_args_list[1].kwargs["parse_mode"] is None


# --- link previews -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("link_preview, extra, expected", [
    (None, {}, False), (None, {"disable_link_previews": True}, True),
    (False, {}, True), (True, {"disable_link_previews": True}, False)])
def test_per_message_preview_choice_beats_the_setting(monkeypatch, link_preview, extra, expected):
    from gateway.config import Platform
    import tools.send_message_tool as smt
    seen = {}

    async def fake_send_telegram(*a, **kw):
        seen.update(kw)
        return {"success": True}
    monkeypatch.setattr(smt, "_send_telegram", fake_send_telegram)
    kw = {} if link_preview is None else {"link_preview": link_preview}
    asyncio.run(smt._send_to_platform(Platform.TELEGRAM, SimpleNamespace(token="t", extra=extra), "1", "hi", **kw))
    assert seen["disable_link_previews"] is expected


@pytest.mark.parametrize("metadata, setting, disabled", [
    (None, False, False), (None, True, True), ({"link_preview": False}, False, True),
    ({"link_preview": True}, True, False), ({"notify": False}, True, True)])
def test_adapter_link_preview_metadata(metadata, setting, disabled):
    from plugins.platforms.telegram.adapter import TelegramAdapter
    a = TelegramAdapter.__new__(TelegramAdapter)
    a._disable_link_previews = setting
    assert bool(a._link_preview_kwargs(metadata)) is disabled


BOARD = "https://mg.sheehan.life/MER/issues/MER-100"


@pytest.mark.parametrize("text, expected", [
    (f"**Done:** [MER-100]({BOARD}) silent notices", False),                              # routine ticket link
    (f"[MER-100]({BOARD}) and [MER-99](https://mg.sheehan.life/MER/issues/MER-99)", False),
    (f"[MER-98]({BOARD}): the paper is https://arxiv.org/abs/2609.01234.", "https://arxiv.org/abs/2609.01234"),
    ("Read https://arxiv.org/abs/1 first, then [MER-1](" + BOARD + ")", None),              # first link is the point
    ("no links at all", None),
    ("https://sub.mg.sheehan.life/x", False),                                               # subdomains count
    ("https://notmg.sheehan.life.evil.example/x", None),                                    # suffix tricks don't
])
def test_skip_hosts_rule(text, expected):
    from plugins.platforms.telegram.link_previews import link_preview_choice, parse_skip_hosts
    assert link_preview_choice(text, parse_skip_hosts(["https://mg.sheehan.life/"])) == expected


def test_markdown_v2_escapes_are_not_part_of_the_url():
    from plugins.platforms.telegram.link_previews import link_preview_choice
    text = "[MER\\-98](https://mg.sheehan.life/MER/issues/MER\\-98) https://arxiv.org/abs/2609\\.01234"
    assert link_preview_choice(text, ("mg.sheehan.life",)) == "https://arxiv.org/abs/2609.01234"


def test_skip_hosts_parsing():
    from plugins.platforms.telegram.link_previews import parse_skip_hosts
    assert parse_skip_hosts("mg.sheehan.life, https://X.example:8443/a  x.example") == ("mg.sheehan.life", "x.example")
    assert parse_skip_hosts(None) == ()


def test_skip_hosts_off_leaves_everything_alone():
    from plugins.platforms.telegram.link_previews import link_preview_choice
    assert link_preview_choice(f"[MER-100]({BOARD})", ()) is None


def test_adapter_applies_skip_hosts_and_the_per_message_choice_still_wins():
    from plugins.platforms.telegram.adapter import TelegramAdapter
    a = TelegramAdapter.__new__(TelegramAdapter)
    a._disable_link_previews, a._link_preview_skip_hosts = False, ("mg.sheehan.life",)
    kw = a._link_preview_kwargs(None, f"**Done:** [MER-100]({BOARD})")
    assert kw and ("link_preview_options" in kw or kw.get("disable_web_page_preview") is True)
    assert a._link_preview_kwargs({"link_preview": True}, f"[MER-100]({BOARD})") == {}
    assert a._link_preview_kwargs(None, "see https://arxiv.org/abs/1") == {}


def test_standalone_send_uses_the_other_link_for_the_card(monkeypatch):
    from tools.send_message_tool import _send_telegram
    bot = _bot(); _install(monkeypatch, bot)
    sys.modules["telegram"].LinkPreviewOptions = lambda **kw: SimpleNamespace(**kw)
    asyncio.run(_send_telegram("tok", "1", f"[MER-98]({BOARD}) https://arxiv.org/abs/1",
                               preview_url="https://arxiv.org/abs/1"))
    assert bot.send_message.await_args.kwargs["link_preview_options"].url == "https://arxiv.org/abs/1"


@pytest.mark.parametrize("extra, link_preview, expected", [
    ({"link_preview_skip_hosts": ["mg.sheehan.life"]}, None, (True, None)),
    ({"link_preview_skip_hosts": ["mg.sheehan.life"]}, True, (False, None)),
    ({}, None, (False, None)),
])
def test_telegram_preview_helper(extra, link_preview, expected):
    from tools.send_message_tool import _telegram_preview
    assert _telegram_preview(SimpleNamespace(extra=extra), f"**Done:** [MER-1]({BOARD})", link_preview) == expected


# --- CLI ---------------------------------------------------------------------------------------------------------------

def _run_cli(monkeypatch, argv):
    from hermes_cli import send_cmd
    calls = []
    mod = types.ModuleType("tools.send_message_tool")
    mod.send_message_tool = lambda args: calls.append(args) or json.dumps({"success": True, "message_id": "7"})
    monkeypatch.setitem(sys.modules, "tools.send_message_tool", mod)
    monkeypatch.setattr(send_cmd, "_load_hermes_env", lambda: None)
    parser = argparse.ArgumentParser()
    send_cmd.register_send_subparser(parser.add_subparsers())
    with pytest.raises(SystemExit) as exit_:
        send_cmd.cmd_send(parser.parse_args(["send", *argv]))
    return exit_.value.code, calls


def test_cli_edit_becomes_an_edit_action(monkeypatch):
    code, calls = _run_cli(monkeypatch, ["-t", "telegram:1", "--edit", "7", "--no-preview", "new text"])
    assert code == 0
    assert calls[0] == {"action": "edit", "target": "telegram:1", "message": "new text", "edit_message_id": "7",
                        "link_preview": False}


def test_cli_reply_to_and_preview(monkeypatch):
    code, calls = _run_cli(monkeypatch, ["-t", "telegram:1", "--reply-to", "42", "--notify", "--preview", "done"])
    assert code == 0
    assert calls[0]["reply_to"] == "42" and calls[0]["notify"] is True and calls[0]["link_preview"] is True
    assert calls[0]["action"] == "send"


@pytest.mark.parametrize("argv", [
    ["-t", "telegram:1", "--edit", "7", "--notify", "x"],        # an edit cannot notify
    ["-t", "telegram:1", "--edit", "7", "--reply-to", "3", "x"],  # nor reply
    ["-t", "telegram:1", "--edit", "abc", "x"],                   # ids are numeric
    ["-t", "telegram:1", "--reply-to", "x1", "x"],
    ["-t", "discord:#ops", "--edit", "7", "x"],                   # Telegram only
    ["-t", "slack:C1", "--no-preview", "x"],
])
def test_cli_refuses_what_it_cannot_do(monkeypatch, argv):
    code, calls = _run_cli(monkeypatch, argv)
    assert code == 2 and calls == []


def test_edit_action_refuses_other_platforms_and_bad_ids():
    from tools.send_message_tool import send_message_tool
    out = json.loads(send_message_tool({"action": "edit", "target": "telegram:1", "message": "x", "edit_message_id": "a"}))
    assert "error" in out
