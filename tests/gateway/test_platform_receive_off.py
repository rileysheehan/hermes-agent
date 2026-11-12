"""``platforms.<p>.receive: false``: a send-only platform.

The home keeps sending on the platform (``send_message``/``hermes send``, cron delivery, notices),
but the gateway starts no adapter for it, because another service owns its inbound updates (a
Telegram bot whose webhook belongs to another service). ``enabled: false`` would switch every send
off too, which is the gap this closes."""
import logging
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from tools import send_message_tool
from gateway.config import GatewayConfig, Platform, PlatformConfig, platform_receives_inbound
from gateway.run import GatewayRunner
from tests.gateway.test_multiplex_adapter_registry import (
    _install_secondary_reconnect_context,
    _SecondaryRecoveryAdapter,
    _secondary_recovery_runner,
)


def _telegram(**extra):
    return PlatformConfig(enabled=True, token="123:synthetic", extra=dict(extra))


class TestReceiveSetting:
    def test_default_receives(self):
        assert platform_receives_inbound(_telegram()) is True
        assert platform_receives_inbound(PlatformConfig(enabled=True)) is True

    @pytest.mark.parametrize("value", [False, "false", "off", "no", 0])
    def test_falsey_values_turn_receiving_off(self, value):
        assert platform_receives_inbound(_telegram(receive=value)) is False

    @pytest.mark.parametrize("value", [True, "true", "yes", None, "garbage"])
    def test_other_values_keep_todays_behaviour(self, value):
        assert platform_receives_inbound(_telegram(receive=value)) is True

    def test_both_yaml_spellings_reach_extra(self):
        top = PlatformConfig.from_dict({"enabled": True, "token": "t", "receive": False})
        nested = PlatformConfig.from_dict({"enabled": True, "token": "t", "extra": {"receive": False}})
        assert platform_receives_inbound(top) is False
        assert platform_receives_inbound(nested) is False
        assert top.enabled and nested.enabled


class TestSendingStaysOn:
    """The regression this closes: with the receiver moved elsewhere, sends must still resolve."""

    def test_send_tool_resolves_a_send_only_platform(self):
        cfg = GatewayConfig(platforms={Platform.TELEGRAM: _telegram(receive=False)})
        _platform, pconfig, _entry, err = send_message_tool._resolve_platform_config("telegram", cfg)
        assert err is None
        assert pconfig is not None and pconfig.token == "123:synthetic"

    def test_disabled_platform_still_refuses_to_send(self):
        """Control: ``enabled: false`` is unchanged (it is still the switch for sending too)."""
        cfg = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=False, token="123:synthetic")})
        _platform, _pconfig, _entry, err = send_message_tool._resolve_platform_config("telegram", cfg)
        assert err and "not configured" in err

    def test_cron_preflight_counts_it_connected(self):
        cfg = GatewayConfig(platforms={Platform.TELEGRAM: _telegram(receive=False)})
        assert Platform.TELEGRAM in cfg.get_connected_platforms()


class _BuildRefused(Exception):
    pass


def _primary_runner(platform_config):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: platform_config})
    runner._abort_startup_if_shutdown_requested = AsyncMock(return_value=False)
    runner._multiplex_on = lambda: False
    built = []

    def create(platform, config):
        built.append(platform)
        raise _BuildRefused(platform)

    runner._create_adapter = create
    return runner, built


class TestPrimaryStartup:
    @pytest.mark.asyncio
    async def test_send_only_platform_builds_no_adapter(self):
        runner, built = _primary_runner(_telegram(receive=False))
        aborted, enabled_count, skipped, pending = await runner._start_prefilter_platforms()
        assert built == []
        assert (aborted, enabled_count, skipped, pending) == (False, 0, [], [])

    @pytest.mark.asyncio
    async def test_control_receiving_platform_is_built(self):
        runner, built = _primary_runner(_telegram())
        with pytest.raises(_BuildRefused):
            await runner._start_prefilter_platforms()
        assert built == [Platform.TELEGRAM]


def _secondary_context(monkeypatch, runner, platform_config):
    _install_secondary_reconnect_context(monkeypatch, runner, _SecondaryRecoveryAdapter())
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: GatewayConfig(multiplex_profiles=True, platforms={Platform.DISCORD: platform_config}),
    )
    built = []
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, config: built.append(platform) or None)
    return built


class TestSecondaryProfile:
    @pytest.mark.asyncio
    async def test_startup_and_rescan_build_no_adapter(self, monkeypatch):
        runner = _secondary_recovery_runner()
        built = _secondary_context(monkeypatch, runner, PlatformConfig(enabled=True, token="t", extra={"receive": False}))
        connected = await runner._start_one_profile_adapters("reviewer", "/tmp/reviewer", {})
        assert connected == 0
        assert built == []
        assert runner._profile_adapters.get("reviewer", {}) == {}
        assert runner._profile_failed_platforms == {}
        assert list(runner._background_tasks) == []

    @pytest.mark.asyncio
    async def test_control_receiving_secondary_is_built(self, monkeypatch):
        runner = _secondary_recovery_runner()
        built = _secondary_context(monkeypatch, runner, PlatformConfig(enabled=True, token="t"))
        await runner._start_one_profile_adapters("reviewer", "/tmp/reviewer", {})
        assert built == [Platform.DISCORD]

    @pytest.mark.asyncio
    async def test_reconnect_stands_down_instead_of_rebuilding(self, monkeypatch):
        """A platform switched to send-only while queued for reconnect is not rebuilt."""
        runner = _secondary_recovery_runner()
        built = _secondary_context(monkeypatch, runner, PlatformConfig(enabled=True, token="t", extra={"receive": False}))
        stood_down = []
        monkeypatch.setattr(runner, "_stand_down_unbuilt_secondary",
                            lambda profile, platform, why: stood_down.append((profile, platform, why)))
        assert await runner._secondary_reconnect_attempt("reviewer", Platform.DISCORD) == (None, None)
        assert built == []
        assert stood_down == [("reviewer", Platform.DISCORD, "receive is false in config")]


class TestRestartOnlyContract:
    """Like ``enabled: false``, ``receive: false`` takes effect when the profile is (re)served. A hot
    rescan leaves an already-running receiver alone and says so (MER-2307): applying the flag is not
    proof the bot's inbound is free for another service."""

    @pytest.mark.asyncio
    async def test_rescan_keeps_a_live_receiver_and_warns(self, monkeypatch, caplog):
        runner = _secondary_recovery_runner()
        built = _secondary_context(monkeypatch, runner, PlatformConfig(enabled=True, token="t", extra={"receive": False}))
        live = _SecondaryRecoveryAdapter()
        runner._profile_adapters["reviewer"] = {Platform.DISCORD: live}
        with caplog.at_level(logging.WARNING):
            await runner._start_one_profile_adapters("reviewer", "/tmp/reviewer", {})
        assert built == []
        assert runner._profile_adapters["reviewer"][Platform.DISCORD] is live
        assert live.disconnected is False
        assert "a rescan does not stop a receiver" in caplog.text
        assert "hermes -p reviewer gateway restart" in caplog.text


class TestPrimaryReconnectQueue:
    @pytest.mark.asyncio
    async def test_queued_send_only_platform_is_dropped_not_built(self, monkeypatch):
        runner = GatewayRunner.__new__(GatewayRunner)
        cfg = _telegram(receive=False)
        runner._failed_platforms = {Platform.TELEGRAM: {"config": cfg, "attempts": 0, "next_retry": 0}}
        runner._flag_reconnect_needs_attention = lambda *a, **k: None
        built = []
        runner._create_adapter = lambda platform, config: built.append(platform)
        await runner._reconnect_failed_platform(Platform.TELEGRAM, 1.0)
        assert built == []
        assert Platform.TELEGRAM not in runner._failed_platforms


def _policy_runner(platforms, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in list(GatewayRunner._BUILTIN_ALLOWED_USERS_VARS) + list(GatewayRunner._BUILTIN_ALLOW_ALL_VARS):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms=platforms, sessions_dir=tmp_path / "sessions")
    return runner


@pytest.mark.parametrize(("receive", "expect_warning"), [(False, False), (True, True)])
def test_allowlist_reminder_skips_a_send_only_platform(monkeypatch, tmp_path, caplog, receive, expect_warning):
    runner = _policy_runner({Platform.TELEGRAM: _telegram(receive=receive)}, tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        assert runner._start_check_access_policy() is False
    assert ("No env user allowlists configured" in caplog.text) is expect_warning
