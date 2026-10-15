"""MER-963: a finite gateway turn of an install with ``oneshot_learning`` keeps its post-turn review.

v0.22.1 runs ``chat -q`` in the gateway and turns every background review off for a finite turn
(the viewer is gone when it would run). Installs whose one-shot runs are turns of one longer-lived
identity opted into ``oneshot_learning`` precisely to keep that review; in the gateway, which outlives
the viewer, it can finish. A plain finite turn keeps upstream's behaviour.
"""
from __future__ import annotations

import types

import pytest

import cli  # module-level: its first import probes the checkout (home_io_guard)
from agent import background_review as br
from agent import oneshot_footprint as of
from gateway.session_finite import finite_turn_scope


def _cfg(monkeypatch, **task):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly",
                        lambda *a, **k: {"auxiliary": {"background_review": dict(task)}})


@pytest.fixture
def tool_policy(monkeypatch):
    # The frozen policy of a `chat -q --source tool` session (an orchestrator's board run).
    monkeypatch.setattr("gateway.session_policy.active_policy", lambda: types.SimpleNamespace(source="tool"))


def test_finite_turn_without_learning_keeps_reviews_off(monkeypatch, tool_policy):
    _cfg(monkeypatch, oneshot_learning=False)
    with finite_turn_scope(True):
        assert br.load_background_review_settings() == (False, {})
        assert of.is_single_query_session() is False


def test_finite_turn_with_learning_reviews_with_the_profile_settings(monkeypatch, tool_policy):
    _cfg(monkeypatch, oneshot_learning=True, carry_owed_reviews=True, linger_timeout_s=600)
    with finite_turn_scope(True):
        enabled, task = br.load_background_review_settings()
        assert enabled is True and task["carry_owed_reviews"] is True
        assert of.is_single_query_session() is True
        assert of.oneshot_learning() is True


def test_learning_respects_an_explicit_disable(monkeypatch, tool_policy):
    _cfg(monkeypatch, oneshot_learning=True, enabled=False)
    with finite_turn_scope(True):
        assert br.load_background_review_settings()[0] is False


def test_non_finite_turn_of_the_same_session_is_unchanged(monkeypatch, tool_policy):
    _cfg(monkeypatch, oneshot_learning=True)
    with finite_turn_scope(False):
        assert of.is_single_query_session() is False
        assert br.load_background_review_settings()[0] is True  # upstream: a live session reviews


def test_oneshot_source_is_single_query_regardless(monkeypatch):
    _cfg(monkeypatch, oneshot_learning=False)
    monkeypatch.setattr("gateway.session_policy.active_policy", lambda: types.SimpleNamespace(source="oneshot"))
    assert of.is_single_query_session() is True


def test_memory_review_due_on_a_substantive_finite_turn(monkeypatch, tool_policy):
    _cfg(monkeypatch, oneshot_learning=True)
    agent = types.SimpleNamespace(_skill_nudge_interval=10)
    with finite_turn_scope(True):
        assert br.oneshot_memory_review_due(agent, 12) is True
        assert br.oneshot_memory_review_due(agent, 3) is False


def test_cli_linger_leaves_a_gateway_turn_to_the_gateway(monkeypatch):
    from hermes_cli.gateway_agent import GatewayAgent
    calls = []
    monkeypatch.setattr(cli, "_active_agent_ref", object.__new__(GatewayAgent))
    monkeypatch.setattr("agent.oneshot_footprint.is_single_query_session", lambda: True)
    monkeypatch.setattr(br, "drain_background_review", lambda *a, **k: calls.append("drain") or True)
    monkeypatch.setattr(br, "drain_owed_review", lambda *a, **k: calls.append("owed") or True)
    cli._linger_for_background_review()
    assert calls == []
    # An in-process agent still lingers.
    monkeypatch.setattr(cli, "_active_agent_ref", types.SimpleNamespace(session_id="s1"))
    monkeypatch.setattr(br, "drain_timeout_s", lambda: 10)
    cli._linger_for_background_review()
    assert calls == ["drain", "owed"]
