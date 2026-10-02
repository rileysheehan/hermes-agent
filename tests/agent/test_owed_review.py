"""Owed post-turn reviews (``auxiliary.background_review.carry_owed_reviews``).

A review that was due but did not finish (killed, cancelled, interrupted) is carried to the session's next
turn or to the profile's next one-shot run, and short one-shot turns add up to the substantive threshold.
"""

from __future__ import annotations

import json
import types
from unittest.mock import MagicMock

import pytest

from agent import background_review as br
from agent import owed_review
from agent.turn_finalizer import finalize_turn
from run_agent import AIAgent  # imported at collection: run_agent's import probes the checkout


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


@pytest.fixture
def oneshot(monkeypatch):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")


def _cfg(monkeypatch, **task):
    enabled = task.pop("enabled", True)
    monkeypatch.setattr(br, "load_background_review_settings", lambda: (enabled, dict(task)))
    monkeypatch.setattr(br, "_background_review_task_config",
                        lambda task_cfg=None: task_cfg if isinstance(task_cfg, dict) else dict(task))


def _marker(session):
    path = owed_review._path(session)
    return json.loads(path.read_text()) if path.exists() else None


def _orphan(session, **kw):
    """An owed marker whose writing process is gone."""
    token = owed_review.mark_owed(session, memory=True, skills=False, reason="review_started", spawning=True, **kw)
    owed_review.release(session, token)  # pid 0: no live owner


# ── the marker ────────────────────────────────────────────────────────────────


def test_settle_needs_the_current_token(home):
    old = owed_review.mark_owed("s", memory=True, skills=False, reason="review_started", spawning=True)
    new = owed_review.mark_owed("s", memory=False, skills=True, reason="review_started", spawning=True)
    assert owed_review.settle("s", old) is False            # an older review cannot clear a newer debt
    assert owed_review.owed_kinds("s") == (True, True)       # kinds merge while owed
    assert owed_review.settle("s", new) is True
    assert _marker("s") is None and owed_review.owed_kinds("s") == (False, False)


def test_attempts_are_capped_and_the_marker_stays_for_monitoring(home):
    for _ in range(owed_review.MAX_ATTEMPTS):
        owed_review.mark_owed("s", memory=True, skills=False, reason="review_started", spawning=True)
    assert owed_review.owed_kinds("s") == (False, False)
    assert [m["session_id"] for m in owed_review.owed_markers()] == ["s"]


def test_short_runs_add_up_and_a_review_resets_the_count(home):
    assert owed_review.carry("s", 4) == 4
    assert owed_review.carry("s", 3) == 7
    assert owed_review.owed_markers() == []                  # a carry is not a debt
    token = owed_review.mark_owed("s", memory=True, skills=False, reason="review_started", spawning=True)
    assert owed_review.carried_tool_calls("s") == 0
    owed_review.carry("s", 2)                                # a short turn while the review runs
    owed_review.settle("s", token)
    assert owed_review.carried_tool_calls("s") == 2 and owed_review.owed_kinds("s") == (False, False)


def test_claim_orphan_takes_only_abandoned_debts_oldest_first(home):
    owed_review.mark_owed("live", memory=True, skills=False, reason="review_started", spawning=True)  # this pid
    assert owed_review.claim_orphan() is None                # its writer is alive
    _orphan("older")
    _orphan("newer")
    data = json.loads(owed_review._path("older").read_text())
    data["owed_since"] -= 100
    owed_review._path("older").write_text(json.dumps(data))
    assert owed_review.claim_orphan(exclude="older")["session_id"] == "newer"
    assert owed_review.claim_orphan()["session_id"] == "older"
    assert owed_review.claim_orphan() is None                # both now stamped with this live pid


def test_claim_orphan_takes_a_stale_debt_of_a_live_process_and_prunes_old_carries(home):
    owed_review.mark_owed("gw", memory=True, skills=False, reason="review_started", spawning=True)
    owed_review.carry("short", 3)
    later = owed_review.read("gw")["updated_at"] + owed_review.CARRY_TTL_S + 1
    assert owed_review.claim_orphan(now=later)["session_id"] == "gw"
    assert _marker("short") is None


@pytest.mark.parametrize(("attrs", "cfg", "expected"), [
    ({}, {"carry_owed_reviews": True}, True),
    ({}, {}, False),                                                       # default: off
    ({}, {"carry_owed_reviews": True, "enabled": False}, False),           # reviews disabled
    ({"skip_background_review": True}, {"carry_owed_reviews": True}, False),  # cron
    ({"_delegate_depth": 1}, {"carry_owed_reviews": True}, False),         # subagent
    ({"_persist_disabled": True}, {"carry_owed_reviews": True}, False),    # the review fork itself
])
def test_applies(monkeypatch, attrs, cfg, expected):
    _cfg(monkeypatch, **cfg)
    assert owed_review.applies(types.SimpleNamespace(session_id="s", **attrs)) is expected


# ── the review fork settles only when it finished ────────────────────────────


@pytest.mark.parametrize(("result", "settled"), [({"interrupted": False}, True), ({"interrupted": True}, False),
                                                 ({"failed": True}, False)])
def test_review_fork_settles_only_an_uninterrupted_review(home, monkeypatch, result, settled):
    fork = types.SimpleNamespace(run_conversation=lambda **kw: result, _session_messages=[])
    monkeypatch.setattr(br, "build_cache_parity_fork", lambda *a, **k: (fork, {}, False))
    monkeypatch.setattr(br, "_review_tool_whitelist", lambda *a, **k: ({"memory"}, set()))
    for name in ("_track_review_fork", "_record_review_usage_to_parent", "_release_fork_clients"):
        monkeypatch.setattr(br, name, lambda *a, **k: None)
    monkeypatch.setattr(br, "_snapshot_review_usage", lambda *a, **k: {})
    agent = types.SimpleNamespace(session_id="s", _background_review_run=None)
    run = br.prepare_background_review_run(agent)
    run.owed_session = "s"
    run.owed_token = owed_review.mark_owed("s", memory=True, skills=False, reason="review_started", spawning=True)
    br._run_review_fork(agent, [], "review", {}, run, br._ReviewForkState(), review_memory=True)
    assert (_marker("s") is None) is settled
    assert run.request_done.is_set()


# ── another session's debt, paid at one-shot exit ─────────────────────────────


class _DB:
    def __init__(self, transcripts):
        self.transcripts = transcripts

    def get_messages_as_conversation(self, session_id, repair_alternation=False):
        return self.transcripts.get(session_id, [])


def _exiting_agent(db):
    agent = types.SimpleNamespace(session_id="current", _session_db=db, valid_tool_names={"memory", "skill_manage"},
                                  _memory_store=object(), _background_review_run=None, spawned=[])

    def _spawn_now(**kw):  # a review that runs to completion
        agent.spawned.append(kw)
        run = br.prepare_background_review_run(agent)
        run.owed_session, run.owed_token = kw.get("owed_session"), kw.get("owed_token")
        br._settle_owed_review(run)
        br.finish_background_review_run(agent, run)
        agent._background_review_run = run  # what the exit linger waits on
    agent._spawn_background_review_now = _spawn_now
    return agent


def test_exit_pays_one_orphaned_review_from_the_stored_transcript(home, monkeypatch):
    _cfg(monkeypatch, carry_owed_reviews=True)
    _orphan("killed")
    transcript = [{"role": "session_meta", "content": "x"}, {"role": "user", "content": "do it"},
                  {"role": "assistant", "content": "done"}]
    agent = _exiting_agent(_DB({"killed": transcript}))
    assert br.drain_owed_review(agent, budget=120) is True
    (kw,) = agent.spawned
    assert kw["owed_session"] == "killed" and kw["review_memory"] is True and kw["review_skills"] is False
    assert [m["role"] for m in kw["messages_snapshot"]] == ["user", "assistant"]
    assert _marker("killed") is None


def test_exit_settles_a_debt_whose_transcript_is_gone_and_respects_budget_and_flag(home, monkeypatch):
    _cfg(monkeypatch, carry_owed_reviews=True)
    _orphan("pruned")
    agent = _exiting_agent(_DB({}))
    assert br.drain_owed_review(agent, budget=10) is False    # too little budget left: not even claimed
    assert owed_review.read("pruned")["pid"] == 0
    assert br.drain_owed_review(agent, budget=120) is False
    assert agent.spawned == [] and _marker("pruned") is None
    _orphan("later")
    _cfg(monkeypatch)                                          # flag off: nothing is touched
    assert br.drain_owed_review(agent, budget=120) is False and _marker("later") is not None


# ── finalize_turn: owed reviews fold in, short turns add up ───────────────────


def _agent(monkeypatch, home):
    agent = AIAgent(model="openai/gpt-4o-mini", provider="openrouter", api_key="sk-dummy",
                    base_url="https://openrouter.ai/api/v1", quiet_mode=True, skip_context_files=True,
                    skip_memory=True, platform="cli")
    for name in ("_spawn_background_review", "_save_trajectory", "_cleanup_task_resources", "_persist_session",
                 "clear_interrupt", "_sync_external_memory_for_turn", "_emit_status", "_safe_print",
                 "_apply_persist_user_message_override"):
        setattr(agent, name, MagicMock())
    agent._session_messages = []
    agent._file_mutation_verifier_enabled = lambda: False
    agent._stream_callback = None
    agent._skill_nudge_interval = 10
    agent._iters_since_skill = 0
    agent.valid_tool_names = {"memory"}
    agent._memory_store = object()
    agent.iteration_budget = MagicMock(remaining=100, used=5, max_total=100)
    agent.max_iterations = 50
    agent.context_compressor = None
    agent._turn_preflight_display_snapshot = None
    agent._turn_received_provider_response = False
    agent.model = "test-model"
    agent.session_id = "board-session"
    agent._turn_failed_file_mutations = {}
    agent._db_flush_scan_prefix = None
    return agent


def _turn(agent, tool_calls, *, interrupted=False, due=False):
    messages = [{"role": "user", "content": "wake"}]
    messages += [{"role": "tool", "content": str(i)} for i in range(tool_calls)]
    messages += [{"role": "assistant", "content": "ok"}]
    agent._spawn_background_review.reset_mock()
    finalize_turn(agent, final_response="ok", api_call_count=1, interrupted=interrupted, failed=False,
                  messages=messages, conversation_history=[], effective_task_id="t", turn_id="t",
                  user_message="wake", original_user_message="wake", _should_review_memory=due,
                  _turn_exit_reason="text_response(1)")
    calls = agent._spawn_background_review.call_args_list
    return calls[0].kwargs if calls else None


def test_short_one_shot_turns_add_up_to_one_review(home, monkeypatch, oneshot):
    _cfg(monkeypatch, oneshot_learning=True, carry_owed_reviews=True)
    agent = _agent(monkeypatch, home)
    assert _turn(agent, 4) is None
    assert _turn(agent, 3) is None
    assert owed_review.carried_tool_calls("board-session") == 7
    kw = _turn(agent, 4)                                        # 4 + 7 >= 10
    assert kw["review_memory"] is True and kw["owed_session"] == "board-session"
    assert kw["owed_token"] == owed_review.read("board-session")["token"]
    assert owed_review.carried_tool_calls("board-session") == 0


def test_an_interrupted_turn_owes_its_review_to_the_next_turn(home, monkeypatch, oneshot):
    _cfg(monkeypatch, oneshot_learning=True, carry_owed_reviews=True)
    agent = _agent(monkeypatch, home)
    assert _turn(agent, 12, interrupted=True) is None
    assert owed_review.read("board-session")["reason"] == "turn_interrupted"
    kw = _turn(agent, 1)                                        # a short follow-up still pays it
    assert kw is not None and kw["review_memory"] is True


def test_flag_off_keeps_todays_behaviour(home, monkeypatch, oneshot):
    _cfg(monkeypatch, oneshot_learning=True)
    agent = _agent(monkeypatch, home)
    assert _turn(agent, 4) is None and _turn(agent, 7) is None  # short turns never add up
    kw = _turn(agent, 10)
    assert kw is not None and "owed_session" not in kw
    assert not (home / owed_review.MARKER_DIRNAME).exists()


def test_resuming_a_session_claims_its_own_debt(home, monkeypatch):
    """The process that resumes a session pays that session's debt in its own review, so it stamps
    the marker and no other exiting run takes it over meanwhile."""
    from agent.turn_context import _hydrate_from_history

    _cfg(monkeypatch, carry_owed_reviews=True)
    _orphan("board-session")
    agent = types.SimpleNamespace(session_id="board-session", _todo_store=MagicMock(has_items=lambda: True),
                                  _user_turn_count=0, context_compressor=None, _memory_nudge_interval=10,
                                  _turns_since_memory=0)
    monkeypatch.setattr("agent.turn_context.restore_usage_anchor", lambda *a, **k: None)
    _hydrate_from_history(agent, [{"role": "user", "content": "earlier"}])
    assert owed_review.claim_orphan() is None
    assert owed_review.owed_kinds("board-session") == (True, False)


def test_exit_linger_spends_what_is_left_of_its_budget_on_an_owed_review(monkeypatch, oneshot):
    import cli

    seen = {}
    monkeypatch.setattr(cli, "_active_agent_ref", object())
    monkeypatch.setattr(br, "drain_background_review", lambda agent: True)
    monkeypatch.setattr(br, "drain_timeout_s", lambda: 240.0)
    monkeypatch.setattr(br, "drain_owed_review", lambda agent, budget: seen.setdefault("budget", budget) and True)
    cli._linger_for_background_review()
    assert 230 < seen["budget"] <= 240
