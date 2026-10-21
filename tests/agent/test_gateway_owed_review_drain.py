"""MER-1031: a finite gateway turn pays one review another session of its profile still owes.

With ``carry_owed_reviews`` an owed review whose session is never resumed was paid by the next one-shot
run's exit linger (cli.py). Once ``chat -q`` runs in the gateway, the client process is only a viewer and
the linger leaves the turn to the gateway, so nothing paid such a debt: a board run that was interrupted
and never resumed stayed owed indefinitely. The gateway, which outlives the viewer, now does what the
linger did, after the turn's own review and off the turn's path.
"""

from __future__ import annotations

import json
import threading
import types
from unittest.mock import MagicMock

import pytest

from agent import background_review as br
from agent import owed_review
from agent.turn_finalizer import finalize_turn
from gateway.session_finite import finite_turn_required, finite_turn_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from run_agent import AIAgent  # imported at collection: run_agent's import probes the checkout

TRANSCRIPT = [{"role": "session_meta", "content": "x"}, {"role": "user", "content": "do it"},
              {"role": "assistant", "content": "done"}]


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


@pytest.fixture
def threads(monkeypatch):
    """Capture drain threads instead of starting them, so each test runs them when it chooses."""
    started = []

    class _Thread:
        def __init__(self, target=None, daemon=None, name=None):
            self.target, self.daemon, self.name = target, daemon, name

        def start(self):
            started.append(self)

    proxy = types.ModuleType("threading")                       # only background_review's view changes
    proxy.__dict__.update(threading.__dict__)
    proxy.Thread = _Thread
    monkeypatch.setattr(br, "threading", proxy)
    return started


def _cfg(monkeypatch, **task):
    enabled = task.pop("enabled", True)
    monkeypatch.setattr(br, "load_background_review_settings", lambda: (enabled, dict(task)))
    monkeypatch.setattr(br, "_background_review_task_config",
                        lambda task_cfg=None: task_cfg if isinstance(task_cfg, dict) else dict(task))


def _marker(session):
    path = owed_review._path(session)
    return json.loads(path.read_text()) if path.exists() else None


def _orphan(session):
    """An owed marker whose writing process is gone (an interrupted board run of an earlier gateway)."""
    token = owed_review.mark_owed(session, memory=False, skills=True, reason="turn_interrupted", spawning=False)
    owed_review.release(session, token)


class _DB:
    def __init__(self, transcripts):
        self.transcripts = transcripts

    def get_messages_as_conversation(self, session_id, repair_alternation=False):
        return self.transcripts.get(session_id, [])


def _gateway_agent(db, session="current"):
    """The gateway's agent for a finite turn; its review fork runs to completion when spawned."""
    agent = types.SimpleNamespace(session_id=session, _session_db=db, valid_tool_names={"memory", "skill_manage"},
                                  _memory_store=object(), _background_review_run=None, spawned=[], seen=[])

    def _spawn_now(**kw):
        agent.spawned.append(kw)
        agent.seen.append((finite_turn_required(), str(owed_review.marker_dir())))
        run = br.prepare_background_review_run(agent)
        run.owed_session, run.owed_token = kw.get("owed_session"), kw.get("owed_token")
        br._settle_owed_review(run)
        br.finish_background_review_run(agent, run)
        agent._background_review_run = run
    agent._spawn_background_review_now = _spawn_now
    return agent


# ── schedule_owed_review_drain ────────────────────────────────────────────────


def test_finite_gateway_turn_pays_one_orphaned_review_from_its_transcript(home, monkeypatch, threads):
    _cfg(monkeypatch, carry_owed_reviews=True)
    _orphan("interrupted-board-run")
    _orphan("another")
    agent = _gateway_agent(_DB({"interrupted-board-run": TRANSCRIPT, "another": TRANSCRIPT}))
    with finite_turn_scope(True):
        assert br.schedule_owed_review_drain(agent) is True
    (thread,) = threads
    assert thread.daemon is True and thread.name == "owed-review-drain"
    assert agent.spawned == []                                  # nothing ran on the turn's path
    thread.target()
    (kw,) = agent.spawned                                       # ONE per turn
    assert kw["owed_session"] == "interrupted-board-run"        # oldest first
    assert kw["review_skills"] is True and kw["review_memory"] is False
    assert [m["role"] for m in kw["messages_snapshot"]] == ["user", "assistant"]
    assert _marker("interrupted-board-run") is None             # settled only because the review finished
    assert _marker("another") is not None


def test_the_drain_waits_for_the_turns_own_review_first(home, monkeypatch, threads):
    _cfg(monkeypatch, carry_owed_reviews=True, linger_timeout_s=600)
    _orphan("old")
    agent = _gateway_agent(_DB({"old": TRANSCRIPT}))
    own = br.prepare_background_review_run(agent)                # this turn's review, still running
    with finite_turn_scope(True):
        assert br.schedule_owed_review_drain(agent) is True
    runner = threading.Thread(target=threads[0].target, daemon=True)
    runner.start()
    runner.join(0.3)
    assert runner.is_alive() and agent.spawned == []            # waiting, nothing claimed
    assert owed_review.read("old")["attempts"] == 0
    br.finish_background_review_run(agent, own)
    runner.join(5)
    assert not runner.is_alive()
    assert [kw["owed_session"] for kw in agent.spawned] == ["old"] and _marker("old") is None


def test_an_own_review_that_outlasts_the_budget_leaves_the_debt_untouched(home, monkeypatch, threads):
    _cfg(monkeypatch, carry_owed_reviews=True, linger_timeout_s=0.05)
    _orphan("old")
    agent = _gateway_agent(_DB({"old": TRANSCRIPT}))
    br.prepare_background_review_run(agent)                      # never finishes
    with finite_turn_scope(True):
        br.schedule_owed_review_drain(agent)
    threads[0].target()
    assert agent.spawned == []
    assert owed_review.read("old")["attempts"] == 0 and owed_review.read("old")["pid"] == 0


def test_the_drain_thread_keeps_the_turns_profile_and_finite_scope(home, monkeypatch, tmp_path):
    """The gateway serves every profile from one process; the profile is a context-local override."""
    _cfg(monkeypatch, carry_owed_reviews=True)
    profile = tmp_path / "profiles" / "noa"
    profile.mkdir(parents=True)
    token = set_hermes_home_override(profile)
    try:
        _orphan("noa-session")
        agent = _gateway_agent(_DB({"noa-session": TRANSCRIPT}))
        with finite_turn_scope(True):
            assert br.schedule_owed_review_drain(agent) is True
    finally:
        reset_hermes_home_override(token)
    for _ in range(100):                                        # a real daemon thread
        if agent.spawned:
            break
        threading.Event().wait(0.05)
    assert agent.seen == [(True, str(profile / owed_review.MARKER_DIRNAME))]
    assert not (profile / owed_review.MARKER_DIRNAME / "noa-session.json").exists()
    assert not (home / owed_review.MARKER_DIRNAME).exists()     # never the process-wide home


@pytest.mark.parametrize("finite", [None, False])
def test_no_drain_outside_a_finite_gateway_turn(home, monkeypatch, threads, finite):
    """None: a standalone one-shot (its exit linger drains). False: a live gateway session."""
    _cfg(monkeypatch, carry_owed_reviews=True)
    _orphan("old")
    agent = _gateway_agent(_DB({"old": TRANSCRIPT}))
    with finite_turn_scope(finite):
        assert br.schedule_owed_review_drain(agent) is False
    assert threads == [] and owed_review.read("old")["attempts"] == 0


def test_an_interrupted_finite_turn_still_pays_one_other_sessions_debt(home, monkeypatch, threads, caplog):
    """MER-1293: Paperclip cancels a board run that reassigns its own issue, so a hand-off ends interrupted."""
    _cfg(monkeypatch, carry_owed_reviews=True)
    _orphan("old")
    owed_review.mark_owed("current", memory=True, skills=True, reason="turn_interrupted", spawning=False)
    agent = _gateway_agent(_DB({"old": TRANSCRIPT, "current": TRANSCRIPT}))
    with finite_turn_scope(True), caplog.at_level("INFO", logger=br.logger.name):
        assert br.schedule_owed_review_drain(agent, interrupted=True) is True
        (thread,) = threads
        thread.target()
    assert [kw["owed_session"] for kw in agent.spawned] == ["old"]  # never its own, still-owed debt
    assert _marker("old") is None
    assert owed_review.owed_kinds("current") == (True, True)       # left for its own session / a later run
    assert "owed review completed (session=current, after an interrupted turn)" in caplog.text


def test_no_drain_with_the_flag_off_or_for_no_agent_or_cron(home, monkeypatch, threads):
    _cfg(monkeypatch, carry_owed_reviews=True)
    _orphan("old")
    agent = _gateway_agent(_DB({"old": TRANSCRIPT}))
    with finite_turn_scope(True):
        _cfg(monkeypatch)                                       # carry_owed_reviews off
        assert br.schedule_owed_review_drain(agent) is False
        assert br.schedule_owed_review_drain(agent, interrupted=True) is False
        _cfg(monkeypatch, carry_owed_reviews=True, enabled=False)
        assert br.schedule_owed_review_drain(agent) is False
        _cfg(monkeypatch, carry_owed_reviews=True)
        assert br.schedule_owed_review_drain(None) is False
        assert br.schedule_owed_review_drain(types.SimpleNamespace(session_id="cron", skip_background_review=True)) is False
    assert threads == []


def test_a_debt_held_by_a_live_writer_is_not_taken(home, monkeypatch, threads):
    """The gateway's own fresh markers (its pid is alive) stay with their session for ORPHAN_AFTER_S."""
    _cfg(monkeypatch, carry_owed_reviews=True)
    owed_review.mark_owed("resumable", memory=False, skills=True, reason="turn_interrupted", spawning=False)
    agent = _gateway_agent(_DB({"resumable": TRANSCRIPT}))
    with finite_turn_scope(True):
        br.schedule_owed_review_drain(agent)
    threads[0].target()
    assert agent.spawned == [] and owed_review.read("resumable")["attempts"] == 0


def test_a_drain_failure_never_escapes_the_thread(home, monkeypatch, threads, caplog):
    _cfg(monkeypatch, carry_owed_reviews=True)
    agent = _gateway_agent(_DB({}))
    monkeypatch.setattr(br, "drain_owed_review", MagicMock(side_effect=RuntimeError("boom")))
    with finite_turn_scope(True):
        br.schedule_owed_review_drain(agent)
    threads[0].target()                                         # no raise
    assert "Owed review drain after a finite gateway turn failed" in caplog.text


# ── finalize_turn wires it in ─────────────────────────────────────────────────


def _agent(monkeypatch):
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


def _turn(agent, *, interrupted=False, final_response="ok"):
    messages = [{"role": "user", "content": "wake"}, {"role": "assistant", "content": "ok"}]
    finalize_turn(agent, final_response=final_response, api_call_count=1, interrupted=interrupted, failed=False,
                  messages=messages, conversation_history=[], effective_task_id="t", turn_id="t",
                  user_message="wake", original_user_message="wake", _should_review_memory=False,
                  _turn_exit_reason="text_response(1)")


@pytest.mark.parametrize("finite, interrupted, final_response, expected", [
    (True, False, "ok", [False]),         # a finished finite gateway turn schedules one drain
    (True, True, "ok", [True]),           # an interrupted one is told so (and still drains, MER-1293)
    (True, False, None, [True]),          # no final response: treated as interrupted
])
def test_finalize_turn_schedules_the_drain(home, monkeypatch, finite, interrupted, final_response, expected):
    _cfg(monkeypatch, oneshot_learning=True, carry_owed_reviews=True)
    calls = []
    monkeypatch.setattr(br, "schedule_owed_review_drain",
                        lambda agent, interrupted=False: calls.append(interrupted) or True)
    agent = _agent(monkeypatch)
    with finite_turn_scope(finite):
        _turn(agent, interrupted=interrupted, final_response=final_response)
    assert calls == expected


def test_finalize_turn_leaves_the_flag_off_and_cron_alone(home, monkeypatch):
    calls = []
    monkeypatch.setattr(br, "schedule_owed_review_drain", lambda agent, interrupted=False: calls.append(1))
    _cfg(monkeypatch, oneshot_learning=True)                    # carry_owed_reviews off
    agent = _agent(monkeypatch)
    with finite_turn_scope(True):
        _turn(agent)
        _cfg(monkeypatch, oneshot_learning=True, carry_owed_reviews=True)
        agent.skip_background_review = True                     # cron
        _turn(agent)
    assert calls == []
