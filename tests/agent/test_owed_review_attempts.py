"""Owed-review attempts (``carry_owed_reviews``) are spent only by a review that ran to an end of its own.

A review that the session's own next live turn supersedes, or that is never admitted because another review is
still in flight, gives back the attempt ``mark_owed`` counted and keeps its owner pid. A teardown cancel, a failed
review and an exit-linger budget cut still count, so ``MAX_ATTEMPTS`` keeps bounding real failures.
"""

from __future__ import annotations

import json
import os
import threading
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


def _cfg(monkeypatch, **task):
    monkeypatch.setattr(br, "load_background_review_settings", lambda: (True, dict(task)))
    monkeypatch.setattr(br, "_background_review_task_config",
                        lambda task_cfg=None: task_cfg if isinstance(task_cfg, dict) else dict(task))


def _marker(session):
    path = owed_review._path(session)
    return json.loads(path.read_text()) if path.exists() else None


def _mark(session="s"):
    return owed_review.mark_owed(session, memory=True, skills=False, reason="review_started", spawning=True)


class _Fork:
    """A review fork whose conversation blocks until it is interrupted (a supersede) or let finish."""

    def __init__(self):
        self.started, self.interrupted, self.finish = threading.Event(), threading.Event(), threading.Event()
        self._session_messages = []

    def interrupt(self, message=None):
        self.interrupted.set()

    def run_conversation(self, **kw):
        self.started.set()
        while not (self.interrupted.is_set() or self.finish.is_set()):
            self.interrupted.wait(0.01)
        return {"interrupted": self.interrupted.is_set()}


@pytest.fixture
def forks(monkeypatch):
    """Every review fork built in the test, in order; the real thread, fork phase and settle path run."""
    built = []

    def _build(*a, **k):
        built.append(_Fork())
        return built[-1], {}, False
    monkeypatch.setattr(br, "build_cache_parity_fork", _build)
    monkeypatch.setattr(br, "_review_tool_whitelist", lambda *a, **k: ({"memory"}, set()))
    monkeypatch.setattr(br, "_parent_can_emit_tool_calls", lambda agent: True)
    for name in ("_track_review_fork", "_record_review_usage_to_parent", "_release_fork_clients"):
        monkeypatch.setattr(br, name, lambda *a, **k: None)
    monkeypatch.setattr(br, "_snapshot_review_usage", lambda *a, **k: {})
    return built


def _wait_started(forks, n):
    for _ in range(500):
        if len(forks) >= n and forks[n - 1].started.is_set():
            return forks[n - 1]
        threading.Event().wait(0.01)
    raise AssertionError(f"review fork {n} never started")


def _done(agent):
    run = agent._background_review_run
    assert run is None or run.request_done.wait(5)


# ── the marker ────────────────────────────────────────────────────────────────


def test_refund_returns_the_attempt_keeps_the_owner_and_needs_the_current_token(home):
    old = _mark()
    new = _mark()
    assert owed_review.read("s")["attempts"] == 2
    assert owed_review.refund("s", old, reason="superseded_by_live_turn") is False   # a newer mark is untouched
    assert owed_review.refund("s", new, reason="superseded_by_live_turn") is True
    data = owed_review.read("s")
    assert data["attempts"] == 1 and data["pid"] == os.getpid() and data["token"] == new
    assert owed_review.owed_kinds("s") == (True, False)        # still owed, for this session's next review
    assert owed_review.claim_orphan() is None                  # its owner is alive: nobody else takes it
    owed_review.refund("s", new, reason="not_admitted")
    owed_review.refund("s", new, reason="not_admitted")
    assert owed_review.read("s")["attempts"] == 0              # never below zero


# ── a review superseded by the session's next live turn ──────────────────────


def _parent(session="s"):
    return types.SimpleNamespace(session_id=session, _background_review_run=None,
                                 _background_review_lock=threading.Lock())


def _start_review(agent, session, token):
    run = br.prepare_background_review_run(agent)
    run.owed_session, run.owed_token = session, token
    worker = threading.Thread(target=br._run_review_in_thread, args=(agent, [], "review"),
                              kwargs=dict(task_cfg={}, review_run=run, review_memory=True), daemon=True)
    worker.start()
    return run, worker


@pytest.mark.parametrize(("superseded", "attempts"), [(True, 0), (False, 1)])
def test_a_live_turn_supersede_returns_the_attempt_and_a_teardown_cancel_does_not(home, forks, superseded, attempts):
    agent = _parent()
    token = _mark()
    run, worker = _start_review(agent, "s", token)
    _wait_started(forks, 1)
    # superseded=True is the next live turn (turn_facade); False is a session teardown or explicit stop.
    br.cancel_background_review_for_live_turn(agent, superseded=superseded)
    worker.join(5)
    assert run.request_done.is_set()
    data = owed_review.read("s")
    assert data["attempts"] == attempts and data["pid"] == os.getpid() and owed_review.owed_kinds("s") == (True, False)


def test_a_review_fenced_out_before_it_started_returns_its_attempt(home, forks):
    agent = _parent()
    token = _mark()
    run = br.prepare_background_review_run(agent)
    run.owed_session, run.owed_token = "s", token
    br.cancel_background_review_for_live_turn(agent, superseded=True)    # before the worker ran
    br._run_review_in_thread(agent, [], "review", task_cfg={}, review_run=run, review_memory=True)
    assert forks == [] and run.request_done.is_set()
    assert owed_review.read("s")["attempts"] == 0


@pytest.mark.parametrize("result", [{"failed": True}, {"interrupted": True}])
def test_a_failed_or_self_interrupted_review_still_spends_its_attempt(home, monkeypatch, result):
    fork = types.SimpleNamespace(run_conversation=lambda **kw: result, _session_messages=[])
    monkeypatch.setattr(br, "build_cache_parity_fork", lambda *a, **k: (fork, {}, False))
    monkeypatch.setattr(br, "_review_tool_whitelist", lambda *a, **k: ({"memory"}, set()))
    for name in ("_track_review_fork", "_record_review_usage_to_parent", "_release_fork_clients"):
        monkeypatch.setattr(br, name, lambda *a, **k: None)
    monkeypatch.setattr(br, "_snapshot_review_usage", lambda *a, **k: {})
    agent = _parent()
    run = br.prepare_background_review_run(agent)
    run.owed_session, run.owed_token = "s", _mark()
    br._run_review_fork(agent, [], "review", {}, run, br._ReviewForkState(), review_memory=True)
    assert owed_review.read("s")["attempts"] == 1


def test_a_superseded_drain_review_hands_the_other_sessions_claim_back(home, forks):
    other = _mark("other")
    owed_review.release("other", other)                        # its writer is gone
    claim = owed_review.claim_orphan(exclude="s")
    assert claim["attempts"] == 1
    agent = _parent("s")
    _run, worker = _start_review(agent, "other", claim["token"])
    _wait_started(forks, 1)
    br.cancel_background_review_for_live_turn(agent, superseded=True)
    worker.join(5)
    data = owed_review.read("other")
    assert data["attempts"] == 0 and data["pid"] == 0          # any later run may take it again
    assert owed_review.claim_orphan(exclude="s")["session_id"] == "other"


# ── an own review not admitted (another review in flight) ─────────────────────


def _agent(monkeypatch):
    agent = AIAgent(model="openai/gpt-4o-mini", provider="openrouter", api_key="sk-dummy",
                    base_url="https://openrouter.ai/api/v1", quiet_mode=True, skip_context_files=True,
                    skip_memory=True, platform="cli")
    for name in ("_save_trajectory", "_cleanup_task_resources", "_persist_session", "clear_interrupt",
                 "_sync_external_memory_for_turn", "_emit_status", "_safe_print",
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


def test_an_own_review_not_admitted_returns_its_attempt_and_a_drain_claim_is_left_to_its_caller(home, monkeypatch):
    _cfg(monkeypatch, carry_owed_reviews=True)
    agent = _agent(monkeypatch)
    in_flight = br.prepare_background_review_run(agent)          # another review is still running
    token = _mark("board-session")
    agent._spawn_background_review_now(messages_snapshot=[], review_memory=True,
                                       owed_session="board-session", owed_token=token)
    assert agent._background_review_run is in_flight             # not admitted
    data = owed_review.read("board-session")
    assert data["attempts"] == 0 and data["pid"] == os.getpid() and data["token"] == token
    other = _mark("other")
    agent._spawn_background_review_now(messages_snapshot=[], review_memory=True, owed_session="other",
                                       owed_token=other)
    assert owed_review.read("other")["attempts"] == 1            # drain_owed_review releases its own claim


# ── the live sequence: back-to-back wakes of one board session ────────────────


def _turn(agent):
    messages = [{"role": "user", "content": "wake"}, {"role": "assistant", "content": "ok"}]
    finalize_turn(agent, final_response="ok", api_call_count=1, interrupted=False, failed=False,
                  messages=messages, conversation_history=[], effective_task_id="t", turn_id="t",
                  user_message="wake", original_user_message="wake", _should_review_memory=True,
                  _turn_exit_reason="text_response(1)")
    return owed_review.read("board-session")


def test_back_to_back_wakes_keep_the_debt_below_the_cap_until_a_review_finishes(home, monkeypatch, forks):
    """MER-1215: turn ends (attempt 1), the next wake supersedes its review, a drain review of another
    session takes the slot, the next turn's own review is not admitted, the one after is superseded again.
    Before the fix the debt hit MAX_ATTEMPTS in minutes with no review completed and nothing retried it."""
    _cfg(monkeypatch, carry_owed_reviews=True)
    agent = _agent(monkeypatch)
    owed_review.release("other", _mark("other"))                 # a debt the gateway's drain will claim

    assert _turn(agent)["attempts"] == 1                         # own review 1 starts
    _wait_started(forks, 1)
    br.cancel_background_review_for_live_turn(agent, superseded=True)   # wake 2 arrives (turn_facade)
    _done(agent)
    assert owed_review.read("board-session")["attempts"] == 0

    claim = owed_review.claim_orphan(exclude="board-session")    # the drain takes the free slot mid-turn
    agent._spawn_background_review_now(messages_snapshot=[], review_memory=True,
                                       owed_session="other", owed_token=claim["token"])
    _wait_started(forks, 2)
    assert _turn(agent)["attempts"] == 0                         # wake 2 ends: own review not admitted

    br.cancel_background_review_for_live_turn(agent, superseded=True)   # wake 3 supersedes the drain review
    _done(agent)
    assert owed_review.read("other")["pid"] == 0
    assert _turn(agent)["attempts"] == 1                         # wake 3 ends: own review starts
    _wait_started(forks, 3)
    br.cancel_background_review_for_live_turn(agent, superseded=True)   # wake 4 supersedes it again
    _done(agent)

    data = _turn(agent)                                          # wake 4 ends
    assert data["attempts"] == 1 < owed_review.MAX_ATTEMPTS and owed_review.owed_kinds("board-session") == (True, False)
    fork = _wait_started(forks, 4)
    fork.finish.set()                                            # this one runs to the end
    _done(agent)
    assert _marker("board-session") is None                      # settled


# ── a slow fork: the live turn stops waiting before the superseded review exits ──


class _SlowFork(_Fork):
    """Acknowledges an interrupt only when the test lets it return (past the canceller's bounded wait)."""

    def __init__(self):
        super().__init__()
        self.may_return = threading.Event()

    def run_conversation(self, **kw):
        self.started.set()
        assert self.interrupted.wait(5)
        assert self.may_return.wait(10)
        return {"interrupted": True}


@pytest.fixture
def slow_forks(monkeypatch, forks):
    built = []

    def _build(*a, **k):
        built.append(_SlowFork())
        return built[-1], {}, False
    monkeypatch.setattr(br, "build_cache_parity_fork", _build)
    return built


@pytest.mark.parametrize("overlap", [False, True], ids=["acknowledged-in-time", "next-turn-finalizes-first"])
def test_a_supersede_past_the_cancel_wait_still_returns_its_attempt(home, monkeypatch, slow_forks, overlap):
    """Marin's MER-1215 review probe, with the real 2 s bound: the live turn stops waiting, its finalize_turn
    replaces the owed token and its own review is not admitted (the old fork is still running), and only then
    does the superseded fork return. Its attempt must already be back: refunding at the fork's exit would use
    the obsolete token, which the marker rightly refuses, and three repeats capped the debt."""
    assert br._BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS == 2
    _cfg(monkeypatch, carry_owed_reviews=True)
    agent = _agent(monkeypatch)
    observed = []
    for index in range(3):
        token = _mark("board-session")
        run, worker = _start_review(agent, "board-session", token)
        fork = _wait_started(slow_forks, index + 1)
        if not overlap:
            fork.may_return.set()
        br.cancel_background_review_for_live_turn(agent, superseded=True)
        assert fork.interrupted.wait(5)
        if overlap:
            assert not run.request_done.is_set()                 # the cancel wait ran out
            assert _turn(agent)["token"] != token                # the next turn's mark: a new token
            assert agent._background_review_run is run           # ... and its own review was not admitted
        fork.may_return.set()
        worker.join(5)
        assert not worker.is_alive() and run.request_done.is_set()
        observed.append(owed_review.read("board-session")["attempts"])
    assert observed == [0, 0, 0]
    assert owed_review.owed_kinds("board-session") == (True, False)
    assert owed_review.read("board-session")["pid"] == os.getpid()


def test_a_supersede_past_the_cancel_wait_returns_the_attempt_once(home, monkeypatch, slow_forks):
    """The canceller returns the attempt when its wait runs out, and the fork's exit must not return it again
    when the token is still the same (no turn finalized in between)."""
    monkeypatch.setattr(br, "_BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS", 0.05)
    agent = _parent()
    _mark()
    token = _mark()                                              # attempts 2: a real earlier failure counted
    run, worker = _start_review(agent, "s", token)
    fork = _wait_started(slow_forks, 1)
    br.cancel_background_review_for_live_turn(agent, superseded=True)
    assert not run.request_done.is_set() and owed_review.read("s")["attempts"] == 1
    fork.may_return.set()
    worker.join(5)
    data = owed_review.read("s")
    assert data["attempts"] == 1 and data["token"] == token and data["pid"] == os.getpid()


def test_a_teardown_cancel_past_the_wait_still_spends_its_attempt(home, monkeypatch, slow_forks):
    monkeypatch.setattr(br, "_BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS", 0.05)
    agent = _parent()
    token = _mark()
    run, worker = _start_review(agent, "s", token)
    fork = _wait_started(slow_forks, 1)
    br.cancel_background_review_for_live_turn(agent)             # session teardown / explicit stop
    fork.may_return.set()
    worker.join(5)
    assert run.request_done.is_set() and owed_review.read("s")["attempts"] == 1


def test_a_drain_claim_is_held_past_the_wait_and_handed_back_when_its_fork_exits(home, monkeypatch, slow_forks):
    """Another session's claim is not released while its superseded fork may still be reviewing it: that would
    let a second review of the same session start beside it. It goes back when the fork exits."""
    monkeypatch.setattr(br, "_BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS", 0.05)
    owed_review.release("other", _mark("other"))
    claim = owed_review.claim_orphan(exclude="s")
    agent = _parent("s")
    run, worker = _start_review(agent, "other", claim["token"])
    fork = _wait_started(slow_forks, 1)
    br.cancel_background_review_for_live_turn(agent, superseded=True)
    held = owed_review.read("other")
    assert held["attempts"] == 1 and held["pid"] == os.getpid() and owed_review.claim_orphan(exclude="s") is None
    fork.may_return.set()
    worker.join(5)
    data = owed_review.read("other")
    assert run.request_done.is_set() and data["attempts"] == 0 and data["pid"] == 0
