"""A stored system prompt cleared out of process must reach the next gateway turn.

``hermes sessions repair-prompts SESSION_ID --apply`` stores NULL and documents that "the next turn
rebuilds". A gateway that already holds the session's AIAgent in its cache kept serving the
in-memory prompt instead (prefix-cache reuse), so the clear only took effect after the idle TTL or a
restart. The turn's cache lookup now treats "row has messages, stored prompt NULL, cached agent
holds a prompt" like the cross-process message_count guard: evict and rebuild.
"""

import logging
import threading
from types import SimpleNamespace

from hermes_state import AsyncSessionDB, SessionDB

from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext

KEY = "agent:main:telegram:dm:1"


def _runner(db):
    runner = SimpleNamespace(
        _agent_cache={},
        _agent_cache_lock=threading.Lock(),
        _session_db=AsyncSessionDB(db),
        _init_cached_agent_for_turn=lambda agent, depth: None,
    )
    return runner


def _turn(runner, session_id="s1"):
    return TurnRunner(runner, TurnContext(session_id=session_id, session_key=KEY))


def _session(tmp_path, *, prompt: str | None = "OLD PROMPT", messages=2):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="telegram")
    for i in range(messages):
        db.append_message("s1", role="user" if i % 2 == 0 else "assistant", content=f"m{i}")
    if prompt is not None:
        db.update_system_prompt("s1", prompt)
    return db


def _cache(runner, agent, db, sid="s1"):
    row = db.get_session("s1")
    runner._agent_cache[KEY] = (agent, "sig", row.get("message_count", 0), sid)


def _lookup(turn, runner):
    msg_count, cleared = turn._session_row_guard()
    return turn._lookup_cached_agent(
        "sig", runner._agent_cache_lock, runner._agent_cache, 90, None, False, msg_count, cleared,
    )


def test_row_guard_reports_a_cleared_prompt(tmp_path):
    db = _session(tmp_path)
    turn = _turn(_runner(db))
    assert turn._session_row_guard() == (2, False)
    db.update_system_prompt("s1", None)  # exactly what repair-prompts --apply writes
    assert turn._session_row_guard() == (2, True)
    assert turn._current_message_count() == 2  # unchanged contract for existing callers


def test_cleared_prompt_evicts_the_cached_agent(tmp_path, caplog):
    db = _session(tmp_path)
    runner = _runner(db)
    agent = SimpleNamespace(_cached_system_prompt="OLD PROMPT", max_iterations=1)
    _cache(runner, agent, db)
    db.update_system_prompt("s1", None)

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        found = _lookup(_turn(runner), runner)

    assert found.agent is None and found.reused is False
    assert found.evicted is agent
    assert KEY not in runner._agent_cache
    assert "stored system prompt for s1 was cleared out of process" in caplog.text


def test_stored_prompt_present_still_reuses(tmp_path):
    db = _session(tmp_path)
    runner = _runner(db)
    agent = SimpleNamespace(_cached_system_prompt="OLD PROMPT", max_iterations=1)
    _cache(runner, agent, db)

    found = _lookup(_turn(runner), runner)

    assert found.agent is agent and found.reused is True
    assert runner._agent_cache[KEY][0] is agent


def test_agent_without_an_in_memory_prompt_is_reused(tmp_path):
    """Nothing stale to serve: the reused agent builds from the NULL row on its own."""
    db = _session(tmp_path, prompt=None)
    runner = _runner(db)
    agent = SimpleNamespace(_cached_system_prompt=None, max_iterations=1)
    _cache(runner, agent, db)

    found = _lookup(_turn(runner), runner)

    assert found.agent is agent and found.reused is True


def test_new_session_without_messages_is_not_a_clear(tmp_path):
    db = _session(tmp_path, prompt=None, messages=0)
    turn = _turn(_runner(db))
    assert turn._session_row_guard() == (0, False)


def test_other_session_id_under_the_same_key_is_not_judged(tmp_path):
    """#54947: a snapshot from another conversation says nothing about this row."""
    db = _session(tmp_path)
    db.create_session("s2", source="telegram")
    db.append_message("s2", role="user", content="x")  # s2: messages, no stored prompt
    runner = _runner(db)
    agent = SimpleNamespace(_cached_system_prompt="OLD PROMPT", max_iterations=1)
    _cache(runner, agent, db, sid="s1")

    found = _lookup(_turn(runner, session_id="s2"), runner)

    assert found.agent is agent and found.reused is True


def test_unreadable_row_fails_safe_to_reuse(tmp_path):
    db = _session(tmp_path)
    runner = _runner(db)
    agent = SimpleNamespace(_cached_system_prompt="OLD PROMPT", max_iterations=1)
    _cache(runner, agent, db)

    def boom(_sid):
        raise RuntimeError("database is locked")

    runner._session_db._db.get_session = boom
    turn = _turn(runner)
    assert turn._session_row_guard() == (None, False)
    found = turn._lookup_cached_agent(
        "sig", runner._agent_cache_lock, runner._agent_cache, 90, None, False, None, False,
    )
    assert found.agent is agent
