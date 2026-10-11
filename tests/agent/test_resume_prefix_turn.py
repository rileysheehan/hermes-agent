"""MER-646 revision 2: the resumed-instruction reference is decided on the turn's FINAL canonical rows.

Vasco's review reproduced, with a real SessionDB and a real uncontended turn-lease admission, a turn whose
pre-admission history still held the instruction block while turn construction adopted a compression child that
did not: revision 1 sent the reference and the block was nowhere in the canonical transcript. These tests drive the
real ``build_turn_context`` (real rotated-session recovery/adoption, real SQLite) and check the receipt Vasco used:
``referenceSent``, ``verbatimBlockPresent`` and the pinned head.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import types
from unittest.mock import patch

import pytest

from agent import resume_prefix_dedupe as rpd
from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX
from agent.turn_facade_lease import admit_durable_turn_lease
from hermes_state import SessionDB
from tests.agent.test_turn_context import _FakeAgent, _build

BLOCK = "# Agent instructions\n" + ("Keep every approval gate. " * 40)
SHA = hashlib.sha256(BLOCK.encode()).hexdigest()
TRAILER = "\nThe above agent instructions were loaded from /runs/{run}/AGENTS.md."
FIRST = BLOCK + TRAILER.format(run="r1") + "\n\n---\n\nFirst wake."
WAKE = BLOCK + TRAILER.format(run="r2") + "\n\n---\n\n## Paperclip Wake Payload\nAnswer the new review comment."
PARENT = [
    {"role": "user", "content": FIRST},
    {"role": "assistant", "content": "Done."},
    {"role": "user", "content": "[Board result] noted"},
    {"role": "assistant", "content": "ok"},
]


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    # Hermetic: no aux-client global, and no liveness watchdog read from the user's config.
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None), \
            patch("agent.turn_liveness.resolve_turn_liveness_settings", lambda _cfg: (None, None)):
        yield


@pytest.fixture
def opt_in(monkeypatch):
    monkeypatch.setenv(rpd.ENV_SHA256, SHA)
    monkeypatch.setenv(rpd.ENV_CHARS, str(len(BLOCK)))


def _compressor():
    c = object.__new__(ContextCompressor)
    c.protect_first_n = 3
    c.protect_last_n = 20
    c.compression_count = 0
    c._previous_summary = None
    c.should_compress = lambda tokens=None: False
    c.should_compress_info = lambda tokens=None: (False, None)
    return c


def _agent(db, session_id):
    agent = _FakeAgent()
    agent.session_id = session_id
    agent._session_db = db
    agent.context_compressor = _compressor()
    agent._session_persist_lock = threading.RLock()
    return agent


def _receipt(agent, ctx):
    rows = [m for m in ctx.messages if m.get("role") != "system"]
    sent = ctx.messages[ctx.current_turn_user_idx]["content"]
    return {
        "session": agent.session_id,
        "referenceSent": sent.startswith("[Agent instructions unchanged"),
        "verbatimBlockPresent": any(isinstance(m.get("content"), str) and m["content"].startswith(BLOCK) for m in rows),
        "pinnedHeadRows": int(getattr(agent.context_compressor, "_pinned_head_rows", 0) or 0),
    }


def _seed_parent(db):
    db.create_session("parent", source="cli")
    db.replace_messages("parent", PARENT)


def _rotate(db):
    """Another holder finishes compression: parent ended 'compression', live child holds summary + tail only."""
    db.end_session("parent", "compression")
    db.create_session("child", source="cli", parent_session_id="parent")
    db.replace_messages("child", [
        {"role": "user", "content": SUMMARY_PREFIX + " earlier work, including the agent instructions"},
        {"role": "assistant", "content": "ok"},
    ])


def test_rotation_before_uncontended_admission_keeps_the_full_block(tmp_path, opt_in, caplog):
    """Vasco's counterexample: history loaded from the parent, rotation completes, admission does not wait."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_parent(db)
    stale = db.get_messages_as_conversation("parent")
    assert stale[0]["content"].startswith(BLOCK)
    _rotate(db)
    agent = _agent(db, "parent")
    admission = admit_durable_turn_lease(
        agent, session_id="parent", relay_turn_id="t1", task_context={"platform": "cli", "session_id": "parent"},
        conversation_history=stale,
    )
    try:
        assert admission.early_result is None
        assert admission.conversation_history is stale  # uncontended: the stale parent history is kept
        with caplog.at_level(logging.INFO, logger="agent.resume_prefix_dedupe"):
            ctx = _build(agent, user_message=WAKE, conversation_history=admission.conversation_history)
    finally:
        if admission.lease is not None:
            admission.lease.release()
    assert _receipt(agent, ctx) == {
        "session": "child", "referenceSent": False, "verbatimBlockPresent": True, "pinnedHeadRows": 0,
    }
    assert ctx.messages[ctx.current_turn_user_idx]["content"] == WAKE
    assert ctx.user_message == WAKE
    assert "kept (compacted_session)" in caplog.text
    assert getattr(agent, "_resume_prefix_reference", None) is None


def test_unrotated_resumed_session_sends_the_reference_with_the_block_in_canonical_rows(tmp_path, opt_in):
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_parent(db)
    history = db.get_messages_as_conversation("parent")
    agent = _agent(db, "parent")
    admission = admit_durable_turn_lease(
        agent, session_id="parent", relay_turn_id="t1", task_context={"platform": "cli", "session_id": "parent"},
        conversation_history=history,
    )
    try:
        ctx = _build(agent, user_message=WAKE, conversation_history=admission.conversation_history)
    finally:
        if admission.lease is not None:
            admission.lease.release()
    assert _receipt(agent, ctx) == {
        "session": "parent", "referenceSent": True, "verbatimBlockPresent": True, "pinnedHeadRows": 1,
    }
    expected = rpd.reference_text(len(BLOCK), SHA) + WAKE[len(BLOCK):]
    assert ctx.messages[ctx.current_turn_user_idx]["content"] == expected
    assert ctx.user_message == expected and ctx.original_user_message == expected
    assert agent._resume_prefix_reference == (SHA, len(BLOCK), expected, BLOCK)
    # The canonical-transcript diagnostic run_conversation performs after the turn.
    assert rpd.verify_final_transcript(agent, ctx.messages) is True


def test_turn_start_compaction_keeps_the_full_block(tmp_path, opt_in, caplog):
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_parent(db)
    history = db.get_messages_as_conversation("parent")
    agent = _agent(db, "parent")
    from agent import turn_context_compaction as tcc
    real = tcc.run_turn_start_compaction

    def _compressed(*a, **k):
        out = real(*a, **k)
        return types.SimpleNamespace(**{**out.__dict__, "compressed": True}) if hasattr(out, "__dict__") else out

    with patch("agent.turn_context_compaction.run_turn_start_compaction", side_effect=_compressed), \
            caplog.at_level(logging.INFO, logger="agent.resume_prefix_dedupe"):
        ctx = _build(agent, user_message=WAKE, conversation_history=history)
    assert ctx.messages[ctx.current_turn_user_idx]["content"] == WAKE
    assert "kept (compacted_this_turn)" in caplog.text


@pytest.mark.parametrize("case", ["already_persisted", "api_content", "staged_cli_input", "no_row"])
def test_apply_to_turn_keeps(opt_in, case):
    # With the turn clock, so the wire projection would pass and each named guard is what keeps the block.
    agent = types.SimpleNamespace(context_compressor=_compressor(), _current_turn_timestamp=1e9)
    messages = [dict(m) for m in PARENT] + [{"role": "user", "content": WAKE}]
    idx = len(messages) - 1
    staged = False
    if case == "already_persisted":
        messages[idx]["_row_id"] = 7
    elif case == "api_content":
        messages[idx]["api_content"] = WAKE
    elif case == "staged_cli_input":
        staged = True
    elif case == "no_row":
        idx = len(messages)
    assert rpd.apply_to_turn(agent, messages, idx, staged_cli_input=staged) is None
    assert messages[-1]["content"] == WAKE
    assert not getattr(agent.context_compressor, "_pinned_head_rows", 0)
    # The opt-in is consumed even when kept: a later turn of the same process never sees it.
    assert rpd.ENV_SHA256 not in __import__("os").environ


def test_canonical_diagnostic_flags_a_lost_block(opt_in, caplog):
    agent = types.SimpleNamespace(context_compressor=_compressor(), _current_turn_timestamp=1e9)
    messages = [dict(m) for m in PARENT] + [{"role": "user", "content": WAKE}]
    sent = rpd.apply_to_turn(agent, messages, len(messages) - 1)
    assert sent is not None
    # A transcript in which the head copy has gone (what revision 1 allowed through adoption).
    lost = [{"role": "user", "content": SUMMARY_PREFIX + " s"}, {"role": "assistant", "content": "ok"},
            {"role": "user", "content": sent}]
    with caplog.at_level(logging.WARNING, logger="agent.resume_prefix_dedupe"):
        assert rpd.verify_final_transcript(agent, lost) is False
    assert "canonical transcript lacks the referenced block" in caplog.text
    assert rpd.verify_final_transcript(types.SimpleNamespace(), messages) is None


def test_revision_1_ordering_reproduces_the_loss(tmp_path, opt_in):
    """Control for the harness: deciding on the pre-admission history (revision 1's call site) loses the block
    exactly as Vasco's receipt showed; the same inputs through revision 2's call site keep it (test above)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_parent(db)
    stale = db.get_messages_as_conversation("parent")
    _rotate(db)
    agent = _agent(db, "parent")
    agent._current_turn_timestamp = 1e9  # revision 1 had no wire projection; the stale rows are untimestamped
    early = rpd.apply_resume_prefix_dedupe(agent, WAKE, stale)  # revision 1: before turn construction
    ctx = _build(agent, user_message=early, conversation_history=stale)
    assert _receipt(agent, ctx) == {
        "session": "child", "referenceSent": True, "verbatimBlockPresent": False, "pinnedHeadRows": 1,
    }


def _full_path_agent(db, monkeypatch, captured):
    """A real AIAgent through the real facade (lease admission) and loop; only the provider is faked."""
    from types import SimpleNamespace

    # A git worktree's .git points into the main checkout; the import-time pull recovery probe would touch it.
    monkeypatch.setattr("hermes_cli._early_recovery.restore_interrupted_pull", lambda *a, **k: False)
    from run_agent import AIAgent

    class _Completions:
        def create(self, **kw):
            captured.append([dict(m) for m in kw.get("messages") or []])
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None, reasoning=None, reasoning_content=None),
                finish_reason="stop")], usage=None, model="test-model")

    monkeypatch.setattr("agent.process_bootstrap.OpenAI",
                        lambda **_kw: SimpleNamespace(chat=SimpleNamespace(completions=_Completions())))
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: [])
    agent = AIAgent(model="test-model", api_key="k", base_url="http://localhost:1/v1", platform="cli",
                    max_iterations=2, quiet_mode=True, skip_memory=True, session_db=db, session_id="parent")
    agent._disable_streaming = True
    return agent


def _wire_user_texts(request):
    out = []
    for m in request:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        out.append(c if isinstance(c, str) else "".join(p.get("text", "") for p in c if isinstance(p, dict)))
    return out


@pytest.mark.parametrize("rotated", [True, False])
def test_full_path_through_the_facade(tmp_path, opt_in, monkeypatch, caplog, rotated):
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_parent(db)
    stale = db.get_messages_as_conversation("parent")
    if rotated:
        _rotate(db)
    captured = []
    agent = _full_path_agent(db, monkeypatch, captured)
    with caplog.at_level(logging.INFO, logger="agent.resume_prefix_dedupe"):
        result = agent.run_conversation(WAKE, conversation_history=stale)
    assert captured, "the fake provider was never called"
    wire = _wire_user_texts(captured[0])
    current = wire[-1]
    if rotated:
        assert agent.session_id == "child"
        assert current.startswith(BLOCK)  # full block on the wire
        assert "[Agent instructions unchanged" not in current
        assert "kept (compacted_session)" in caplog.text
    else:
        assert current.startswith("[Agent instructions unchanged")
        assert any(t.startswith(BLOCK) for t in wire[:-1])  # the referenced copy is in the same request
        assert "wire verified block before the reference in 1/1 requests" in caplog.text
        assert "canonical transcript retains the block" in caplog.text
    # Persisted canonical transcript: every clause of the block is verbatim somewhere the model sees.
    stored = db.get_messages_as_conversation(agent.session_id)
    assert any(isinstance(m.get("content"), str) and m["content"].startswith(BLOCK) for m in stored)
    assert "RESTORED" not in caplog.text and "lacks the referenced block" not in caplog.text
    assert isinstance(result, dict)
