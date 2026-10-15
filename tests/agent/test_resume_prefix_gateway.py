"""MER-963: the MER-646 resume-prefix reference on a gateway-run finite turn.

In the gateway the opt-in arrives through the turn's forwarded client env (gateway.client_env_passthrough),
and the gateway always hands its clean persist text through (equal to the message). Both must still give
the reference, and the transcript must hold exactly what the model received.
"""
from __future__ import annotations

from agent import resume_prefix_dedupe as rpd
from agent.turn_facade_lease import admit_durable_turn_lease
from gateway import session_client_env as sce
from hermes_state import SessionDB
from tests.agent.test_resume_prefix_turn import BLOCK, SHA, WAKE, _agent, _build, _receipt, _seed_parent
from tests.agent.test_resume_prefix_turn import _stub_runtime_main  # autouse fixture


def _turn(tmp_path, *, persist):
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_parent(db)
    history = db.get_messages_as_conversation("parent")
    agent = _agent(db, "parent")
    admission = admit_durable_turn_lease(
        agent, session_id="parent", relay_turn_id="t1", task_context={"platform": "local", "session_id": "parent"},
        conversation_history=history,
    )
    try:
        ctx = _build(agent, user_message=WAKE, conversation_history=admission.conversation_history,
                     persist_user_message=persist)
    finally:
        if admission.lease is not None:
            admission.lease.release()
    return agent, ctx


def test_forwarded_opt_in_and_equal_persist_text_send_and_store_the_reference(tmp_path, monkeypatch):
    monkeypatch.delenv(rpd.ENV_SHA256, raising=False)
    monkeypatch.delenv(rpd.ENV_CHARS, raising=False)
    sce.stash("parent", "req1", {rpd.ENV_SHA256: SHA, rpd.ENV_CHARS: str(len(BLOCK))})
    with sce.client_env_scope({"target_session_id": "parent", "request_id": "req1"}):
        agent, ctx = _turn(tmp_path, persist=WAKE)
    expected = rpd.reference_text(len(BLOCK), SHA) + WAKE[len(BLOCK):]
    assert _receipt(agent, ctx)["referenceSent"] is True
    assert ctx.messages[ctx.current_turn_user_idx]["content"] == expected
    # The persisted row is what the model received, not the full block the gateway handed through.
    assert agent._persist_user_message_override == expected


def test_a_different_persist_text_still_keeps_the_full_block(tmp_path, monkeypatch, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="agent.resume_prefix_dedupe")
    sce.stash("parent", "req1", {rpd.ENV_SHA256: SHA, rpd.ENV_CHARS: str(len(BLOCK))})
    with sce.client_env_scope({"target_session_id": "parent", "request_id": "req1"}):
        agent, ctx = _turn(tmp_path, persist="clean text the caller stores instead")
    assert _receipt(agent, ctx)["referenceSent"] is False
    assert "kept (separate persisted text)" in caplog.text


def test_without_a_forwarded_opt_in_nothing_changes(tmp_path, monkeypatch):
    monkeypatch.delenv(rpd.ENV_SHA256, raising=False)
    monkeypatch.delenv(rpd.ENV_CHARS, raising=False)
    agent, ctx = _turn(tmp_path, persist=WAKE)
    assert _receipt(agent, ctx)["referenceSent"] is False
    assert ctx.messages[ctx.current_turn_user_idx]["content"] == WAKE
