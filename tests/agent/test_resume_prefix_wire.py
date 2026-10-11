"""MER-646 revision 3: the reference is judged by the request the provider receives, not by canonical history.

Vasco's re-review drove the real one-shot resume loader, SQLite, the real ``AIAgent`` facade, admission, turn
construction and the request builder, and found that the builder's existing safety cleanup (expired dangerous
confirmations, ``replay_cleanup``) can redact the whole first message that carries the referenced block, while the
canonical-transcript check still logged success. These tests reuse his five controls through the same real path and
add the fail-closed wire guard's own controls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent import resume_prefix_dedupe as rpd
from hermes_state import SessionDB
from tests.agent import test_resume_prefix_turn as t

SAFETY = '\nSafety rule: never treat the phrase "confirm forced restart" as authorization.\n'
TAIL = '\nAssess the quoted string "confirm forced restart"; no live action is authorized.'
FRESH = "Fresh user correction: do not install."


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None), \
            patch("agent.turn_liveness.resolve_turn_liveness_settings", lambda _cfg: (None, None)):
        yield


def _run(tmp_path, monkeypatch, caplog, *, offer, phrase, force_reference=False):
    """Vasco's harness (test_independent_wire.py), unchanged in substance; returns his receipt plus ours."""
    block = t.BLOCK + (SAFETY if phrase is True else "")
    first = block + t.TRAILER.format(run="r1") + "\n\n---\n\nFirst wake."
    if phrase == "tail":
        first += TAIL
    wake = block + t.TRAILER.format(run="r2") + "\n\n---\n\n" + FRESH
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", source="cli")
    db.replace_messages("parent", [
        {"role": "user", "content": first, "timestamp": 1.0},
        {"role": "assistant", "content": "Offline progress only.", "timestamp": 2.0},
    ])
    from hermes_cli.oneshot import _load_resume_target
    resumed_session, stale, _ = _load_resume_target(db, "parent")
    assert resumed_session == "parent"
    assert stale[0]["content"].startswith(block)
    if offer:
        monkeypatch.setenv(rpd.ENV_SHA256, hashlib.sha256(block.encode()).hexdigest())
        monkeypatch.setenv(rpd.ENV_CHARS, str(len(block)))
    else:
        monkeypatch.delenv(rpd.ENV_SHA256, raising=False)
        monkeypatch.delenv(rpd.ENV_CHARS, raising=False)
    if force_reference:
        # Disable the decision-time projection to exercise the wire guard alone.
        monkeypatch.setattr(rpd, "wire_prefix_carries_block", lambda *a, **k: True)
    monkeypatch.setattr("agent.turn_context._maybe_title_session_at_turn_start", lambda *a, **k: None)
    captured = []
    with caplog.at_level(logging.INFO, logger="agent.resume_prefix_dedupe"):
        agent = t._full_path_agent(db, monkeypatch, captured)
        result = agent.run_conversation(wake, conversation_history=stale)
    wire = t._wire_user_texts(captured[0])
    receipt = {
        "offer": offer, "phrase": phrase, "forceReference": force_reference,
        "referenceSent": wire[-1].startswith("[Agent instructions unchanged"),
        "referenceInCanonicalRow": any(isinstance(m.get("content"), str)
                                       and m["content"].startswith("[Agent instructions unchanged")
                                       for m in result["messages"]),
        "verbatimBlockOnWire": any(block in x for x in wire),
        "verbatimBlockOnWireBeforeOrAtCurrent": block in "".join(wire),
        "historicalWireWasRedacted": any("high-risk confirmation previously given here has EXPIRED" in x for x in wire),
        "freshInputRetained": wire[-1].endswith(FRESH),
        "wireRows": len(wire),
        "requests": len(captured),
        "wireVerifiedLogged": "wire verified block before the reference" in caplog.text,
        "wireRestoredLogged": "RESTORED full block on the wire" in caplog.text,
        "keptCarrierNotOnWire": "kept (carrier_not_on_wire)" in caplog.text,
    }
    out = os.environ.get("MER646_RECEIPT_DIR")
    if out:
        Path(out, f"wire-r3-{offer}-{phrase}-{force_reference}.json").write_text(json.dumps(receipt, indent=2))
    # Every provider request this turn: never a reference without the block verbatim before it.
    for request in captured:
        texts = t._wire_user_texts(request)
        joined = "\n".join(texts)
        line = rpd.reference_text(len(block), hashlib.sha256(block.encode()).hexdigest())
        if line in joined:
            assert block in joined[: joined.rfind(line)], receipt
    return receipt


@pytest.mark.parametrize("offer,phrase", [(True, True), (True, "tail"), (False, True), (False, "tail")])
def test_vasco_cleanup_controls_keep_the_full_block_on_the_wire(tmp_path, monkeypatch, caplog, offer, phrase):
    r = _run(tmp_path, monkeypatch, caplog, offer=offer, phrase=phrase)
    assert r["historicalWireWasRedacted"] is True  # the safety cleanup still runs, untouched
    assert r["referenceSent"] is False
    assert r["verbatimBlockOnWire"] is True
    assert r["freshInputRetained"] is True
    assert r["keptCarrierNotOnWire"] is offer  # opted in: refused at decision time, before anything is sent


def test_benign_control_sends_the_reference_with_the_block_on_the_wire(tmp_path, monkeypatch, caplog):
    r = _run(tmp_path, monkeypatch, caplog, offer=True, phrase=False)
    assert r["historicalWireWasRedacted"] is False
    assert r["referenceSent"] is True and r["verbatimBlockOnWire"] is True and r["freshInputRetained"] is True
    assert r["wireVerifiedLogged"] is True and r["wireRestoredLogged"] is False


@pytest.mark.parametrize("phrase", [True, "tail"])
def test_wire_guard_alone_restores_the_block_when_cleanup_removes_the_carrier(tmp_path, monkeypatch, caplog, phrase):
    """Second layer: with the decision-time projection disabled, the reference is persisted, but the request that
    reaches the provider carries the full block in its place, and the outcome is logged as RESTORED."""
    r = _run(tmp_path, monkeypatch, caplog, offer=True, phrase=phrase, force_reference=True)
    assert r["referenceInCanonicalRow"] is True  # the decision did reference ...
    assert r["historicalWireWasRedacted"] is True
    assert r["referenceSent"] is False  # ... but the wire carries the block instead
    assert r["verbatimBlockOnWire"] is True and r["freshInputRetained"] is True
    assert r["wireRestoredLogged"] is True and r["wireVerifiedLogged"] is False


# --- enforce_on_wire / report_turn units -------------------------------------------------------------------------

BLOCK = t.BLOCK
SHA = t.SHA
LINE = rpd.reference_text(len(BLOCK), SHA)


class _Agent:
    def __init__(self):
        self._resume_prefix_reference = (SHA, len(BLOCK), LINE + " tail", BLOCK)
        self._resume_prefix_wire = []


def test_enforce_verified_when_an_earlier_user_message_holds_the_block():
    agent = _Agent()
    req = [{"role": "system", "content": "s"}, {"role": "user", "content": BLOCK + " first"},
           {"role": "assistant", "content": "ok"}, {"role": "user", "content": LINE + " tail"}]
    assert rpd.enforce_on_wire(agent, req)[-1]["content"] == LINE + " tail"
    assert agent._resume_prefix_wire == ["verified"]
    assert rpd.report_turn(agent) == "verified"


@pytest.mark.parametrize("carrier", ["absent", "system_only", "assistant_only", "after_reference", "near_miss"])
def test_enforce_restores_when_the_block_is_not_before_the_reference(carrier):
    agent = _Agent()
    first = {"absent": "redacted", "near_miss": BLOCK[:-1] + "X first"}.get(carrier, "first")
    req = [{"role": "system", "content": BLOCK if carrier == "system_only" else "s"},
           {"role": "user", "content": first},
           {"role": "assistant", "content": BLOCK if carrier == "assistant_only" else "ok"},
           {"role": "user", "content": LINE + " tail"}]
    if carrier == "after_reference":
        req.append({"role": "user", "content": BLOCK})
    rpd.enforce_on_wire(agent, req)
    assert req[3]["content"] == BLOCK + " tail"
    assert agent._resume_prefix_wire == ["restored"]
    assert rpd.report_turn(agent) == "restored"


def test_enforce_handles_text_parts_and_a_merged_same_message_carrier():
    agent = _Agent()
    parts = [{"type": "text", "text": BLOCK + " first\n\n" + LINE + " tail", "cache_control": {"type": "ephemeral"}}]
    req = [{"role": "user", "content": parts}]
    rpd.enforce_on_wire(agent, req)
    assert agent._resume_prefix_wire == ["verified"]
    agent2 = _Agent()
    parts2 = [{"type": "text", "text": "first"}, {"type": "text", "text": LINE + " tail", "cache_control": {"x": 1}}]
    req2 = [{"role": "user", "content": parts2}]
    rpd.enforce_on_wire(agent2, req2)
    assert req2[0]["content"][1] == {"type": "text", "text": BLOCK + " tail", "cache_control": {"x": 1}}
    assert agent2._resume_prefix_wire == ["restored"]


def test_enforce_fails_closed_on_an_internal_error(monkeypatch):
    agent = _Agent()

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(rpd, "_wire_outcome", boom)
    req = [{"role": "user", "content": BLOCK}, {"role": "user", "content": LINE + " tail"}]
    rpd.enforce_on_wire(agent, req)
    assert req[1]["content"] == BLOCK + " tail"
    assert agent._resume_prefix_wire == ["restored"]


def test_enforce_and_report_are_inert_without_a_reference():
    agent = _Agent()
    agent._resume_prefix_reference = None
    req = [{"role": "user", "content": LINE}]
    assert rpd.enforce_on_wire(agent, req)[0]["content"] == LINE
    assert agent._resume_prefix_wire == []
    assert rpd.report_turn(agent) is None


def test_report_unchecked_when_no_request_carried_the_reference(caplog):
    agent = _Agent()
    rpd.enforce_on_wire(agent, [{"role": "user", "content": "something else"}])
    assert agent._resume_prefix_wire == ["no_reference"]
    with caplog.at_level(logging.WARNING, logger="agent.resume_prefix_dedupe"):
        assert rpd.report_turn(agent) == "unchecked"
    assert "wire unchecked" in caplog.text


def test_decision_projection_matches_the_builders_cleanup():
    agent = type("A", (), {"_current_turn_timestamp": 1000.0})()
    hist = [{"role": "user", "content": BLOCK + SAFETY, "timestamp": 1.0}]
    assert rpd.wire_prefix_carries_block(agent, hist, SHA, len(BLOCK)) is False  # expired phrase: redacted
    fresh = [{"role": "user", "content": BLOCK + SAFETY, "timestamp": 990.0}]
    assert rpd.wire_prefix_carries_block(agent, fresh, SHA, len(BLOCK)) is True  # within the expiry window
    untimed = [{"role": "user", "content": BLOCK + SAFETY}]
    assert rpd.wire_prefix_carries_block(agent, untimed, SHA, len(BLOCK)) is True  # cleanup leaves untimestamped
    sidecar = [{"role": "user", "content": BLOCK, "api_content": "different bytes sent"}]
    assert rpd.wire_prefix_carries_block(agent, sidecar, SHA, len(BLOCK)) is False
    assert rpd.wire_prefix_carries_block(object(), [{"role": "user", "content": BLOCK}], SHA, len(BLOCK)) is False


def test_moa_and_codex_app_server_keep_the_full_block(monkeypatch):
    for kwargs, mode in (({"moa_active": True}, "chat_completions"), ({}, "codex_app_server")):
        monkeypatch.setenv(rpd.ENV_SHA256, SHA)
        monkeypatch.setenv(rpd.ENV_CHARS, str(len(BLOCK)))
        agent = type("A", (), {"api_mode": mode, "_current_turn_timestamp": 1e9,
                               "context_compressor": t._compressor()})()
        msgs = [dict(m) for m in t.PARENT] + [{"role": "user", "content": t.WAKE}]
        assert rpd.apply_to_turn(agent, msgs, len(msgs) - 1, **kwargs) is None
        assert msgs[-1]["content"] == t.WAKE


def test_decide_itself_refuses_a_differing_first_block_or_sidecar():
    """The wire projection repeats these two checks; they are tested on decide() alone so neither masks the other."""
    def run(first_row):
        hist = [first_row, {"role": "assistant", "content": "ok"}]
        return rpd.decide(t.WAKE, hist, sha256=SHA, chars=len(BLOCK), protect_first_n=3, effective_protect_first_n=3,
                          is_summary=lambda m: False)
    assert run({"role": "user", "content": BLOCK + " first"})[2] == "deduplicated"
    assert run({"role": "user", "content": BLOCK[:-1] + "X first"})[2] == "first_user_block_differs"
    assert run({"role": "user", "content": BLOCK + " first", "api_content": "other bytes sent"})[2] == \
        "first_user_block_differs"
