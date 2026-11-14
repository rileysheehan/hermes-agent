"""A finite (-q/-Q) classic CLI resumes without the transcript (``session.resume omit_messages``).

The whole transcript in one ``session.resume`` frame can exceed the client's 8 MiB ``max_size`` (close 1009), after
which that session never starts again. A finite CLI restores its history from the local SessionDB and never shows the
snapshot's transcript, so its startup attach, reconnect resubscribe and replay-gap re-attach all ask the owner to omit
it, and a snapshot that omitted it keeps what the viewer already held. Interactive viewers keep the full snapshot.
Scripted gateway, no live one.
"""
import asyncio
import types
from contextlib import asynccontextmanager

import pytest

from hermes_cli.gateway_connection import GatewayConnection, SessionState
from hermes_cli.gateway_session_store import GatewaySessionStore

ROWS = [{"role": "user", "content": "hello", "row_id": 1}, {"role": "assistant", "content": "hi", "row_id": 2}]


def snapshot(sid, params):
    omitted = bool(params.get("omit_messages"))
    return {"session_id": sid, "stored_session_id": sid, "revision": 7, "execution_generation": 3,
            "subscription_id": "sub-" + sid, "info": {"model": "m", "title": "T"},
            "pending": [{"admission_id": "adm-1"}], "prompts": [],
            "messages": [] if omitted else list(ROWS), "message_count": len(ROWS), "messages_omitted": omitted}


class FakeClient:
    def __init__(self):
        self.events = asyncio.Queue()
        self.calls = []

    async def rpc(self, method, *, _timeout=None, **params):
        self.calls.append((method, params))
        if method == "session.resume":
            return snapshot(params["session_id"], params)
        return {"status": "ok"}


@pytest.fixture
def store():
    client = FakeClient()

    @asynccontextmanager
    async def connect():
        yield client
    conn = GatewayConnection(connect=connect)
    conn.start(5)
    yield GatewaySessionStore(conn), client
    conn.close()


def resumes(client):
    return [params for method, params in client.calls if method == "session.resume"]


def test_finite_cli_startup_resume_omits_the_transcript(store, monkeypatch):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    db, client = store
    assert db.open_session(types.SimpleNamespace(session_id="local-a", _resumed=True)) == "local-a"
    (params,) = resumes(client)
    assert params == {"session_id": "local-a", "omit_messages": True}
    state = db.connection.state("local-a")
    # Same stored session, generation, revision, subscription and pending controls as a full resume would give.
    assert (state.revision, state.execution_generation, state.subscription_id) == (7, 3, "sub-local-a")
    assert state.pending == [{"admission_id": "adm-1"}]
    assert state.message_count == len(ROWS) and state.messages == []


def test_interactive_cli_keeps_the_full_snapshot(store, monkeypatch):
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    db, client = store
    db.open_session(types.SimpleNamespace(session_id="local-b", _resumed=True))
    (params,) = resumes(client)
    assert "omit_messages" not in params
    assert db.connection.state("local-b").messages == ROWS


def test_finite_reattach_and_keyed_resume_omit_it_too(store, monkeypatch):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    db, client = store
    db.open_session(types.SimpleNamespace(session_id="local-c", _resumed=True), {"api_key": "k"})
    db.connection.reattach("local-c")
    first, second = resumes(client)
    assert first == {"session_id": "local-c", "api_key": "k", "omit_messages": True}
    assert second.get("omit_messages") is True and second["session_id"] == "local-c"


def test_reconnect_resubscribe_omits_it_for_a_finite_viewer():
    client = FakeClient()
    conn = GatewayConnection(connect=None)
    conn.omit_resume_messages = True
    assert asyncio.run(conn._resume(client, "local-d"))["messages_omitted"] is True
    assert resumes(client) == [{"session_id": "local-d", "omit_messages": True}]
    conn.omit_resume_messages = False
    asyncio.run(conn._resume(client, "local-d"))
    assert resumes(client)[-1] == {"session_id": "local-d"}


def test_an_omitted_snapshot_keeps_the_viewers_transcript_and_a_full_one_replaces_it():
    state = SessionState("local-e")
    state.adopt_snapshot(snapshot("local-e", {}))
    assert state.messages == ROWS and state.message_count == 2
    state.adopt_snapshot(snapshot("local-e", {"omit_messages": True}))
    assert state.messages == ROWS, "an omission is not an empty chat"
    full = snapshot("local-e", {})
    full["messages"], full["message_count"] = [], 0
    state.adopt_snapshot(full)
    assert state.messages == [] and state.message_count == 0


def test_reset_session_state_counts_an_omitted_transcript_as_history(monkeypatch):
    from hermes_cli import gateway_agent
    state = SessionState("local-f")
    state.adopt_snapshot(snapshot("local-f", {"omit_messages": True}))
    refreshed = []
    agent = types.SimpleNamespace(_reset_session_figures=lambda: None, _state=lambda: state,
                                  _refresh_stats=lambda: refreshed.append(1) or True)
    gateway_agent.GatewayAgent.reset_session_state(agent)
    assert refreshed == [1], "a resumed session with an omitted transcript still reads its figures"
