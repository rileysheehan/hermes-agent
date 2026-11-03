"""cua-driver session recovery: a dead ``cua-driver mcp`` child (the SDK's ``MCPError`` "Connection closed") and an
ended IMPLICIT session on a call that carries no ``session`` label (``bring_to_front``).

The first half drives the real ``_CuaDriverSession`` and ``_AsyncBridge`` over a real MCP stdio transport to a
stand-in driver (``cua_session_fake_server.py``), so the exception the SDK really raises after the child dies, and the
driver's real rejection text, decide the tests. The second half pins the edge cases with a fake bridge.
"""

import concurrent.futures
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any, cast

import pytest

from tests.waiting import wait_until
from tools.computer_use import cua_backend_driver as _driver
from tools.computer_use.cua_backend_session import _AsyncBridge, _CuaDriverSession

SERVER = Path(__file__).with_name("cua_session_fake_server.py")
LABEL = "hermes-label"


class _FakeDaemon:
    """Embedded-daemon seam: ``_lifecycle_coro`` launches whatever ``proxy_invocation`` names."""

    socket_path = ""

    def __init__(self, state: Path, mode: str) -> None:
        self.state, self.mode = state, mode

    def proxy_invocation(self):
        return sys.executable, [str(SERVER), str(self.state), self.mode]

    def child_env(self):
        return dict(os.environ)


@pytest.fixture
def driver(tmp_path, monkeypatch):
    monkeypatch.setattr(_driver, "resolve_cua_driver_cmd", lambda: "/fake/cua-driver")
    made = []

    def make(mode: str) -> _CuaDriverSession:
        bridge = _AsyncBridge()
        session = _CuaDriverSession(bridge, _FakeDaemon(tmp_path, mode))
        session.start()
        made.append((session, bridge))
        assert session.call_tool("start_session", {"session": LABEL}).get("isError") is not True
        return session

    yield make
    for session, bridge in made:
        session.stop()
        bridge.stop()


def _calls(state: Path) -> list[tuple[int, str, str]]:
    path = state / "calls.log"
    if not path.exists():
        return []
    return [(int(p), tool, outcome) for p, tool, outcome in (line.split() for line in path.read_text().splitlines())]


def _pids(state: Path) -> list[int]:
    return [int((state / f"pid.{n}").read_text()) for n in range(1, 10) if (state / f"pid.{n}").exists()]


def _kill_child(state: Path, session: _CuaDriverSession) -> int:
    pid = _pids(state)[0]
    os.kill(pid, signal.SIGKILL)

    def saw_eof() -> bool:  # the SDK's reader saw EOF: later requests get its "Connection closed" contract
        dispatcher = getattr(session._session, "_dispatcher", None)
        return dispatcher is None or getattr(dispatcher, "_closed", True) is True

    wait_until(saw_eof, timeout=10.0, what="the MCP client to see the dead child's EOF")
    return pid


def _text(result: dict[str, Any]) -> str:
    return str(result.get("data"))


# ── Closed MCP connection (the child died) ─────────────────────────────────────────────────────────────────────


def test_dead_child_reconnects_once_and_replays_a_read(driver, tmp_path):
    session = driver("healthy")
    dead = _kill_child(tmp_path, session)

    result = session.call_tool("list_apps", {"session": LABEL})

    assert result.get("isError") is not True, result
    assert "apps: Music" in _text(result)
    first, second = _pids(tmp_path)
    assert first == dead and second != dead
    # The replacement transport restored the public label before the replayed read.
    assert [(t, o) for p, t, o in _calls(tmp_path) if p == second] == [
        ("start_session", "executed"), ("list_apps", "executed")]


def test_dead_child_reconnects_without_replaying_a_mutation(driver, tmp_path):
    session = driver("healthy")
    _kill_child(tmp_path, session)

    result = session.call_tool("click", {"x": 1, "y": 2, "session": LABEL})

    assert result["isError"] is True
    assert result["structuredContent"]["code"] == "transport_outcome_unknown"
    assert result["structuredContent"]["next_step"] == "fresh_state"
    assert [c for c in _calls(tmp_path) if c[1] == "click"] == []  # never replayed
    # Reconnected exactly once: the next call runs on the replacement child without another restart.
    follow = session.call_tool("list_apps", {"session": LABEL})
    assert follow.get("isError") is not True, follow
    assert len(_pids(tmp_path)) == 2


# ── Ended implicit session on a call with no session label ─────────────────────────────────────────────────────


def test_ended_implicit_session_is_revived_by_the_id_the_driver_names(driver, tmp_path):
    session = driver("revivable")

    result = session.call_tool("bring_to_front", {"pid": 1359, "window_id": 58})

    assert result.get("isError") is not True, result
    assert "brought 1359/58 to front" in _text(result)
    assert len(_pids(tmp_path)) == 1  # revived in place, no transport restart
    assert [(t, o) for _, t, o in _calls(tmp_path)] == [
        ("start_session", "executed"),            # the fixture's public label
        ("bring_to_front", "rejected"),
        ("start_session", "executed"),            # revive the named implicit session
        ("bring_to_front", "executed"),
    ]
    assert session._declared_session_id == LABEL  # the implicit id never replaces our public label


def test_implicit_session_that_stays_ended_gets_a_fresh_transport(driver, tmp_path):
    session = driver("stubborn")

    result = session.call_tool("bring_to_front", {"pid": 1359, "window_id": 58})

    assert result.get("isError") is not True, result
    first, second = _pids(tmp_path)
    fronts = [(p, o) for p, t, o in _calls(tmp_path) if t == "bring_to_front"]
    assert fronts == [(first, "rejected"), (first, "rejected"), (second, "executed")]  # executed exactly once
    # The replacement transport restored the public label before the replay.
    assert [t for p, t, _ in _calls(tmp_path) if p == second] == ["start_session", "bring_to_front"]


# ── Edge cases (fake bridge) ───────────────────────────────────────────────────────────────────────────────────


def _ended(session_id: str, tool: str) -> dict[str, Any]:
    return {"isError": True, "data": (
        f"session '{session_id}' has ended; tool call '{tool}' was rejected. Call start_session with this id to "
        "revive it before issuing further actions, or use a new session id."), "structuredContent": None}


class _Bridge:
    def __init__(self, effects):
        self.calls, self.effects = [], list(effects)

    def run(self, value, timeout=None):
        self.calls.append(value)
        effect = self.effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return effect


def _session(bridge, declared: str | None = LABEL):
    session = cast(Any, _CuaDriverSession.__new__(_CuaDriverSession))
    session._bridge, session._session, session._lock, session._started = bridge, object(), threading.Lock(), True
    session._capabilities, session._tool_schemas, session._capability_version = {}, {}, ""
    session._declared_session_id = declared
    session._transport_reset_callback = None
    session._call_tool_async = lambda name, args: ("call", name, args)
    session._reconnect_log = []
    session._stop_lifecycle_locked = lambda: session._reconnect_log.append("stop")
    session._start_lifecycle_locked = lambda: session._reconnect_log.append("start")
    return session


def _mcp_error(code: int, message: str) -> Exception:
    from mcp.shared.exceptions import MCPError
    return MCPError(code=code, message=message)


@pytest.mark.parametrize(("exc", "closed"), [
    (lambda: _mcp_error(-32000, "Connection closed"), True),
    # -32000 is also JSON-RPC's generic server error, returned by a LIVE peer: no reconnect.
    (lambda: _mcp_error(-32000, "Internal server error"), False),
    (lambda: _mcp_error(-32603, "Connection closed"), False),
    (lambda: RuntimeError("Connection closed"), False),
])
def test_closed_connection_classification(exc, closed):
    assert _CuaDriverSession._is_closed_session_error(exc()) is closed


def test_older_sdk_error_shape_is_matched_structurally():
    class McpError(Exception):  # mcp 1.x name, same ``.error`` shape
        def __init__(self):
            super().__init__("Connection closed")
            self.error = type("ErrorData", (), {"code": -32000, "message": "Connection closed"})()

    assert _CuaDriverSession._is_closed_session_error(McpError()) is True


def test_live_server_error_with_closed_code_is_not_a_reconnect():
    session = _session(_Bridge([_mcp_error(-32000, "Internal server error")]))
    with pytest.raises(Exception, match="Internal server error"):
        session.call_tool("list_apps", {})
    assert session._reconnect_log == []


def test_explicitly_labelled_call_still_ended_surfaces_without_transport_restart():
    bridge = _Bridge([_ended("other-label", "list_apps"), {"isError": False}, _ended("other-label", "list_apps")])
    session = _session(bridge)

    result = session.call_tool("list_apps", {"session": "other-label"})

    assert result["isError"] is True
    assert bridge.calls == [("call", "list_apps", {"session": "other-label"}),
                            ("call", "start_session", {"session": "other-label"}),
                            ("call", "list_apps", {"session": "other-label"})]
    assert session._reconnect_log == []


def test_ended_declared_session_keeps_the_existing_revive():
    bridge = _Bridge([_ended(LABEL, "click"), {"isError": False}, {"isError": False, "data": "clicked"}])
    session = _session(bridge)

    assert session.call_tool("click", {"session": LABEL})["data"] == "clicked"
    assert bridge.calls[1] == ("call", "start_session", {"session": LABEL})
    assert session._reconnect_log == []


def test_unnamed_ended_session_without_declared_label_surfaces_as_is():
    old_wording = {"isError": True, "data": "this session has ended; call start_session explicitly to reuse its label"}
    bridge = _Bridge([old_wording])
    session = _session(bridge, declared=None)

    assert session.call_tool("bring_to_front", {"pid": 1, "window_id": 2}) is old_wording
    assert len(bridge.calls) == 1


def test_lifecycle_call_is_never_revived():
    bridge = _Bridge([_ended(LABEL, "end_session")])
    session = _session(bridge)

    assert session.call_tool("end_session", {"session": LABEL})["isError"] is True
    assert len(bridge.calls) == 1


def test_replay_that_times_out_fails_closed_and_marks_session_suspect():
    bridge = _Bridge([_ended("mcp-1-1", "bring_to_front"), {"isError": False}, concurrent.futures.TimeoutError()])
    session = _session(bridge)

    result = session.call_tool("bring_to_front", {"pid": 1, "window_id": 2})

    assert result["structuredContent"]["code"] == "timeout_outcome_unknown"
    assert session._timeout_suspect is True
    assert len(bridge.calls) == 3


def test_replay_on_a_closed_transport_fails_closed():
    bridge = _Bridge([_ended("mcp-1-1", "bring_to_front"), {"isError": False}, _mcp_error(-32000, "Connection closed")])
    session = _session(bridge)

    result = session.call_tool("bring_to_front", {"pid": 1, "window_id": 2})

    assert result["structuredContent"]["code"] == "transport_outcome_unknown"
    assert len(bridge.calls) == 3
