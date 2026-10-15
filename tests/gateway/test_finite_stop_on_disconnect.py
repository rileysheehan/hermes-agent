"""MER-963: ``gateway.finite_stop_on_disconnect`` stops a finite turn whose submitting client is gone.

In-process, a killed ``chat -q`` run took its work with it. In the gateway the turn's authority outlives
the viewer, so an orchestrator that cancels a run by killing it (SIGKILL after its grace) left the
turn and its tools running. With the setting on, the connection's close stops each finite admission it
submitted that has not settled; settled ones, and everything with the setting off, are untouched.
"""
from __future__ import annotations

import asyncio
import os
import types

import pytest

import gateway.run  # module-level: its first import probes the checkout (home_io_guard)
from gateway.session_contract import SessionRef
from gateway.session_controls import AuthorityConnection
from gateway import session_finite


@pytest.fixture(autouse=True)
def _plain_config_reader(monkeypatch):
    from ruamel.yaml import YAML

    def read(path=None):
        return YAML(typ="safe").load(open(path)) if path and os.path.exists(path) else {}
    monkeypatch.setattr("gateway.run._load_gateway_config", read)


class _Authority:
    def __init__(self, home, statuses):
        self.db = types.SimpleNamespace(db_path=str(home / "state.db"))
        self.statuses = statuses
        self.submitted = []

    async def submit(self, actor, submission):
        self.submitted.append(submission)
        return types.SimpleNamespace(status="queued", admission_id=f"adm-{len(self.submitted)}",
                                     execution_generation=None)

    async def receipt(self, actor, ref, admission_id):
        status, gen = self.statuses[admission_id]
        return types.SimpleNamespace(status=status, execution_generation=gen, admission_id=admission_id)


def _conn(tmp_path, *, on=True, native=True, statuses=None):
    (tmp_path / "config.yaml").write_text(f"gateway:\n  finite_stop_on_disconnect: {'true' if on else 'false'}\n")
    conn = object.__new__(AuthorityConnection)
    conn.authority = _Authority(tmp_path, statuses or {})
    conn.actor = object()
    conn.native_owner = native
    conn.subscriptions = {"s1": object()}
    conn._finite_owned = {}
    conn._siblings = {}
    conn.stopped = []

    async def cancel(ref, params):
        conn.stopped.append(("cancel", params["admission_id"]))

    async def interrupt(ref, params):
        conn.stopped.append(("interrupt", params["execution_generation"]))
    conn.cancel, conn.interrupt = cancel, interrupt
    return conn


def _submit(conn, finite=True, rid="r1"):
    import gateway.session_controls as sc

    async def run():
        orig = sc.asdict
        sc.asdict = lambda r: vars(r)
        try:
            return await conn.submit(SessionRef("keely", "s1"), {"session_id": "s1", "submission_id": rid,
                                                                  "text": "hi", "finite": finite})
        finally:
            sc.asdict = orig
    return asyncio.run(run())


def test_setting_is_read_from_the_profile_and_defaults_off(tmp_path):
    a = types.SimpleNamespace(db=types.SimpleNamespace(db_path=str(tmp_path / "state.db")))
    assert session_finite.stop_on_disconnect(a) is False
    (tmp_path / "config.yaml").write_text("gateway:\n  finite_stop_on_disconnect: true\n")
    assert session_finite.stop_on_disconnect(a) is True


@pytest.mark.parametrize("status,expected", [
    ("started", [("interrupt", 7)]), ("queued", [("cancel", "adm-1")]), ("terminal", []), ("unknown", []),
])
def test_close_stops_an_unsettled_finite_admission(tmp_path, status, expected):
    conn = _conn(tmp_path, statuses={"adm-1": (status, 7)})
    _submit(conn)
    assert conn._finite_owned == {"adm-1": SessionRef("keely", "s1")}
    asyncio.run(conn._stop_unsettled_finite())
    assert conn.stopped == expected
    assert conn._finite_owned == {}


@pytest.mark.parametrize("kw", [dict(on=False), dict(native=False)])
def test_nothing_is_tracked_when_off_or_for_a_remote_viewer(tmp_path, kw):
    conn = _conn(tmp_path, statuses={"adm-1": ("started", 7)}, **kw)
    _submit(conn)
    assert conn._finite_owned == {}


def test_a_non_finite_submission_is_never_tracked(tmp_path):
    conn = _conn(tmp_path, statuses={"adm-1": ("started", 7)})
    _submit(conn, finite=False)
    assert conn._finite_owned == {}


def test_a_failed_stop_never_breaks_close(tmp_path, caplog):
    conn = _conn(tmp_path, statuses={"adm-1": ("started", 7)})
    _submit(conn)

    async def boom(ref, params):
        raise RuntimeError("owner gone")
    conn.interrupt = boom
    asyncio.run(conn._stop_unsettled_finite())
    assert "could not stop admission adm-1" in caplog.text
