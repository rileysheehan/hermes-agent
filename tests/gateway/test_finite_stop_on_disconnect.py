"""MER-963: ``gateway.finite_stop_on_disconnect`` stops a finite turn whose submitting client is gone.

In-process, a killed ``chat -q`` run took its work with it. In the gateway the turn's authority outlives
the viewer, so an orchestrator that cancels a run by killing it (SIGKILL after its grace) left the
turn and its tools running. With the setting on, the connection's close stops each finite admission it
submitted that has not settled, following that admission through its state changes (sanity review of
plan rev 1: a claim between two reads made the stop refuse, and the admission was dropped). The store
double refuses exactly as the runtime store and authority do: cancelling a started row is
``stale_generation`` (``cancel_or_claimed`` then returns the started row), and an interrupt whose
generation is not the live handle's is ``stale_generation``.
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
from hermes_state_runtime import RuntimeStoreError

REF = SessionRef("keely", "s1")


@pytest.fixture(autouse=True)
def _plain_config_reader(monkeypatch):
    from ruamel.yaml import YAML

    def read(path=None):
        return YAML(typ="safe").load(open(path)) if path and os.path.exists(path) else {}
    monkeypatch.setattr("gateway.run._load_gateway_config", read)


@pytest.fixture(autouse=True)
def _doubles(monkeypatch):
    async def instant(_s):
        return None
    monkeypatch.setattr("gateway.session_controls.asyncio.sleep", instant)
    monkeypatch.setattr("gateway.session_controls.asdict", lambda r: dict(vars(r)))


class _Store:
    """One admission and the session's live handle, with scripted transitions at named call points:
    ``script={"cancel": [fn, ...]}`` runs fn(store) on the first, second... call of that point."""

    def __init__(self, home, status="queued", generation=None, handle=None, script=None):
        self.db = types.SimpleNamespace(db_path=str(home / "state.db"))
        self.status, self.generation, self.handle = status, generation, handle
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls, self.interrupted, self.submitted = [], [], []

    def _hook(self, point):
        self.calls.append(point)
        steps = self.script.get(point)
        if steps:
            step = steps.pop(0)
            if step:
                step(self)

    # transitions the drain / turn make
    def claim(self, gen=7):
        self.status, self.generation, self.handle = "started", gen, gen

    def claim_unpublished(self, gen=7):
        self.status, self.generation, self.handle = "started", gen, None

    def finish(self):
        self.status, self.handle = "terminal", None

    def successor(self, gen=8):
        self.status, self.handle = "terminal", gen

    # authority surface
    async def submit(self, actor, submission, **kwargs):  # queue_while_restarting (v0.22.5)
        self.submitted.append(submission)
        return types.SimpleNamespace(status="queued", admission_id="adm-1", execution_generation=None)

    async def receipt(self, actor, ref, admission_id):
        # The read is taken first; a "receipt" step is a transition right AFTER it returns.
        read = types.SimpleNamespace(status=self.status, admission_id=admission_id,
                                     execution_generation=self.generation if self.status == "started" else None)
        self._hook("receipt")
        return read

    async def cancel_or_claimed(self, actor, ref, admission_id):
        self._hook("cancel")
        if self.status == "queued":
            self.status = "terminal"
            return None
        if self.status == "started":
            return {"admission_id": admission_id, "status": "started", "generation": self.generation}
        raise RuntimeStoreError("stale_generation")

    async def cancel_queued(self, actor, ref, admission_id):  # the refusal the old handler hit
        raise AssertionError("the disconnect stop must not take the started-row cancel path")

    async def interrupt(self, actor, ref, generation):
        self._hook("interrupt")
        if self.handle != generation:
            raise RuntimeStoreError("stale_generation")
        self.interrupted.append(generation)
        return types.SimpleNamespace(execution_generation=generation)


def _conn(tmp_path, *, on=True, native=True, **store):
    (tmp_path / "config.yaml").write_text(f"gateway:\n  finite_stop_on_disconnect: {'true' if on else 'false'}\n")
    conn = object.__new__(AuthorityConnection)
    conn.authority = _Store(tmp_path, **store)
    conn.actor = types.SimpleNamespace(subject="owner")  # v0.22.5 submit keys voice stops by subject
    conn._operator = None
    conn._typed_voice_stop = _no_typed_voice_stop
    conn.native_owner = native
    conn.subscriptions = {"s1": object()}
    conn._finite_owned = {}
    conn._siblings = {}
    return conn


def _submit(conn, finite=True, rid="r1"):
    return asyncio.run(conn.submit(REF, {"session_id": "s1", "submission_id": rid, "text": "hi", "finite": finite}))


def _stop(conn):
    _submit(conn)
    assert conn._finite_owned == {"adm-1": REF}
    asyncio.run(conn._stop_unsettled_finite())
    assert conn._finite_owned == {}
    return conn.authority


def test_setting_is_read_from_the_profile_and_defaults_off(tmp_path):
    a = types.SimpleNamespace(db=types.SimpleNamespace(db_path=str(tmp_path / "state.db")))
    assert session_finite.stop_on_disconnect(a) is False
    (tmp_path / "config.yaml").write_text("gateway:\n  finite_stop_on_disconnect: true\n")
    assert session_finite.stop_on_disconnect(a) is True


# --- healthy controls -------------------------------------------------------------------------------

def test_queued_is_withdrawn(tmp_path, caplog):
    caplog.set_level("INFO")
    store = _stop(_conn(tmp_path, status="queued"))
    assert store.status == "terminal" and store.interrupted == []
    assert "withdrew queued admission adm-1" in caplog.text


def test_started_is_interrupted_at_its_generation(tmp_path, caplog):
    caplog.set_level("INFO")
    store = _stop(_conn(tmp_path, status="started", generation=7, handle=7))
    assert store.interrupted == [7]
    assert "interrupted started generation 7 admission adm-1" in caplog.text


@pytest.mark.parametrize("status", ["terminal", "unknown"])
def test_settled_or_unknown_is_left_alone(tmp_path, status):
    store = _stop(_conn(tmp_path, status=status, generation=7, handle=7))
    assert store.interrupted == [] and store.calls == ["receipt"]


# --- races ------------------------------------------------------------------------------------------

def test_claim_during_the_cancel_commit_is_interrupted(tmp_path):
    # The review's second interleaving: the drain claims during the cancel commit.
    store = _stop(_conn(tmp_path, status="queued", script={"cancel": [lambda s: s.claim(7)]}))
    assert store.interrupted == [7]
    assert store.calls == ["receipt", "cancel", "interrupt"]


def test_claim_right_after_the_read_is_interrupted(tmp_path):
    # The review's first interleaving: the receipt reads queued and the drain claims before the cancel
    # is attempted. No second status read decides the path; cancel_or_claimed hands back the started row.
    store = _stop(_conn(tmp_path, status="queued", script={"receipt": [lambda s: s.claim(9)]}))
    assert store.interrupted == [9]
    assert store.calls == ["receipt", "cancel", "interrupt"]


def test_claimed_and_finished_before_the_cancel_stops_nothing(tmp_path):
    store = _stop(_conn(tmp_path, status="queued", script={"cancel": [lambda s: (s.claim(7), s.finish())]}))
    assert store.interrupted == []
    assert store.calls == ["receipt", "cancel", "receipt"]


def test_a_successor_turn_is_never_interrupted(tmp_path):
    # Read started at 7; before the Stop lands that turn ends and a successor runs at 8.
    store = _stop(_conn(tmp_path, status="started", generation=7, handle=7,
                        script={"interrupt": [lambda s: s.successor(8)]}))
    assert store.interrupted == []
    assert store.handle == 8
    assert store.calls == ["receipt", "interrupt", "receipt"]


def test_a_successor_after_a_claim_race_is_never_interrupted(tmp_path):
    store = _stop(_conn(tmp_path, status="queued",
                        script={"cancel": [lambda s: s.claim(7)], "interrupt": [lambda s: s.successor(8)]}))
    assert store.interrupted == []


def test_claim_whose_handle_publishes_late_is_interrupted_on_retry(tmp_path):
    store = _stop(_conn(tmp_path, status="queued",
                        script={"cancel": [lambda s: s.claim_unpublished(7)],
                                "receipt": [None, lambda s: setattr(s, "handle", 7)]}))
    assert store.interrupted == [7]


def test_bounded_and_loud_when_it_cannot_stop(tmp_path, caplog):
    store = _stop(_conn(tmp_path, status="started", generation=7, handle=None))
    assert store.interrupted == []
    assert store.calls.count("receipt") == AuthorityConnection._STOP_ATTEMPTS
    assert "gave up after" in caplog.text


def test_an_unexpected_error_never_breaks_close(tmp_path, caplog):
    conn = _conn(tmp_path, status="started", generation=7, handle=7)

    async def boom(actor, ref, generation):
        raise RuntimeError("owner gone")
    conn.authority.interrupt = boom
    _stop(conn)
    assert "could not stop admission adm-1" in caplog.text


# --- what gets tracked ------------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [dict(on=False), dict(native=False)])
def test_nothing_is_tracked_when_off_or_for_a_remote_viewer(tmp_path, kw):
    conn = _conn(tmp_path, status="started", generation=7, handle=7, **kw)
    _submit(conn)
    assert conn._finite_owned == {}


def test_a_non_finite_submission_is_never_tracked(tmp_path):
    conn = _conn(tmp_path, status="started", generation=7, handle=7)
    _submit(conn, finite=False)
    assert conn._finite_owned == {}


async def _no_typed_voice_stop(ref, params):
    """v0.22.5 submit first checks for a typed voice-chat stop phrase; these fakes have no voice chat."""
    return False
