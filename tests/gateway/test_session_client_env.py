"""MER-963: a finite (``chat -q``) gateway turn gets its client's allowlisted per-run environment.

v0.22.1 runs every ``chat -q`` turn in the profile's gateway, so the run's own environment (an
orchestrator's run credential, its issue id, the MER-646 opt-in) never reached the turn's tools.
``gateway.client_env_passthrough`` forwards the allowlisted names; these tests pin the filter, the
in-memory handoff (never the durable payload), the per-admission binding, the child-env bridge and the
snapshot exclusion that keeps one run's value from reaching a later run.
"""
from __future__ import annotations

import asyncio
import os
import time
import types

import pytest

import gateway.run  # module-level: its first import probes the checkout (home_io_guard)
import gateway.session_controls
from gateway import session_client_env as sce
from gateway.session_context import client_env, get_session_env, take_client_env_value
from hermes_state_runtime import RuntimeStoreError

ALLOW = ("PAPERCLIP_*", "HERMES_RESUME_UNCHANGED_PREFIX_*", "EXACT_NAME")


@pytest.fixture(autouse=True)
def _clean_registry():
    with sce._lock:
        sce._pending.clear()
    yield
    with sce._lock:
        sce._pending.clear()


def test_patterns_read_the_gateway_list_and_ignore_bare_star():
    assert sce.patterns({"gateway": {"client_env_passthrough": ["PAPERCLIP_*", " X ", "*", ""]}}) == ("PAPERCLIP_*", "X")
    assert sce.patterns({"gateway": {"client_env_passthrough": "A_*, B"}}) == ("A_*", "B")
    assert sce.patterns({"gateway": {}}) == ()
    assert sce.patterns({}) == ()
    assert sce.patterns(None) == ()


@pytest.mark.parametrize("name,ok", [
    ("PAPERCLIP_API_KEY", True), ("PAPERCLIP_", True), ("EXACT_NAME", True), ("EXACT_NAME_2", False),
    ("HERMES_RESUME_UNCHANGED_PREFIX_SHA256", True), ("OTHER", False),
    # Always refused, whatever the allowlist says.
    ("PATH", False), ("HOME", False), ("HERMES_HOME", False), ("PYTHONPATH", False), ("LD_PRELOAD", False),
    ("DYLD_INSERT_LIBRARIES", False), ("_HERMES_FORCE_OPENAI_API_KEY", False), ("HERMES_SESSION_ID", False),
    ("BAD=NAME", False), ("", False),
])
def test_allowed(name, ok):
    assert sce.allowed(name, ALLOW + ("PATH", "LD_*", "HERMES_SESSION_*", "_HERMES_FORCE_*", "HOME")) is ok


def test_select_takes_only_allowed_names_and_nothing_when_list_is_empty():
    environ = {"PAPERCLIP_RUN_ID": "r1", "PAPERCLIP_API_KEY": "k", "OPENAI_API_KEY": "secret", "PATH": "/bin"}
    assert sce.select(environ, ALLOW) == {"PAPERCLIP_API_KEY": "k", "PAPERCLIP_RUN_ID": "r1"}
    assert sce.select(environ, ()) == {}
    assert sce.select({"PAPERCLIP_BIG": "x" * (sce.MAX_VALUE + 1)}, ALLOW) == {}


def test_admit_filters_by_the_owner_list_and_needs_a_finite_native_turn():
    params = {"finite": True, "client_env": {"PAPERCLIP_RUN_ID": "r1", "OPENAI_API_KEY": "sneaky"}}
    assert sce.admit(params, native_owner=True, allow=ALLOW) == {"PAPERCLIP_RUN_ID": "r1"}
    assert sce.admit(params, native_owner=False, allow=ALLOW) == {}
    assert sce.admit({**params, "finite": False}, native_owner=True, allow=ALLOW) == {}
    assert sce.admit(params, native_owner=True, allow=()) == {}
    assert sce.admit({"finite": True}, native_owner=True, allow=ALLOW) == {}


@pytest.mark.parametrize("bad", [["PAPERCLIP_RUN_ID"], {"PAPERCLIP_RUN_ID": 1}, {1: "x"},
                                 {"PAPERCLIP_RUN_ID": "a\0b"}, {f"PAPERCLIP_{i}": "v" for i in range(sce.MAX_VARS + 1)}])
def test_admit_refuses_a_malformed_value(bad):
    with pytest.raises(RuntimeStoreError):
        sce.admit({"finite": True, "client_env": bad}, native_owner=True, allow=ALLOW)


def test_stash_is_taken_once_and_expires(monkeypatch):
    sce.stash("s1", "req1", {"PAPERCLIP_RUN_ID": "r1"})
    assert sce.take("s1", "req1") == {"PAPERCLIP_RUN_ID": "r1"}
    assert sce.take("s1", "req1") == {}
    sce.stash("s1", "req2", {"PAPERCLIP_RUN_ID": "r2"})
    now = time.monotonic()
    monkeypatch.setattr(sce.time, "monotonic", lambda: now + sce.TTL_S + 1)
    assert sce.take("s1", "req2") == {}
    assert sce.pending_count() == 0


def test_stash_is_bounded():
    for i in range(sce.MAX_PENDING + 5):
        sce.stash("s", f"r{i}", {"PAPERCLIP_X": str(i)})
    assert sce.pending_count() == sce.MAX_PENDING
    assert sce.take("s", "r0") == {}  # the oldest went first
    assert sce.take("s", f"r{sce.MAX_PENDING + 4}") == {"PAPERCLIP_X": str(sce.MAX_PENDING + 4)}


def test_scope_binds_for_the_admission_and_resets(monkeypatch):
    monkeypatch.setenv("PAPERCLIP_RUN_ID", "gateway-process-value")
    sce.stash("s1", "req1", {"PAPERCLIP_RUN_ID": "run-1", "HERMES_RESUME_UNCHANGED_PREFIX_CHARS": "900"})
    row = {"target_session_id": "s1", "request_id": "req1", "admission_id": "a1"}
    with sce.client_env_scope(row) as env:
        assert env["PAPERCLIP_RUN_ID"] == "run-1"
        assert client_env()["PAPERCLIP_RUN_ID"] == "run-1"
        assert get_session_env("PAPERCLIP_RUN_ID") == "run-1"
        assert take_client_env_value("HERMES_RESUME_UNCHANGED_PREFIX_CHARS") == "900"
        assert take_client_env_value("HERMES_RESUME_UNCHANGED_PREFIX_CHARS") is None  # read once
    assert client_env() == {}
    assert get_session_env("PAPERCLIP_RUN_ID") == "gateway-process-value"
    # A second execution of the same admission (an owner replay) gets nothing.
    with sce.client_env_scope(row) as env:
        assert env == {} and client_env() == {}


def test_scope_binding_reaches_a_thread_that_copies_the_context():
    import contextvars
    import threading
    sce.stash("s1", "req1", {"PAPERCLIP_TASK_ID": "t1"})
    seen = {}
    with sce.client_env_scope({"target_session_id": "s1", "request_id": "req1"}):
        ctx = contextvars.copy_context()
        t = threading.Thread(target=lambda: seen.update(ctx.run(client_env)))
        t.start()
        t.join()
    assert seen == {"PAPERCLIP_TASK_ID": "t1"}


def test_child_env_bridge_carries_the_forwarded_values(monkeypatch):
    from tools.environments.local import _inject_session_context_env
    sce.stash("s1", "req1", {"PAPERCLIP_API_URL": "http://run", "PAPERCLIP_RUN_ID": "run-1"})
    env = {"PAPERCLIP_RUN_ID": "stale-owner-value", "KEEP": "1"}
    with sce.client_env_scope({"target_session_id": "s1", "request_id": "req1"}):
        _inject_session_context_env(env)
    assert env["PAPERCLIP_RUN_ID"] == "run-1" and env["PAPERCLIP_API_URL"] == "http://run" and env["KEEP"] == "1"
    env2 = {"KEEP": "1"}
    _inject_session_context_env(env2)  # no turn bound: nothing added
    assert "PAPERCLIP_RUN_ID" not in env2


def test_forwarded_names_never_persist_in_the_shared_snapshot():
    from tools.environments.base import BaseEnvironment
    from tools.environments.base_session_env import _wrap_command_script

    class _Env(BaseEnvironment):
        def cleanup(self):
            pass

        def _run_bash(self, *a, **k):
            raise AssertionError("not used")
    env = object.__new__(_Env)
    env._profile_scoped_passthrough = False
    env._snapshot_passthrough_names = set()
    assert env._snapshot_excluded_passthrough_names() == ()
    sce.stash("s1", "req1", {"PAPERCLIP_API_KEY": "k1", "PAPERCLIP_RUN_ID": "r1"})
    with sce.client_env_scope({"target_session_id": "s1", "request_id": "req1"}):
        names = env._snapshot_excluded_passthrough_names()
    assert names == ("PAPERCLIP_API_KEY", "PAPERCLIP_RUN_ID")
    # Monotonic: a later turn without them still excludes (and unsets) them, so an earlier run's value
    # sourced from an old snapshot can never reach it.
    assert env._snapshot_excluded_passthrough_names() == names
    script = _wrap_command_script("true", quoted_cwd="/tmp", quoted_snap="/tmp/s.sh", snap_tmp_template="/tmp/s.XXX",
                                  passthrough_names=names, snapshot_ready=True, cwd_marker="M")
    assert "PAPERCLIP_API_KEY" in script and "k1" not in script


class _Authority:
    def __init__(self, db_path, status="queued"):
        self.db = types.SimpleNamespace(db_path=str(db_path))
        self.submitted = []

    async def submit(self, actor, submission):
        self.submitted.append(submission)
        return types.SimpleNamespace(status="queued", admission_id="adm-1", execution_generation=None)


@pytest.fixture(autouse=True)
def _plain_config_reader(monkeypatch):
    # The owner reads the profile's config.yaml beside its state.db; read it plainly here (the suite's
    # home isolation keeps the effective-config loader away from temp homes).
    from ruamel.yaml import YAML

    def read(path=None):
        return YAML(typ="safe").load(open(path)) if path and os.path.exists(path) else {}
    monkeypatch.setattr("gateway.run._load_gateway_config", read)


def _controls(tmp_path, native=True, config="gateway:\n  client_env_passthrough: ['PAPERCLIP_*']\n"):
    from gateway.session_controls import AuthorityConnection
    (tmp_path / "config.yaml").write_text(config)
    conn = object.__new__(AuthorityConnection)
    conn.authority = _Authority(tmp_path / "state.db")
    conn.actor = object()
    conn.native_owner = native
    conn.subscriptions = {"s1": object()}
    conn._finite_owned = {}
    return conn


def _submit(conn, params):
    from gateway.session_contract import SessionRef
    from dataclasses import asdict

    async def run():
        import gateway.session_controls as sc
        orig = sc.asdict
        sc.asdict = lambda r: vars(r)
        try:
            return await conn.submit(SessionRef("keely", "s1"), params)
        finally:
            sc.asdict = orig
    return asyncio.run(run())


def test_submit_keeps_client_env_out_of_the_durable_payload(tmp_path):
    conn = _controls(tmp_path)
    _submit(conn, {"session_id": "s1", "submission_id": "req1", "text": "hi", "finite": True,
                   "client_env": {"PAPERCLIP_API_KEY": "k1", "OPENAI_API_KEY": "nope"}})
    (submission,) = conn.authority.submitted
    assert "client_env" not in submission.payload
    assert "k1" not in repr(submission.payload)
    assert sce.take("s1", "req1") == {"PAPERCLIP_API_KEY": "k1"}


def test_submit_forwards_nothing_without_an_owner_allowlist_or_for_a_remote_viewer(tmp_path):
    conn = _controls(tmp_path, config="gateway: {}\n")
    _submit(conn, {"session_id": "s1", "submission_id": "req1", "text": "hi", "finite": True,
                   "client_env": {"PAPERCLIP_API_KEY": "k1"}})
    assert sce.take("s1", "req1") == {}
    remote_home = tmp_path / "remote"
    remote_home.mkdir()
    conn = _controls(remote_home, native=False)
    _submit(conn, {"session_id": "s1", "submission_id": "req2", "text": "hi", "finite": True,
                   "client_env": {"PAPERCLIP_API_KEY": "k1"}})
    assert sce.take("s1", "req2") == {}


def test_submit_discards_the_stash_when_admission_fails(tmp_path):
    conn = _controls(tmp_path)

    async def boom(actor, submission):
        raise RuntimeStoreError("session_busy")
    conn.authority.submit = boom
    with pytest.raises(RuntimeStoreError):
        _submit(conn, {"session_id": "s1", "submission_id": "req1", "text": "hi", "finite": True,
                       "client_env": {"PAPERCLIP_API_KEY": "k1"}})
    assert sce.pending_count() == 0


def test_client_sends_only_its_allowlisted_names(monkeypatch):
    from hermes_cli import gateway_agent
    monkeypatch.setattr("hermes_cli.config.load_config_readonly",
                        lambda *a, **k: {"gateway": {"client_env_passthrough": ["MER963T_*"]}})
    monkeypatch.setenv("MER963T_RUN_ID", "r1")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    assert gateway_agent._client_env_for_turn() == {"MER963T_RUN_ID": "r1"}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda *a, **k: {})
    assert gateway_agent._client_env_for_turn() == {}


def test_submit_params_carry_client_env_only_on_a_finite_turn(monkeypatch):
    from hermes_cli import gateway_agent
    monkeypatch.setattr(gateway_agent, "_client_env_for_turn", lambda: {"PAPERCLIP_RUN_ID": "r1"})
    agent = object.__new__(gateway_agent.GatewayAgent)
    agent.__dict__.update(_voice_turn_pending=False, _pending_cli_user_message=None, clarify_callback=None)
    agent.__dict__["_finite"] = True
    monkeypatch.setattr(gateway_agent.GatewayAgent, "session_id", property(lambda self: "s1"))
    params = agent._submit_params("hello", None, None)
    assert params["client_env"] == {"PAPERCLIP_RUN_ID": "r1"} and params["finite"] is True
    agent.__dict__["_finite"] = False
    assert "client_env" not in agent._submit_params("hello", None, None)


def test_wire_contract_accepts_client_env():
    from tui_gateway.contracts.canonical_sessions import CanonicalSubmitParams
    parsed = CanonicalSubmitParams.model_validate({"session_id": "s1", "text": "hi", "finite": True,
                                                   "client_env": {"PAPERCLIP_RUN_ID": "r1"}})
    assert parsed.client_env == {"PAPERCLIP_RUN_ID": "r1"}
