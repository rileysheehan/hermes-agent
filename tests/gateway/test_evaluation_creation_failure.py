"""An orphan enrollment is a quarantined identity, never an ordinary session.

Real disposable SQLite/owner and fsync paths for MER-1030's failure window.
No memory provider import, model call, or bank request.
"""
import asyncio
import copy
import json
import os
from pathlib import Path
import sqlite3
import stat
import sys

import pytest

from agent import evaluation_policy as ep
from gateway.session_controls import AuthorityConnection
from gateway.session_evaluation import policy_for
from gateway.session_local import create_local_session, stage_local_session
from gateway.session_local_recovery import local_identity
from hermes_state_local import POLICY_PREFIX, commit_local_session
from hermes_state_runtime import RuntimeStoreError
pytest_plugins = ('tests.gateway.test_evaluation_create_retry',)


def assert_unpublished(owner, sid, route):
    assert owner.db.get_session(sid) is None
    assert sid not in owner.sessions
    with owner.db._read_ctx() as conn:
        assert conn.execute('SELECT 1 FROM state_meta WHERE key=?', (POLICY_PREFIX + sid,)).fetchone() is None
        assert conn.execute('SELECT 1 FROM gateway_routing WHERE session_key=?', (route,)).fetchone() is None


def fail_after_marker(owner, monkeypatch, failure):
    """Return a restorer after arming one real failure after the marker exists."""
    if failure in ('file-fsync', 'directory-fsync'):
        original = os.fsync

        def fail(fd):
            mode = os.fstat(fd).st_mode
            if stat.S_ISREG(mode) if failure == 'file-fsync' else stat.S_ISDIR(mode):
                raise OSError('synthetic fsync failure after marker creation')
            return original(fd)

        monkeypatch.setattr(os, 'fsync', fail)
        return RuntimeStoreError, lambda: monkeypatch.setattr(os, 'fsync', original)
    table = {'session-insert': 'sessions', 'route-insert': 'gateway_routing', 'policy-insert': 'state_meta'}[failure]
    owner.db._execute_write(lambda conn: conn.execute(
        f"CREATE TRIGGER abort_creation BEFORE INSERT ON {table} BEGIN SELECT RAISE(ABORT, 'synthetic insert failure'); END"))
    return sqlite3.IntegrityError, lambda: owner.db._execute_write(
        lambda conn: conn.execute('DROP TRIGGER abort_creation'))


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['file-fsync', 'directory-fsync', 'session-insert', 'route-insert', 'policy-insert'])
async def test_failed_evaluation_quarantines_production_but_can_recover_as_evaluation(world, monkeypatch, failure):
    home, owner, actor, params = world
    # Even a production receipt staged BEFORE the failed evaluation must refuse.
    staged_production = stage_local_session(owner, actor, params)
    sid, route = staged_production['session_id'], staged_production['route']
    error, restore = fail_after_marker(owner, monkeypatch, failure)
    with pytest.raises(error):
        create_local_session(owner, actor, {**params, 'evaluation_isolation': 1})
    restore()
    assert_unpublished(owner, sid, route)
    marker = ep.marker_path(home, sid)
    before = marker.stat()
    assert ep.enrolled(home, sid)
    for make in (
        lambda: commit_local_session(owner.db, epoch=owner.epoch, receipt=staged_production),
        lambda: create_local_session(owner, actor, params),
    ):
        with pytest.raises(RuntimeStoreError, match='evaluation_isolation_unproven'):
            make()
        assert_unpublished(owner, sid, route)
    connection = object.__new__(AuthorityConnection)
    connection.authority, connection.actor = owner, actor
    with pytest.raises(RuntimeStoreError, match='evaluation_isolation_unproven'):
        await connection.create(None, {**params, 'title': 'ordinary retry after failed evaluation'})
    assert_unpublished(owner, sid, route)
    after = marker.stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)
    # Production uses a fresh request identity. Its receipt and marker stay ordinary.
    production = create_local_session(owner, actor, {**params, 'request_id': 'fresh-production'})
    assert policy_for(owner, production).evaluation_isolation is False
    assert not ep.enrolled(home, production.session_id)
    receipt = copy.deepcopy(owner.db.get_session(production.session_id))
    # Reuse of the quarantined identity is supported only as evaluation, not by deleting its mark.
    recovered = create_local_session(owner, actor, {**params, 'evaluation_isolation': 1})
    assert recovered.session_id == sid
    assert policy_for(owner, recovered).evaluation_isolation is True
    assert ep.enrolled(home, sid)
    assert create_local_session(owner, actor, {**params, 'evaluation_isolation': 1}) == recovered
    assert owner.db.get_session(production.session_id) == receipt


@pytest.mark.asyncio
async def test_failure_before_marker_does_not_quarantine_unmarked_production(world, monkeypatch):
    home, owner, actor, params = world
    sid = local_identity(owner.profile_id, actor.subject, params['request_id'])
    def refuse(*args):
        raise OSError('synthetic failure before marker')
    with monkeypatch.context() as patch:
        patch.setattr(ep, 'enroll', refuse)
        with pytest.raises(RuntimeStoreError, match='evaluation_isolation_unproven'):
            create_local_session(owner, actor, {**params, 'evaluation_isolation': 1})
    assert not ep.enrolled(home, sid)
    ref = create_local_session(owner, actor, params)
    assert ref.session_id == sid
    assert policy_for(owner, ref).evaluation_isolation is False
    assert not ep.enrolled(home, sid)


@pytest.mark.asyncio
@pytest.mark.parametrize('window', ['after-enrollment', 'before-commit'])
async def test_creator_process_crash_quarantines_identity_across_cold_recovery(world, window):
    home, owner, actor, params = world
    staged = stage_local_session(owner, actor, {**params, 'evaluation_isolation': 1})
    sid, route = staged['session_id'], staged['route']
    worker = home / 'crashing-creator.py'
    worker.write_text('''import json, os, sys
from pathlib import Path
from hermes_state import SessionDB
from hermes_state_local import commit_local_session
from agent import evaluation_policy as ep
request = json.loads(sys.stdin.read())
db = SessionDB(db_path=Path(request['db']))
if request['window'] == 'after-enrollment':
    original = ep.enroll
    def crash_after_enrollment(*args):
        original(*args)
        os._exit(73)
    ep.enroll = crash_after_enrollment
else:
    original_write = db._execute_write
    def crash_before_commit(fn, *args, **kwargs):
        def callback(conn):
            fn(conn)
            os._exit(73)
        return original_write(callback, *args, **kwargs)
    db._execute_write = crash_before_commit
commit_local_session(db, epoch=request['epoch'], receipt=request['receipt'])
raise AssertionError('crash window not reached')
''')
    env = {**os.environ, 'HERMES_HOME': str(home), 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONPATH': str(Path(__file__).resolve().parents[2])}
    child = await asyncio.create_subprocess_exec(
        sys.executable, str(worker), cwd=home, env=env, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(child.communicate(json.dumps({
            'db': str(owner.db.db_path), 'epoch': owner.epoch, 'receipt': staged, 'window': window,
        }).encode()), timeout=20)
        assert child.returncode == 73, (stdout.decode(), stderr.decode())
    finally:
        if child.returncode is None:
            child.kill()
            await asyncio.wait_for(child.wait(), timeout=10)
    assert_unpublished(owner, sid, route)
    assert ep.enrolled(home, sid)
    from gateway.session_authority import initialize_session_authority
    runner = owner.runner
    runner.adapters = {}
    cold = await initialize_session_authority(runner, profile_id=str(home), instance_id='after-crash')
    with pytest.raises(RuntimeStoreError, match='evaluation_isolation_unproven'):
        create_local_session(cold, actor, params)
    assert_unpublished(cold, sid, route)
    recovered = create_local_session(cold, actor, {**params, 'evaluation_isolation': 1})
    assert recovered.session_id == sid
    assert policy_for(cold, recovered).evaluation_isolation is True
    assert ep.enrolled(home, sid)


@pytest.mark.asyncio
@pytest.mark.parametrize('obstruction', ['regular', 'directory', 'symlink', 'broken-symlink', 'stat-error'])
async def test_production_requires_positive_absence_not_enrollment_proof(world, monkeypatch, obstruction):
    home, owner, actor, params = world
    staged = stage_local_session(owner, actor, params)
    sid = staged['session_id']
    marker = ep.marker_path(home, sid)
    marker.parent.mkdir(parents=True)
    if obstruction == 'regular':
        marker.touch()
    elif obstruction == 'directory':
        marker.mkdir()
    elif obstruction in ('symlink', 'broken-symlink'):
        target = home / 'synthetic-target'
        if obstruction == 'symlink':
            target.touch()
        marker.symlink_to(target)
    else:
        original = Path.lstat
        def fail(path, *args, **kwargs):
            if path == marker:
                raise PermissionError('synthetic unknown marker state')
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'lstat', fail)
    # enrolled() intentionally refuses non-regular/unknown proof, but they must
    # not thereby certify the opposite policy as writable.
    with pytest.raises(RuntimeStoreError, match='evaluation_isolation_unproven'):
        commit_local_session(owner.db, epoch=owner.epoch, receipt=staged)
    assert_unpublished(owner, sid, staged['route'])
    if obstruction != 'stat-error':
        marker.lstat()  # refusal preserves even malformed enrollment evidence


@pytest.mark.parametrize('sid', ['ordinary', '20261016_203000_abc123', 'legacy.with-dots'])
def test_proved_absence_keeps_safe_legacy_and_other_profile_identity_ordinary(tmp_path, sid):
    other = tmp_path / 'another-profile/hindsight/read-only-sessions'
    other.mkdir(parents=True)
    (other / sid).touch()
    ep.require_unenrolled(tmp_path, sid)
    ep.require_unenrolled(tmp_path, sid)  # read-only check creates nothing
    assert not (tmp_path / 'hindsight').exists()


@pytest.mark.parametrize('sid', ['', '.', '..', '../foreign', '/absolute', None])
def test_ordinary_absence_check_refuses_unprovable_identity(tmp_path, sid):
    with pytest.raises(ep.EvaluationIsolationError, match='identity'):
        ep.require_unenrolled(tmp_path, sid)
