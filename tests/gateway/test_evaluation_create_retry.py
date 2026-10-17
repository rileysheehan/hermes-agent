"""Refused evaluation creation must not enroll another frozen production route.

Disposable real owner/SQLite paths; no provider import, model or bank traffic.
The owner and new-title regressions preserve Marin's independent probe steps.
"""
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import pytest_asyncio

from agent import evaluation_policy as ep
from gateway import run
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionStore
from gateway.session_authority import initialize_session_authority
from gateway.session_contract import Principal
from gateway.session_controls import AuthorityConnection
from gateway.session_evaluation import policy_for
from gateway.session_local import create_local_session, stage_local_session
from hermes_state_local import commit_local_session, local_receipt
from hermes_state_runtime import RuntimeStoreError


@pytest_asyncio.fixture
async def world(tmp_path, monkeypatch):
    import hermes_constants
    monkeypatch.setattr(hermes_constants, 'get_hermes_home', lambda: tmp_path)
    plugin = tmp_path / 'plugins/hindsight/__init__.py'
    plugin.parent.mkdir(parents=True)
    plugin.write_text('# synthetic provider, never imported\n')
    monkeypatch.setattr(ep, 'PLUGIN_SHA256', hashlib.sha256(plugin.read_bytes()).hexdigest())
    config = {'model': {'default': 'fixture', 'provider': 'custom'},
              'memory': {'provider': 'hindsight'}, 'platform_toolsets': {'cli': []}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a, **k: config)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={},
                             _draining=False, _cached_agent_for=lambda *a, **k: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(Platform.LOCAL)
    owner = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='retry')
    actor = Principal('human', str(tmp_path), frozenset({
        'session:create', 'session:read', 'session:submit', 'session:control'}), 'wire')
    params = {'request_id': 'stable-request', 'source': 'oneshot',
              'model': 'fixture', 'cwd': str(tmp_path)}
    return tmp_path, owner, actor, params


@pytest.mark.asyncio
@pytest.mark.parametrize('canonical', [False, True], ids=['owner', 'canonical-new-title'])
async def test_conflicting_evaluation_retry_leaves_production_unchanged(world, canonical):
    home, owner, actor, params = world
    ref = create_local_session(owner, actor, params)
    before = copy.deepcopy(local_receipt(owner.db, ref.session_id))
    assert not ep.enrolled(home, ref.session_id)
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        if canonical:
            connection = object.__new__(AuthorityConnection)
            connection.authority, connection.actor = owner, actor
            await connection.create(None, {**params, 'evaluation_isolation': 1,
                                           'title': 'new neutral evaluation title'})
        else:
            create_local_session(owner, actor, {**params, 'evaluation_isolation': 1})
    assert local_receipt(owner.db, ref.session_id) == before
    assert policy_for(owner, ref).evaluation_isolation is False
    assert not ep.enrolled(home, ref.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize('isolated', [False, True], ids=['production', 'evaluation'])
async def test_exact_retry_preserves_original_policy_and_marker(world, isolated):
    home, owner, actor, params = world
    if isolated:
        params = {**params, 'evaluation_isolation': 1}
    ref = create_local_session(owner, actor, params)
    before = copy.deepcopy(local_receipt(owner.db, ref.session_id))
    marker_stat = ep.marker_path(home, ref.session_id).stat() if isolated else None
    assert create_local_session(owner, actor, params) == ref
    assert local_receipt(owner.db, ref.session_id) == before
    assert policy_for(owner, ref).evaluation_isolation is isolated
    assert ep.enrolled(home, ref.session_id) is isolated
    if isolated:
        after = ep.marker_path(home, ref.session_id).stat()
        assert (after.st_ino, after.st_mtime_ns) == (marker_stat.st_ino, marker_stat.st_mtime_ns)


@pytest.mark.asyncio
async def test_new_evaluation_enrolled_before_commit_returns_or_owner_publishes(world, monkeypatch):
    home, owner, actor, params = world
    from gateway import session_local_recovery
    restore = session_local_recovery.restore_local_session
    observed = []

    def checked_restore(authority, sid):
        assert ep.enrolled(home, sid)
        assert local_receipt(authority.db, sid)['policy']['evaluation_isolation'] is True
        assert sid not in authority.sessions
        observed.append(sid)
        return restore(authority, sid)

    monkeypatch.setattr(session_local_recovery, 'restore_local_session', checked_restore)
    ref = create_local_session(owner, actor, {**params, 'evaluation_isolation': 1})
    assert observed == [ref.session_id]
    assert owner.agent(ref) is None


@pytest.mark.asyncio
async def test_same_policy_bad_credential_retry_does_not_touch_enrollment(world, monkeypatch):
    _home, owner, actor, params = world
    params = {**params, 'evaluation_isolation': 1, 'api_key': 'synthetic-launch-key'}
    ref = create_local_session(owner, actor, params)
    before = copy.deepcopy(local_receipt(owner.db, ref.session_id))
    enrollments = []
    monkeypatch.setattr(ep, 'enroll', lambda *args: enrollments.append(args))
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        create_local_session(owner, actor, {**params, 'api_key': 'different-synthetic-key'})
    assert enrollments == []
    assert local_receipt(owner.db, ref.session_id) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('first_isolated', [False, True], ids=['production-wins', 'evaluation-wins'])
async def test_conflicting_staged_creates_enroll_only_the_committed_policy(world, first_isolated):
    home, owner, actor, params = world
    production = stage_local_session(owner, actor, params)
    evaluation = stage_local_session(owner, actor, {**params, 'evaluation_isolation': 1})
    sid = production['session_id']
    assert sid == evaluation['session_id']
    assert not ep.enrolled(home, sid), 'staging must not enroll a competing uncommitted route'
    first, second = (evaluation, production) if first_isolated else (production, evaluation)
    commit_local_session(owner.db, epoch=owner.epoch, receipt=first)
    before = copy.deepcopy(local_receipt(owner.db, sid))
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        commit_local_session(owner.db, epoch=owner.epoch, receipt=second)
    assert local_receipt(owner.db, sid) == before
    assert before['policy']['evaluation_isolation'] is first_isolated
    assert ep.enrolled(home, sid) is first_isolated


@pytest.mark.asyncio
async def test_concurrent_exact_evaluation_creates_enroll_once(world, monkeypatch):
    home, owner, actor, params = world
    one = stage_local_session(owner, actor, {**params, 'evaluation_isolation': 1})
    two = stage_local_session(owner, actor, {**params, 'evaluation_isolation': 1})
    enroll = ep.enroll
    calls = []

    def record(*args):
        calls.append(args)
        return enroll(*args)

    monkeypatch.setattr(ep, 'enroll', record)
    results = await asyncio.gather(*[
        asyncio.to_thread(commit_local_session, owner.db, epoch=owner.epoch, receipt=receipt)
        for receipt in (one, two)])
    assert results[0] == results[1]
    assert calls == [(str(home), one['session_id'])]
    assert ep.enrolled(home, one['session_id'])


@pytest.mark.asyncio
@pytest.mark.parametrize('mixed', [False, True], ids=['exact-evaluation', 'mixed-policy'])
async def test_process_creators_agree_on_one_committed_enrollment(world, mixed):
    home, owner, actor, params = world
    receipts = [stage_local_session(owner, actor, {**params, 'evaluation_isolation': 1}),
                stage_local_session(owner, actor, params if mixed else {**params, 'evaluation_isolation': 1})]
    worker = home / 'commit-worker.py'
    worker.write_text('''import json, sys
from pathlib import Path
from hermes_state import SessionDB
from hermes_state_local import commit_local_session
from hermes_state_runtime import RuntimeStoreError
request = json.loads(sys.stdin.readline())
db = SessionDB(db_path=Path(request['db']))
print('ready', flush=True)
assert sys.stdin.readline().strip() == 'commit'
try:
    commit_local_session(db, epoch=request['epoch'], receipt=request['receipt'])
except RuntimeStoreError as exc:
    print(exc.reason, flush=True)
else:
    print('committed', flush=True)
finally:
    db.close()
''')
    env = {**os.environ, 'HERMES_HOME': str(home),
           'PYTHONPATH': str(Path(__file__).resolve().parents[2]), 'PYTHONDONTWRITEBYTECODE': '1'}
    children = []
    try:
        for receipt in receipts:
            child = await asyncio.create_subprocess_exec(
                sys.executable, str(worker), cwd=home, env=env,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
            children.append(child)
            child.stdin.write((json.dumps({'db': str(owner.db.db_path), 'epoch': owner.epoch,
                                           'receipt': receipt}) + '\n').encode())
            await child.stdin.drain()
            ready = await asyncio.wait_for(child.stdout.readline(), timeout=20)
            assert ready.strip() == b'ready', (await child.stderr.read()).decode()
        for child in children:
            child.stdin.write(b'commit\n')
            await child.stdin.drain()
        outcomes = []
        for child in children:
            stdout, stderr = await asyncio.wait_for(child.communicate(), timeout=20)
            assert child.returncode == 0, stderr.decode()
            outcomes.append(stdout.decode().strip())
    finally:
        for child in children:
            if child.returncode is None:
                child.kill()
                await asyncio.wait_for(child.wait(), timeout=10)
    assert sorted(outcomes) == (['committed', 'invalid_params'] if mixed else ['committed', 'committed'])
    saved = local_receipt(owner.db, receipts[0]['session_id'])
    assert ep.enrolled(home, saved['session_id']) is saved['policy']['evaluation_isolation']
    assert saved['session_id'] not in owner.sessions


@pytest.mark.asyncio
async def test_enrollment_failure_leaves_no_route_or_publication(world, monkeypatch):
    home, owner, actor, params = world
    from gateway.session_local_recovery import local_identity
    sid = local_identity(owner.profile_id, actor.subject, params['request_id'])

    def refuse(*args):
        raise OSError('synthetic enrollment failure')

    monkeypatch.setattr(ep, 'enroll', refuse)
    with pytest.raises(RuntimeStoreError, match='evaluation_isolation_unproven'):
        create_local_session(owner, actor, {**params, 'evaluation_isolation': 1})
    assert owner.db.get_session(sid) is None
    assert sid not in owner.sessions
    assert not ep.enrolled(home, sid)
