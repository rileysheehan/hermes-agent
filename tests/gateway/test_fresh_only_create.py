"""Opt-in creation must not return an old executable session."""
from types import SimpleNamespace

import pytest

from gateway import run
from gateway.session_controls import AuthorityConnection
from hermes_state_runtime import RuntimeStoreError


async def owner(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    # Snapshot metadata needs no provider-capability discovery in an admission test.
    monkeypatch.setattr('gateway.session_local._reasoning_fields', lambda *a, **kw: {})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture')
    return authority, AuthorityConnection(authority, object(), {'user_id': 'owner'})


def request(tmp_path, **extra):
    return dict(request_id='first', source='oneshot', cwd=str(tmp_path), model='fixture',
                toolsets=[], title='bounded worker', **extra)


@pytest.mark.asyncio
async def test_fresh_only_title_collision_never_returns_existing(tmp_path, monkeypatch):
    authority, connection = await owner(tmp_path, monkeypatch)
    try:
        first = await connection.create(None, request(tmp_path))
        again = request(tmp_path, fresh_only=True)
        again['request_id'] = 'second'
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await connection.create(None, again)
        assert authority.db.get_session(first['stored_session_id'])['title'] == 'bounded worker'
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['foreign', 'hidden', 'numbered', 'literal-id', 'ghost'])
async def test_fresh_only_conflicts_leave_all_rows_unchanged(tmp_path, monkeypatch, case):
    authority, connection = await owner(tmp_path, monkeypatch)
    db = authority.db
    try:
        sid = 'bounded worker' if case == 'literal-id' else 'foreign-session'
        db.create_session(sid, source='telegram', user_id='another-user')
        if case != 'literal-id':
            db.set_session_title(sid, 'bounded worker #2' if case == 'numbered' else 'bounded worker')
        if case == 'hidden':
            db.set_session_hidden(sid, True)
        if case == 'ghost':
            db.end_session(sid, 'abandoned')
        before = db.get_session(sid)
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await connection.create(None, request(tmp_path, fresh_only=True))
        assert db.get_session(sid) == before
        with db._read_ctx() as conn:
            assert conn.execute('SELECT count(*) FROM sessions').fetchone()[0] == 1
    finally:
        await connection.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('same_identity', [False, True])
async def test_fresh_only_atomic_two_connections_same_title_or_identity(tmp_path, monkeypatch, same_identity):
    import asyncio
    import copy
    from gateway.session_local import stage_local_session
    from hermes_state import SessionDB
    from hermes_state_local import commit_local_session
    authority, connection = await owner(tmp_path, monkeypatch)
    other = SessionDB(db_path=authority.db.db_path)
    try:
        # Both stage BEFORE either commits; actual SQLite must serialize the final decision.
        params = request(tmp_path, fresh_only=True)
        title = params.pop('title')
        one = stage_local_session(authority, connection.actor, params)
        two = stage_local_session(authority, connection.actor, dict(params, request_id='first' if same_identity else 'second'))
        one['fresh_title'] = two['fresh_title'] = title
        async def commit(db, receipt):
            try:
                await asyncio.to_thread(commit_local_session, db, epoch=authority.epoch, receipt=receipt)
                return 'created'
            except RuntimeStoreError as exc:
                return exc.reason
        assert sorted(await asyncio.gather(commit(authority.db, one), commit(other, two))) == ['admission_conflict', 'created']
        winner = one if authority.db.get_session(one['session_id']) else two
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            # v0.22.5 receipts carry an on_create publication (holds a lock): copy the plain mapping.
            commit_local_session(other, epoch=authority.epoch, receipt=copy.deepcopy(dict(winner)))
        with authority.db._read_ctx() as conn:
            assert conn.execute('SELECT count(*) FROM sessions').fetchone()[0] == 1
            assert conn.execute('SELECT title_source FROM sessions').fetchone()[0] == 'user'
    finally:
        other.close()
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_default_resume_create_and_idempotency_remain_unchanged(tmp_path, monkeypatch):
    authority, connection = await owner(tmp_path, monkeypatch)
    try:
        first = await connection.create(None, request(tmp_path))
        sid = first['stored_session_id']
        authority.db.append_message(sid, 'user', 'old history')
        repeated = await connection.create(None, dict(request(tmp_path), request_id='second', model='other'))
        assert repeated['stored_session_id'] == sid
        assert repeated['messages'][0]['content'] == 'old history'
        from gateway.session_local import create_local_session
        params = request(tmp_path)
        params.pop('title')
        assert create_local_session(authority, connection.actor, params).session_id == sid
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('value', ['true', 1, None, [], {}])
async def test_fresh_only_malformed_values_refuse_even_existing_title(tmp_path, monkeypatch, value):
    authority, connection = await owner(tmp_path, monkeypatch)
    try:
        await connection.create(None, request(tmp_path))
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            await connection.create(None, request(tmp_path, fresh_only=value))
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_fresh_only_stale_epoch_rolls_back_without_row(tmp_path, monkeypatch):
    from hermes_state_runtime import begin_runtime_epoch
    authority, connection = await owner(tmp_path, monkeypatch)
    try:
        begin_runtime_epoch(authority.db, instance_id='new-owner')
        with pytest.raises(RuntimeStoreError):
            await connection.create(None, request(tmp_path, fresh_only=True))
        with authority.db._read_ctx() as conn:
            assert conn.execute('SELECT count(*) FROM sessions').fetchone()[0] == 0
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_fresh_only_new_title_is_committed_with_user_provenance(tmp_path, monkeypatch):
    authority, connection = await owner(tmp_path, monkeypatch)
    try:
        snapshot = await connection.create(None, request(tmp_path, fresh_only=True))
        sid = snapshot['stored_session_id']
        row = authority.db.get_session(sid)
        assert (row['title'], row['title_source'], snapshot['messages']) == ('bounded worker', 'user', [])
        from hermes_state_local import local_receipt
        assert local_receipt(authority.db, sid)['fresh_title'] == 'bounded worker'
        # Repeated creation identity refuses even if a different title is requested.
        repeated = request(tmp_path, fresh_only=True)
        repeated['title'] = 'another title'
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await connection.create(None, repeated)
    finally:
        await connection.close()
        authority.db.close()
