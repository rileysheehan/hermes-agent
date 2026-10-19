"""Watch matches through the REAL local source and admission path, driven by the real drain (MER-1177).

Only the clock is replaced. A suspended routing entry is the real refusal used for the transient and
never-clearing cases (``_owner`` refuses it ``not_found``); nothing bypasses authorization.
"""
import logging
import queue

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_contract import Principal
from hermes_state_runtime import list_session_admissions


async def _authority(tmp_path, monkeypatch, db=None, instance='watch-fixture'):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    from hermes_state import SessionDB
    db = db or SessionDB(tmp_path / 'state.db')
    runner.session_store._db = db
    runner._session_db = db
    runner._load_background_notifications_mode = lambda: 'all'
    authority = await initialize_session_authority(runner, profile_id='default', instance_id=instance, db=db)
    return runner, authority


async def _local_session(tmp_path, monkeypatch, request_id='board'):
    from gateway.session_local import create_local_session
    runner, authority = await _authority(tmp_path, monkeypatch)
    actor = Principal('test-owner', 'default', frozenset({'session:create', 'session:submit', 'session:read'}),
                      'fixture')
    ref = create_local_session(authority, actor, dict(request_id=request_id, source='cli', model='fixture',
                                                      toolsets=[]))
    return runner, authority, ref, authority.sessions[ref.session_id].route


def _event(route, sid, n=1, proc='proc_watch'):
    """The fields ``ProcessRegistry._watch_event_base`` + ``_check_watch_patterns`` put on a match."""
    return {'type': 'watch_match', 'event_id': f'evt-{n}', 'started_at': 1.0, 'parent_session_id': sid,
            'session_id': proc, 'session_key': route, 'task_id': '', 'owner_task_id': '', 'command': 'sleep 1',
            'pattern': 'READY', 'output': 'READY', 'suppressed': 0, 'platform': 'local', 'chat_type': 'dm',
            'chat_id': sid, 'user_id': '', 'user_name': '', 'thread_id': ''}


def _automation_rows(authority, sid):
    return [r for r in list_session_admissions(authority.db, session_id=sid, pending_only=False)
            if 'local_automation_v1' in r['payload']]


async def _drive(runner, q, clock, monkeypatch, until, refuse_until=None, entry=None, step=2.0):
    """Drain every ``step`` seconds of fake time (the watcher's tick), as the gateway loop does."""
    from gateway import run_notifications
    monkeypatch.setattr(run_notifications, '_watch_retry_clock', lambda: clock['now'])
    while clock['now'] <= until and not q.empty():
        if entry is not None:
            entry.suspended = refuse_until is None or clock['now'] < refuse_until
        await runner._drain_watch_notifications(q)
        clock['now'] += step


def _lines(caplog, word):
    return [r.getMessage() for r in caplog.records if f'Watch notification {word}' in r.getMessage()]


@pytest.mark.asyncio
async def test_live_local_session_admits_on_the_first_drain(tmp_path, monkeypatch, caplog):
    runner, authority, ref, route = await _local_session(tmp_path, monkeypatch)
    q, clock = queue.Queue(), {'now': 0.0}
    q.put(_event(route, ref.session_id))
    try:
        with caplog.at_level(logging.INFO, logger='gateway.run'):
            await _drive(runner, q, clock, monkeypatch, until=10)
        assert q.empty()
        assert len(_automation_rows(authority, ref.session_id)) == 1
        assert _lines(caplog, 'refused') == []
        assert _lines(caplog, 'admitted') == [
            f'Watch notification admitted for local chat={ref.session_id} event=evt-1 proc=proc_watch '
            'type=watch_match attempts=1']
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('cold', [False, True])
async def test_ended_local_session_is_not_refused_warm_or_cold(tmp_path, monkeypatch, caplog, cold):
    """``cli_close`` alone does not make the real admission path refuse, in the same gateway or after a
    restart's cold restore. So "ended" is not, by itself, the permanent condition behind Mercury's loop."""
    runner, authority, ref, route = await _local_session(tmp_path, monkeypatch)
    authority.db.end_session(ref.session_id, 'cli_close')
    if cold:
        runner, authority = await _authority(tmp_path, monkeypatch, db=authority.db, instance='after-restart')
    q, clock = queue.Queue(), {'now': 0.0}
    q.put(_event(route, ref.session_id))
    try:
        with caplog.at_level(logging.INFO, logger='gateway.run'):
            await _drive(runner, q, clock, monkeypatch, until=10)
        assert q.empty()
        assert len(_automation_rows(authority, ref.session_id)) == 1
        assert _lines(caplog, 'refused') == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_transient_real_refusal_for_300s_still_delivers(tmp_path, monkeypatch, caplog):
    """The review's control: a refusal that clears at 300 s must still deliver (the 80bb92a head dropped
    it at 240 s). Backoff only delays the retry; the event is admitted once, within one 60 s step."""
    runner, authority, ref, route = await _local_session(tmp_path, monkeypatch)
    entry = runner.session_store.lookup_by_session_key(route)
    q, clock = queue.Queue(), {'now': 0.0}
    q.put(_event(route, ref.session_id))
    try:
        with caplog.at_level(logging.INFO, logger='gateway.run'):
            await _drive(runner, q, clock, monkeypatch, until=600, refuse_until=300, entry=entry)
        assert q.empty()
        assert len(_automation_rows(authority, ref.session_id)) == 1
        refused = _lines(caplog, 'refused')
        assert refused and all('code=not_found' in line and 'event=evt-1' in line for line in refused)
        assert len(refused) <= 9  # 0, 4, 12, 28, 60, 120, 180, 240, 300 s; the base tree tried 150 times
        admitted = _lines(caplog, 'admitted')
        assert len(admitted) == 1 and admitted[0].endswith(f'attempts={len(refused) + 1}')
        assert clock['now'] <= 362
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_refusal_that_never_clears_is_kept_and_bounded_per_event(tmp_path, monkeypatch, caplog):
    """Two events of one process on one chat plus a second chat: each is kept (never dropped) and retried
    at most about once a minute, and every refusal line names its own event."""
    runner, authority, ref, route = await _local_session(tmp_path, monkeypatch)
    entry = runner.session_store.lookup_by_session_key(route)
    q, clock = queue.Queue(), {'now': 0.0}
    for n in (1, 2):
        q.put(_event(route, ref.session_id, n))
    try:
        with caplog.at_level(logging.INFO, logger='gateway.run'):
            await _drive(runner, q, clock, monkeypatch, until=2 * 3600, refuse_until=None, entry=entry)
        assert q.qsize() == 2  # still queued: nothing dropped
        refused = _lines(caplog, 'refused')
        for n in (1, 2):
            mine = [line for line in refused if f'event=evt-{n} ' in line]
            # 2 h: 5 ramp attempts (0-60 s), then one a minute, so at most about 125, never 3,600.
            assert 100 <= len(mine) <= 126, len(mine)
        assert _automation_rows(authority, ref.session_id) == []
        # Once the route comes back, both deliver exactly once.
        with caplog.at_level(logging.INFO, logger='gateway.run'):
            await _drive(runner, q, clock, monkeypatch, until=clock['now'] + 120, refuse_until=0, entry=entry)
        assert q.empty()
        assert len(_automation_rows(authority, ref.session_id)) == 2
        assert sorted(line.split('event=')[1].split()[0] for line in _lines(caplog, 'admitted')) == ['evt-1', 'evt-2']
    finally:
        authority.db.close()
