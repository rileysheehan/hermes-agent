"""MER-1371: a producer input queued behind a finite turn never starts an unattended turn.

With ``gateway.finite_stop_on_disconnect`` on, a finite turn (``chat -q``, a board run) belongs to its
submitting process. Its background process's completion was admitted behind it and, claimed the moment
the turn settled while its client was still attached, ran as a fresh turn with no client and no board
run (49 model calls after a timed-out run was stopped). Now every producer input whose newest preceding
client input is finite is a note (a liveness tick: settled with neither), while a successor client turn,
finite or not, runs as before.
"""
import json
import os

import pytest

from tests.gateway.test_session_end_delegations import _PROCESS, _admissions, _deliver, _local, _notes
from tests.gateway.test_unviewed_completion_notes import _target, _turns


@pytest.fixture(autouse=True)
def _plain_config_reader(monkeypatch):
    from ruamel.yaml import YAML

    def read(path=None):
        return YAML(typ="safe").load(open(path)) if path and os.path.exists(path) else {}
    monkeypatch.setattr("gateway.run._load_gateway_config", read)


def _setting(tmp_path, on=True):
    (tmp_path / "config.yaml").write_text(f"gateway:\n  finite_stop_on_disconnect: {'true' if on else 'false'}\n")


def _client(authority, ref, request_id, text, finite):
    from hermes_state_runtime import admit_session_input
    payload = {'text': text, 'finite': True} if finite else {'text': text}
    return admit_session_input(authority.db, epoch=authority.epoch, principal_id='test-owner',
                               session_id=ref.session_id, request_id=request_id, payload=payload)


def _attach(authority, ref, actor):
    """The finite client is still subscribed when the drain claims the row behind its turn."""
    authority.sessions[ref.session_id].subscribers['finite-client'] = actor


async def _admit(runner, authority, ref, identity, text):
    from gateway.platforms.event import MessageEvent
    live = authority.sessions[ref.session_id]
    entry = runner.session_store.lookup_by_session_key(live.route)
    event = MessageEvent(text=text, source=live.source, internal=True, metadata={
        'gateway_session_key': live.route, 'gateway_session_id': entry.session_id})
    await authority.admit_automation(runner._adapter_for_source(live.source), event, identity)


_COMPLETION_TEXT = f'[finished] RESULT_{_PROCESS[1]}'


@pytest.mark.asyncio
async def test_completion_behind_a_finite_turn_is_a_note_with_its_client_attached(tmp_path, monkeypatch):
    """The MER-1371 gap: viewer still attached at claim, so the viewer rule alone ran it as a turn."""
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    _setting(tmp_path)
    try:
        target = _target(runner, authority, ref)
        _attach(authority, ref, actor)
        _client(authority, ref, 'board-run', 'BOARD', finite=True)
        await _deliver(runner, authority, ref, _PROCESS)
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert seen == [('BOARD', 0)], 'the completion behind the finite turn ran as its own turn'
        assert [m['content'] for m in _notes(authority, target)] == [_COMPLETION_TEXT]
        assert [r['outcome'] for r in _admissions(authority, ref)] == ['completed']
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_late_completion_after_the_stopped_client_left_is_a_note(tmp_path, monkeypatch):
    """The incident's own order: the finite turn ran, its client left, then the completion arrived."""
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    _setting(tmp_path)
    try:
        target = _target(runner, authority, ref)
        _attach(authority, ref, actor)
        _client(authority, ref, 'board-run', 'BOARD', finite=True)
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        authority.sessions[ref.session_id].subscribers.clear()
        await _deliver(runner, authority, ref, _PROCESS)
        await authority._drain(ref)
        assert seen == [('BOARD', 0)]
        assert len(_notes(authority, target)) == 1
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_successor_finite_turn_runs_and_owns_the_completions_after_it(tmp_path, monkeypatch):
    """A retry that resumes the session (``--resume``) is a client input: it runs, and neither the old
    run's completion nor its own becomes a turn that overlaps or outlives it."""
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    _setting(tmp_path)
    try:
        target = _target(runner, authority, ref)
        _attach(authority, ref, actor)
        _client(authority, ref, 'run-1', 'FIRST RUN', finite=True)
        await _deliver(runner, authority, ref, _PROCESS)
        _client(authority, ref, 'run-2', 'RETRY RUN', finite=True)
        await _deliver(runner, authority, ref, ('completion', 'proc_retry', 1700000001.0))
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert seen == [('FIRST RUN', 0), ('RETRY RUN', 1)], 'the successor saw the old note and ran'
        assert len(_notes(authority, target)) == 2
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_interactive_input_after_a_finite_turn_makes_completions_turns_again(tmp_path, monkeypatch):
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    _setting(tmp_path)
    try:
        target = _target(runner, authority, ref)
        _attach(authority, ref, actor)
        _client(authority, ref, 'run-1', 'BOARD', finite=True)
        _client(authority, ref, 'chat', 'HELLO', finite=False)
        await _deliver(runner, authority, ref, _PROCESS)
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert [text for text, _ in seen] == ['BOARD', 'HELLO', _COMPLETION_TEXT]
        assert _notes(authority, target) == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_setting_off_keeps_the_viewer_rule(tmp_path, monkeypatch):
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    _setting(tmp_path, on=False)
    try:
        target = _target(runner, authority, ref)
        _attach(authority, ref, actor)
        _client(authority, ref, 'run-1', 'BOARD', finite=True)
        await _deliver(runner, authority, ref, _PROCESS)
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert [text for text, _ in seen] == ['BOARD', _COMPLETION_TEXT]
        assert _notes(authority, target) == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_liveness_tick_behind_a_finite_turn_settles_with_neither_note_nor_turn(tmp_path, monkeypatch):
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    _setting(tmp_path)
    try:
        target = _target(runner, authority, ref)
        _attach(authority, ref, actor)
        _client(authority, ref, 'run-1', 'BOARD', finite=True)
        await _admit(runner, authority, ref, '["heartbeat","proc_tick","evt-1"]', '[running] RESULT_tick')
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert seen == [('BOARD', 0)]
        assert _notes(authority, target) == []
        assert [r['outcome'] for r in _admissions(authority, ref)] == ['completed']
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_liveness_tick_after_an_interactive_turn_still_runs(tmp_path, monkeypatch):
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    _setting(tmp_path)
    try:
        target = _target(runner, authority, ref)
        _attach(authority, ref, actor)
        _client(authority, ref, 'chat', 'HELLO', finite=False)
        await _admit(runner, authority, ref, '["heartbeat","proc_tick","evt-1"]', '[running] RESULT_tick')
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert [text for text, _ in seen] == ['HELLO', '[running] RESULT_tick']
    finally:
        authority.db.close()


def test_only_a_three_part_heartbeat_is_a_process_tick():
    from gateway.session_completion_notes import _process_tick

    def row(identity):
        return {'payload': {'local_automation_v1': {'identity': identity}}}
    assert _process_tick(row('["heartbeat","proc_x","evt-1"]'))
    assert not _process_tick(row(json.dumps(['heartbeat', 'sess', 1.0, 3])))  # the user's /heartbeat wake
    assert not _process_tick(row('not json'))
    assert not _process_tick({'payload': {'text': 'x'}})


def test_unreadable_row_fails_closed_only_with_the_setting_on(tmp_path):
    import types
    from gateway.session_completion_notes import follows_finite_turn
    authority = types.SimpleNamespace(db=types.SimpleNamespace(db_path=str(tmp_path / 'state.db')))
    _setting(tmp_path, on=False)
    assert follows_finite_turn(authority, {}) is False
    _setting(tmp_path)
    assert follows_finite_turn(authority, {}) is True
    assert follows_finite_turn(authority, {'seq': True, 'target_session_id': 's'}) is True


def _settle_claimed(authority, ref, outcome):
    """Claim the FIFO head and settle it as its run would (``interrupted``: the stopped board run)."""
    from hermes_state_runtime import claim_session_input, settle_session_input
    claimed = claim_session_input(authority.db, epoch=authority.epoch, session_id=ref.session_id)
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=claimed['admission_id'],
                         generation=claimed['generation'], outcome=outcome)


def _withdraw(authority, ref, request_id, text, finite):
    from hermes_state_runtime import cancel_session_input
    row = _client(authority, ref, request_id, text, finite)
    cancelled = cancel_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'])
    assert cancelled['outcome'] == 'cancelled' and cancelled['generation'] is None
    return cancelled


@pytest.mark.asyncio
@pytest.mark.parametrize('tick', [False, True], ids=['completion-attached', 'tick-unviewed'])
@pytest.mark.parametrize('interrupted', [False, True], ids=['completed-finite', 'interrupted-finite'])
async def test_withdrawn_interactive_input_leaves_the_finite_boundary(tmp_path, monkeypatch, tick, interrupted):
    """MER-1528 review: an interactive prompt cancelled while queued never ran, so it is not the client
    turn that makes producer inputs turns again."""
    from hermes_state_runtime import get_session_admission
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    _setting(tmp_path)
    try:
        target = _target(runner, authority, ref)
        _attach(authority, ref, actor)
        _client(authority, ref, 'run-1', 'BOARD', finite=True)
        seen = _turns(runner, authority, target)
        if interrupted:
            _settle_claimed(authority, ref, 'interrupted')
        else:
            await authority._drain(ref)
        withdrawn = _withdraw(authority, ref, 'withdrawn', 'NEVER RUN THIS', finite=False)
        if tick:
            authority.sessions[ref.session_id].subscribers.clear()
            await _admit(runner, authority, ref, '["heartbeat","proc_tick","evt-1"]', '[running] RESULT_tick')
        else:
            await _deliver(runner, authority, ref, _PROCESS)
        await authority._drain(ref)
        assert seen == ([] if interrupted else [('BOARD', 0)]), f'a withdrawn prompt re-enabled a producer turn: {seen!r}'
        assert get_session_admission(authority.db, admission_id=withdrawn['admission_id'])['outcome'] == 'cancelled'
        assert len(_notes(authority, target)) == (0 if tick else 1)
        assert [r['outcome'] for r in _admissions(authority, ref)] == ['completed']
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_withdrawn_finite_successor_does_not_override_an_interactive_turn(tmp_path, monkeypatch):
    """The incident's row 1158 shape the other way round: a finite run withdrawn before it started is
    not a finite turn either, so the interactive turn before it still decides."""
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    _setting(tmp_path)
    try:
        target = _target(runner, authority, ref)
        _attach(authority, ref, actor)
        _client(authority, ref, 'chat', 'HELLO', finite=False)
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        _withdraw(authority, ref, 'run-2', 'WITHDRAWN RUN', finite=True)
        await _deliver(runner, authority, ref, _PROCESS)
        await authority._drain(ref)
        assert [text for text, _ in seen] == ['HELLO', _COMPLETION_TEXT]
        assert _notes(authority, target) == []
    finally:
        authority.db.close()
