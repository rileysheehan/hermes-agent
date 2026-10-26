"""A completion for a local session no viewer is watching is written into it as a note (#134889 LC-7).

Main's CLI and TUI hosted async delegations and background processes in their own process, so once
the user left nothing started a turn for them. Under the gateway the owner outlives its viewers. The
completion is admitted like any producer input. Producers still coalesce a batch into one admission
under its primary identity, the siblings' identities in its descriptor: a lookup by any of them finds
it (``completion_admission``), results already admitted are left out of a new batch, and the one
outcome, note or turn, covers the whole batch. When that admission comes up for execution with no
viewer attached (the authority's own subscriber records) the executor records it in the session's
history, where the next turn sees it, and settles the admission without a model call. Queue order, restart survival and
exactly-once are the admission contract's. With a viewer attached it runs as the turn it always was;
messaging, cron, Kanban, ``/loop`` and heartbeat wakes never reach here.
"""
import json
import logging
from functools import partial

logger = logging.getLogger(__name__)

# ``_completion_delivery_identity`` kinds: an async delegation's result, a background process's exit
# and its watch-pattern notices.
_NOTED_PRODUCERS = frozenset({'async_delegation', 'completion', 'watch_match', 'watch_disabled'})


def _producer(identity):
    try:
        return json.loads(identity)[0]
    except (ValueError, TypeError, IndexError, KeyError):
        return None


def _process_tick(row):
    """*row* is a background process's liveness tick, ``["heartbeat", process, event_id]``; the
    ``/heartbeat`` wake a user schedules on the session has four parts and is not one."""
    descriptor = (row or {}).get('payload', {}).get('local_automation_v1')
    try:
        identity = json.loads(descriptor['identity'])
    except (ValueError, TypeError, KeyError):
        return False
    return isinstance(identity, list) and len(identity) == 3 and identity[0] == 'heartbeat'


def noted_completion(row, authority=None):
    """*row* is a local completion admission: a note when it runs with no viewer attached. With
    *authority*, also a process liveness tick queued behind a finite turn (``follows_finite_turn``),
    which runs as neither a note nor a turn."""
    descriptor = (row or {}).get('payload', {}).get('local_automation_v1')
    if descriptor is None:
        return False
    if _producer(descriptor['identity']) in _NOTED_PRODUCERS:
        return True
    return authority is not None and _process_tick(row) and follows_finite_turn(authority, row)


def follows_finite_turn(authority, row):
    """``gateway.finite_stop_on_disconnect``: the newest client input before producer *row* was a
    finite turn (``chat -q``), so *row* belongs to that run and is never a turn of its own (MER-1371).

    A finite turn is its submitting process's: an orchestrator that kills or times out that process
    expects the work to stop, as when the turn ran in-process and its background processes' wakes
    died with it. The viewer check alone misses two cases: a completion queued behind the finite turn
    is claimed the moment that turn settles, while its client is still attached (it detaches only
    after reading the result), and a liveness tick is never noted. Either one started an unattended
    turn with no client and no board run (MER-1371: 49 model calls after the stop). Only a later client
    input (an interactive turn, a message) makes producer inputs turns again; a successor finite turn
    (``--resume``) is a client input and runs normally, and the producer inputs after it are its own.
    Unreadable: treated as following a finite turn, so the failure is a note, never an unattended turn."""
    from gateway.session_finite import stop_on_disconnect
    if not stop_on_disconnect(authority):
        return False
    seq, target = row.get('seq'), row.get('target_session_id')
    if type(seq) is not int or not isinstance(target, str):
        return True
    try:
        with authority.db._read_ctx() as conn:
            prior = conn.execute(
                "SELECT payload_json FROM session_admissions WHERE target_session_id=? AND seq<? "
                "AND substr(principal_id,1,11) != 'automation:' ORDER BY seq DESC LIMIT 1",
                (target, seq)).fetchone()
        if prior is None:
            return False
        payload = json.loads(prior[0])
    except Exception:  # health: allow BLE001 -- unreadable means fail closed: a note, not a turn
        return True
    return not isinstance(payload, dict) or payload.get('finite') is True


async def note_if_unviewed(authority, ref, row):
    """The claimed *row*'s response when it was written as a note instead of running, else None.
    It is the FIFO head, so the note lands after every input committed before it and before every
    input committed after; the viewer check and the decision share one loop step. A producer input
    queued behind a finite turn is noted even with that turn's client still attached, and its
    liveness tick is settled with neither a note nor a turn (``follows_finite_turn``)."""
    noted, tick = noted_completion(row), _process_tick(row)
    if not (noted or tick):
        return None
    after_finite = follows_finite_turn(authority, row)
    if tick and not after_finite:
        return None
    from gateway.session_runtime_workers import tracked_write
    live = authority.sessions.get(ref.session_id)
    if tick:
        logger.info("Process tick %s queued behind finite turn on %s: settled without a turn (MER-1371)",
                    row['admission_id'], ref.session_id)
        authority.pending_results[row['admission_id']] = {'result': {'final_response': '', 'messages': []}, 'usage': {}}
        return ''
    if live is None or (live.subscribers and not after_finite):
        # The claim left a closed row closed for a note; the turn reopens it as any first turn does.
        from gateway.session_local_recovery import reopen_local_session
        await tracked_write(authority, partial(reopen_local_session, authority, ref))
        return None
    from gateway.session_viewer_close import _physical_target
    descriptor = row['payload']['local_automation_v1']
    metadata = {'completion_note': sorted(set(descriptor.get('identities') or ()) | {descriptor['identity']})}
    metadata.update({key: descriptor[key] for key in ('display_text', 'notification_category') if key in descriptor})
    target = _physical_target(authority, ref.session_id)
    # Keyed by its admission: a claim is never replayed, and this keeps a write retry exact.
    await tracked_write(authority, partial(
        authority.db.append_completion_note, target, row['admission_id'], row['payload']['text'],
        descriptor.get('display_kind', 'internal_notification'), metadata))
    refresh = getattr(authority.runner, '_refresh_agent_cache_message_count', None)
    if refresh is not None:  # the owner's own row: a cached agent stays, as after a turn's writes
        await refresh(live.route, target)
    authority.pending_results[row['admission_id']] = {'result': {'final_response': '', 'messages': []}, 'usage': {}}
    return ''
