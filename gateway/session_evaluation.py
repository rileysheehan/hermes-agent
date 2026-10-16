"""Owner-side admission/transport guards for an evaluation's frozen policy."""
from agent import evaluation_policy as ep
from hermes_state_runtime import RuntimeStoreError


def policy_for(authority, ref):
    # Read the durable receipt, not a viewer's request or a mutable agent attribute.
    from hermes_state_local import local_receipt
    from gateway.session_policy import restore_policy
    receipt = local_receipt(authority.db, ref.session_id)
    return restore_policy(receipt['policy']) if receipt else None


def locked(authority, ref):
    # Non-local production sessions have no local receipt. Missing/corrupt local
    # receipts still raise from local_receipt; never catch that as "not isolated".
    live = authority.sessions.get(ref.session_id)
    if live is None or live.source is None:
        return False
    from gateway.config import Platform
    if live.source.platform != Platform.LOCAL:
        return False
    policy = policy_for(authority, ref)
    return policy is not None and policy.evaluation_isolation


def proof(authority, ref, policy):
    if not policy or not policy.evaluation_isolation:
        return {}
    try:
        ep.require_provider(authority.profile_id, policy.config(authority, secrets=False))
        if not ep.enrolled(authority.profile_id, ref.session_id):
            raise ep.EvaluationIsolationError('missing enrollment')
    except (OSError, ep.EvaluationIsolationError) as exc:
        raise RuntimeStoreError('evaluation_isolation_unproven') from exc
    return {'evaluation_isolation': {'version': 1, 'session_id': ref.session_id,
            'enforced': True, 'read_only': True, 'tools': [],
            'auxiliary_generation': False, 'background_review': False}}


SAFE_METHODS = frozenset({'session.resume', 'session.info', 'session.detach', 'session.events.since',
                         'prompt.submit', 'prompt.receipt', 'prompt.cancel', 'session.interrupt',
                         'prompt.resolve_unknown', 'session.stats'})


def guard_method(authority, ref, method):
    if locked(authority, ref) and method not in SAFE_METHODS:
        raise RuntimeStoreError('evaluation_policy_locked')


def guard_submit(authority, actor, request):
    if not locked(authority, request.ref):
        return
    policy = policy_for(authority, request.ref)
    proof(authority, request.ref, policy)
    # No attachments, API overrides, operator rebind, hidden turns or review debt.
    if (set(request.payload) - {'text', 'finite', 'unattended'}
            or request.payload.get('finite') is not True
            or not isinstance(request.payload.get('text'), str)
            or request.payload['text'].lstrip().startswith('/')):
        raise RuntimeStoreError('evaluation_policy_locked')
    from hermes_state_runtime import list_session_admissions
    rows = list_session_admissions(authority.db, session_id=request.ref.session_id, pending_only=False)
    if any(row['request_id'] != request.request_id or row['principal_id'] != actor.subject for row in rows):
        raise RuntimeStoreError('evaluation_turn_already_admitted')
