"""Real owner/database controls for evaluation creation and durable admission."""
import hashlib
from types import SimpleNamespace

import pytest

from agent import evaluation_policy as ep
from gateway import run
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionStore
from gateway.session_authority import initialize_session_authority
from gateway.session_contract import Principal, Submission
from gateway.session_local import create_local_session, local_session_info
from gateway.session_evaluation import guard_submit, guard_method
from hermes_state_runtime import RuntimeStoreError


@pytest.mark.asyncio
async def test_real_creation_precedes_prewarm_and_cold_resume_preserves_lock(tmp_path, monkeypatch):
    import hermes_constants
    monkeypatch.setattr(hermes_constants, 'get_hermes_home', lambda: tmp_path)
    plugin = tmp_path / 'plugins/hindsight/__init__.py'
    plugin.parent.mkdir(parents=True)
    plugin.write_text('# synthetic provider, never imported\n')
    monkeypatch.setattr(ep, 'PLUGIN_SHA256', hashlib.sha256(plugin.read_bytes()).hexdigest())
    cfg = {'model': {'default': 'fixture', 'provider': 'custom'}, 'memory': {'provider': 'hindsight'},
           'platform_toolsets': {'cli': []}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a, **k: cfg)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda *a, **k: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(Platform.LOCAL)
    owner = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')
    actor = Principal('human', str(tmp_path), frozenset({'session:create','session:read','session:submit','session:control'}), 'wire')
    params = {'request_id': 'evaluation', 'source': 'oneshot', 'model': 'fixture',
              'cwd': str(tmp_path), 'evaluation_isolation': 1}
    ref = create_local_session(owner, actor, params)
    assert owner.agent(ref) is None
    assert ep.enrolled(tmp_path, ref.session_id)
    assert local_session_info(owner, ref)['evaluation_isolation']['enforced'] is True
    with pytest.raises(RuntimeStoreError):
        create_local_session(owner, actor, {k:v for k,v in params.items() if k != 'evaluation_isolation'})
    guard_submit(owner, actor, Submission('one', ref, {'text': 'synthetic', 'finite': True}, 'queue'))
    with pytest.raises(RuntimeStoreError):
        guard_submit(owner, actor, Submission('one', ref, {'text': '/curator', 'finite': True}, 'queue'))
    # Committed terminal evidence (not just pending rows) prevents a second model turn.
    from hermes_state_runtime import admit_session_input
    row = admit_session_input(owner.db, epoch=owner.epoch, principal_id=actor.subject,
                              session_id=ref.session_id, request_id='one',
                              payload={'text': 'synthetic', 'finite': True}, intent='queue')
    owner.db._execute_write(lambda c: c.execute("UPDATE session_admissions SET status='terminal' WHERE admission_id=?", (row['admission_id'],)))
    with pytest.raises(RuntimeStoreError, match='already_admitted'):
        guard_submit(owner, actor, Submission('two', ref, {'text': 'synthetic', 'finite': True}, 'queue'))
    # New authority restores the policy, never a profile default or an empty mutable cache.
    runner.adapters = {}
    cold = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='second')
    from gateway.session_local_recovery import restore_local_session
    restore_local_session(cold, ref.session_id)
    assert local_session_info(cold, ref)['evaluation_isolation']['enforced'] is True
    with pytest.raises(RuntimeStoreError, match='already_admitted'):
        guard_submit(cold, actor, Submission('two', ref, {'text': 'synthetic', 'finite': True}, 'queue'))
    for method in ('slash.exec', 'session.tools', 'session.mutate', 'session.compress'):
        with pytest.raises(RuntimeStoreError, match='locked'):
            guard_method(cold, ref, method)
    ep.marker_path(tmp_path, ref.session_id).unlink()
    with pytest.raises(RuntimeStoreError, match='unproven'):
        local_session_info(cold, ref)


@pytest.mark.asyncio
async def test_wrong_provider_refuses_before_session_commit_or_enrollment(tmp_path, monkeypatch):
    import hermes_constants
    monkeypatch.setattr(hermes_constants, 'get_hermes_home', lambda: tmp_path)
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a, **k:
                        {'model': {'default':'fixture'}, 'memory': {'provider':'other'}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda *a, **k: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(Platform.LOCAL)
    owner = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')
    actor = Principal('human', str(tmp_path), frozenset({'session:create','session:read'}), 'wire')
    def census():
        with owner.db._read_ctx() as c:
            return [tuple(row) for row in c.execute('SELECT id FROM sessions ORDER BY id')]
    before = census()
    with pytest.raises(RuntimeStoreError, match='unproven'):
        create_local_session(owner, actor, {'request_id':'no', 'source':'oneshot','model':'fixture',
                             'cwd':str(tmp_path), 'evaluation_isolation':1})
    assert census() == before
    assert not (tmp_path/'hindsight/read-only-sessions').exists()
