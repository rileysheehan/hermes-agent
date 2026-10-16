"""Offline opt-in evaluation policy gates. Real functions, no model/bank/network."""
from dataclasses import asdict
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import evaluation_policy as ep
from agent.agent_init import _load_tools
from agent.background_review import drain_owed_review, schedule_owed_review_drain
from agent.owed_review import applies
from agent.turn_context import _maybe_title_session_at_turn_start, _refresh_mcp_tools_between_turns
from agent.conversation_loop import _refresh_bot_chat_tools
from agent.conversation_compression import _refresh_agent_tool_definitions
from gateway import session_policy as sp
from gateway import session_evaluation as se
from hermes_state_runtime import RuntimeStoreError
from run_agent import AIAgent

SID = 'local-' + 'a' * 64


@pytest.fixture
def world(tmp_path, monkeypatch):
    import hermes_constants
    monkeypatch.setattr(hermes_constants, 'get_hermes_home', lambda: tmp_path)
    plugin = tmp_path / 'plugins/hindsight/__init__.py'
    plugin.parent.mkdir(parents=True)
    plugin.write_bytes(b'# synthetic pinned provider, never imported\n')
    monkeypatch.setattr(ep, 'PLUGIN_SHA256', hashlib.sha256(plugin.read_bytes()).hexdigest())
    cfg = {'model': {'default': 'fixture', 'provider': 'custom'}, 'memory': {'provider': 'hindsight'},
           'platform_toolsets': {'cli': ['bot_room']}}
    params = {'source': 'oneshot', 'model': 'fixture', 'cwd': str(tmp_path), 'evaluation_isolation': 1}
    return tmp_path, cfg, params


def policy(world):
    _, cfg, params = world
    return sp.build_policy(params, cfg, private_secrets={})


def test_frozen_policy_survives_durable_roundtrip_and_cannot_follow_live_config(world):
    p = policy(world)
    assert p.evaluation_isolation is True
    assert sp.restore_policy(asdict(p)).evaluation_isolation is True
    with pytest.raises(Exception):
        p.evaluation_isolation = False
    data = asdict(p)
    data['evaluation_isolation'] = 'true'
    with pytest.raises(RuntimeStoreError):
        sp.restore_policy(data)
    assert sp.build_policy({k: v for k, v in world[2].items() if k != 'evaluation_isolation'},
                           world[1], private_secrets={}).evaluation_isolation is False


@pytest.mark.parametrize('extra', [
    {'evaluation_isolation': True}, {'evaluation_isolation': 2}, {'evaluation_isolation': None},
    {'source': 'cli'}, {'follow_profile_config': True}, {'ignore_rules': True}, {'safe_mode': True},
    {'ignore_user_config': True}, {'skills': ['x']}, {'checkpoints': True}, {'accept_hooks': True},
    {'editor': {'x': 1}},
])
def test_unsupported_creation_refuses_before_provider_import(world, extra):
    _, cfg, params = world
    with pytest.raises(RuntimeStoreError):
        sp.build_policy({**params, **extra}, cfg, private_secrets={})


def test_wrong_provider_and_managed_executor_are_refused(world):
    _, cfg, params = world
    for bad in ({**cfg, 'memory': {'provider': 'other'}}, {**cfg, 'gateway': {'managed_workers': True}}):
        with pytest.raises(RuntimeStoreError):
            sp.build_policy(params, bad, private_secrets={})


def test_missing_or_changed_provider_fails_closed(world):
    home, cfg, params = world
    (home / 'plugins/hindsight/__init__.py').write_text('# changed')
    with pytest.raises(RuntimeStoreError):
        sp.build_policy(params, cfg, private_secrets={})


def test_construction_before_enrollment_cannot_touch_provider(world):
    home, _, _ = world
    p = policy(world)
    agent = SimpleNamespace()
    with sp.policy_scope(p):
        with pytest.raises(ep.EvaluationIsolationError, match='enrollment'):
            ep.bind_agent(agent, SID)
        ep.enroll(home, SID)
        ep.bind_agent(agent, SID)
    assert agent._evaluation_isolation is True
    assert agent._evaluation_session_id == SID


def test_symlink_marker_is_not_evidence(world):
    home, _, _ = world
    path = ep.marker_path(home, SID)
    path.parent.mkdir(parents=True)
    (home / 'elsewhere').touch()
    path.symlink_to(home / 'elsewhere')
    assert not ep.enrolled(home, SID)
    with pytest.raises(OSError):
        ep.enroll(home, SID)


def locked_agent():
    return SimpleNamespace(_evaluation_isolation=True, _evaluation_session_id=SID, session_id=SID,
                           tools=[], valid_tool_names=set())


def test_deferred_and_provider_tool_injection_cannot_reopen_surface(world, monkeypatch):
    import model_tools
    discover = Mock(side_effect=AssertionError('discovery must not run'))
    monkeypatch.setattr(model_tools, 'get_tool_definitions', discover)
    agent = locked_agent()
    _load_tools(agent, ['bot_room'], [])
    assert agent.tools == [] and agent.valid_tool_names == set()
    assert _refresh_agent_tool_definitions(agent) is False
    _refresh_bot_chat_tools(agent)
    _refresh_mcp_tools_between_turns(agent)
    discover.assert_not_called()


def test_hallucinated_tools_refuse_before_dispatch(world):
    with pytest.raises(ep.EvaluationIsolationError, match='tool dispatch'):
        AIAgent._execute_tool_calls(locked_agent(), Mock(), [], 'synthetic')


def test_no_title_review_claim_or_spawn_even_without_context_or_config(world, monkeypatch):
    from agent import owed_review
    claim = Mock(side_effect=AssertionError('must not claim debt'))
    monkeypatch.setattr(owed_review, 'claim_orphan', claim)
    agent = locked_agent()
    agent._session_db = Mock(side_effect=AssertionError('title must not touch db'))
    _maybe_title_session_at_turn_start(agent, [{'role': 'user', 'content': 'synthetic'}])
    assert not applies(agent, {'carry_owed_reviews': True})
    assert not drain_owed_review(agent, budget=90)
    assert not schedule_owed_review_drain(agent)
    AIAgent._spawn_background_review(agent, [])
    AIAgent._spawn_background_review_now(agent, [])
    claim.assert_not_called()


@pytest.mark.asyncio
async def test_auxiliary_sync_and_async_refuse_before_resolution(world, monkeypatch):
    from agent.auxiliary_client import _relay_auxiliary_call, _relay_auxiliary_call_async
    seen = []
    @_relay_auxiliary_call
    def sync():
        seen.append('sync')
    @_relay_auxiliary_call_async
    async def async_():
        seen.append('async')
    with sp.policy_scope(policy(world)):
        with pytest.raises(ep.EvaluationIsolationError):
            sync()
        with pytest.raises(ep.EvaluationIsolationError):
            await async_()
    assert seen == []
    sync()
    await async_()
    assert seen == ['sync', 'async']


def test_model_payload_gate_detects_changed_tools_marker_or_identity(world):
    home, _, _ = world
    ep.enroll(home, SID)
    agent = locked_agent()
    ep.assert_request(agent, [])
    for mutation in ('tools', 'valid_tool_names', 'identity', 'marker'):
        agent = locked_agent()
        if mutation == 'tools':
            agent.tools = [{'function': {'name': 'tool_call'}}]
        elif mutation == 'valid_tool_names':
            agent.valid_tool_names = {'tool_call'}
        elif mutation == 'identity':
            agent.session_id = 'other'
        else:
            ep.marker_path(home, SID).unlink()
        with pytest.raises(ep.EvaluationIsolationError):
            ep.assert_request(agent, [])
        ep.enroll(home, SID)
    with pytest.raises(ep.EvaluationIsolationError):
        ep.assert_request(locked_agent(), [{'function': {'name': 'tool_call'}}])


@pytest.mark.parametrize('method', ['session.mutate', 'slash.exec', 'session.tools', 'session.model',
                                   'session.config', 'image.attach_bytes', 'session.compress', 'command.dispatch'])
def test_transport_reconfiguration_is_refused(world, monkeypatch, method):
    monkeypatch.setattr(se, 'locked', lambda *a: True)
    with pytest.raises(RuntimeStoreError, match='evaluation_policy_locked'):
        se.guard_method(None, None, method)


def test_second_admission_and_extended_payload_refuse_before_any_dispatch(world, monkeypatch):
    import hermes_state_runtime as rt
    monkeypatch.setattr(se, 'locked', lambda *a: True)
    monkeypatch.setattr(se, 'policy_for', lambda *a: policy(world))
    monkeypatch.setattr(se, 'proof', lambda *a: {})
    authority, actor = SimpleNamespace(db=object()), SimpleNamespace(subject='owner')
    request = SimpleNamespace(ref=SimpleNamespace(session_id=SID), request_id='first',
                              payload={'text': 'synthetic', 'finite': True})
    monkeypatch.setattr(rt, 'list_session_admissions', lambda *a, **kw: [])
    se.guard_submit(authority, actor, request)
    monkeypatch.setattr(rt, 'list_session_admissions', lambda *a, **kw:
                        [] if kw.get('pending_only', True) else [{'request_id': 'first', 'principal_id': 'owner'}])
    se.guard_submit(authority, actor, request)  # exact retry remains idempotent
    request.request_id = 'second'
    with pytest.raises(RuntimeStoreError, match='already_admitted'):
        se.guard_submit(authority, actor, request)
    request.request_id = 'first'
    for payload in ({'text': ' /curator', 'finite': True}, {'text': 'x'}, {'text': 'x', 'finite': True, 'attachments': []},
                    {'text': 'x', 'finite': True, 'model_override': {}},
                    {'text': 'x', 'finite': True, 'api_text': 'y'}):
        request.payload = payload
        with pytest.raises(RuntimeStoreError):
            se.guard_submit(authority, actor, request)


@pytest.mark.parametrize('providers', [None, [], [SimpleNamespace(_session_id='other')]])
def test_unproven_provider_adoption_stops_before_recall_and_model(world, providers):
    agent = locked_agent()
    agent._memory_manager = SimpleNamespace(providers=providers)
    with pytest.raises(ep.EvaluationIsolationError, match='adoption'):
        ep.assert_provider(agent)


def test_read_only_provider_adoption_requires_the_exact_session_and_dispatch_latch(world):
    agent = locked_agent()
    provider = SimpleNamespace(_session_id=SID, _read_only=False, _writes_refused=lambda: False)
    agent._memory_manager = SimpleNamespace(providers=[provider])
    with pytest.raises(ep.EvaluationIsolationError):
        ep.assert_provider(agent)
    provider._read_only = True
    provider._writes_refused = lambda: True
    ep.assert_provider(agent)


def test_dispatch_and_discovery_are_fenced_at_the_central_bridge(world, monkeypatch):
    import model_tools
    run = Mock(side_effect=AssertionError('bridge compute must not run'))
    monkeypatch.setattr(model_tools, '_compute_tool_definitions', run)
    with sp.policy_scope(policy(world)):
        assert model_tools.get_tool_definitions() == []
        for tool in ('tool_search', 'tool_describe', 'tool_call'):
            with pytest.raises(ep.EvaluationIsolationError):
                model_tools.handle_function_call(tool, {})
    run.assert_not_called()


def test_deferred_title_thread_never_starts_for_a_locked_agent(world):
    from agent.turn_context import start_deferred_title_upgrade
    agent = locked_agent()
    agent._deferred_title_upgrade = Mock()
    start_deferred_title_upgrade(agent)
    agent._deferred_title_upgrade.start.assert_not_called()


def test_title_gate_independent_of_the_existing_title(world, monkeypatch):
    from agent import title_generator
    agent = locked_agent()
    agent._session_db = object()
    agent._session_db_created = True
    agent.platform = 'cli'
    dispatch = Mock(return_value=None)
    monkeypatch.setattr(title_generator, 'maybe_auto_title', dispatch)
    _maybe_title_session_at_turn_start(agent, [{'role': 'user', 'content': 'synthetic'}])
    dispatch.assert_not_called()


def test_direct_debt_drain_gate_does_not_rely_on_other_review_gates(world, monkeypatch):
    from agent import background_review as br, owed_review
    monkeypatch.setattr(owed_review, 'applies', lambda *a, **k: True)
    monkeypatch.setattr(br, 'load_background_review_settings', lambda: (True, {}))
    dispatch = Mock(return_value=None)
    monkeypatch.setattr(owed_review, 'claim_orphan', dispatch)
    agent = locked_agent()
    agent._session_db = object()
    assert br.drain_owed_review(agent, budget=90) is False
    dispatch.assert_not_called()


def test_scheduled_debt_gate_does_not_rely_on_other_review_gates(world, monkeypatch):
    from agent import background_review as br, owed_review
    from gateway.session_finite import finite_turn_scope
    monkeypatch.setattr(owed_review, 'applies', lambda *a, **k: True)
    monkeypatch.setattr(br, 'load_background_review_settings', lambda: (True, {}))
    import threading
    proxy = SimpleNamespace(Thread=Mock())
    monkeypatch.setattr(br, 'threading', proxy)
    with finite_turn_scope(True):
        assert br.schedule_owed_review_drain(locked_agent()) is False
    proxy.Thread.assert_not_called()
