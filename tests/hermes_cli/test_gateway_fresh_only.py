"""Fresh-only classic client refuses an unsupported owner before create/query."""
import contextlib
from unittest.mock import AsyncMock

import pytest

from hermes_cli import gateway_chat, gateway_chat_view, main


def arguments(tmp_path):
    argv = ['chat', '--cli', '--ignore-user-config', '--ignore-rules', '--model', 'gpt-6.1-sol',
            '--provider', 'openai-codex', '--reasoning', 'medium', '--toolsets', 'terminal,file',
            '--max-turns', '12', '--oneshot', '-Q', '--in', str(tmp_path), '-c', 'bounded worker',
            '--create-if-missing', '--fresh-only', '-q', 'focused spec']
    try:
        return main._light_chat_parser().parse_args(argv)
    except SystemExit:
        pytest.fail('the classic parser does not carry the opt-in fresh-only request')


@pytest.mark.asyncio
async def test_classic_fresh_only_reaches_wire_and_refuses_unsupported_owner(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    gateway_chat.validate_options(args)
    from gateway.session_policy import CREATE_FIELDS
    from types import SimpleNamespace
    events = []
    async def rpc(method, **params):
        events.append((method, params))
        return {'session_create': {'sources': ['cli', 'oneshot'], 'parameters': sorted((CREATE_FIELDS | {'title'}) - {'fresh_only'})}}
    @contextlib.asynccontextmanager
    async def connected():
        yield SimpleNamespace(rpc=rpc)
    monkeypatch.setattr(gateway_chat, 'connect_gateway', connected)
    render = AsyncMock()
    monkeypatch.setattr(gateway_chat_view.GatewayChatView, 'run', render)
    with pytest.raises(gateway_chat.GatewayClientError, match='fresh_only'):
        await gateway_chat.run_gateway_chat(args)
    assert [e[0] for e in events] == ['runtime.describe']
    render.assert_not_called()


@pytest.mark.parametrize('change', ['no-title', 'bare-continue', 'resume'])
def test_fresh_only_cannot_turn_into_resume(tmp_path, change):
    args = arguments(tmp_path)
    if change == 'no-title':
        args.continue_last = None
    elif change == 'bare-continue':
        args.continue_last = True
    else:
        args.resume = 'old-session'
    with pytest.raises(gateway_chat.GatewayClientError, match='fresh-only'):
        gateway_chat.validate_options(args)


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['lost-reply', 'collision', 'created', 'wrong-policy', 'old-history'])
async def test_fresh_only_client_never_resends_uncertain_creation(tmp_path, monkeypatch, outcome):
    args = arguments(tmp_path)
    from gateway.session_policy import CREATE_FIELDS
    from types import SimpleNamespace
    events = []
    async def rpc(method, **params):
        events.append((method, params))
        if method == 'runtime.describe':
            return {'session_create': {'sources': ['cli','oneshot'], 'parameters': sorted(CREATE_FIELDS | {'title'})}}
        assert method == 'session.create' and params['fresh_only'] is True
        if outcome in {'lost-reply', 'collision'}:
            raise gateway_chat.GatewayClientError(outcome)
        frozen = gateway_chat._requested_policy(args)
        if outcome == 'wrong-policy':
            frozen['provider'] = 'other'
        return {'stored_session_id':'fresh-session', 'messages':([{'role':'user','content':'old'}] if outcome == 'old-history' else []),
                'info':{'launch_request':frozen,'cwd':str(tmp_path)}}
    @contextlib.asynccontextmanager
    async def connected():
        yield SimpleNamespace(rpc=rpc)
    monkeypatch.setattr(gateway_chat, 'connect_gateway', connected)
    monkeypatch.setattr(gateway_chat_view.GatewayChatView, '__init__', lambda *a, **kw: None)
    render=AsyncMock(return_value=0)
    monkeypatch.setattr(gateway_chat_view.GatewayChatView, 'run', render)
    if outcome == 'created':
        assert await gateway_chat.run_gateway_chat(args) == 0
        render.assert_awaited_once_with('focused spec', oneshot=True)
    else:
        with pytest.raises(gateway_chat.GatewayClientError):
            await gateway_chat.run_gateway_chat(args)
        render.assert_not_called()
    assert [e[0] for e in events] == ['runtime.describe','session.create']


def test_fresh_only_cannot_be_dropped_by_ordinary_cli_dispatch(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.ignore_user_config = False
    args.safe_mode = False
    import hermes_cli.gateway_chat_startup as startup
    calls=[]
    monkeypatch.setattr(startup, 'launch_gateway_chat', lambda a: calls.append(a.fresh_only) or 0)
    monkeypatch.setattr(main, '_resolve_chat_session_args', lambda *a: pytest.fail('fresh-only lost to ordinary resume dispatch'))
    monkeypatch.setattr(main, '_guard_noninteractive_user_config', lambda *a: None)
    with pytest.raises(SystemExit) as exc:
        main.cmd_chat(args)
    assert exc.value.code == 0 and calls == [True]
