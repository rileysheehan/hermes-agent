"""A finite viewer must deliver the owner's settled answer, not its empty send-suppression text."""
import asyncio
import json

import pytest

from hermes_cli.gateway_chat_view import GatewayChatView


@pytest.mark.asyncio
@pytest.mark.parametrize('emitting', [False, True])
async def test_streamed_suppression_delivers_admission_bound_receipt(emitting, capsys):
    from hermes_cli.stream_json import StreamJsonEmitter

    class Peer:
        def __init__(self):
            self.events = asyncio.Queue()
            self.calls = []

        async def rpc(self, method, **params):
            self.calls.append((method, params))
            if method == 'prompt.receipt':
                assert params == {'session_id': 's', 'admission_id': 'ours', 'include_result': True}
                return {'result': {'completed': True, 'final_response': 'Neutral streamed reply.', 'api_calls': 1}}
            assert method == 'prompt.submit'
            for kind, payload in [('message.delta', {'text': 'Neutral streamed '}),
                                  ('message.delta', {'text': 'reply.'}),
                                  ('message.complete', {'outcome': 'completed', 'text': ''})]:
                self.events.put_nowait({'method': 'event', 'params': {'session_id': 's', 'type': kind,
                    'admission_id': 'ours', 'payload': payload}})
            return {'admission_id': 'ours'}

    peer = Peer()
    emitter = StreamJsonEmitter(model='m', session_id='s') if emitting else None
    view = GatewayChatView(peer, {'stored_session_id': 's'}, quiet=True, emitter=emitter)
    view.unattended = True
    assert await asyncio.wait_for(view.run('neutral work', oneshot=True), 5) == 0
    out = capsys.readouterr().out
    if emitting:
        result = [json.loads(line) for line in out.splitlines()][-1]
        assert result['type'] == 'result'
        assert result['text'] == 'Neutral streamed reply.'
    else:
        assert out == 'Neutral streamed reply.\n'


@pytest.mark.asyncio
async def test_successful_usage_without_answer_does_not_make_stdout():
    class Peer:
        def __init__(self):
            self.events = asyncio.Queue()

        async def rpc(self, method, **params):
            if method == 'prompt.receipt':
                return {'result': {'completed': True, 'api_calls': 1}}
            self.events.put_nowait({'method': 'event', 'params': {'session_id': 's',
                'type': 'message.complete', 'admission_id': 'ours',
                'payload': {'outcome': 'completed', 'text': ''}}})
            return {'admission_id': 'ours'}

    view = GatewayChatView(Peer(), {'stored_session_id': 's'}, quiet=True)
    view.unattended = True
    assert await asyncio.wait_for(view.run('neutral work', oneshot=True), 5) == 1
