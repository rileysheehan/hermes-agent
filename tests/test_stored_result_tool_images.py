"""A stored turn result keeps tool-result text, never a tool-result screenshot's inline payload."""
import copy
import json

from hermes_state_terminal import compact_result

_PNG = 'data:image/png;base64,' + 'A' * 4096


def _tool_image(call_id='t1'):
    return {'role': 'tool', 'tool_call_id': call_id, 'content': [
        {'type': 'text', 'text': 'capture mode=vision (shareable screenshot saved to /tmp/computer_use_x.png)'},
        {'type': 'image_url', 'image_url': {'url': _PNG}}]}


def _turn(*, compressed=False):
    user = {'role': 'user', 'content': [{'type': 'text', 'text': 'look'}, {'type': 'image_url', 'image_url': {'url': _PNG}}],
            'api_content': 'look (as the model saw it)'}
    messages = [{'role': 'user', 'content': 'earlier'}, {'role': 'assistant', 'content': 'ok'}, user,
                {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 't1', 'type': 'function',
                 'function': {'name': 'computer_use', 'arguments': '{}'}}]},
                _tool_image(), {'role': 'assistant', 'content': 'done'}]
    value = {'final_response': 'done', 'messages': messages, 'current_turn_user_idx': 2}
    if compressed:
        value['_compressed'] = True
    return {'result': value, 'usage': {}}


def test_turn_suffix_drops_tool_screenshot_payload_and_keeps_its_text():
    original = _turn()
    snapshot = copy.deepcopy(original)
    stored = compact_result(original, user_message='look')['result']
    tool = next(m for m in stored['messages'] if m['role'] == 'tool')
    assert _PNG not in json.dumps(tool), 'tool screenshot payload stored in the turn result'
    assert 'shareable screenshot saved to /tmp/computer_use_x.png' in json.dumps(tool)
    assert original == snapshot, 'compact_result mutated the live result'


def test_user_attached_image_on_the_turn_row_is_kept_for_chained_replay():
    stored = compact_result(_turn(), user_message='look')['result']
    assert _PNG in json.dumps(stored['_turn_user_row']['content'])


def test_compressed_snapshot_is_kept_whole_but_without_tool_screenshots():
    stored = compact_result(_turn(compressed=True), user_message='look')['result']
    assert len(stored['messages']) == 6, 'compressed replacement transcript must stay whole'
    assert _PNG not in json.dumps([m for m in stored['messages'] if m['role'] == 'tool'])
    assert _PNG in json.dumps(stored['messages'][2]), 'user upload in the compressed snapshot must stay'


def test_multimodal_envelope_tool_result_is_reduced_to_its_summary():
    result = _turn()
    result['result']['messages'][4] = {'role': 'tool', 'tool_call_id': 't1', 'content': {
        '_multimodal': True, 'content': [{'type': 'image_url', 'image_url': {'url': _PNG}}],
        'text_summary': 'shareable screenshot saved to /tmp/computer_use_y.png'}}
    stored = compact_result(result, user_message='look')['result']
    tool = next(m for m in stored['messages'] if m['role'] == 'tool')
    assert _PNG not in json.dumps(tool) and 'computer_use_y.png' in json.dumps(tool)


def test_result_without_images_is_unchanged_by_the_strip():
    result = _turn()
    result['result']['messages'][4] = {'role': 'tool', 'tool_call_id': 't1', 'content': 'plain output'}
    stored = compact_result(result, user_message='look')['result']
    assert stored['messages'][1] == {'role': 'tool', 'tool_call_id': 't1', 'content': 'plain output'}
