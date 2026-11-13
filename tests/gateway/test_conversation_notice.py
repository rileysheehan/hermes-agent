"""A Paperclip Agent Chat turn's learning summary becomes that turn's conversation notice (MER-2063).

The gateway serves ``hermes chat -q … -Q`` as a local viewer; the quiet client drops the ``review.summary`` frame and
has usually exited before the review finishes, so on a bot moved to Paperclip the "💾 Memory updated" notice reached
neither the Agent Chat nor Telegram. Paperclip accepts one post-answer notice from the turn that answered and
publishes it to the chat that turn came from."""
import io
import json
import sys
import threading
import urllib.error
from types import SimpleNamespace

import pytest

from gateway import run_turn_runner as rtr
from gateway.config import Platform
from gateway.session_context import bind_client_env, reset_client_env

ENV = {"PAPERCLIP_CONVERSATION_TURN": "1", "PAPERCLIP_API_URL": "http://127.0.0.1:3100",
       "PAPERCLIP_API_KEY": "synthetic-run-key", "PAPERCLIP_RUN_ID": "run-1", "PAPERCLIP_TASK_ID": "issue 1"}


@pytest.fixture
def client_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    tokens = []

    def bind(env):
        tokens.append(bind_client_env(env))
    yield bind
    for token in reversed(tokens):
        reset_client_env(token)


class _Response:
    def __init__(self, status, body=b"{}"):
        self.status, self._body = status, body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _error(status, code):
    return urllib.error.HTTPError("u", status, "x", {}, io.BytesIO(json.dumps({"details": {"code": code}}).encode()))


class _Opener:
    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_target_only_for_an_agent_chat_turn_with_its_run_identity(client_env):
    assert rtr._conversation_notice_target() is None                       # no forwarded env: not a chat turn
    client_env({**ENV, "PAPERCLIP_CONVERSATION_TURN": "0"})
    assert rtr._conversation_notice_target() is None                       # a board run
    client_env({k: v for k, v in ENV.items() if k != "PAPERCLIP_RUN_ID"})
    assert rtr._conversation_notice_target() is None                       # no run identity
    client_env(ENV)
    assert rtr._conversation_notice_target() == {
        "url": "http://127.0.0.1:3100/api/issues/issue%201/comments", "key": "synthetic-run-key", "run_id": "run-1"}


def test_waits_for_the_answer_then_records_the_notice_once():
    opener = _Opener([_error(409, "conversation_notice_before_answer"), OSError("reset"), _Response(201)])
    sleeps = []
    target = {"url": "http://x/api/issues/i/comments", "key": "k", "run_id": "run-1"}
    assert rtr._post_conversation_notice(target, "💾 Memory updated: one fact", opener=opener,
                                         sleep=sleeps.append, clock=lambda: 0.0) is True
    assert len(opener.requests) == 3 and len(sleeps) == 2
    request = opener.requests[-1]
    assert json.loads(request.data) == {"body": "💾 Memory updated: one fact", "conversationNotice": True}
    assert request.get_header("X-paperclip-run-id") == "run-1"
    assert request.get_header("Authorization") == "Bearer k"


def test_an_existing_notice_is_success_and_a_refusal_is_not_retried():
    target = {"url": "http://x/api/issues/i/comments", "key": "k", "run_id": "r"}
    assert rtr._post_conversation_notice(target, "t", opener=_Opener([_error(409, "conversation_notice_exists")]),
                                         sleep=lambda s: None) is True
    refused = _Opener([_error(403, "conversation_notice_not_allowed")])
    assert rtr._post_conversation_notice(target, "t", opener=refused, sleep=lambda s: None) is False
    assert len(refused.requests) == 1


def test_gives_up_after_the_budget():
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds
    opener = _Opener([_error(409, "conversation_notice_before_answer") for _ in range(1000)])
    assert rtr._post_conversation_notice({"url": "http://x/api/issues/i/comments", "key": "k", "run_id": "r"}, "t", opener=opener,
                                         sleep=sleep, clock=lambda: now[0]) is False
    assert len(opener.requests) == int(rtr._NOTICE_BUDGET_S / rtr._NOTICE_RETRY_S) + 1


def test_a_bad_url_is_reported_not_raised():
    assert rtr._post_conversation_notice({"url": "not a url", "key": "k", "run_id": "r"}, "t",
                                         opener=_Opener([])) is False


def test_long_text_is_capped():
    opener = _Opener([_Response(201)])
    rtr._post_conversation_notice({"url": "http://x/api/issues/i/comments", "key": "k", "run_id": "r"}, "x" * 5000, opener=opener)
    assert len(json.loads(opener.requests[0].data)["body"]) == rtr._NOTICE_MAX_CHARS


def _runner(platform):
    published = []
    live = SimpleNamespace(event_stream=SimpleNamespace(
        execution={}, publish=lambda sid, payload, event_type: published.append((event_type, payload["text"]))))
    authority = SimpleNamespace(sessions={"s1": live})
    runner = SimpleNamespace(_ctx=SimpleNamespace(source=SimpleNamespace(platform=platform)),
                             _approval_owner=(authority, "s1", 1), _status_live=lambda: False)
    return runner, published


@pytest.mark.parametrize("env, expected", [(ENV, 1), ({**ENV, "PAPERCLIP_CONVERSATION_TURN": "0"}, 0), (None, 0)])
def test_the_local_viewer_summary_also_goes_to_the_conversation(client_env, monkeypatch, env, expected):
    sent = []
    monkeypatch.setattr(rtr, "_send_conversation_notice", lambda target, text: sent.append((target["run_id"], text)))
    # The local-viewer branch never reaches gateway.run's metadata helpers; stub the module so the import does not
    # touch the real home (a worktree's update check).
    monkeypatch.setitem(sys.modules, "gateway.run", SimpleNamespace(
        _interim_metadata=lambda m: m, _non_conversational_metadata=lambda m, platform=None: m))
    if env:
        client_env(env)
    runner, published = _runner(Platform.LOCAL)
    send, _release = rtr.TurnRunner._make_bg_review_callbacks(runner)
    # The review reports from its own thread, where the turn's forwarded env is not bound.
    worker = threading.Thread(target=send, args=("💾 Memory updated: one fact",))
    worker.start()
    worker.join(5)
    assert published == [("review.summary", "💾 Memory updated: one fact")]   # the viewer frame is unchanged
    assert sent == [("run-1", "💾 Memory updated: one fact")] * expected
