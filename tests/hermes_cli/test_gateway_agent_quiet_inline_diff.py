"""-Q stdout carries only the final response: the gateway's rendered edit diff is printed only when the CLI
asked for edit previews (it registered tool_complete_callback), never once -Q has cleared it."""
import cli as cli_module
import pytest

from hermes_cli.cli_single_query import _configure_quiet_agent
from tests.hermes_cli.test_gateway_agent import frame, make_agent  # make_agent is the fixture

DIFF = "a/notes.sh → b/notes.sh\n@@ -0,0 +1,2 @@\n+#!/bin/bash\n+set -u\n"


def _script(name="write_file"):
    return [frame("tool.start", {"tool_id": "c1", "name": name, "args": {"path": "notes.sh"}, "context": ""}),
            frame("tool.complete", {"tool_id": "c1", "name": name, "args": {"path": "notes.sh"},
                                    "result": "{}", "is_error": False, "duration_s": 0.1, "inline_diff": DIFF}),
            frame("message.complete", {"text": "Needs you: one choice.", "outcome": "completed"})]


@pytest.fixture
def printed(monkeypatch):
    lines = []
    monkeypatch.setattr(cli_module, "_cprint", lambda text, *a, **k: lines.append(text))
    return lines


def test_quiet_single_query_agent_prints_no_gateway_diff(make_agent, printed):
    completes = []
    agent, _ = make_agent(_script(), tool_complete_callback=lambda *a: completes.append(a))
    _configure_quiet_agent(agent)           # the real -Q path: clears the CLI's edit-preview callback
    agent.run_conversation("write it")
    assert printed == []
    assert completes == []


def test_no_preview_callback_prints_no_gateway_diff(make_agent, printed):
    agent, _ = make_agent(_script())        # display.inline_diffs: false registers no callback
    agent.run_conversation("write it")
    assert printed == []


def test_interactive_preview_still_prints_the_gateway_diff_once(make_agent, printed):
    completes = []
    agent, _ = make_agent(_script(), tool_complete_callback=lambda *a: completes.append(a))
    agent.run_conversation("write it")
    assert [line for line in printed if "@@ -0,0 +1,2 @@" in line or "set -u" in line]
    assert completes == []                  # the gateway's diff stands in; main's callback would re-render it


def test_delegate_task_with_a_diff_still_reaches_the_callback(make_agent, printed):
    completes = []
    agent, _ = make_agent(_script("delegate_task"), tool_complete_callback=lambda *a: completes.append(a))
    agent.run_conversation("delegate it")
    assert printed and completes and completes[0][1] == "delegate_task"


def test_quiet_delegate_task_prints_nothing_and_calls_nothing(make_agent, printed):
    completes = []
    agent, _ = make_agent(_script("delegate_task"), tool_complete_callback=lambda *a: completes.append(a))
    _configure_quiet_agent(agent)
    agent.run_conversation("delegate it")
    assert printed == [] and completes == []
