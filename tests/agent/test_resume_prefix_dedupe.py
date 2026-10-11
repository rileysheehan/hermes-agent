"""MER-646: resumed one-shot turns reference an unchanged leading instruction block instead of repeating it."""

import hashlib

import pytest

from agent import resume_prefix_dedupe as rpd
from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX

BLOCK = "# Agent instructions\n" + ("Keep every approval gate. " * 40)
SHA = hashlib.sha256(BLOCK.encode()).hexdigest()
TRAILER = "\nThe above agent instructions were loaded from /runs/{run}/AGENTS.md. Resolve any relative file references from /runs/{run}/."
WAKE = TRAILER.format(run="r2") + "\n\n---\n\n## Paperclip Wake Payload\nAnswer the new review comment."


def compressor(compressions=0, protect=3):
    c = object.__new__(ContextCompressor)
    c.protect_first_n = protect
    c.protect_last_n = 20
    c.compression_count = compressions
    c._previous_summary = None
    return c


class Agent:
    def __init__(self, comp):
        self.context_compressor = comp
        # The turn's frozen expiry clock (build_turn_context stamps it before deciding); the wire projection
        # needs it, and untimestamped rows are left alone by the replay cleanup.
        self._current_turn_timestamp = 1e9


def history(first=None):
    return [
        {"role": "user", "content": (first if first is not None else BLOCK) + TRAILER.format(run="r1") + "\n\n---\n\nFirst wake."},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "[Board result] noted"},
        {"role": "assistant", "content": "ok"},
    ]


@pytest.fixture
def opt_in(monkeypatch):
    monkeypatch.setenv(rpd.ENV_SHA256, SHA)
    monkeypatch.setenv(rpd.ENV_CHARS, str(len(BLOCK)))


def test_references_block_and_pins_first_user_message(opt_in):
    comp = compressor()
    out = rpd.apply_resume_prefix_dedupe(Agent(comp), BLOCK + WAKE, history())
    assert out == rpd.reference_text(len(BLOCK), SHA) + WAKE
    assert comp._pinned_head_rows == 1
    # After a compaction inside this turn, decay no longer drops the referenced copy.
    comp.compression_count = 1
    api = [{"role": "system", "content": "s"}] + history() + [{"role": "user", "content": out}]
    assert comp._effective_protect_first_n(api) == 0
    assert comp._protect_head_size(api) == 2  # system + first user message


def test_env_is_consumed_once(opt_in):
    agent = Agent(compressor())
    assert rpd.apply_resume_prefix_dedupe(agent, BLOCK + WAKE, history()) != BLOCK + WAKE
    assert rpd.apply_resume_prefix_dedupe(agent, BLOCK + WAKE, history()) == BLOCK + WAKE


def test_no_opt_in_is_byte_for_byte_unchanged(monkeypatch):
    monkeypatch.delenv(rpd.ENV_SHA256, raising=False)
    monkeypatch.delenv(rpd.ENV_CHARS, raising=False)
    comp = compressor()
    assert rpd.apply_resume_prefix_dedupe(Agent(comp), BLOCK + WAKE, history()) == BLOCK + WAKE
    assert not hasattr(comp, "_pinned_head_rows")


@pytest.mark.parametrize("case", [
    "fresh_session", "summary_in_history", "summary_deep_in_history", "decayed_after_compaction", "first_user_differs", "first_user_one_char",
    "first_user_unprotected", "message_block_mismatch", "wrong_sha_env", "short_block_env", "bad_chars_env",
    "separate_persisted_text", "list_content", "sidecar_differs",
])
def test_keeps_full_message(monkeypatch, case):
    sha, chars = SHA, str(len(BLOCK))
    hist, message, comp, persist = history(), BLOCK + WAKE, compressor(), None
    if case == "fresh_session":
        hist = []
    elif case == "summary_in_history":
        hist = history() + [{"role": "user", "content": SUMMARY_PREFIX + " earlier work"}]
    elif case == "summary_deep_in_history":
        # Past the compressor's restart-handoff probe, so only the full-history scan sees it.
        filler = [{"role": "assistant" if i % 2 else "user", "content": f"turn {i}"} for i in range(30)]
        hist = history() + filler + [{"role": "user", "content": SUMMARY_PREFIX + " earlier work"}]
        assert comp._effective_protect_first_n([{"role": "system", "content": ""}] + hist) == 3
    elif case == "decayed_after_compaction":
        comp = compressor(compressions=1)
    elif case == "first_user_differs":
        hist = history(first="# Older instructions\n" + "x" * 900)
    elif case == "first_user_one_char":
        hist = history(first=BLOCK[:-1] + "!")
    elif case == "first_user_unprotected":
        comp = compressor(protect=1)
        hist = [{"role": "assistant", "content": "hello"}] + history()
    elif case == "message_block_mismatch":
        message = BLOCK.replace("Keep", "Drop", 1) + WAKE
    elif case == "wrong_sha_env":
        sha = "0" * 64
    elif case == "short_block_env":
        chars = "100"
    elif case == "bad_chars_env":
        chars = "lots"
    elif case == "separate_persisted_text":
        persist = "stored text"
    elif case == "list_content":
        message = [{"type": "text", "text": BLOCK + WAKE}]
    elif case == "sidecar_differs":
        hist[0]["api_content"] = "something else entirely" * 100
    monkeypatch.setenv(rpd.ENV_SHA256, sha)
    monkeypatch.setenv(rpd.ENV_CHARS, chars)
    out = rpd.apply_resume_prefix_dedupe(Agent(comp), message, hist, persist)
    assert out == message
    assert not getattr(comp, "_pinned_head_rows", 0)


def test_error_keeps_message(opt_in):
    class Broken:
        protect_first_n = 3

        def _effective_protect_first_n(self, _messages):
            raise RuntimeError("boom")

    assert rpd.apply_resume_prefix_dedupe(Agent(Broken()), BLOCK + WAKE, history()) == BLOCK + WAKE


def test_reference_keeps_wake_specific_text_intact(opt_in):
    out = rpd.apply_resume_prefix_dedupe(Agent(compressor()), BLOCK + WAKE, history())
    assert out.endswith(WAKE)
    assert BLOCK not in out
