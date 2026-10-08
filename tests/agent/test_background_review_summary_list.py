"""The messaging form of a background-review summary: one change stays on one line; several become a list with
each change on its own line, applied ones marked done and a staged proposal marked as not applied yet."""
from types import SimpleNamespace

from agent import background_review as br


def _published(actions, profile=None, monkeypatch=None):
    if monkeypatch is not None:
        import hermes_cli.profiles as profiles
        monkeypatch.setattr(profiles, "current_profile_name", lambda default=None: profile)
    got = []
    agent = SimpleNamespace(suppress_status_output=True, background_review_callback=got.append,
                            _safe_print=lambda *_a, **_k: None)
    br._publish_review_summary(agent, actions)
    return got


def test_one_change_keeps_the_one_line_form(monkeypatch):
    assert _published(["Skill 'deploy' patched"], "keely", monkeypatch) == ["💾 Self-improvement review: Skill 'deploy' patched"]


def test_several_changes_are_a_list_naming_the_agent(monkeypatch):
    [text] = _published(["Skill 'deploy' patched", "Memory ➕ the NAS mount needs a remount after sleep"], "keely",
                        monkeypatch)
    assert text == ("💾 Keely's self-improvement review:\n"
                    "• ✅ Skill 'deploy' patched\n"
                    "• ✅ Memory ➕ the NAS mount needs a remount after sleep")
    assert " · " not in text


def test_a_staged_proposal_is_not_read_as_applied(monkeypatch):
    staged = ("Background review may not delete memory entries unattended. The proposed replace was staged for your "
              "approval — review it with /memory pending (approve to apply, discard to drop).")
    [text] = _published(["Memory updated", staged], "aria", monkeypatch)
    lines = text.splitlines()
    assert lines[1] == "• ✅ Memory updated"
    assert lines[2].startswith("• ⏳ not applied yet: Background review may not delete")


def test_the_default_profile_is_not_named(monkeypatch):
    [text] = _published(["A", "B"], "default", monkeypatch)
    assert text.splitlines()[0] == "💾 Self-improvement review:"


def test_duplicates_collapse_and_the_cli_log_line_is_unchanged(monkeypatch, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="agent.background_review"):
        [text] = _published(["A", "A", "B"], None, monkeypatch)
    assert text.count("• ") == 2
    assert "Background review: A · B" in caplog.text
