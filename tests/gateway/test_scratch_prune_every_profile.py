"""Idle scratch is pruned in EVERY served profile's home, hourly, by the gateway housekeeping tick.

``get_scratch_dir``'s boot prune is guarded by one process-wide flag, so a multiplexed gateway
(one long-lived process serving every profile) pruned only the first home it resolved after
start, and never again for the rest of its life. Every other served home's ``cache/scratch``
kept its idle entries until the next restart picked a different first home, if it ever did.
"""

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
import hermes_constants


class _Ticks:
    """Stop event that lets the housekeeping loop run exactly ``n`` ticks with no sleeping."""

    def __init__(self, n: int):
        self.n, self.left = 0, n

    def is_set(self):
        return self.n >= self.left

    def wait(self, timeout=None):
        self.n += 1
        return True


@pytest.fixture
def served_homes(tmp_path, monkeypatch):
    """Launch home A (multiplex ``default``) and served named profiles B and C, each with one
    entry idle for two days and one fresh entry in its scratch dir."""
    fake_home = tmp_path / "home"
    a = fake_home / ".hermes"
    homes = [a, a / "profiles" / "b", a / "profiles" / "c"]
    old = time.time() - 48 * 3600
    for home in homes:
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model:\n  provider: nous\n", encoding="utf-8")
        scratch = home / "cache" / "scratch"
        (scratch / "idle-tree").mkdir(parents=True)
        (scratch / "idle-tree" / "copy.db").write_text("x", encoding="utf-8")
        for path in (scratch / "idle-tree" / "copy.db", scratch / "idle-tree"):
            os.utime(path, (old, old))
        (scratch / "fresh.txt").write_text("y", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HERMES_HOME", str(a))
    # The process-wide boot guard is already spent in any real gateway; spend it here too so only
    # the housekeeping path can do the pruning these tests look for.
    monkeypatch.setattr(hermes_constants, "_scratch_pruned_once", True)
    return homes


def _run(ticks: int) -> None:
    from agent.secret_scope import set_multiplex_active

    set_multiplex_active(True)
    try:
        gateway_run._start_gateway_housekeeping(
            _Ticks(ticks), interval=0, runner=SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True)))
    finally:
        set_multiplex_active(False)


def test_hourly_tick_prunes_every_served_home(served_homes):
    """Tick 60 sweeps A, B and C alike: idle entries go, fresh ones stay, and each home's
    ``.last_prune`` is stamped so other processes' hourly gate and outside checks can see it."""
    before = time.time() - 1
    _run(60)
    for home in served_homes:
        scratch = home / "cache" / "scratch"
        assert not (scratch / "idle-tree").exists(), f"{home} kept its idle scratch entry"
        assert (scratch / "fresh.txt").exists(), f"{home} lost a fresh scratch entry"
        assert (scratch / ".last_prune").stat().st_mtime >= before
        assert "idle-tree" in (home / "logs" / "scratch-prune.log").read_text(encoding="utf-8")
    from hermes_constants import get_hermes_home
    assert get_hermes_home() == served_homes[0]


def test_scratch_sweep_rides_the_hourly_cadence_not_every_tick(served_homes):
    """59 ticks sweep nothing: the sweep walks trees, so it runs hourly like the media caches."""
    _run(59)
    for home in served_homes:
        assert (home / "cache" / "scratch" / "idle-tree").exists()
        assert not (home / "cache" / "scratch" / ".last_prune").exists()


def test_sweep_scratch_dir_stamps_then_prunes(tmp_path):
    """The helper stamps ``.last_prune`` and keeps it (the stamp is never pruned as idle)."""
    scratch = tmp_path / "cache" / "scratch"
    (scratch / "idle").mkdir(parents=True)
    old = time.time() - 48 * 3600
    os.utime(scratch / "idle", (old, old))
    assert hermes_constants.sweep_scratch_dir(scratch) == 1
    assert (scratch / ".last_prune").exists()
    assert not (scratch / "idle").exists()
