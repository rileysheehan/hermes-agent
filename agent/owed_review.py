"""Owed post-turn reviews: a memory/skill review that was due but did not finish is carried, not lost.

Opt in with ``auxiliary.background_review.carry_owed_reviews: true`` (default off: today's behaviour).

Two gaps this closes, both common where every turn is its own short-lived process (``hermes chat -q``
resumed once per wake by an orchestrator, a kanban worker):

* **A review that never finishes.** The post-turn review runs after delivery, on a daemon thread. A
  process that is killed, cancelled, or interrupted by a new message before the fork returns loses it,
  and nothing records that it was owed. Here a marker is written *before* the review spawns and
  removed only when the fork returns without being interrupted. A marker left behind is owed:

  - the next turn of the **same session** folds the owed kinds into its own post-turn review (the
    snapshot is the whole conversation, so it covers the turns the lost review missed);
  - otherwise the next **one-shot run in the same profile** reviews it at exit, inside the existing exit
    linger and its budget, from the transcript in ``state.db`` (one per run, oldest first).

  A turn that was itself interrupted while a review was due is owed the same way.

* **Short runs.** With ``oneshot_learning`` a one-shot turn is reviewed only when it made at least
  ``oneshot_min_tool_calls`` tool calls. Turns below that add their calls to the session's count, so
  several short wakes of one session reach the threshold together.

One small JSON file per session under ``<HERMES_HOME>/review_owed/``; read-modify-write under a
cross-process lock. The reviewing process stamps its pid, so another process takes an owed review over
only when its writer is gone (or the marker is older than ``ORPHAN_AFTER_S``). Each attempt is counted;
after ``MAX_ATTEMPTS`` the marker is left in place, unreviewed, for monitoring to see (an owed marker
older than a day means learning has stopped for that session).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

MARKER_DIRNAME = "review_owed"
# A marker whose writer pid is still alive is taken over after this long anyway: long-lived
# processes (a gateway) keep their pid forever, and a review takes minutes, not hours.
ORPHAN_AFTER_S = 2 * 3600
# Attempts (spawned reviews) per owed stretch before it is left for monitoring instead of retried.
MAX_ATTEMPTS = 3
# Markers that only carry a short-run count (nothing owed) are dropped after this long.
CARRY_TTL_S = 14 * 24 * 3600


def enabled(task_cfg: Optional[Dict[str, Any]] = None) -> bool:
    """``auxiliary.background_review.carry_owed_reviews`` (default off). Never raises."""
    try:
        from agent.background_review import _background_review_task_config
        from utils import is_truthy_value

        return is_truthy_value(_background_review_task_config(task_cfg).get("carry_owed_reviews"), default=False)
    except Exception:  # noqa: BLE001 — a bad knob keeps the default
        return False


def applies(agent: Any, task_cfg: Optional[Dict[str, Any]] = None) -> bool:
    """True when this agent's automatic reviews are carried: opted in, a top-level persisted session,
    and not a caller that suppresses reviews (cron)."""
    if agent is None or not getattr(agent, "session_id", None):
        return False
    if getattr(agent, "skip_background_review", False) or getattr(agent, "_delegate_depth", 0) > 0:
        return False
    if getattr(agent, "_persist_disabled", False):
        return False
    try:
        from agent.background_review import load_background_review_settings

        reviews_on, cfg = load_background_review_settings() if task_cfg is None else (True, task_cfg)
        return bool(reviews_on) and enabled(cfg)
    except Exception:  # noqa: BLE001
        return False


# ── storage ────────────────────────────────────────────────────────────────────


def marker_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / MARKER_DIRNAME


def _path(session_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(session_id))
    return marker_dir() / f"{safe}.json"


def _lock():
    from tools.skill_usage import skill_file_lock

    return skill_file_lock(marker_dir() / ".lock")


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write(path: Path, data: Dict[str, Any], *, pid: Optional[int] = None) -> None:
    from hermes_constants import mkdir_under_hermes_home

    mkdir_under_hermes_home(path.parent)
    data["updated_at"] = time.time()
    data["pid"] = os.getpid() if pid is None else pid
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _pid_alive(pid: Any) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _is_owed(data: Optional[Dict[str, Any]]) -> bool:
    return bool(data) and bool(data.get("memory") or data.get("skills"))


def read(session_id: str) -> Optional[Dict[str, Any]]:
    if not session_id:
        return None
    return _read(_path(session_id))


# ── the owed marker ───────────────────────────────────────────────────────────


def owed_kinds(session_id: str) -> Tuple[bool, bool]:
    """``(memory, skills)`` still owed for this session; ``(False, False)`` once the attempt cap is hit
    (the marker stays for monitoring)."""
    data = read(session_id) or {}
    if not _is_owed(data) or int(data.get("attempts", 0) or 0) >= MAX_ATTEMPTS:
        return False, False
    return bool(data.get("memory")), bool(data.get("skills"))


def mark_owed(session_id: str, *, memory: bool, skills: bool, reason: str, spawning: bool) -> Optional[str]:
    """Record that a review of *session_id* is due, BEFORE it spawns. Merges with kinds already owed
    and clears the short-run count (this review covers those turns). Returns the token the review
    passes to :func:`settle`; a later mark replaces it, so an older review cannot clear a newer debt."""
    if not session_id or not (memory or skills):
        return None
    try:
        with _lock():
            path = _path(session_id)
            data = _read(path) or {}
            was_owed = _is_owed(data)
            token = uuid.uuid4().hex
            data.update(
                session_id=session_id,
                memory=bool(memory or (was_owed and data.get("memory"))),
                skills=bool(skills or (was_owed and data.get("skills"))),
                reason=reason,
                owed_since=data.get("owed_since") if was_owed and data.get("owed_since") else time.time(),
                attempts=int(data.get("attempts", 0) or 0) + (1 if spawning else 0),
                token=token,
                carried_tool_calls=0,
            )
            _write(path, data)
        logger.info("Review owed: session=%s memory=%s skills=%s reason=%s attempts=%d",
                    session_id, data["memory"], data["skills"], reason, data["attempts"])
        return token
    except Exception:  # noqa: BLE001 — bookkeeping must never break a turn
        logger.debug("Could not record owed review for %s", session_id, exc_info=True)
        return None


def settle(session_id: Optional[str], token: Optional[str]) -> bool:
    """The review carrying *token* finished: remove the marker, unless a newer debt replaced it."""
    if not session_id or not token:
        return False
    try:
        with _lock():
            path = _path(session_id)
            data = _read(path)
            if not data or data.get("token") != token:
                return False
            if int(data.get("carried_tool_calls", 0) or 0) > 0:
                # A short turn landed after this review started: keep its count, drop the debt.
                for key in ("memory", "skills", "reason", "owed_since", "token"):
                    data.pop(key, None)
                data["attempts"] = 0
                _write(path, data)
            else:
                path.unlink(missing_ok=True)
        logger.info("Owed review settled: session=%s", session_id)
        return True
    except Exception:  # noqa: BLE001
        logger.debug("Could not settle owed review for %s", session_id, exc_info=True)
        return False


def touch(session_id: str) -> None:
    """A live process resumed *session_id*: stamp its pid so no other process takes its debt over;
    this session's own next review will pay it."""
    if not session_id:
        return
    try:
        path = _path(session_id)
        if not path.exists():
            return
        with _lock():
            data = _read(path)
            if data:
                _write(path, data)
    except Exception:  # noqa: BLE001
        logger.debug("Could not touch owed review for %s", session_id, exc_info=True)


# ── short runs add up ─────────────────────────────────────────────────────────


def carried_tool_calls(session_id: str) -> int:
    data = read(session_id)
    try:
        return max(0, int((data or {}).get("carried_tool_calls", 0) or 0))
    except (TypeError, ValueError):
        return 0


def carry(session_id: str, tool_calls: int) -> int:
    """Add a short turn's tool calls to the session's count; returns the new total."""
    if not session_id or tool_calls <= 0:
        return carried_tool_calls(session_id)
    try:
        with _lock():
            path = _path(session_id)
            data = _read(path) or {"session_id": session_id}
            total = max(0, int(data.get("carried_tool_calls", 0) or 0)) + int(tool_calls)
            data["carried_tool_calls"] = total
            _write(path, data)
        logger.debug("Short run carried: session=%s tool_calls=+%d total=%d", session_id, tool_calls, total)
        return total
    except Exception:  # noqa: BLE001
        logger.debug("Could not carry tool calls for %s", session_id, exc_info=True)
        return 0


# ── another session's debt ────────────────────────────────────────────────────


def claim_orphan(exclude: Optional[str] = None, *, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Take over the oldest owed review whose writer is gone (or which is older than
    ``ORPHAN_AFTER_S``), stamping this pid and a fresh token. Prunes stale carry-only markers while
    scanning. Returns the claimed marker, or ``None``."""
    now = time.time() if now is None else now
    try:
        directory = marker_dir()
        if not directory.is_dir():
            return None
        with _lock():
            candidates = []
            for path in directory.glob("*.json"):
                data = _read(path)
                if not data:
                    continue
                age = now - float(data.get("updated_at", 0) or 0)
                if not _is_owed(data):
                    if age > CARRY_TTL_S:
                        with suppress(OSError):
                            path.unlink()
                    continue
                if data.get("session_id") == exclude or int(data.get("attempts", 0) or 0) >= MAX_ATTEMPTS:
                    continue
                if _pid_alive(data.get("pid")) and age < ORPHAN_AFTER_S:
                    continue
                candidates.append((float(data.get("owed_since", 0) or 0), path, data))
            if not candidates:
                return None
            _since, path, data = min(candidates, key=lambda c: c[0])
            data["token"] = uuid.uuid4().hex
            data["attempts"] = int(data.get("attempts", 0) or 0) + 1
            data["reason"] = "carried_to_next_run"
            _write(path, data)
        logger.info("Claimed owed review: session=%s memory=%s skills=%s attempts=%d",
                    data.get("session_id"), data.get("memory"), data.get("skills"), data["attempts"])
        return data
    except Exception:  # noqa: BLE001
        logger.debug("Could not scan owed reviews", exc_info=True)
        return None


def release(session_id: Optional[str], token: Optional[str]) -> None:
    """A claimed review was not started after all: give the claim back (pid 0 = any process may take it)."""
    if not session_id or not token:
        return
    with suppress(Exception):
        with _lock():
            path = _path(session_id)
            data = _read(path)
            if data and data.get("token") == token:
                data["attempts"] = max(0, int(data.get("attempts", 0) or 0) - 1)
                _write(path, data, pid=0)


def owed_markers() -> list:
    """Every marker that still owes a review (for status and health checks)."""
    with suppress(Exception):
        return [d for p in sorted(marker_dir().glob("*.json")) if _is_owed(d := _read(p))]
    return []
