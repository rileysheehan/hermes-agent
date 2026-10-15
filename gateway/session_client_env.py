"""Per-run client environment for a finite turn (``gateway.client_env_passthrough``).

A ``chat -q`` client used to run its agent in its own process, so an orchestrator that launches one
run per wake could hand that run per-run values through the environment: the run's API URL, its
short-lived credential, the issue it is working (Paperclip's ``PAPERCLIP_*``), or an opt-in such as
``HERMES_RESUME_UNCHANGED_PREFIX_*``. Now the profile's gateway runs the turn, and its tools inherit
the gateway's environment, never the client's.

``gateway.client_env_passthrough`` (default empty: nothing is forwarded) lists the names a finite
client may forward to its own turn: exact names, or a prefix ending in ``*``. Both ends filter: the
client sends only the names its profile allows, and the owner keeps only the names the owning
profile's current config allows, from a native local viewer of that session.

The values are never persisted. Admissions are durable (``session_admissions``), so they do not ride
the submit payload: the owner holds them in memory, keyed by the admission's (session, request id),
for at most :data:`TTL_S`, and binds them for that admission's execution only. A turn replayed after
an owner restart runs without them, exactly like a client launched without them.
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
import threading
import time
from typing import Any, Iterable, Mapping

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

TTL_S = 6 * 3600
MAX_VARS = 64
MAX_NAME = 128
MAX_VALUE = 32_768
MAX_PENDING = 512
# Names a client may never set on the owner's tools, whatever the allowlist says: they change how
# every child process resolves programs, libraries or its own home.
_DENY_NAMES = frozenset({"PATH", "HOME", "SHELL", "USER", "LOGNAME", "TMPDIR", "HERMES_HOME", "PYTHONPATH",
                         "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "NODE_OPTIONS", "BASH_ENV", "ENV"})
_DENY_PREFIXES = ("LD_", "DYLD_", "_HERMES_FORCE_", "HERMES_SESSION_")

_lock = threading.Lock()
_pending: dict[tuple[str, str], tuple[float, dict[str, str]]] = {}


def patterns(config: Mapping[str, Any] | None) -> tuple[str, ...]:
    """``gateway.client_env_passthrough`` from *config* as a tuple of names / ``PREFIX*`` patterns."""
    gateway = (config or {}).get("gateway") if isinstance(config, Mapping) else None
    raw = gateway.get("client_env_passthrough") if isinstance(gateway, Mapping) else None
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(p for p in (str(item).strip() for item in raw) if p and p != "*")


def allowed(name: str, allow: Iterable[str]) -> bool:
    if not isinstance(name, str) or not name or len(name) > MAX_NAME or "=" in name or "\0" in name:
        return False
    if name in _DENY_NAMES or name.startswith(_DENY_PREFIXES):
        return False
    for pattern in allow:
        if pattern.endswith("*"):
            if len(pattern) > 1 and name.startswith(pattern[:-1]):
                return True
        elif name == pattern:
            return True
    return False


def select(environ: Mapping[str, str], allow: Iterable[str]) -> dict[str, str]:
    """Client side: the allowed names present in *environ* (bounded)."""
    allow = tuple(allow)
    if not allow:
        return {}
    picked = {k: v for k, v in environ.items()
              if allowed(k, allow) and isinstance(v, str) and len(v) <= MAX_VALUE and "\0" not in v}
    return dict(sorted(picked.items())[:MAX_VARS])


def profile_gateway_config(authority) -> Mapping[str, Any]:
    """The owning profile's CURRENT ``gateway`` section (its config.yaml beside the authority's
    state.db), so a session frozen before a setting existed still follows it. ``{}`` on any failure."""
    try:
        from pathlib import Path
        from gateway.run import _load_gateway_config
        config = _load_gateway_config(Path(authority.db.db_path).resolve().parent / "config.yaml")
        section = config.get("gateway") if isinstance(config, Mapping) else None
        return section if isinstance(section, Mapping) else {}
    except Exception:  # health: allow BLE001 -- an unreadable config means the defaults
        logger.debug("client env: profile config unreadable; using defaults", exc_info=True)
        return {}


def profile_patterns(authority) -> tuple[str, ...]:
    """The owning profile's current allowlist; ``()`` on any failure (forwards nothing)."""
    return patterns({"gateway": profile_gateway_config(authority)})


def admit(params: Mapping[str, Any], *, native_owner: bool, allow: Iterable[str]) -> dict[str, str]:
    """Owner side: validate a submission's ``client_env`` and keep what this owner allows. A malformed
    value is refused; a well-formed but disallowed name, a non-finite turn or a non-native viewer just
    forwards nothing (the turn runs as if the client had sent none)."""
    raw = params.get("client_env")
    if raw is None:
        return {}
    if (not isinstance(raw, Mapping) or len(raw) > MAX_VARS
            or any(not isinstance(k, str) or not isinstance(v, str) or len(v) > MAX_VALUE or "\0" in v
                   for k, v in raw.items())):
        raise RuntimeStoreError("invalid_params")
    if params.get("finite") is not True or not native_owner:
        return {}
    allow = tuple(allow)
    return {k: v for k, v in raw.items() if allowed(k, allow)}


def _sweep(now: float) -> None:
    for key in [k for k, (at, _env) in _pending.items() if now - at > TTL_S]:
        _pending.pop(key, None)


def stash(session_id: str, request_id: str, env: Mapping[str, str]) -> None:
    if not env or not session_id or not request_id:
        return
    now = time.monotonic()
    with _lock:
        _sweep(now)
        if len(_pending) >= MAX_PENDING:
            _pending.pop(min(_pending, key=lambda k: _pending[k][0]), None)
        _pending[(str(session_id), str(request_id))] = (now, dict(env))


def discard(session_id: str, request_id: str) -> None:
    with _lock:
        _pending.pop((str(session_id), str(request_id)), None)


def take(session_id: str | None, request_id: str | None) -> dict[str, str]:
    if not session_id or not request_id:
        return {}
    with _lock:
        _sweep(time.monotonic())
        held = _pending.pop((str(session_id), str(request_id)), None)
    return dict(held[1]) if held else {}


def pending_count() -> int:
    with _lock:
        return len(_pending)


@contextmanager
def client_env_scope(row: Mapping[str, Any], session_id: str | None = None):
    """Bind the stashed env of the admission *row* for its execution, then drop it."""
    from gateway.session_context import bind_client_env, reset_client_env
    env = take(row.get("target_session_id") or session_id, row.get("request_id"))
    token = bind_client_env(env)
    if env:
        logger.info("client env: %d forwarded name(s) bound for admission %s", len(env), row.get("admission_id"))
    try:
        yield env
    finally:
        reset_client_env(token)
