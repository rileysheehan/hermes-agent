"""Opt-in native worker environment scoping, not a host/credential sandbox.

Only removes ambient parent board/run/resume authority. Native admission RPC
identity and provider access are independent. The marker propagates this denial
to cooperating native descendants; it never grants authority. Trusted shell code
can still read host files or deliberately construct an unrelated environment.
"""
from functools import wraps
import os

MARKER = 'HERMES_MANAGED_AUTHORITY_EXCLUDED'
# Captured before dotenv/profile overlays; a later overlay cannot switch denial off.
_enabled = os.environ.get(MARKER) == '1'
_PARENT_KEYS = frozenset({'HERMES_KANBAN_TASK', 'HERMES_KANBAN_RUN_ID',
                         'HERMES_KANBAN_CLAIM_LOCK', 'HERMES_YOLO_MODE'})
_PARENT_PREFIXES = ('PAPERCLIP_', 'HERMES_RESUME_UNCHANGED_PREFIX_')


def parent_authority_name(name):
    name = name.upper()
    name = name.removeprefix('_HERMES_FORCE_')
    return name in _PARENT_KEYS or name.startswith(_PARENT_PREFIXES)


def exclusion_enabled():
    return _enabled


def exclude_parent_authority(env, *, enabled=None):
    """Apply LAST, after context, profile secrets and caller/terminal extras."""
    if enabled is None:
        enabled = exclusion_enabled()
    if not enabled:
        return env
    result = {k: v for k, v in env.items() if not parent_authority_name(k)}
    result[MARKER] = '1'
    return result


def authority_scoped_startup(function):
    """Reapply inherited denial even when startup returns early or an overlay fails."""
    @wraps(function)
    def scoped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        finally:
            if exclusion_enabled():
                cleaned = exclude_parent_authority(os.environ)
                for name in set(os.environ) - set(cleaned):
                    del os.environ[name]
                os.environ.update(cleaned)
    return scoped
