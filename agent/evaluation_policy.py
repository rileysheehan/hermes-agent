"""Opt-in, session-frozen evaluation boundary. Ordinary sessions are unchanged.

Only the gateway's persisted creation policy enables this; a toolset choice or a
client environment variable cannot. The currently supported provider is pinned
because initialization is third-party code and cannot be assumed read-only.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat

CAPABILITY = 'evaluation-isolation-v1'
PLUGIN_SHA256 = 'f96295c1153a90c3217a14cfa4294d76acf63875833c5c497d7e7bf9832b3e07'


class EvaluationIsolationError(RuntimeError):
    """Refused before the prohibited effect, never a retrospective verdict."""


def active(agent=None):
    if agent is not None and getattr(agent, '_evaluation_isolation', False) is True:
        return True
    from gateway.session_policy import active_policy
    policy = active_policy()
    return policy is not None and policy.evaluation_isolation is True


def require_provider(home, config):
    """A source proof before creation/build, not a bank probe or provider import."""
    path = Path(home) / 'plugins/hindsight/__init__.py'
    memory = config.get('memory') or {}
    if (not isinstance(memory, dict) or memory.get('provider') != 'hindsight'
            or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != PLUGIN_SHA256):
        raise EvaluationIsolationError('evaluation read-only provider is unproven')


def marker_path(home, session_id):
    if not isinstance(session_id, str) or not re.fullmatch(r'local-[a-f0-9]{64}', session_id):
        raise EvaluationIsolationError('evaluation session identity is unproven')
    return Path(home) / 'hindsight/read-only-sessions' / session_id


def enrolled(home, session_id):
    path = marker_path(home, session_id)
    try:
        return stat.S_ISREG(path.lstat().st_mode)  # a symlink is never enrollment evidence
    except OSError:
        return False


def require_unenrolled(home, session_id):
    """A fresh ordinary creation needs proved absence, not failed enrollment proof.

    Enrollment is monotonic and outside SQLite's rollback. A failed/crashed
    evaluation may leave only its marker: keep that identity quarantined rather
    than publish production with retention silently disabled. Legacy IDs use the
    companion provider's safe filename alphabet, not the evaluation-only format.
    """
    if (not isinstance(session_id, str) or session_id in {'.', '..'}
            or not re.fullmatch(r'[A-Za-z0-9_.-]+', session_id)):
        raise EvaluationIsolationError('ordinary session identity is unproven')
    path = Path(home) / 'hindsight/read-only-sessions' / session_id
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise EvaluationIsolationError('ordinary session enrollment absence is unproven') from exc
    raise EvaluationIsolationError('enrolled identity cannot become an ordinary session')


def enroll(home, session_id):
    """Owner enrolls before publishing creation. No provider/model is constructed."""
    path = marker_path(home, session_id)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise EvaluationIsolationError('evaluation enrollment is not a regular file')
        os.fsync(fd)
    finally:
        os.close(fd)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    if not enrolled(home, session_id):
        raise EvaluationIsolationError('evaluation enrollment failed')


def bind_agent(agent, session_id):
    """Called BEFORE any constructor work; prewarm cannot outrun this gate."""
    agent._evaluation_isolation = active()
    if not agent._evaluation_isolation:
        return
    from hermes_constants import get_hermes_home
    from gateway.session_policy import active_policy
    home = get_hermes_home()
    require_provider(home, active_policy().config(secrets=False))
    if not enrolled(home, session_id):
        raise EvaluationIsolationError('evaluation enrollment is missing before agent construction')
    agent._evaluation_session_id = session_id
    import logging
    logging.getLogger(__name__).info('Evaluation isolation locked before construction (session=%s)', session_id)


def assert_provider(agent):
    if not active(agent):
        return
    providers = getattr(getattr(agent, '_memory_manager', None), 'providers', None)
    if (not isinstance(providers, list) or len(providers) != 1
            or getattr(providers[0], '_session_id', None) != agent.session_id
            or not providers[0]._writes_refused()
            or getattr(providers[0], '_read_only', False) is not True):
        raise EvaluationIsolationError('evaluation provider adoption is unproven; model call refused')
    import logging
    logging.getLogger(__name__).info('Evaluation isolation provider read-only adopted (session=%s)', agent.session_id)


def empty_tools(agent):
    """Creation-time surface, never a mid-conversation prompt-cache mutation."""
    agent.tools = []
    agent.valid_tool_names = set()
    agent._tool_snapshot_generation = 0
    agent._kanban_worker_guidance = ''
    agent._skip_mcp_refresh = True


def assert_request(agent, tools):
    if active(agent):
        from hermes_constants import get_hermes_home
        if (getattr(agent, 'session_id', None) != getattr(agent, '_evaluation_session_id', None)
                or not enrolled(get_hermes_home(), agent.session_id)
                or tools or agent.tools or agent.valid_tool_names):
            raise EvaluationIsolationError('evaluation request policy changed; model call refused')


def refuse_effect(effect, agent=None):
    if active(agent):
        raise EvaluationIsolationError(f'evaluation forbids {effect}')
