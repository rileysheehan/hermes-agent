"""A stdio MCP stand-in for ``cua-driver mcp``, for ``test_computer_use_session_recovery.py``.

Usage: ``python cua_session_fake_server.py <state_dir> <mode>``. Every process appends ``<pid> <tool> <outcome>``
to ``<state_dir>/calls.log`` (``executed`` or ``rejected``) and writes its pid to ``<state_dir>/pid.<n>``.

Like the real driver, a call that carries no ``session`` runs in the transport's implicit ``mcp-<pid>-1`` session.
Modes (only the FIRST process starts with its implicit session ended; later processes are healthy):

- ``healthy``: nothing ends.
- ``revivable``: the implicit session is ended until ``start_session`` names it.
- ``stubborn``: the implicit session stays ended for this process whatever ``start_session`` is told.
"""

import os
import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

STATE_DIR, MODE = sys.argv[1], sys.argv[2]
_PROCESS_NO = len([n for n in os.listdir(STATE_DIR) if n.startswith("pid.")]) + 1
with open(os.path.join(STATE_DIR, f"pid.{_PROCESS_NO}"), "w") as fh:
    fh.write(str(os.getpid()))

IMPLICIT = f"mcp-{os.getpid()}-1"
_ended = {IMPLICIT} if (MODE in {"revivable", "stubborn"} and _PROCESS_NO == 1) else set()

app = MCPServer("cua-driver-fake")


def _log(tool: str, outcome: str) -> None:
    with open(os.path.join(STATE_DIR, "calls.log"), "a") as fh:
        fh.write(f"{os.getpid()} {tool} {outcome}\n")


def _admit(tool: str, session: str | None) -> None:
    sid = session or IMPLICIT
    if sid in _ended:
        _log(tool, "rejected")
        raise ToolError(f"session '{sid}' has ended; tool call '{tool}' was rejected. Call start_session with "
                        "this id to revive it before issuing further actions, or use a new session id.")
    _log(tool, "executed")


@app.tool()
def start_session(session: str) -> str:
    if MODE != "stubborn":
        _ended.discard(session)
    _log("start_session", "executed")
    return f"session {session} started"


@app.tool()
def end_session(session: str) -> str:
    _log("end_session", "executed")
    return f"session {session} ended"


@app.tool()
def bring_to_front(pid: int, window_id: int) -> str:
    """Strict schema with no ``session`` property, like the real driver's."""
    _admit("bring_to_front", None)
    return f"brought {pid}/{window_id} to front"


@app.tool()
def list_apps(session: str | None = None) -> str:
    _admit("list_apps", session)
    return "apps: Music"


@app.tool()
def click(x: int, y: int, session: str | None = None) -> str:
    _admit("click", session)
    return f"clicked {x},{y}"


app.run()
