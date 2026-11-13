"""A local owner's socket survives a stalled owner; real loss on either end still ends it.

The owner answers WebSocket pings on its agent event loop, which a GIL-holding turn can starve. The
owner's own listener already sends no ping on loopback (``build_listener_server``); the client's
default keepalive closed a healthy local socket with ``1011 keepalive ping timeout`` instead, and the
owner then stops the submitter's finite turn as if the submitter had gone.

Every case drives ``connect_gateway`` against a real WebSocket owner in its own process, so a stall
is a real ``SIGSTOP`` and a death a real ``SIGKILL``. The library keepalive is shortened to 0.3 s for
callers that leave it at its default, which keeps the cases fast without changing what is decided.
"""
import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

import hermes_cli.gateway_client as gc

FAST = 0.3          # the shortened library keepalive (interval and timeout)
STALL = 2.0         # several keepalive periods: the default client gives up well inside it

OWNER = textwrap.dedent("""
    import asyncio, json, sys
    from websockets.asyncio.server import serve

    async def handler(ws):
        try:
            async for raw in ws:
                frame = json.loads(raw)
                await ws.send(json.dumps({"jsonrpc": "2.0", "id": frame["id"], "result": "ok"}))
        finally:
            print(f"closed {ws.close_code} {ws.close_reason}", flush=True)

    def pick(ws, offered):
        return "hermes-gateway-v1" if "hermes-gateway-v1" in offered else None

    async def main():
        # The owner's loopback listener: no server ping (build_listener_server).
        async with serve(handler, "127.0.0.1", 0, ping_interval=None, select_subprotocol=pick) as server:
            print(f"port {server.sockets[0].getsockname()[1]}", flush=True)
            await asyncio.Future()

    asyncio.run(main())
""")

SUBMITTER = textwrap.dedent("""
    import asyncio, sys
    from types import SimpleNamespace
    import websockets.asyncio.client as wac
    import hermes_cli.gateway_client as gc
    real = wac.connect
    def connect(*args, **kwargs):  # the same shortened library keepalive as fast_library_keepalive
        if kwargs.get("ping_interval", 0.3) is not None:
            kwargs["ping_interval"] = min(kwargs.get("ping_interval", 0.3), 0.3)
            kwargs.setdefault("ping_timeout", 0.3)
        return real(*args, **kwargs)
    wac.connect = connect
    gc._session_ticket = lambda *a, **k: "t"
    async def main():
        async with gc.connect_gateway(SimpleNamespace(api_origin="http://127.0.0.1:" + sys.argv[1])) as c:
            print(await c.rpc("ping"), flush=True)
            await asyncio.Future()
    asyncio.run(main())
""")


class Owner:
    def __init__(self):
        self.proc = subprocess.Popen([sys.executable, "-c", OWNER], stdout=subprocess.PIPE, text=True)
        line = self.proc.stdout.readline().split()
        assert line[:1] == ["port"], line
        self.port = int(line[1])

    def endpoint(self):
        return SimpleNamespace(api_origin=f"http://127.0.0.1:{self.port}")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGCONT)
            self.proc.kill()
        self.proc.wait(5)


@pytest.fixture
def owner():
    o = Owner()
    yield o
    o.stop()


@pytest.fixture(autouse=True)
def fast_library_keepalive(monkeypatch):
    """A caller that leaves the keepalive at the library default gets FAST; an explicit value stands."""
    import websockets.asyncio.client as wac
    real = wac.connect
    seen = []

    def connect(*args, **kwargs):
        seen.append(dict(kwargs))
        if kwargs.get("ping_interval", FAST) is not None:
            kwargs["ping_interval"] = min(kwargs.get("ping_interval", FAST), FAST)
            kwargs.setdefault("ping_timeout", FAST)
        return real(*args, **kwargs)

    monkeypatch.setattr(wac, "connect", connect)
    monkeypatch.setattr(gc, "_session_ticket", lambda *a, **k: "t")
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    return seen


async def _stall(owner, seconds=STALL):
    owner.proc.send_signal(signal.SIGSTOP)
    try:
        await asyncio.sleep(seconds)
    finally:
        owner.proc.send_signal(signal.SIGCONT)


@pytest.mark.asyncio
async def test_local_owner_stall_keeps_the_socket_and_the_turn(owner, fast_library_keepalive):
    async with gc.connect_gateway(owner.endpoint()) as client:
        assert await client.rpc("ping") == "ok"
        await _stall(owner)
        assert client.websocket.close_code is None
        assert await client.rpc("ping") == "ok"
    assert fast_library_keepalive[0]["ping_interval"] is None


@pytest.mark.asyncio
async def test_explicit_url_keeps_its_keepalive_and_finds_the_stall(owner, monkeypatch):
    """An explicit URL may be a tunnel (the local end healthy, the far end gone): only a ping finds it."""
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", f"ws://127.0.0.1:{owner.port}/api/ws")
    async with gc.connect_gateway() as client:
        assert await client.rpc("ping") == "ok"
        await _stall(owner)
        with pytest.raises(gc.GatewayClientError):
            await client.rpc("ping")
    # What the owner logged in the incident: the client's own close, then its finite-stop.
    assert owner.proc.stdout.readline().strip() == "closed 1011 keepalive ping timeout"


@pytest.mark.asyncio
async def test_owner_death_still_ends_the_local_socket(owner):
    async with gc.connect_gateway(owner.endpoint()) as client:
        assert await client.rpc("ping") == "ok"
        owner.proc.kill()
        owner.proc.wait(5)
        started = time.monotonic()
        with pytest.raises(gc.GatewayClientError):
            for _ in range(50):
                await client.rpc("ping", _timeout=0.5)
                await asyncio.sleep(0.05)
        assert time.monotonic() - started < 3


def test_submitter_death_still_reaches_the_owner_at_once(owner):
    """A killed submitter (a cancelled or timed-out run) closes its socket with a real FIN: the owner's
    finite-stop still sees the viewer go, with no ping involved."""
    env = {**os.environ, "PYTHONPATH": os.getcwd()}
    sub = subprocess.Popen([sys.executable, "-c", SUBMITTER, str(owner.port)], stdout=subprocess.PIPE,
                           text=True, env=env)
    try:
        assert sub.stdout.readline().strip() == "ok"
        sub.kill()
        started = time.monotonic()
        line = owner.proc.stdout.readline()
        assert line.startswith("closed"), line
        assert time.monotonic() - started < 3
        assert "keepalive" not in line
    finally:
        sub.kill()
        sub.wait(5)


def test_submitter_through_a_stall_is_not_reported_gone(owner):
    """The incident's order: the owner stalls while a local submitter is attached. The owner must see
    no disconnect, so it has nothing to stop."""
    env = {**os.environ, "PYTHONPATH": os.getcwd()}
    sub = subprocess.Popen([sys.executable, "-c", SUBMITTER, str(owner.port)], stdout=subprocess.PIPE,
                           text=True, env=env)
    try:
        assert sub.stdout.readline().strip() == "ok"
        owner.proc.send_signal(signal.SIGSTOP)
        time.sleep(STALL)
        owner.proc.send_signal(signal.SIGCONT)
        os.set_blocking(owner.proc.stdout.fileno(), False)
        time.sleep(0.5)
        assert not (owner.proc.stdout.readline() or "").startswith("closed")
        assert sub.poll() is None
    finally:
        sub.kill()
        sub.wait(5)
