"""Per-run authority exclusion is separate from freshness and provider credentials."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent import secret_scope
from gateway.session_context import bind_client_env, reset_client_env
from gateway.session_local import _bypass_policy
from gateway.session_managed_worker import _worker_env
from gateway.session_policy import policy_scope
from tools.environments import local

MARKERS = {'PAPERCLIP_API_KEY': 'synthetic-owner', 'PAPERCLIP_TASK_ID': 'synthetic-task',
           'PAPERCLIP_RUN_ID': 'synthetic-run', 'HERMES_RESUME_UNCHANGED_PREFIX_X': 'synthetic-prefix'}


def selected(tmp_path, excluded=True):
    policy = _bypass_policy({'source': 'oneshot', 'cwd': str(tmp_path), 'model': 'gpt-6.1-sol',
        'provider': 'openai-codex', 'reasoning': 'medium', 'toolsets': ['terminal', 'file'],
        'ignore_user_config': True, 'ignore_rules': True, 'fresh_only': True}, private_secrets={})
    request = json.loads(policy.request_json)
    if excluded:
        request['exclude_parent_authority'] = True
    return replace(policy, request_json=json.dumps(request))


@pytest.mark.parametrize('routed', [False, True])
def test_owner_process_authority_removed_after_profile_overlay(tmp_path, monkeypatch, routed):
    monkeypatch.setattr(secret_scope, 'is_multiplex_active', lambda: routed)
    monkeypatch.setattr(local, '_is_routed_home', lambda *a: routed)
    monkeypatch.setattr(secret_scope, 'build_profile_secret_scope', lambda *a: {
        **MARKERS, 'OPENAI_API_KEY': 'synthetic-provider'})
    for name, value in MARKERS.items():
        monkeypatch.setenv(name, value)
    token = bind_client_env({'PAPERCLIP_API_KEY': 'synthetic-client',
                             'HERMES_RESUME_UNCHANGED_PREFIX_Y': 'synthetic-client-prefix'})
    try:
        with policy_scope(selected(tmp_path)):
            env = _worker_env(SimpleNamespace(profile_id=str(tmp_path)))
        actual = dict(__import__('os').environ) if env is None else env
        assert not any(k.startswith(('PAPERCLIP_', 'HERMES_RESUME_UNCHANGED_PREFIX_')) for k in actual)
        if routed:
            assert actual['OPENAI_API_KEY'] == 'synthetic-provider'
        assert actual['HERMES_HOME'] == str(tmp_path)
    finally:
        reset_client_env(token)


@pytest.mark.parametrize('scrub', [False, True])
def test_native_terminal_child_does_not_restore_client_or_explicit_authority(tmp_path, monkeypatch, scrub):
    from agent import managed_authority
    monkeypatch.setattr(managed_authority, '_enabled', True)
    token = bind_client_env(MARKERS)
    try:
        env = local.build_subprocess_env(base={'PATH': '/usr/bin', **MARKERS},
            scrub_secrets=scrub, extra={**MARKERS, '_HERMES_FORCE_PAPERCLIP_API_KEY': 'synthetic-force'})
        assert not any(managed_authority.parent_authority_name(k) for k in env)
        assert env[managed_authority.MARKER] == '1'
    finally:
        reset_client_env(token)


@pytest.mark.parametrize('excluded', [False, True])
def test_startup_overlays_cannot_restore_excluded_authority(tmp_path, monkeypatch, excluded):
    import os
    from agent import managed_authority
    from hermes_cli import env_loader
    monkeypatch.setattr(managed_authority, '_enabled', excluded)
    for name in MARKERS:
        monkeypatch.delenv(name, raising=False)
    for name in ('OPENAI_API_KEY', managed_authority.MARKER):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / '.env').write_text('PAPERCLIP_API_KEY=synthetic-file\nOPENAI_API_KEY=synthetic-provider\n'
                                  + managed_authority.MARKER + '=0\n')
    monkeypatch.setattr(env_loader, '_apply_external_secret_sources', lambda *a: os.environ.update(MARKERS))
    monkeypatch.setattr(env_loader, '_apply_managed_env', lambda **kw: os.environ.update(MARKERS))
    monkeypatch.setattr(env_loader, '_reapply_terminal_config_bridge', lambda *a: None)
    env_loader.load_hermes_dotenv(hermes_home=tmp_path)
    assert os.environ['OPENAI_API_KEY'] == 'synthetic-provider'
    assert ('PAPERCLIP_API_KEY' in os.environ) is (not excluded)
    if excluded:
        assert not any(managed_authority.parent_authority_name(k) for k in os.environ)
        assert os.environ[managed_authority.MARKER] == '1'


@pytest.mark.parametrize('excluded', [False, True])
def test_scoped_secret_reader_cannot_rehydrate_board_authority(monkeypatch, excluded):
    from agent import managed_authority
    monkeypatch.setattr(managed_authority, '_enabled', excluded)
    token = secret_scope.set_secret_scope({**MARKERS, 'OPENAI_API_KEY': 'synthetic-provider'})
    try:
        assert secret_scope.get_secret('OPENAI_API_KEY') == 'synthetic-provider'
        assert secret_scope.get_secret('PAPERCLIP_API_KEY') == (None if excluded else 'synthetic-owner')
    finally:
        secret_scope.reset_secret_scope(token)


def test_served_native_child_applies_denial_after_profile_secret_source(tmp_path, monkeypatch):
    from agent import managed_authority
    monkeypatch.setattr(managed_authority, '_enabled', True)
    monkeypatch.setattr(local, '_is_routed_home', lambda *a: False)
    monkeypatch.setattr(secret_scope, 'build_profile_secret_scope', lambda *a: {
        **MARKERS, 'OPENAI_API_KEY': 'synthetic-provider'})
    env = local.served_profile_child_env(base={'PATH': '/usr/bin'}, target_home=tmp_path, inherit_credentials=True)
    assert not any(managed_authority.parent_authority_name(k) for k in env)
    assert env['OPENAI_API_KEY'] == 'synthetic-provider'


def test_fresh_bootstrap_catalog_and_denial_do_not_load_profile_behavior(tmp_path):
    import os
    from dataclasses import asdict
    from pathlib import Path
    import subprocess
    import sys
    from agent.managed_authority import MARKER
    root = Path(__file__).resolve().parents[2]
    home = tmp_path/'home'
    home.mkdir()
    (home/'config.yaml').write_text('plugins:\n  enabled: [sentinel]\nmodel: hostile\n')
    (home/'.env').write_text('PAPERCLIP_API_KEY=synthetic-profile\n'+MARKER+'=0\n')
    plugin = home/'plugins'/'sentinel'
    plugin.mkdir(parents=True)
    (plugin/'plugin.yaml').write_text('name: sentinel\nversion: 1.0.0\nkind: standalone\n')
    (plugin/'__init__.py').write_text("import os; from pathlib import Path\nPath(os.environ['HERMES_HOME'],'plugin-executed').touch()\ndef register(ctx): pass\n")
    record = tmp_path/'policy.json'
    record.write_text(json.dumps(asdict(selected(tmp_path))))
    env = {k:os.environ[k] for k in ('HOME','PATH','LANG','PYTHONPATH') if k in os.environ}
    env.update(HERMES_HOME=str(home), MER1106_SOURCE_ROOT=str(root), PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(tmp_path), **{MARKER:'1'})
    run = subprocess.run([sys.executable, str(root/'tests/gateway/fixtures/managed_authority_probe.py'), str(record)],
                         cwd=root, env=env, capture_output=True, text=True, timeout=45, check=False)
    assert run.returncode == 0, run.stdout+run.stderr
    print(run.stdout)
    receipt = json.loads(run.stdout.strip().splitlines()[-1])
    assert 'sentinel' not in receipt['plugins'] and receipt['authority_names'] == []
    assert receipt['provider'] == 'openai-codex'
    assert 'terminal' in receipt['full'] and 'process_manage' in receipt['full']


@pytest.mark.parametrize('bad', [None, 'cli', 'follow', 'rules', 'config', 'hooks', 'fresh'])
def test_owner_refuses_unproven_exclusion_policy(tmp_path, bad):
    from hermes_state_runtime import RuntimeStoreError
    request = json.loads(selected(tmp_path).request_json)
    key, value = {None:('exclude_parent_authority','yes'), 'cli':('source','cli'),
                  'follow':('follow_profile_config',True), 'hooks':('accept_hooks',True),
                  'rules':('ignore_rules',False), 'config':('ignore_user_config',False),
                  'fresh':('fresh_only',False)}[bad]
    request[key] = value
    with pytest.raises(RuntimeStoreError):
        _bypass_policy(request, private_secrets={})


def test_owner_without_absolute_home_refuses_exclusion(tmp_path):
    from hermes_state_runtime import RuntimeStoreError
    with policy_scope(selected(tmp_path)), pytest.raises(RuntimeStoreError, match='authority_exclusion_unproven'):
        _worker_env(SimpleNamespace(profile_id='unbound'))


@pytest.mark.parametrize('source', ['process', 'client', 'profile'])
@pytest.mark.parametrize('excluded', [False, True])
def test_independent_marker_sources_and_ordinary_provider_control(tmp_path, monkeypatch, source, excluded):
    import os
    from gateway.session_managed_worker import _profile_worker_env
    from agent import managed_authority
    monkeypatch.setattr(managed_authority, '_enabled', False)
    for key in MARKERS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(secret_scope, 'is_multiplex_active', lambda: True)
    monkeypatch.setattr(local, '_is_routed_home', lambda *a: True)
    profile = {**(MARKERS if source == 'profile' else {}), 'OPENAI_API_KEY':'synthetic-provider'}
    monkeypatch.setattr(secret_scope, 'build_profile_secret_scope', lambda *a: profile)
    if source == 'process':
        for key,value in MARKERS.items():
            monkeypatch.setenv(key,value)
    token = bind_client_env(MARKERS if source == 'client' else {})
    before = dict(os.environ)
    try:
        ordinary = _profile_worker_env(SimpleNamespace(profile_id=str(tmp_path)))
        with policy_scope(selected(tmp_path, excluded=excluded)):
            actual = _worker_env(SimpleNamespace(profile_id=str(tmp_path)))
        assert actual['OPENAI_API_KEY'] == 'synthetic-provider'
        assert dict(os.environ) == before
        if excluded:
            assert not any(managed_authority.parent_authority_name(k) for k in actual)
            assert actual[managed_authority.MARKER] == '1'
        else:
            assert actual == ordinary
            assert actual['PAPERCLIP_TASK_ID'] == 'synthetic-task'
    finally:
        reset_client_env(token)


def test_exact_process_cleanup_preserves_unrelated_group(tmp_path):
    import subprocess
    import sys
    import psutil
    from gateway.session_managed_worker import ManagedWorker, WORKER_POPEN
    command = [sys.executable, '-c', 'import sys; print("ready",flush=True); sys.stdin.read()']
    children = [subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, **WORKER_POPEN) for _ in range(2)]
    try:
        assert all(p.stdout.readline() == b'ready\n' for p in children)
        victim, unrelated = children
        wrapper = ManagedWorker(victim)
        # A reused PID / wrong birth never authorizes signalling that interpreter.
        wrapper.worker = (unrelated.pid, psutil.Process(unrelated.pid).create_time()-1)
        wrapper.close()
        assert victim.poll() is not None
        assert unrelated.poll() is None
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
            child.stdin.close(); child.stdout.close()


def test_private_bootstrap_refuses_missing_owner_denial(tmp_path, monkeypatch):
    from dataclasses import asdict
    from agent import managed_authority
    from agent.managed_worker import bind_worker_policy
    monkeypatch.setattr(managed_authority, '_enabled', False)
    with pytest.raises(ValueError, match='authority_exclusion_unproven'):
        bind_worker_policy({'policy':asdict(selected(tmp_path)), 'ignore_user_config':True,'safe_mode':False})
