"""Fresh import/bootstrap/catalog probe, no agent construction or owner transport."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

sys.path.insert(0, os.environ['MER1106_SOURCE_ROOT'])
sys.dont_write_bytecode = True
def denied(*a, **kw):
    raise AssertionError('offline dispatch denied')
socket.socket.connect = denied
socket.socket.connect_ex = denied
# Reuse native runtime; this imports only bootstrap/stdlib before frozen policy binding.
from agent.managed_worker import bind_worker_policy
policy = json.loads(Path(sys.argv[1]).read_text())
bind_worker_policy({'safe_mode':False, 'ignore_user_config':True, 'policy':policy})
# Resolve read-only checkout version metadata before fencing all subsequent process dispatch.
from hermes_cli.version_info import get_version_info
get_version_info()
subprocess.Popen = denied
from hermes_cli.config import load_config
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_cli.plugins import discover_plugins, get_plugin_manager
from agent import managed_authority
load_hermes_dotenv()
discover_plugins()
from model_tools import get_tool_definitions
WORKER_DISABLED_TOOLSETS = ()  # v0.22.5's managed worker disables no toolset (delegation runs there now)
from tools import tool_search
full = get_tool_definitions(['terminal', 'file'], list(WORKER_DISABLED_TOOLSETS), quiet_mode=True, skip_tool_search_assembly=True)
visible = get_tool_definitions(['terminal', 'file'], list(WORKER_DISABLED_TOOLSETS), quiet_mode=True)
full_names = sorted(d['function']['name'] for d in full)
visible_names = sorted(d['function']['name'] for d in visible)
search = json.loads(tool_search.dispatch_tool_search({'queries':['browser','gmail send','paperclip','process_manage']}, current_tool_defs=full,
                                                   connector_search=denied, call=denied))
describe = json.loads(tool_search.dispatch_tool_describe({'names':['browser_vault','connectors__gmail__send','paperclip']},
                                                       current_tool_defs=full, connector_describe=denied, call=denied))
assert not any(managed_authority.parent_authority_name(k) for k in os.environ)
assert load_config()['model']['provider'] == 'openai-codex'
plugin_states = {k:{'enabled':v.enabled, 'error':v.error} for k,v in get_plugin_manager()._plugins.items()}
print(json.dumps({'plugin_states':plugin_states,'full':full_names,'visible':visible_names,
                  'sentinel_executed':Path(os.environ['HERMES_HOME'], 'plugin-executed').exists()}))
assert not plugin_states.get('sentinel', {}).get('enabled', False)
assert not Path(os.environ['HERMES_HOME'], 'plugin-executed').exists()
assert not tool_search.connections_in_scope(full)
assert not describe['tools']
assert set(full_names) <= {'terminal','process_manage','read_file','write_file','patch','search_files'}
from model_tools import _dispatch_bridge_tool
bridge_refusals = {}
for name in ('browser_vault', 'connectors__gmail__send', 'paperclip', 'delegate_task'):
    error, resolved = _dispatch_bridge_tool('tool_call', {'calls':[{'name':name,'arguments':{}}]},
                                           ['terminal','file'],list(WORKER_DISABLED_TOOLSETS))
    assert error is not None and resolved is None
    bridge_refusals[name] = json.loads(error)
allowed_error, allowed = _dispatch_bridge_tool('tool_call',
    {'calls':[{'name':'process_manage','arguments':{'action':'list'}}]}, ['terminal','file'],list(WORKER_DISABLED_TOOLSETS))
assert allowed_error is None and allowed == ('process_manage', {'action':'list'})
print(json.dumps({'bridge_refusals':bridge_refusals,'allowed_not_dispatched':allowed,
                 'full':full_names,'visible':visible_names,'search':search,'describe':describe,
    'plugins':sorted(k for k,v in plugin_states.items() if v['enabled']), 'plugin_states':plugin_states,
    'provider':load_config()['model']['provider'],
    'authority_names':[], 'scope':'native config/plugin/schema consumers, no model/worker execution'}, sort_keys=True))
