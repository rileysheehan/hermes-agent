"""The macOS installer's own launchd wrapper is a recognized gateway identity.

``gateway_launchd.launchd_program_arguments`` runs the gateway under
``/usr/bin/osascript`` (JXA ``system()``) for Local Network Privacy (#71206),
and the source install's stderr wrapper is ``<launcher> --run-module
hermes_cli.stderr_timestamp``. Without recognizing both, a client that finds
the gateway absent cannot start an installed service:
``service_identity_unverified``.
"""
import json

import pytest

from hermes_cli.gateway_launchd import launchd_program_arguments
from hermes_cli.gateway_runtime_service_identity import (
    verify_gateway_argv, verify_launchd_loaded, verify_launchd_plist,
)


def _install(tmp_path, *, profile=None, python=False):
    root = tmp_path / "home" / ".hermes"
    home = root / "profiles" / profile if profile else root
    home.mkdir(parents=True, exist_ok=True)
    launcher = str(root / "hermes-agent" / ".hermes" / "bin" / "hermes")
    logs = home / "logs"
    inner = [launcher, *(["--profile", profile] if profile else []), "gateway", "run", "--external-supervisor"]
    if python:
        wrapper = ["/opt/venv/bin/python3.14", "-m", "hermes_cli.stderr_timestamp"]
    else:
        wrapper = [launcher, "--run-module", "hermes_cli.stderr_timestamp"]
    command = [*wrapper, "--error-log", str(logs / "gateway.error.log"), "--", *inner]
    argv = launchd_program_arguments(command, logs / "gateway.log", logs / "gateway.error.log")
    return home, argv


def _with_shell(argv, edit):
    """Re-encode the template around an edited shell string, exactly as the installer would."""
    prefix, _, rest = argv[4].partition("$.system(")
    literal, _, suffix = rest.partition("); const signal")
    shell = edit(json.loads(literal))
    return [*argv[:4], f"{prefix}$.system({json.dumps(shell)}); const signal{suffix}"]


def _loaded(argv, home, *, label="ai.hermes.gateway", uid=501):
    return (f"gui/{uid}/{label} = {{\n\tstate = not running\n\tprogram = {argv[0]}\n\targuments = {{\n"
            + "\n".join("\t\t" + arg for arg in argv)
            + f"\n\t}}\n\tenvironment = {{\n\t\tHERMES_SUPERVISED_CHILD => 1\n\t\tHERMES_HOME => {home}\n\t}}\n}}")


@pytest.mark.parametrize("profile,python", [(None, False), (None, True), ("worker", False)])
def test_installer_launchd_wrapper_is_verified_on_disk_and_loaded(tmp_path, profile, python):
    home, argv = _install(tmp_path, profile=profile, python=python)
    verify_gateway_argv(argv, home)
    label = "ai.hermes.gateway" + (f"-{profile}" if profile else "")
    definition = dict(Label=label, ProgramArguments=argv, RunAtLoad=True,
                      EnvironmentVariables=dict(HERMES_HOME=str(home), HERMES_SUPERVISED_CHILD="1"))
    verify_launchd_plist(definition, label, home)
    verify_launchd_loaded(_loaded(argv, home, label=label), home, uid=501, username="owner")


def test_mercury_live_shape(tmp_path):
    """The exact ProgramArguments the v0.22 installer wrote on Mercury, paths re-rooted."""
    root = tmp_path / ".hermes"
    root.mkdir()
    hermes = f"{root}/hermes-agent/.hermes/bin/hermes"
    script = ('ObjC.import("stdlib"); const status=$.system("exec ' + hermes
              + ' --run-module hermes_cli.stderr_timestamp --error-log ' + f"{root}/logs/gateway.error.log -- "
              + hermes + f' gateway run --external-supervisor >> {root}/logs/gateway.log 2>> {root}/logs/gateway.error.log");'
              ' const signal=status & 127; $.exit(status === -1 ? 1 : signal === 0 ? (status >> 8) & 255 : 128 + signal);')
    verify_gateway_argv(["/usr/bin/osascript", "-l", "JavaScript", "-e", script], root)


def test_paths_needing_quotes_round_trip(tmp_path):
    home, argv = _install(tmp_path / "a dir with 'quotes' & $signs")
    verify_gateway_argv(argv, home)


TAMPERED = {
    "second_command": lambda s: s.replace(" >> ", " ; /bin/sh -c id >> ", 1),
    "and_command": lambda s: s + " && /usr/bin/true",
    "substitution": lambda s: s.replace("exec ", "exec $(id) ", 1),
    "backticks": lambda s: s.replace("exec ", "exec `id` ", 1),
    "pipe": lambda s: s.replace(" >> ", " | /usr/bin/tee >> ", 1),
    "no_exec": lambda s: s[len("exec "):],
    "quoted_redirect": lambda s: s.replace(" >> ", " '>>' ", 1),
    "relative_log": lambda s: s.rsplit(" 2>> ", 1)[0] + " 2>> gateway.error.log",
    "extra_redirect": lambda s: s + " < /dev/null",
    "not_gateway": lambda s: "exec /bin/echo gateway run" + s[s.index(" >> "):],
    "wrong_subcommand": lambda s: s.replace(" gateway run", " gateway restart", 1),
    "extra_flag": lambda s: s.replace("--external-supervisor", "--external-supervisor --replace", 1),
    "hermes_with_dash_m": lambda s: s.replace("--run-module", "-m", 1),
    "other_module": lambda s: s.replace("hermes_cli.stderr_timestamp", "hermes_cli.other", 1),
    "empty": lambda s: "",
}


@pytest.mark.parametrize("case", sorted(TAMPERED))
def test_tampered_shell_is_unverified(tmp_path, case):
    home, argv = _install(tmp_path)
    with pytest.raises(ValueError, match="service_identity_unverified"):
        verify_gateway_argv(_with_shell(argv, TAMPERED[case]), home)


def test_python_with_run_module_is_unverified(tmp_path):
    home, argv = _install(tmp_path, python=True)
    edited = _with_shell(argv, lambda s: s.replace(" -m hermes_cli.stderr_timestamp", " --run-module hermes_cli.stderr_timestamp", 1))
    with pytest.raises(ValueError, match="service_identity_unverified"):
        verify_gateway_argv(edited, home)


@pytest.mark.parametrize("case", [
    "other_osascript", "applescript", "extra_arg", "trailing_js", "leading_js", "two_systems", "unquoted_literal",
])
def test_tampered_wrapper_is_unverified(tmp_path, case):
    home, argv = _install(tmp_path)
    argv = list(argv)
    if case == "other_osascript":
        argv[0] = str(tmp_path / "osascript")
    elif case == "applescript":
        argv[2] = "AppleScript"
    elif case == "extra_arg":
        argv.append("x")
    elif case == "trailing_js":
        argv[4] += ' $.system("id");'
    elif case == "leading_js":
        argv[4] = '$.system("id"); ' + argv[4]
    elif case == "two_systems":
        argv[4] = argv[4].replace('$.system("', '$.system("id") + $.system("', 1)
    elif case == "unquoted_literal":
        argv[4] = argv[4].replace('$.system("', "$.system(String.fromCharCode(105)+\"", 1)
    with pytest.raises(ValueError, match="service_identity_unverified"):
        verify_gateway_argv(argv, home)


def test_wrapped_profile_must_match(tmp_path):
    home, argv = _install(tmp_path, profile="worker")
    with pytest.raises(ValueError, match="profile_mismatch"):
        verify_gateway_argv(_with_shell(argv, lambda s: s.replace("--profile worker", "--profile stranger", 1)), home)


def test_wrapped_home_must_match(tmp_path):
    home, argv = _install(tmp_path)
    other = tmp_path / "other" / ".hermes"
    with pytest.raises(ValueError):
        verify_launchd_loaded(_loaded(argv, other), home, uid=501, username="owner")
