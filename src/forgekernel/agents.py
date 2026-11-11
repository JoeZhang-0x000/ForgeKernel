"""Agent commands from a task's ``[agent]`` (merged per role: forgekernel.task).

The harness builds the command line itself (scripts/agent-claude.sh): the task names the
backend, the provider, the model, the effort and a turn's timeout, and the user never writes
a CLI invocation. One script serves both a ``claude -p`` turn and, when the caller asks for
one (FK_AGENT_DRIVER=goal), a /goal session.
"""

from __future__ import annotations

import shlex

from forgekernel import provider as provider_mod
from forgekernel.paths import repo_root

AGENT_SCRIPT = "scripts/agent-claude.sh"
# One launcher per backend, all with the same contract: <model> [effort] [timeout_s], prompt on
# stdin, reply on stdout. pi's provider is pi's own configuration, so a pi agent uses
# provider = "native" (forgekernel.provider has no channel for it).
AGENT_SCRIPTS = {"claude": AGENT_SCRIPT, "pi": "scripts/agent-pi.sh"}


def agent_command(spec, role: str, env: dict | None = None) -> str:
    """The shell command the loop runs for one ``role`` turn (prompt on stdin). A provider
    other than ``native`` rides as environment assignments in front of it: the endpoint and
    the name of the variable that holds the key, never the key."""
    if spec.backend not in AGENT_SCRIPTS:
        raise ValueError(f"backend {spec.backend!r} is not supported ({', '.join(AGENT_SCRIPTS)})")
    grant = provider_mod.grant(spec.backend, spec.provider, env)
    if spec.backend == "pi":
        grant = {**grant, "FK_AGENT_ROLE": role}     # for the turn's line in the usage log
    argv = ["bash", str(repo_root() / AGENT_SCRIPTS[spec.backend]), spec.model, spec.effort or "",
            str(spec.timeout_s)]
    prefix = ["env", *(f"{k}={v}" for k, v in grant.items())] if grant else []
    return " ".join(shlex.quote(a) for a in [*prefix, *argv])
