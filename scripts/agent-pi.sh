#!/usr/bin/env bash
# agent-pi.sh — one agent turn on pi, with the contract of agent-claude.sh.
#
#   bash agent-pi.sh <model> [effort] [timeout_s]
#     prompt on stdin, reply on stdout.
#
# <model> is "provider/id" as pi names it; the provider's endpoint and the variable that holds its
# key are pi's own configuration (PI_CODING_AGENT_DIR/models.json), so a task uses provider = "native".
# The turn runs under the same wall-clock timeout as a claude turn. When the loop's deadline is known
# (FK_DEADLINE_EPOCH, seconds since the epoch), the turn is cut to what is left of it: the loop only
# looks at its deadline between turns, and one turn may otherwise run hours past a fixed budget.
#
# THE AGENT POLICY IS NOT ENFORCED YET. The policy the loop installs before every turn
# (loop/agent_policy.py: the student offline and unable to read the reference) is delivered as
# Claude Code settings, which pi does not read. Until the pi extension and sandbox that carry the
# same policy exist, this script refuses to run unless FK_PI_UNSANDBOXED=1 says the caller knows:
# a result produced that way is not comparable with one produced under the policy.
set -u
MODEL=${1:?model as pi names it, e.g. bailian/qwen3.8-max}
EFFORT=${2:-}
TIMEOUT=${3:-7200}
if [ "${FK_PI_UNSANDBOXED:-}" != "1" ]; then
    echo "agent-pi.sh: the agent policy (offline student, no read of the reference) is not enforced on pi yet;" >&2
    echo "agent-pi.sh: set FK_PI_UNSANDBOXED=1 to run a turn without it" >&2
    exit 2
fi
HERE="$(dirname "${BASH_SOURCE[0]}")"
PY="${FK_AGENT_PY:-python3}"
if [ -n "${FK_DEADLINE_EPOCH:-}" ]; then
    LEFT=$("$PY" -c 'import sys, time; print(max(60, int(float(sys.argv[1]) - time.time())))' "$FK_DEADLINE_EPOCH") || exit 2
    [ "$LEFT" -lt "$TIMEOUT" ] && TIMEOUT=$LEFT
fi
exec "$PY" "$HERE/timeout-run.py" "$TIMEOUT" "$PY" "$HERE/agent-pi.py" "$MODEL" "$EFFORT"
