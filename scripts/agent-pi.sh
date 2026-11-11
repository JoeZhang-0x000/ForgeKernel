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
# The agent policy is NOT enforced on pi. The policy the loop installs before every turn
# (loop/agent_policy.py: the student offline and unable to read the reference) is delivered as
# Claude Code settings, which pi does not read, and this backend adds no sandbox of its own. On pi
# the policy is only what the loop's notice at the head of each prompt tells the agent. A result
# from this backend was produced without the isolation a claude run has; say so where it is reported.
set -u
MODEL=${1:?model as pi names it, e.g. bailian/qwen3.8-max}
EFFORT=${2:-}
TIMEOUT=${3:-7200}
HERE="$(dirname "${BASH_SOURCE[0]}")"
PY="${FK_AGENT_PY:-python3}"
if [ -n "${FK_DEADLINE_EPOCH:-}" ]; then
    LEFT=$("$PY" -c 'import sys, time; print(max(60, int(float(sys.argv[1]) - time.time())))' "$FK_DEADLINE_EPOCH") || exit 2
    [ "$LEFT" -lt "$TIMEOUT" ] && TIMEOUT=$LEFT
fi
exec "$PY" "$HERE/timeout-run.py" "$TIMEOUT" "$PY" "$HERE/agent-pi.py" "$MODEL" "$EFFORT"
