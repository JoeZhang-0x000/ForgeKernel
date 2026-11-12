#!/usr/bin/env python3
"""One agent turn on pi (https://github.com/badlogic/pi-mono), with the contract of a `claude -p` turn.

    agent-pi.py <model> [effort]        prompt on stdin, the reply (the last assistant text) on stdout

`pi --print --mode json` streams one JSON event per line. This reads the stream, prints the final assistant
text, and appends one line per turn to a usage log, which `claude -p` in text mode never offered: tokens, cost,
provider retries and wall time. A turn the provider ended with an error exits non-zero with the provider's
message on stderr, so the loop's own retry (loop/cli.py, _TRANSIENT_AGENT_ERR) sees a 429 as it does for claude.

Every turn is a fresh session, as with `claude -p`: state lives in the working directory, not in the agent.
Sessions and the usage log go under FK_PI_STATE_DIR (default ~/.cache/forgekernel-pi/<hash of the working
directory>/), outside the tree the loop commits.

What is passed to pi:
  --model <model>            "provider/id" as pi names it (pi's own models.json / PI_CODING_AGENT_DIR)
  --thinking <effort>        the task's effort, when it has one
  --skill <dir>              every skill the harness mirrored into ./.claude/skills/ for this turn, then
                             every directory in FK_PI_SKILLS (os.pathsep-separated): material from outside
                             the repository, e.g. a knowledge base every compared system reads
  --no-context-files         no AGENTS.md / CLAUDE.md from the working directory or its parents
  --no-extensions, -e FILE   only the extensions named in FK_PI_EXTENSIONS (os.pathsep-separated)
Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EFFORTS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


def state_dir(cwd: Path) -> Path:
    root = Path(os.environ.get("FK_PI_STATE_DIR") or Path.home() / ".cache" / "forgekernel-pi")
    d = root / hashlib.sha256(str(cwd.resolve()).encode()).hexdigest()[:16]
    d.mkdir(parents=True, exist_ok=True)
    return d


def pi_argv(model: str, effort: str, cwd: Path, sessions: Path, prompt: str) -> list[str]:
    argv = [os.environ.get("FK_PI_BIN", "pi"), "--print", "--mode", "json", "--session-dir", str(sessions),
            "--no-context-files", "--no-extensions", "--no-prompt-templates", "--no-themes", "--no-mcp",
            "--no-approve", "--offline", "--model", model]
    if effort:
        if effort not in EFFORTS:
            raise SystemExit(f"agent-pi.py: effort {effort!r} is not a pi thinking level {list(EFFORTS)}")
        argv += ["--thinking", effort]
    for ext in filter(None, os.environ.get("FK_PI_EXTENSIONS", "").split(os.pathsep)):
        argv += ["--extension", ext]
    skills = sorted(p for p in (cwd / ".claude" / "skills").glob("*") if (p / "SKILL.md").is_file())
    skills += [Path(p) for p in filter(None, os.environ.get("FK_PI_SKILLS", "").split(os.pathsep))]
    for s in skills:
        argv += ["--skill", str(s)]
    if not skills:
        argv.append("--no-skills")
    return [*argv, "--", prompt]


class Turn:
    """What the event stream says: the last assistant text, the usage summed over the assistant messages, the
    provider retries, and how it ended."""

    def __init__(self) -> None:
        self.text = ""
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0, "cost": 0.0}
        self.messages = self.tool_calls = self.retries = 0
        self.retry_wait_s = 0.0
        self.errors: list[str] = []
        self.failed = False
        self._retry_started: float | None = None

    def observe(self, e: dict, now: float) -> None:
        kind = e.get("type")
        if kind == "message_end" and (e.get("message") or {}).get("role") == "assistant":
            m = e["message"]
            u = m.get("usage") or {}
            for ours, theirs in (("input", "input"), ("output", "output"), ("cache_read", "cacheRead"),
                                 ("cache_write", "cacheWrite"), ("total", "totalTokens")):
                self.usage[ours] += u.get(theirs) or 0
            self.usage["cost"] += (u.get("cost") or {}).get("total") or 0.0
            self.messages += 1
            texts = [c.get("text", "") for c in m.get("content") or [] if c.get("type") == "text"]
            if "".join(texts).strip():
                self.text = "".join(texts)
            self.failed = m.get("stopReason") in ("error", "aborted")
            if m.get("stopReason") == "error":
                self.errors.append(str(m.get("errorMessage") or "provider error"))
        elif kind == "tool_execution_start":
            self.tool_calls += 1
        elif kind == "auto_retry_start":
            self.retries += 1
            self._retry_started = now
        elif kind == "auto_retry_end":
            if self._retry_started is not None:
                self.retry_wait_s += now - self._retry_started
                self._retry_started = None
            if not e.get("success"):
                self.failed = True
                self.errors.append(str(e.get("finalError") or "retry failed"))


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    model, effort = argv[0], (argv[1] if len(argv) > 1 else "")
    cwd = Path.cwd()
    state = state_dir(cwd)
    prompt = sys.stdin.read()
    if not prompt.strip():
        print("agent-pi.py: empty prompt on stdin", file=sys.stderr)
        return 2
    t0, turn = time.time(), Turn()
    # stdin closed: in --print mode pi reads a piped stdin as more prompt and waits for its end
    proc = subprocess.Popen(pi_argv(model, effort, cwd, state / "sessions", prompt), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    for raw in proc.stdout:                                  # binary: pi frames events with LF only
        line = raw.strip()
        if not line:
            continue
        try:
            turn.observe(json.loads(line.decode("utf-8")), time.time())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    err = proc.stderr.read().decode("utf-8", "replace")
    rc = proc.wait()
    if rc == 0 and (turn.failed or not turn.text.strip()):
        rc = 1
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)), "role": os.environ.get("FK_AGENT_ROLE", ""),
              "model": model, "effort": effort, "wall_s": round(time.time() - t0, 1), "rc": rc, **turn.usage,
              "assistant_messages": turn.messages, "tool_calls": turn.tool_calls, "retries": turn.retries,
              "retry_wait_s": round(turn.retry_wait_s, 1), "errors": turn.errors[-3:], "cwd": str(cwd)}
    with (state / "usage.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    if rc != 0:
        print("agent-pi.py: " + ("; ".join(turn.errors[-3:]) or f"pi exited with {rc} and no reply"), file=sys.stderr)
        sys.stderr.write(err[-4000:])
    sys.stdout.write(turn.text)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
