"""The pi backend: scripts/agent-pi.sh has the contract of scripts/agent-claude.sh (model, effort, timeout;
prompt on stdin, reply on stdout), and a task may name it. pi itself is a stand-in here that prints the event
stream a real `pi --print --mode json` would."""
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from forgekernel import agents
from forgekernel.loop import agentctx
from forgekernel.task import BACKENDS, AgentSpec

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "agent-pi.sh"

FAKE_PI = """#!{python}
import json, os, sys
argv = sys.argv[1:]
prompt = argv[argv.index("--") + 1]
open(os.environ["FAKE_PI_ARGV"], "w").write(json.dumps(argv))
def emit(**e): print(json.dumps(e), flush=True)
def msg(text, stop="stop", **kw):
    emit(type="message_end", message={{"role": "assistant", "content": [{{"type": "text", "text": text}}],
         "stopReason": stop, "usage": {{"input": 100, "output": 10, "cacheRead": 5, "cacheWrite": 0,
         "totalTokens": 115, "cost": {{"total": 0.01}}}}, **kw}})
mode = os.environ.get("FAKE_PI_MODE", "ok")
if mode == "ok":
    emit(type="message_end", message={{"role": "user", "content": [{{"type": "text", "text": prompt}}]}})
    msg("let me look")
    emit(type="tool_execution_start", toolName="bash", args={{"command": "ls"}})
    emit(type="auto_retry_start"); emit(type="auto_retry_end", success=True)
    msg("the reply to: " + prompt)
elif mode == "error":
    msg("", stop="error", errorMessage="429 rate limit exceeded")
elif mode == "silent":
    emit(type="agent_end")
elif mode == "hang":
    import time; time.sleep(60)
"""


@pytest.fixture
def pi(tmp_path, monkeypatch):
    fake = tmp_path / "pi"
    fake.write_text(FAKE_PI.format(python=sys.executable))
    fake.chmod(stat.S_IRWXU)
    work = tmp_path / "ws"
    work.mkdir()
    env = {**os.environ, "FK_PI_BIN": str(fake), "FK_PI_STATE_DIR": str(tmp_path / "state"),
           "FAKE_PI_ARGV": str(tmp_path / "argv.json"), "FK_AGENT_PY": sys.executable}
    for k in ("FK_DEADLINE_EPOCH", "FK_PI_EXTENSIONS", "FK_PI_SKILLS", "FAKE_PI_MODE", "FK_AGENT_ROLE"):
        env.pop(k, None)

    def turn(prompt="do it", *args, **more):
        r = subprocess.run(["bash", str(SCRIPT), *(args or ("bailian/qwen3.8-max", "high", "30"))], input=prompt,
                           capture_output=True, text=True, cwd=work, env={**env, **more}, check=False)
        argv_file = tmp_path / "argv.json"
        usage = sorted((tmp_path / "state").glob("*/usage.jsonl"))
        return r, (json.loads(argv_file.read_text()) if argv_file.exists() else None), \
            ([json.loads(ln) for ln in usage[0].read_text().splitlines()] if usage else [])

    turn.work = work
    return turn


def test_a_turn_prints_the_last_assistant_text_and_logs_its_usage(pi):
    for name in ("sass", "timeline"):
        (pi.work / ".claude" / "skills" / name).mkdir(parents=True)
        (pi.work / ".claude" / "skills" / name / "SKILL.md").write_text("# skill\n")
    (pi.work / ".claude" / "skills" / "not-a-skill").mkdir()
    r, argv, usage = pi("optimize the kernel", FK_AGENT_ROLE="student")
    assert r.returncode == 0 and r.stdout == "the reply to: optimize the kernel"
    assert argv[-2:] == ["--", "optimize the kernel"] and argv[:3] == ["--print", "--mode", "json"]
    assert argv[argv.index("--model") + 1] == "bailian/qwen3.8-max" and argv[argv.index("--thinking") + 1] == "high"
    assert [Path(argv[i + 1]).name for i, a in enumerate(argv) if a == "--skill"] == ["sass", "timeline"]
    assert {"--no-context-files", "--no-extensions", "--offline"} <= set(argv) and "--extension" not in argv
    assert "--session-id" not in argv                        # a fresh session every turn, as `claude -p`
    (rec,) = usage
    assert (rec["role"], rec["model"], rec["rc"]) == ("student", "bailian/qwen3.8-max", 0)
    assert (rec["input"], rec["output"], rec["cache_read"], rec["total"]) == (200, 20, 10, 230)
    assert rec["cost"] == pytest.approx(0.02) and (rec["assistant_messages"], rec["tool_calls"], rec["retries"]) == (2, 1, 1)
    assert not list(pi.work.rglob("usage.jsonl")) and not (pi.work / ".pi").exists()   # nothing in the committed tree


def test_extensions_are_only_the_named_ones_and_no_effort_means_pis_default(pi):
    r, argv, _ = pi("x", "bailian/qwen3.8-max", "", "30", FK_PI_EXTENSIONS=os.pathsep.join(["/a/guard.ts", "/b/policy.ts"]))
    assert r.returncode == 0 and "--thinking" not in argv and "--no-skills" in argv
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--extension"] == ["/a/guard.ts", "/b/policy.ts"]
    r, _, _ = pi("x", "bailian/qwen3.8-max", "ultra", "30")
    assert r.returncode != 0 and "not a pi thinking level" in r.stderr


def test_skills_from_outside_the_repository_follow_the_mirrored_ones(pi):
    r, argv, _ = pi("x", "bailian/qwen3.8-max", "", "30", FK_PI_SKILLS=os.pathsep.join(["/kb/KernelWiki", "/kb/other"]))
    assert r.returncode == 0 and "--no-skills" not in argv
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--skill"][-2:] == ["/kb/KernelWiki", "/kb/other"]


def test_a_provider_error_fails_the_turn_with_the_message_the_loop_retries_on(pi):
    from forgekernel.loop.cli import _TRANSIENT_AGENT_ERR
    r, _, usage = pi(FAKE_PI_MODE="error")
    assert r.returncode != 0 and r.stdout == "" and _TRANSIENT_AGENT_ERR.search(r.stderr)
    assert usage[-1]["rc"] != 0 and usage[-1]["errors"] == ["429 rate limit exceeded"]
    r, _, _ = pi(FAKE_PI_MODE="silent")                      # pi exits 0 having said nothing: not a reply
    assert r.returncode != 0 and "no reply" in r.stderr
    r, _, _ = pi("   ")
    assert r.returncode == 2 and "empty prompt" in r.stderr


def test_the_turn_is_cut_at_its_timeout_and_at_the_loops_deadline(pi):
    t0 = time.monotonic()
    r, _, _ = pi("x", "bailian/qwen3.8-max", "", "2", FAKE_PI_MODE="hang")
    assert r.returncode == 124 and time.monotonic() - t0 < 30
    # the deadline is nearer than the timeout: what is left of it bounds the turn (never under a minute)
    probe = "import sys, time; print(max(60, int(float(sys.argv[1]) - time.time())))"
    left = subprocess.run([sys.executable, "-c", probe, str(time.time() + 600)], capture_output=True, text=True, check=True)
    assert 590 <= int(left.stdout) <= 600
    assert 'LEFT" -lt "$TIMEOUT"' in SCRIPT.read_text()


def test_it_runs_without_a_sandbox_and_says_the_policy_is_not_enforced(pi):
    """This backend adds no isolation of its own (the policy is Claude Code settings, which pi does not read):
    nothing has to be set for a turn to run, and the launcher says so where a reader of it looks."""
    r, argv, _ = pi()
    assert r.returncode == 0 and argv is not None
    assert "NOT enforced on pi" in SCRIPT.read_text()


def test_a_task_may_name_the_pi_backend_and_gets_its_launcher(monkeypatch):
    assert BACKENDS == ("claude", "pi")
    spec = AgentSpec(backend="pi", model="bailian/qwen3.8-max", effort="high", provider="native", timeout_s=7200)
    cmd = agents.agent_command(spec, "student")
    assert cmd == f"env FK_AGENT_ROLE=student bash {REPO / 'scripts' / 'agent-pi.sh'} bailian/qwen3.8-max high 7200"
    claude = AgentSpec(backend="claude", model="claude-opus-5-5", effort=None, provider="native", timeout_s=60)
    assert agents.agent_command(claude, "teacher") == f"bash {REPO / 'scripts' / 'agent-claude.sh'} claude-opus-5-5 '' 60"
    with pytest.raises(ValueError, match="not supported"):
        agents.agent_command(AgentSpec(backend="codex", model="m", effort=None, provider="native", timeout_s=1), "student")
    # the context-window gate is about claude model ids behind a gateway: a pi command has no such leg
    facts = agentctx.inspect_agent_cmd(cmd, {"ANTHROPIC_BASE_URL": "https://gateway.example"}, cwd=REPO)
    assert agentctx.window_findings(facts) == []


def test_a_run_on_pi_goes_without_the_agent_policy(tmp_path):
    """The policy is Claude Code settings; a pi agent would run beside it, not under it. So a run with a pi agent
    is told to go without: nothing installed, no notice in front of the prompt, no refusal at launch."""
    from types import SimpleNamespace as NS

    from forgekernel import launch
    from forgekernel.loop import cli as loop_cli
    pi_spec = AgentSpec(backend="pi", model="bailian/qwen3.8-max", effort=None)
    claude = AgentSpec(backend="claude", model="claude-opus-5-5", effort=None)
    assert launch.policy_free(NS(teacher=pi_spec, student=pi_spec))
    assert launch.policy_free(NS(teacher=claude, student=pi_spec))      # half a policy is not a policy
    assert not launch.policy_free(NS(teacher=claude, student=claude))
    # the backend the loop builds for such a run: the prompt reaches the agent as written, nothing in .claude/
    script = tmp_path / "echo.sh"
    script.write_text("#!/bin/sh\ncat\n")
    script.chmod(stat.S_IRWXU)
    work = tmp_path / "ws"
    work.mkdir()
    be = loop_cli.RealBackend("op", work, str(script), "n", "n@localhost", agent_policy=None)
    assert be.run_agent("the prompt as written") == "the prompt as written"
    assert not (work / ".claude").exists()


def test_a_board_card_with_a_source_reference_gives_it_to_the_teacher(tmp_path):
    """The dataset's baseline solution ships in the card as source. The loop config names its files as the
    oracle, entry first, at their place in the run's workspace, so the run distils a milestone from the
    reference; a card whose reference is the definition's own, or sealed, gets none."""
    import tomllib
    from types import SimpleNamespace as NS

    from forgekernel import taskcard
    op, ws = tmp_path / "card", tmp_path / "run" / "ws"
    base = op / "eval" / "leaderboard" / "baseline"
    (base / "kernel").mkdir(parents=True)
    (base / "main.py").write_text("def run(): ...\n")
    (base / "kernel" / "impl.py").write_text("x = 1\n")
    (base / "__pycache__").mkdir()
    (base / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"\0")
    problem = {"reference_impl": {"kind": "baseline_solution", "name": "seed", "path": "baseline/main.py", "entry": "run"}}
    (op / "eval" / "leaderboard" / "problem.json").write_text(json.dumps(problem))
    (op / "loop.toml.example").write_text('[loop]\nlanguage = "cutedsl"\nref_cmd = "python3 eval/bench.py --mode ref"\n')
    task = NS(arch="sm90", target=NS(timing="device"))
    run = NS(ws=ws)
    loop = tomllib.loads(taskcard.write_board_loop_toml(run, task, op, deny=[]).read_text())["loop"]
    want = [str(ws / "eval" / "leaderboard" / "baseline" / "main.py"),
            str(ws / "eval" / "leaderboard" / "baseline" / "kernel" / "impl.py")]
    assert loop["oracle"] == want and loop["kb_arch"] == "sm90" and "oracle_mode" not in loop
    from forgekernel.loop.cli import _resolve_oracle, _resolve_oracle_mode
    mode = _resolve_oracle_mode(False, None, {"loop": loop})
    assert mode == "source" and _resolve_oracle(mode, None, {"loop": loop}) == ", ".join(want)

    for other in ({"kind": "definition"}, {"kind": "sealed", "name": "x"}):
        (op / "eval" / "leaderboard" / "problem.json").write_text(json.dumps({"reference_impl": other}))
        assert "oracle" not in tomllib.loads(taskcard.write_board_loop_toml(run, task, op, deny=[]).read_text())["loop"]
    (op / "eval" / "leaderboard" / "problem.json").write_text(json.dumps(problem))
    (op / "loop.toml.example").write_text('[loop]\noracle_mode = "blackbox"\n')        # a card that says otherwise keeps its say
    assert "oracle" not in tomllib.loads(taskcard.write_board_loop_toml(run, task, op, deny=[]).read_text())["loop"]
