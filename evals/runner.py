"""Run every trial of an eval.

The runner's main job, beyond calling the agent, is to keep two things apart that are easy to
blur: what the model got wrong, and what the plumbing got wrong. A trial that produced an
answer is graded and written to `results.jsonl`. A trial that never got that far (the API
failed, a timeout, the wrong model answered) goes to `errors.jsonl` with the reason and what it
had already cost, is never scored, and is tried again on the next run.

Output, under the run directory:

    manifest.json   what was run: model, settings, and hashes of the cases and the database
    results.jsonl   one graded row per trial, appended as each finishes
    errors.jsonl    one row per trial that could not be completed
    traces/         the full conversation of every trial
    summary.json    the numbers
    summary.md      the same, as a short report
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import socket
import subprocess
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
import uvicorn
from anthropic.lib.credentials import WorkloadIdentityError
from mcp import Client

from alltouchdowns_mcp import legal
from alltouchdowns_mcp.app import create_app
from alltouchdowns_mcp.config import Settings
from evals import agent, grade, report, scripted
from evals import truth as truths
from evals.schema import Case, load_cases

EXIT_OK = 0
EXIT_INVALID = 2  # the run did not complete well enough to be a measurement


@dataclass(frozen=True)
class RunConfig:
    model: str
    out: Path
    cases_dir: Path
    db: Path
    effort: str = "medium"
    reps: int = 3
    tools: bool = True  # False runs the no-tools baseline: what can the model answer unaided?
    only: str | None = None  # category or id prefix
    split: str | None = None  # "dev" or "heldout"
    limit: int | None = None  # first N cases only, for a pilot
    concurrency: int = 4
    max_turns: int = 10
    trial_timeout_s: float = 300.0
    max_cost_usd: float = 5.0


def run(config: RunConfig) -> int:
    return asyncio.run(_run(config))


def is_scripted(model: str) -> bool:
    return model in (scripted.ORACLE, scripted.NULL)


async def _run(config: RunConfig) -> int:
    if config.model not in report.PRICES:
        # Without a price the cost cap could not work, so refuse before anything is spent.
        raise SystemExit(f"no price is known for {config.model}: add it to report.PRICES")
    conn = truths.open_database(config.db)
    try:
        cases = _select_cases(config, conn)
        answers = {c.id: truths.compute(conn, c) for c in cases if c.expect == "answer"}
        traps = {c.id: truths.trap_value(conn, c) for c in cases}
        franchises = truths.franchises(conn)
    finally:
        conn.close()

    config.out.mkdir(parents=True, exist_ok=True)
    (config.out / "traces").mkdir(exist_ok=True)

    with _serving(config.db) as url:
        async with Client(url) as probe:
            instructions = probe.instructions or ""
            tools = agent.tool_definitions((await probe.list_tools()).tools)
        if config.tools:
            system = f"{instructions}\n\n{agent.ANSWER_RULES}"
        else:
            system, tools = f"{agent.ANSWER_RULES}\n\n{agent.NO_TOOLS_RULES}", []

        manifest = _manifest(config, cases, system, tools)
        _check_resumable(config.out, manifest)
        earlier_rows = report.read_jsonl(config.out / "results.jsonl")
        earlier_errors = report.read_jsonl(config.out / "errors.jsonl")
        done = {(row["case_id"], row["rep"]) for row in earlier_rows}
        pending = [(c, rep) for c in cases for rep in range(config.reps) if (c.id, rep) not in done]
        print(f"{len(cases)} cases x {config.reps} trials on {config.model}: "
              f"{len(pending)} to run, {len(done)} already done")  # fmt: skip

        if is_scripted(config.model):
            api: Any = scripted.ScriptedApi(cases, answers)  # tests the harness, free
        else:
            # max_retries=0: the agent retries itself, so retries are counted and not timed.
            api = anthropic.AsyncAnthropic(max_retries=0)
            await _require_sign_in(api, config.model)
        state = _RunState(config, url, api, system, tools, answers, traps, franchises)
        # The cost cap covers the whole run directory, including what earlier runs spent.
        state.spent = report.total_cost(earlier_rows, earlier_errors)
        if pending:
            # The first trial runs alone, so there is a measured cost per trial before several
            # run side by side.
            await state.trial(*pending[0])
            await asyncio.gather(*(state.trial(case, rep) for case, rep in pending[1:]))
        await api.close()

    manifest["stopped"] = state.stopped
    manifest["finished_at"] = _now()
    _write_json(config.out / "manifest.json", manifest)
    summary = report.summarise(config.out)
    _write_json(config.out / "summary.json", summary)
    (config.out / "summary.md").write_text(report.markdown(summary))
    print(report.markdown(summary))
    if state.stopped:
        print(f"STOPPED EARLY: {state.stopped}")
    return EXIT_OK if summary["valid"] else EXIT_INVALID


class _RunState:
    """What the concurrent trials share: the budget, the output files and the stop flag."""

    def __init__(
        self,
        config: RunConfig,
        url: str,
        api: Any,
        system: str,
        tools: list[dict[str, Any]],
        answers: Mapping[str, truths.Truth],
        traps: Mapping[str, int | None],
        franchises: Mapping[str, str],
    ) -> None:
        self.config, self.url, self.api = config, url, api
        self.system, self.tools = system, tools
        self.answers, self.traps, self.franchises = answers, traps, franchises
        self.slots = asyncio.Semaphore(config.concurrency)
        self.spent = 0.0
        self.costliest_trial = 0.0
        self.in_flight = 0
        self.stopped: str | None = None

    async def trial(self, case: Case, rep: int) -> None:
        async with self.slots:
            # Stop before the cap, not after it: assume every trial already running will cost
            # as much as the most expensive one seen so far.
            projected = self.spent + (self.in_flight + 1) * self.costliest_trial
            if self.stopped is None and projected > self.config.max_cost_usd:
                self.stopped = f"cost cap of ${self.config.max_cost_usd:.2f} reached"
            if self.stopped:
                return
            self.in_flight += 1
            try:
                await self._run_one(case, rep)
            finally:
                self.in_flight -= 1

    async def _run_one(self, case: Case, rep: int) -> None:
        config = self.config
        settings = {
            "model": config.model,
            "effort": config.effort,
            "question": case.question,
            "system": self.system,
            "tools": self.tools,
            "max_turns": config.max_turns,
        }
        # Created here and filled in by the agent, so that what a trial spent is still known
        # if it fails or times out half-way.
        trial = agent.Trial()
        try:
            # A hard ceiling on the whole trial. A stalled connection can look alive forever,
            # so a timeout on inactivity alone would never free the slot.
            async with asyncio.timeout(config.trial_timeout_s):
                if config.tools:
                    async with Client(self.url) as mcp:  # a fresh connection per trial
                        await agent.run_trial(self.api, mcp, trial=trial, **settings)
                else:
                    await agent.run_trial(self.api, None, trial=trial, **settings)
        except Exception as raised:
            error = _root_cause(raised)
            # Failures that every later trial would repeat: stop at once.
            if _sign_in_refused(error):
                self.stopped = "the API rejected the credentials"
            if isinstance(error, agent.ModelMismatch):
                self.stopped = "the API answered with a different model than the one asked for"
            self._error(case, rep, _error_class(error), error, self._charge(trial))
        else:
            self._charge(trial)
            self._record(case, rep, trial)

    def _charge(self, trial: agent.Trial) -> float:
        """Count what a trial cost, finished or not. Priced as the model that was asked for:
        when another model answered, that is the best estimate there is."""
        cost = report.cost_usd(self.config.model, trial.usage)
        self.spent += cost
        self.costliest_trial = max(self.costliest_trial, cost)
        return cost

    def _record(self, case: Case, rep: int, trial: agent.Trial) -> None:
        verdict = grade.grade(
            case,
            trial.answer,
            self.answers.get(case.id),
            franchises=self.franchises,
            trap=self.traps.get(case.id),
        )
        answer = trial.answer
        # Everything the model itself wrote: its text (the final answer included) and the
        # arguments it passed to tools.
        written = " ".join(
            turn["content"] for turn in trial.transcript if turn["role"] in _WRITTEN_BY_THE_MODEL
        )
        row = {
            "case_id": case.id,
            "rep": rep,
            "category": case.category,
            "split": case.split,
            "expect": case.expect,
            "model": trial.model,
            "status": trial.status,
            "stop_reason": trial.stop_reason,
            "answer": answer.model_dump() if answer else None,
            "answer_kind_given": answer.kind if answer else None,
            "correct": verdict.correct,
            "failure": verdict.failure,
            "grounded": grade.grounded(case, answer, trial.tool_results, self.franchises),
            "expected_tools_used": grade.used_expected_tools(
                case, [call.name for call in trial.tool_calls]
            ),
            "tool_calls": [
                {k: v for k, v in dataclasses.asdict(call).items() if k != "arguments"}
                for call in trial.tool_calls
            ],
            "over_budget": len(trial.tool_calls) > case.max_calls,
            "forced_final": trial.forced_final,
            "turns": trial.turns,
            "api_retries": trial.api_retries,
            "model_seconds": round(trial.model_seconds, 3),
            "usage": trial.usage,
            # Does the model use terms the server never would? Reported, not graded.
            "banned_terms_from_model": len(legal.find_banned(written)),
        }
        _append(self.config.out / "results.jsonl", row)
        _write_json(self.config.out / "traces" / f"{case.id}_rep{rep}.json", trial.transcript)

    def _error(self, case: Case, rep: int, kind: str, error: BaseException, cost: float) -> None:
        row = {"case_id": case.id, "rep": rep, "class": kind, "detail": repr(error)[:300],
               "cost_usd": cost, "at": _now()}  # fmt: skip
        _append(self.config.out / "errors.jsonl", row)


_WRITTEN_BY_THE_MODEL = ("assistant", "tool_call")


def _root_cause(error: BaseException) -> BaseException:
    """The MCP client runs inside task groups, which wrap whatever is raised within them in
    an ExceptionGroup. Dig out the one real error."""
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return error


def _sign_in_refused(error: BaseException) -> bool:
    """The credentials themselves were turned down, as opposed to a request failing."""
    if isinstance(error, anthropic.AuthenticationError | anthropic.CredentialsError):
        return True
    # Signing in by federation: the exchange of GitHub's token for an access token was refused.
    return isinstance(error, WorkloadIdentityError) and error.status_code == 401


def _error_class(error: BaseException) -> str:
    """Why a trial could not be completed. None of these is the model getting it wrong."""
    if isinstance(error, agent.ModelMismatch):
        return "model_mismatch"
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, anthropic.APIError | WorkloadIdentityError | RuntimeError):
        return "api"
    return "harness"  # a bug in this code or in the server


# --- set-up ----------------------------------------------------------------------------------


async def _require_sign_in(api: Any, model: str) -> None:
    """Before anything is spent, ask the API which models these credentials may use.

    The question is free. A refused key, a federation rule that does not match and a model
    that is not offered all end the run here, with the reason, instead of failing every trial
    one after another.
    """
    try:
        offered = {listed.id async for listed in api.models.list(limit=1000)}
    except Exception as error:
        reason = f"{type(error).__name__}: {str(error)[:400]}"
        raise SystemExit(f"could not sign in to the API: {reason}") from None
    if model not in offered:
        raise SystemExit(f"the API does not offer {model} to these credentials")


def _select_cases(config: RunConfig, conn) -> list[Case]:
    cases = truths.cases_that_hold(conn, load_cases(config.cases_dir))
    if not config.tools:
        cases = [c for c in cases if c.expect == "answer"]
    if config.only:
        cases = [
            c for c in cases if c.category.startswith(config.only) or c.id.startswith(config.only)
        ]
    if config.split:
        cases = [c for c in cases if c.split == config.split]
    return cases[: config.limit] if config.limit else cases


@contextmanager
def _serving(db: Path) -> Iterator[str]:
    """Run the real server on a free local port, the same app production runs.

    Rate limits are a setting of the app; this starts it without any, since the eval makes
    far more calls per minute than one real client would.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings = Settings(db_path=db, port=port, rate_limits=False)
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("the server did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _manifest(
    config: RunConfig, cases: list[Case], system: str, tools: list[dict[str, Any]]
) -> dict[str, Any]:
    def digest(*parts: bytes) -> str:
        return hashlib.sha256(b"\0".join(parts)).hexdigest()[:16]

    root = Path(__file__).parent.parent
    code = sorted([*root.glob("evals/*.py"), *root.glob("src/alltouchdowns_mcp/**/*.py")])
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return {
        "model": config.model,
        "effort": config.effort,
        "tools": config.tools,
        "reps": config.reps,
        "max_turns": config.max_turns,
        "cases": len(cases),
        # If any of these change, earlier results are not comparable and must not be resumed.
        "cases_hash": digest(*(c.model_dump_json().encode() for c in cases)),
        "database_hash": digest(config.db.read_bytes()),
        "prompt_hash": digest(system.encode(), json.dumps(tools, sort_keys=True).encode()),
        # The harness and the server: rows are graded as they are written, so rows from two
        # versions of the grader or the tools must not end up in one score.
        "code_hash": digest(*(path.read_bytes() for path in code)),
        "git_commit": git.stdout.strip() or "uncommitted",
        "started_at": _now(),
    }


_MUST_MATCH = ("model", "effort", "tools", "reps", "max_turns", "cases_hash", "database_hash",
               "prompt_hash", "code_hash")  # fmt: skip


def _check_resumable(out: Path, manifest: dict[str, Any]) -> None:
    path = out / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        changed = [key for key in _MUST_MATCH if previous.get(key) != manifest[key]]
        if changed:
            raise SystemExit(
                f"{out} holds results from a different set-up (changed: {', '.join(changed)}). "
                "Use a new --out directory."
            )
        manifest["started_at"] = previous["started_at"]
    _write_json(path, manifest)


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as file:
        file.write(json.dumps(row) + "\n")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
