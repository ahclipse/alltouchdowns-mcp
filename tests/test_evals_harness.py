"""The eval harness end to end, with scripted stand-ins for the model. No API calls."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from mcp.types import TextContent

from evals import agent, preflight, report, runner, scripted
from evals import truth as truths
from evals.__main__ import main as evals_main
from evals.schema import CASES_DIR, load_cases
from tests.conftest import FIXTURE_DB


def config(out: Path, model: str = scripted.ORACLE, **overrides: Any) -> runner.RunConfig:
    settings = {"model": model, "out": out, "cases_dir": CASES_DIR, "db": FIXTURE_DB, "reps": 1}
    return runner.RunConfig(**{**settings, **overrides})


def rows(out: Path) -> list[dict[str, Any]]:
    return report.read_jsonl(out / "results.jsonl")


# --- The committed cases ---------------------------------------------------------------------


def reported_failures(capsys: pytest.CaptureFixture[str]) -> list[str]:
    """The reasons a check printed, so a failing test says why and not just that."""
    return [line for line in capsys.readouterr().out.splitlines() if line.startswith("FAILED")]


def test_committed_cases_pass_the_free_preflight(capsys: pytest.CaptureFixture[str]) -> None:
    assert evals_main(["preflight"]) == 0, reported_failures(capsys)


def test_oracle_scores_everything_and_null_scores_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert evals_main(["selftest"]) == 0, reported_failures(capsys)


def test_cases_are_a_usable_set() -> None:
    cases = load_cases()
    ids = [case.id for case in cases]

    assert len(ids) == len(set(ids))
    assert {case.split for case in cases} <= {"dev", "heldout"}
    assert any(case.expect == "decline" for case in cases)


# --- The runner ------------------------------------------------------------------------------


def test_a_run_writes_rows_traces_and_a_summary(tmp_path: Path) -> None:
    code = runner.run(config(tmp_path, reps=2))

    cases = load_cases()
    results = rows(tmp_path)
    assert code == runner.EXIT_OK
    assert len(results) == 2 * len(cases)
    assert {(row["case_id"], row["rep"]) for row in results} == {
        (case.id, rep) for case in cases for rep in (0, 1)
    }
    assert all(
        (tmp_path / "traces" / f"{c.id}_rep{rep}.json").exists() for c in cases for rep in (0, 1)
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["valid"] and summary["correct_and_grounded"] == 1.0
    assert "Correct and grounded: 100.0%" in (tmp_path / "summary.md").read_text()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["model"] == scripted.ORACLE and manifest["cases"] == len(cases)


def test_the_copy_that_ci_uploads_has_none_of_the_model_s_own_words(tmp_path: Path) -> None:
    """A public repository's workflow artifacts are public, and model text is unchecked text."""
    run_dir, shared = tmp_path / "run", tmp_path / "shared"
    runner.run(config(run_dir, only="leaderboard-ties-01"))
    (run_dir / "errors.jsonl").write_text(json.dumps({**failed("x"), "detail": "MODEL SAID"}))

    report.shareable_copy(run_dir, shared)

    assert sorted(path.name for path in shared.iterdir()) == [
        "errors.jsonl", "manifest.json", "results.jsonl", "summary.json", "summary.md",
    ]  # fmt: skip
    (row,) = report.read_jsonl(shared / "results.jsonl")
    assert "answer" not in row and row["correct"] is True and row["case_id"]
    everything = "".join(path.read_text() for path in shared.iterdir())
    assert "antrel-rolle" not in everything and "MODEL SAID" not in everything
    assert report.summarise(shared)["correct_and_grounded"] == 1.0  # still enough to score


def test_a_trace_shows_the_whole_conversation(tmp_path: Path) -> None:
    runner.run(config(tmp_path, only="leaderboard-ties-01"))

    trace = json.loads((tmp_path / "traces" / "leaderboard-ties-01_rep0.json").read_text())
    assert [turn["role"] for turn in trace] == [
        "system",
        "user",
        "tool_call",
        "tool_result",
        "assistant",
    ]
    assert trace[2]["name"] == "get_leaderboard"
    assert "antrel-rolle" in trace[3]["content"]


def test_a_finished_run_resumes_without_repeating_trials(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner.run(config(tmp_path))
    before = (tmp_path / "results.jsonl").read_text()

    runner.run(config(tmp_path))

    assert (tmp_path / "results.jsonl").read_text() == before
    assert "0 to run" in capsys.readouterr().out


def test_results_from_a_different_setup_cannot_be_resumed(tmp_path: Path) -> None:
    runner.run(config(tmp_path))

    with pytest.raises(SystemExit, match="different set-up"):
        runner.run(config(tmp_path, model=scripted.NULL))


def test_filters_select_cases(tmp_path: Path) -> None:
    runner.run(config(tmp_path / "a", only="decline"))
    runner.run(config(tmp_path / "b", limit=1))

    assert {row["expect"] for row in rows(tmp_path / "a")} == {"decline"}
    assert len(rows(tmp_path / "b")) == 1


def test_no_tools_baseline_asks_only_answerable_questions(tmp_path: Path) -> None:
    runner.run(config(tmp_path, model=scripted.NULL, tools=False))

    results = rows(tmp_path)
    assert results and {row["expect"] for row in results} == {"answer"}
    assert all(row["tool_calls"] == [] and not row["grounded"] for row in results)


def test_plumbing_failures_go_to_the_error_file_and_invalidate_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(*args: Any, **kwargs: Any) -> None:
        raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://example.test"))

    monkeypatch.setattr(agent, "run_trial", broken)

    code = runner.run(config(tmp_path))

    errors = report.read_jsonl(tmp_path / "errors.jsonl")
    assert code == runner.EXIT_INVALID
    assert rows(tmp_path) == []  # nothing was scored as a wrong answer
    assert errors and {error["class"] for error in errors} == {"api"}
    assert json.loads((tmp_path / "summary.json").read_text())["valid"] is False


def test_a_reply_from_the_wrong_model_stops_the_run_at_once(tmp_path: Path, monkeypatch) -> None:
    """Every later trial would be served by the same wrong model, and billed."""
    real_create = scripted.ScriptedApi.create

    async def served_by_another_model(self, *, model: str, **kwargs: Any):
        response = await real_create(self, model=model, **kwargs)
        response.model = "some-other-model"
        return response

    monkeypatch.setattr(scripted.ScriptedApi, "create", served_by_another_model)

    assert runner.run(config(tmp_path)) == runner.EXIT_INVALID
    errors = report.read_jsonl(tmp_path / "errors.jsonl")
    assert [error["class"] for error in errors] == ["model_mismatch"]  # one trial, not all 90
    assert rows(tmp_path) == []
    assert "different model" in json.loads((tmp_path / "manifest.json").read_text())["stopped"]


def test_a_trial_that_takes_too_long_is_an_error_not_a_score(tmp_path: Path, monkeypatch) -> None:
    real_create = scripted.ScriptedApi.create

    async def slow(self, **kwargs: Any):
        await asyncio.sleep(0.5)
        return await real_create(self, **kwargs)

    monkeypatch.setattr(scripted.ScriptedApi, "create", slow)

    code = runner.run(config(tmp_path, limit=2, trial_timeout_s=0.05))

    assert code == runner.EXIT_INVALID and rows(tmp_path) == []
    assert {e["class"] for e in report.read_jsonl(tmp_path / "errors.jsonl")} == {"timeout"}


def test_a_model_without_a_price_is_refused_before_anything_is_spent(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no price"):
        runner.run(config(tmp_path, model="a-model-nobody-priced"))


def test_cost_cap_stops_the_run_before_it_is_exceeded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(report, "cost_usd", lambda model, usage: 1.0)  # every trial "costs" $1

    code = runner.run(config(tmp_path, reps=5, max_cost_usd=3.0))

    assert code == runner.EXIT_INVALID
    assert 1 <= len(rows(tmp_path)) <= 3
    assert "cost cap" in json.loads((tmp_path / "manifest.json").read_text())["stopped"]


def test_what_a_failed_trial_spent_counts_towards_the_cap_and_the_total(
    tmp_path: Path, monkeypatch
) -> None:
    """A trial can fail after the API has already billed for its first requests."""

    async def billed_then_broken(api: Any, mcp: Any, *, trial: agent.Trial, **settings: Any):
        trial.usage["input_tokens"] += 1000
        raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://example.test"))

    monkeypatch.setattr(agent, "run_trial", billed_then_broken)
    monkeypatch.setattr(report, "cost_usd", lambda model, usage: 1.0)

    code = runner.run(config(tmp_path, max_cost_usd=3.0))

    errors = report.read_jsonl(tmp_path / "errors.jsonl")
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert code == runner.EXIT_INVALID
    assert [error["cost_usd"] for error in errors] == [1.0, 1.0, 1.0]  # then the cap stopped it
    assert summary["cost_usd"] == 3.0
    assert "cost cap" in json.loads((tmp_path / "manifest.json").read_text())["stopped"]


def test_results_with_a_different_number_of_trials_cannot_be_resumed(tmp_path: Path) -> None:
    runner.run(config(tmp_path, limit=2))

    with pytest.raises(SystemExit, match="changed: reps"):
        runner.run(config(tmp_path, limit=2, reps=2))


# --- The agent loop --------------------------------------------------------------------------


class _Block:
    def __init__(self, type: str, **fields: Any) -> None:
        self.type = type
        self.__dict__.update(fields)


class _Usage:
    input_tokens, output_tokens = 100, 20
    cache_read_input_tokens, cache_creation_input_tokens = 50, 0


class _Reply:
    def __init__(self, stop_reason: str, *content: _Block, model: str = "m") -> None:
        self.stop_reason, self.content, self.model, self.usage = (
            stop_reason,
            list(content),
            model,
            _Usage(),
        )


class _Api:
    """Returns the given replies in turn and records each request."""

    def __init__(self, *replies: Any) -> None:
        self._replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    async def create(self, **request: Any) -> _Reply:
        self.requests.append(request)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


FINAL = '{"kind":"number","number":3,"items":[],"boolean":null,"not_covered_reason":null,"note":""}'
TOOLS = [{"name": "get_leaderboard", "description": "", "input_schema": {"type": "object"}}]


class _Mcp:
    """A tool server that answers every call with the name of the tool."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        text = TextContent(type="text", text=f"result of {name}")
        return SimpleNamespace(content=[text], is_error=False)


def tool_use(call_id: str, name: str) -> _Block:
    return _Block("tool_use", id=call_id, name=name, input={"season": 2007})


async def trial(api: _Api, mcp: _Mcp | None = None, **overrides: Any) -> agent.Trial:
    settings = {"model": "m", "effort": "medium", "question": "?", "system": "s", "tools": [],
                "max_turns": 2}  # fmt: skip
    return await agent.run_trial(api, mcp, **{**settings, **overrides})  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_tool_calls_made_together_are_answered_in_one_message() -> None:
    api = _Api(
        _Reply("tool_use", tool_use("a", "get_leaderboard"), tool_use("b", "list_games")),
        _Reply("end_turn", _Block("text", text=FINAL)),
    )
    mcp = _Mcp()

    result = await trial(api, mcp, tools=TOOLS)

    assert [name for name, _ in mcp.calls] == ["get_leaderboard", "list_games"]
    conversation = api.requests[-1]["messages"]
    assert [message["role"] for message in conversation] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [(block["tool_use_id"], block["content"]) for block in conversation[2]["content"]] == [
        ("a", "result of get_leaderboard"),
        ("b", "result of list_games"),
    ]
    assert result.tool_results == ["result of get_leaderboard", "result of list_games"]
    assert result.answer is not None and not result.forced_final


@pytest.mark.anyio
async def test_when_the_turns_run_out_the_model_must_answer_without_tools() -> None:
    api = _Api(
        _Reply("tool_use", tool_use("a", "get_leaderboard")),
        _Reply("tool_use", tool_use("b", "get_leaderboard")),
        _Reply("end_turn", _Block("text", text=FINAL)),
    )

    result = await trial(api, _Mcp(), tools=TOOLS, max_turns=2)

    assert [request["tool_choice"]["type"] for request in api.requests] == ["auto", "auto", "none"]
    assert result.forced_final and result.turns == 3 and result.answer is not None
    assert len(api.requests) == 3  # two rounds of tools plus the forced answer, and no more


@pytest.mark.anyio
async def test_a_model_that_still_asks_for_tools_on_its_last_turn_gets_no_more() -> None:
    """Whatever the reply to the forced last request, the trial ends there, with no answer."""
    api = _Api(*(_Reply("tool_use", tool_use(str(n), "get_leaderboard")) for n in range(3)))
    mcp = _Mcp()

    result = await trial(api, mcp, tools=TOOLS, max_turns=2)

    assert len(api.requests) == 3 and len(mcp.calls) == 2
    assert result.forced_final and result.status == "ok" and result.answer is None


@pytest.mark.anyio
async def test_a_caller_s_trial_keeps_what_was_spent_when_the_trial_fails() -> None:
    request = httpx2.Request("POST", "https://example.test")
    bad = anthropic.BadRequestError("no", response=httpx2.Response(400, request=request), body=None)
    api = _Api(_Reply("tool_use", tool_use("a", "get_leaderboard")), bad)
    kept = agent.Trial()

    with pytest.raises(anthropic.BadRequestError):
        await trial(api, _Mcp(), tools=TOOLS, trial=kept)

    assert kept.usage["input_tokens"] == 100 and kept.turns == 1  # the first request was billed


@pytest.mark.anyio
async def test_final_answer_is_parsed_and_usage_added_up() -> None:
    api = _Api(_Reply("end_turn", _Block("text", text=FINAL)))

    result = await trial(api)

    assert result.answer is not None and result.answer.number == 3
    assert result.status == "ok" and result.turns == 1 and not result.forced_final
    assert result.usage == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_input_tokens": 50,
        "cache_creation_input_tokens": 0,
    }
    request = api.requests[0]
    assert request["output_config"]["effort"] == "medium"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in request and "temperature" not in request


@pytest.mark.anyio
async def test_truncated_and_refused_replies_are_not_answers() -> None:
    cut_off = await trial(_Api(_Reply("max_tokens", _Block("text", text='{"kind":"num'))))
    refused = await trial(_Api(_Reply("refusal")))

    assert (cut_off.status, cut_off.answer) == ("truncated", None)
    assert (refused.status, refused.answer) == ("refused", None)


@pytest.mark.anyio
async def test_text_that_is_not_the_answer_format_is_no_answer() -> None:
    result = await trial(_Api(_Reply("end_turn", _Block("text", text="I think it is three."))))

    assert result.status == "ok" and result.answer is None


@pytest.mark.anyio
async def test_overload_is_retried_and_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(seconds: float) -> None:
        pass

    monkeypatch.setattr(agent.asyncio, "sleep", no_sleep)
    request = httpx2.Request("POST", "https://example.test")
    busy = anthropic.RateLimitError(
        "busy", response=httpx2.Response(429, request=request), body=None
    )
    api = _Api(busy, busy, _Reply("end_turn", _Block("text", text=FINAL)))

    result = await trial(api)

    assert result.api_retries == 2 and result.answer is not None


@pytest.mark.anyio
async def test_the_wait_before_a_retry_is_what_the_api_asks_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(agent.asyncio, "sleep", record)
    request = httpx2.Request("POST", "https://example.test")
    told = httpx2.Response(429, request=request, headers={"retry-after": "7"})
    untold = httpx2.Response(529, request=request)
    api = _Api(
        anthropic.RateLimitError("busy", response=told, body=None),
        anthropic.InternalServerError("overloaded", response=untold, body=None),
        _Reply("end_turn", _Block("text", text=FINAL)),
    )

    result = await trial(api)

    assert result.api_retries == 2 and result.answer is not None
    assert waits[0] == 7.0  # the API said when to come back
    assert 1.0 <= waits[1] <= 3.0  # it did not: the second wait of a doubling series


@pytest.mark.anyio
async def test_a_bad_request_is_not_retried() -> None:
    request = httpx2.Request("POST", "https://example.test")
    bad = anthropic.BadRequestError("no", response=httpx2.Response(400, request=request), body=None)

    with pytest.raises(anthropic.BadRequestError):
        await trial(_Api(bad))


@pytest.mark.anyio
async def test_a_different_model_in_the_reply_stops_the_trial() -> None:
    with pytest.raises(agent.ModelMismatch):
        await trial(_Api(_Reply("end_turn", _Block("text", text=FINAL), model="other")))


# --- Reporting -------------------------------------------------------------------------------


def test_cost_comes_from_the_tokens_and_the_model_that_served() -> None:
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 100_000,
        "cache_read_input_tokens": 2_000_000,
        "cache_creation_input_tokens": 400_000,
    }

    # 1M in at $4 + 0.1M out at $20 + 2M cache reads at $0.20 + 0.4M cache writes at $5
    assert report.cost_usd("claude-opus-5-5", usage) == pytest.approx(4 + 2 + 0.4 + 2)
    with pytest.raises(KeyError):
        report.cost_usd("a-model-with-no-price", usage)  # never silently free


def test_baseline_sets_the_gate_below_the_mean_by_the_noise() -> None:
    def run_scoring(score: float) -> dict[str, Any]:
        return {"correct_and_grounded": score, "model": "m", "effort": "medium", "cases": 90,
                "cases_hash": "abc", "trials_per_case": 3, "valid": True}  # fmt: skip

    steady = report.make_baseline([run_scoring(0.90), run_scoring(0.90), run_scoring(0.90)])
    noisy = report.make_baseline([run_scoring(0.80), run_scoring(0.90), run_scoring(1.00)])

    assert steady["noise_floor"] == pytest.approx(1 / 270**0.5)  # about 6 points
    assert steady["gate"] == pytest.approx(0.90 - steady["noise_floor"])
    assert noisy["gate"] == pytest.approx(0.90 - 2 * 0.10)  # run-to-run spread dominates

    assert report.gate_verdict(run_scoring(0.86), steady)[0] == 0
    assert report.gate_verdict(run_scoring(0.80), steady)[0] == 1
    assert report.gate_verdict({**run_scoring(0.99), "valid": False}, steady)[0] == 2
    # A run of a different experiment says nothing about this baseline.
    for change in ({"cases": 80}, {"cases_hash": "xyz"}, {"trials_per_case": 1}, {"model": "n"}):
        assert report.gate_verdict({**run_scoring(0.99), **change}, steady)[0] == 2
        with pytest.raises(ValueError, match="differ"):
            report.make_baseline([run_scoring(0.9), {**run_scoring(0.9), **change}])


def graded(case_id: str, *, correct: bool = True, status: str = "ok") -> dict[str, Any]:
    """One row of results.jsonl, as the runner writes it."""
    return {
        "case_id": case_id, "rep": 0, "category": "c", "split": "dev", "expect": "answer",
        "model": scripted.ORACLE, "status": status, "stop_reason": "end_turn", "answer": None,
        "answer_kind_given": "number" if status == "ok" else None,
        "correct": correct, "failure": None if correct else "no_answer", "grounded": correct,
        "expected_tools_used": True, "tool_calls": [], "over_budget": False,
        "forced_final": False, "turns": 1, "api_retries": 0, "model_seconds": 1.0,
        "usage": dict(agent.Trial().usage), "banned_terms_from_model": 0,
    }  # fmt: skip


def failed(case_id: str) -> dict[str, Any]:
    """One row of errors.jsonl."""
    return {"case_id": case_id, "rep": 0, "class": "api", "detail": "", "cost_usd": 0.0, "at": ""}


def summary_of(
    run_dir: Path, results: list[dict], errors: list[dict], *, planned: int, **manifest: Any
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    described = {"model": scripted.ORACLE, "effort": "medium", "tools": True, "reps": 1,
                 "cases": planned, "cases_hash": "abc", **manifest}  # fmt: skip
    (run_dir / "manifest.json").write_text(json.dumps(described))
    (run_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in results))
    (run_dir / "errors.jsonl").write_text("".join(json.dumps(row) + "\n" for row in errors))
    return report.summarise(run_dir)


def test_a_reply_cut_off_or_refused_counts_as_wrong(tmp_path: Path) -> None:
    """Leaving them out would score a model only on the questions it managed to finish."""
    results = [
        graded("a"),
        graded("b"),
        graded("c", correct=False, status="truncated"),
        graded("d", correct=False, status="refused"),
    ]

    summary = summary_of(tmp_path, results, [], planned=4)

    assert summary["correct_and_grounded"] == 0.5 and summary["pass_every_trial"] == 0.5
    assert (summary["truncated"], summary["refused"], summary["trials"]) == (1, 1, 4)
    assert summary["valid"]


def test_a_run_is_a_measurement_only_if_nearly_every_planned_trial_was_graded(
    tmp_path: Path,
) -> None:
    ids = [f"case-{n}" for n in range(100)]

    def run(name: str, graded_count: int, **manifest: Any) -> dict[str, Any]:
        results = [graded(case_id) for case_id in ids[:graded_count]]
        errors = [failed(case_id) for case_id in ids[graded_count:]]
        return summary_of(tmp_path / name, results, errors, planned=100, **manifest)

    assert run("two-missing", 98)["valid"]
    assert not run("three-missing", 97)["valid"]
    assert not run("nothing-ran", 0)["valid"]
    assert not run("stopped", 100, stopped="cost cap of $3.00 reached")["valid"]
    assert run("three-missing", 97)["errors"] == 3


def test_an_error_that_a_later_run_made_good_no_longer_counts(tmp_path: Path) -> None:
    ids = [f"case-{n}" for n in range(10)]
    results = [graded(case_id) for case_id in ids]
    earlier_failures = [failed(case_id) for case_id in ids[:4]]  # 40% failed the first time

    summary = summary_of(tmp_path, results, earlier_failures, planned=10)

    assert summary["valid"] and summary["errors"] == 0 and summary["error_attempts"] == 4


def test_reference_path() -> None:
    result = {
        "matched": {"total": 7},
        "rows": [{"player": {"slug": "a"}}, {"player": {"slug": "b"}}],
    }

    assert preflight.resolve_path(result, "matched.total") == 7
    assert preflight.resolve_path(result, "rows[*].player.slug") == ["a", "b"]
    with pytest.raises(KeyError):
        preflight.resolve_path(result, "rows[*].team")


def test_reference_check_catches_a_tool_that_disagrees_with_the_truth() -> None:
    conn = truths.open_database(FIXTURE_DB)
    try:
        (case,) = [c for c in load_cases() if c.id == "leaderboard-ties-01"]
        wrong = case.model_copy(update={"truth_sql": "SELECT 'randy-moss', 1"})
        problems = evals_main  # keep the import used; the check below is the point
        import anyio

        problems = anyio.run(
            preflight.reference, [wrong], FIXTURE_DB, conn, truths.franchises(conn)
        )
    finally:
        conn.close()

    assert len(problems) == 1 and "the tools returned" in problems[0]
