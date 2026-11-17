# AllTouchdowns MCP

A free, public, read-only [Model Context Protocol](https://modelcontextprotocol.io) server that answers questions about every NFL touchdown since 1999. It is the data behind [alltouchdowns.com](https://alltouchdowns.com), made available to AI assistants.

> **Status: not deployed yet.** The server, its tests and the eval suite are complete and run locally. The public endpoint, the sign-in tier and the published eval scores are still to come.

Ask an assistant connected to it things like:

- "How many touchdowns did LaDainian Tomlinson score in the 2007 regular season?"
- "Who scored in overtime in week 9 of 2019?"
- "Which San Diego players had the most rushing touchdowns in 2007?"

## What it shows

This is a portfolio project. It is small on purpose, so that every part can be read and explained:

- **A production MCP server in Python**, on the official `mcp` SDK, with typed tool results, a limit on the work a query may do, rate limiting and OAuth.
- **An eval suite with real ground truth.** An LLM agent is given the server's tools and about 100 questions. Each answer is graded against SQL run on the same database, not against a stored answer key.
- **A data pipeline with rules that can be read in one place**: what may be published, what is held back, and what is never copied.

## Tools

| Tool | What it does |
|---|---|
| `find_players` | Looks a player up by name and returns slugs. Tells namesakes apart by position, seasons and teams. |
| `get_player` | One player's touchdowns by type, season and team. |
| `search_touchdowns` | Individual touchdowns filtered by player, team, season, week, type, quarter and length. Returns the full count, so "how many" takes one call. |
| `get_game` | One finished game: final score and every touchdown in order. |
| `list_games` | Finished games in a season, by week or team. |
| `get_leaderboard` | Top scorers for a season or across all seasons. Keeps every player tied at the cut-off. |

All tools are read-only. Every tool result and every protocol error carries the data attribution.

## Design decisions

**Scorers, not passers.** The data credits each touchdown to the player who scored it. On a pass, that is the receiver. So the types are called `receiving` and `rushing`, and every description says that passes thrown are not recorded. A question about a quarterback's touchdown passes is declined, because the data would give a confident wrong number: a quarterback's "receiving" touchdowns are the rare passes he caught.

**Regular season and postseason are always separate.** Every count comes back as three numbers: regular season, postseason and total. Mixing them changes the season leader in 6 of 27 seasons.

**Slugs, not names.** Several players share a name, and the best-known one often does not have the plain slug: the Adrian Peterson with 131 touchdowns is `adrian-peterson-2`. Tools that take a player take a slug, and `find_players` exists to get it.

**Cities, not nicknames.** Teams are identified by city and abbreviation, with the city of the era: San Diego in 2007, Los Angeles (LAC) in 2019. A search for "San Diego" finds the whole franchise and labels each season correctly. "Los Angeles" and "New York" each have two teams, so the server asks which.

**Errors a model can act on.** An error is a short fixed sentence with a code, such as `ambiguous_team: More than one team plays in that city. Use the abbreviation. Choose Los Angeles (LA) or Los Angeles (LAC).` Errors never repeat what the caller sent.

**Two endpoints instead of optional sign-in.** MCP clients only offer to sign in after a server answers 401. A single URL that also accepts anonymous calls would never trigger that, so the sign-in tier would be unreachable. Anonymous access and signed-in access therefore live at two URLs, with a higher rate limit on the second.

**The database is replaced, never modified.** A daily export builds a new file, checks it, and swaps it in with an atomic rename. The server opens the file fresh for each call, so it needs no reload logic and a call in flight is never disturbed. One thing is read only at start-up: the season range quoted in the server's instructions. Every result reports the range actually loaded, and the instructions say to trust that one.

**Recent games are held back until the source has caught up.** Play-by-play data can lag a finished game. A game from the last 7 days is published only once its touchdown records have arrived. A handful of early games whose records are missing upstream are flagged as incomplete rather than reported as having no touchdowns.

## Evals

The eval suite lives in [`evals/`](evals). An agent (a Claude model) gets the server's tools and one question per case, and must return a structured answer.

- **Ground truth is SQL.** Each case is graded against a query, not a stored answer. The query runs at eval time against the same database the server is serving, and may only use columns copied unchanged from the source, so a bug in the server's own derived data cannot hide itself. (A case's notes record the answer for human readers; the grader never reads them.)
- **Grading is exact and uses no model.** Answers are numbers, yes/no, or lists of identifiers (player slugs, game ids, team abbreviations). Ranked lists accept any order within a tie.
- **Declining is tested too.** About a quarter of the cases must be answered "not covered": passes thrown, seasons outside the data, statistics that are not touchdowns, requests for a bulk export.
- **The headline is "correct and grounded"**: the answer is right, and it came from a tool result rather than the model's memory.
- **Plumbing failures are not scores, and cannot hide them.** A trial the harness could not complete (an API error, a timeout) is recorded separately, retried on the next run, and never counted as a wrong answer. A run counts as a measurement only if at least 98% of its planned trials were graded. A reply that was cut off or refused is the model's own, so it counts as wrong.
- **The suite checks itself for free.** On every pull request, a scripted "oracle" agent must score 100% and a "null" agent 0% through the real runner, server and grader, and each case's scripted solution must reproduce its ground truth through the real tools.
- **The pass mark is measured, not chosen.** The gate is set from repeated runs of the current code, below their mean by the larger of twice the run-to-run spread and the noise floor for that many trials.

In CI the suite runs against a committed two-season fixture (2007 and 2019), so every question is about those seasons. That limits what it can show: true career totals are not tested, and paging is tested only up to six pages.

In CI the paid runs sign in without a stored API key. GitHub issues each job a short-lived signed token, and the API exchanges it for a short-lived access token under a rule that accepts only this repository's protected environment (workload identity federation).

Scores will be published here once the first baseline has been measured.

```bash
uv run python -m evals preflight   # free: are the cases and grader sound?
uv run python -m evals selftest    # free: does the harness score known agents correctly?
uv run python -m evals sign-in     # free: are the API credentials accepted, and the models offered?
uv run python -m evals run --model claude-haiku-5-5 --reps 1 --sample 10 --out evals/runs/pilot   # a 10-case pilot; costs money
```

## Run it locally

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra evals
```

```bash
uv run pytest
```

Start the server on the fixture database:

```bash
ATD_DB_PATH=tests/fixtures/seasons-2007-2019.db uv run python -m alltouchdowns_mcp
```

Connect Claude Code to it:

```bash
claude mcp add --transport http alltouchdowns-local http://127.0.0.1:8000/mcp
```

## Layout

```
src/alltouchdowns_mcp/
  server.py        the six tools as the client model sees them
  tools/           the queries behind them
  guard.py         every tool result passes through here: attribution, no echoed input, no banned terms
  envelope.py      the same guarantee for everything else the endpoint can say (protocol errors)
  export/          builds the public database from the website's database
  legal.py         the banned-term filter
  auth.py limits.py   token verification and rate limiting
  logs.py          what is logged, and what never is
evals/             cases, ground truth, grader, runner, report
tests/             unit tests, and a two-season fixture database
deploy/            container configuration
```

## Data and attribution

Play-by-play data from nflverse (https://github.com/nflverse), licensed CC BY 4.0. Processed and reformatted. Not affiliated with, endorsed by, or sponsored by the NFL or any NFL team.

What "processed" means here: play descriptions are removed, rows are limited to finished games, team abbreviations are mapped to city names, and touchdown types are relabelled. Team locations and abbreviations are used for identification only. See [NOTICE](NOTICE).

### Known gaps

- The data starts with the 1999 season.
- Three games from 1999 and 2000 have no touchdown records in the source although their scores show touchdowns were scored. They are flagged as incomplete.
- One touchdown from 2005 is credited to a scorer who could not be linked to a player record.

## Privacy

The server keeps no access log and does not record IP addresses, tokens, user identifiers or anything a caller typed. Its log holds one line per tool call (the tool, whether the caller was signed in, the outcome, the duration) and a count of requests refused by each rate limit. Rate limiting counts requests per caller in memory only; those counts are lost on restart and are never written anywhere.

## License

Code: [MIT](LICENSE). Data: see above.
