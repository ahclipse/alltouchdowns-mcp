"""Worked solutions for the cases whose answer no single field of a result holds.

Most cases prove they can be answered with a `reference_path`: the scripted calls are run and
one field of the last result is the answer. Others cannot do that. Their answer is a count of
rows, a difference of two counts, a tally by team, or the rows that two results share. For
those, the proof is one of the functions below. It gets the structured results of the case's
reference calls, in order, and works the answer out from them alone, the way an agent has to.
The free pre-flight grades what it returns against the case's truth SQL.

A function returns what its case answers with: a number, a yes/no, a set of identifiers, or
a list in rank order.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

Result = Mapping[str, Any]
Solution = Callable[[Sequence[Result]], int | bool | Collection[str]]


def _touchdowns(results: Sequence[Result]) -> list[dict[str, Any]]:
    """Every touchdown row in these results, e.g. the pages of one search."""
    return [row for result in results for row in result.get("touchdowns", [])]


def _most(tally: Mapping[str, int]) -> set[str]:
    """Everything level on the highest count."""
    return {item for item, count in tally.items() if count == max(tally.values())}


# --- One number from two results --------------------------------------------------------------


def both_namesakes(results: Sequence[Result]) -> int:
    """After the name search: one regular-season count per namesake, added."""
    return sum(result["matched"]["regular_season"] for result in results[1:])


def given_up_in_a_season(results: Sequence[Result]) -> int:
    """Every touchdown in the team's games, less the ones the team scored itself."""
    games, own = results
    return sum(game["touchdown_count"] for game in games["games"]) - own["matched"]["total"]


def exactly_one_yard(results: Sequence[Result]) -> int:
    """Touchdowns of at least 1 yard, less those of at least 2."""
    at_least_one, at_least_two = results
    return at_least_one["matched"]["total"] - at_least_two["matched"]["total"]


# --- Rows filtered by something only the row shows -------------------------------------------


def scored_against(opponent: str) -> Solution:
    """A search has no opponent filter, but every row names the opponent."""

    def count(results: Sequence[Result]) -> int:
        return sum(row["opponent"]["abbr"] == opponent for row in results[-1]["touchdowns"])

    return count


def scored_as_the_visitor(results: Sequence[Result]) -> int:
    """A touchdown row does not say who was at home; the team's list of games does."""
    _, scored, games = results
    visitor = {game["game_id"]: game["away"]["abbr"] for game in games["games"]}
    return sum(row["team"]["abbr"] == visitor[row["game_id"]] for row in scored["touchdowns"])


def played_in_january_2020(results: Sequence[Result]) -> int:
    (playoffs,) = results
    return sum(
        game["touchdown_count"] for game in playoffs["games"] if game["date"].startswith("2020-01")
    )


def combined_length(results: Sequence[Result]) -> int:
    return sum(row["yards"] for row in results[-1]["touchdowns"])


def on_the_leaderboard_or_zero(results: Sequence[Result]) -> int:
    """The name search finds the player's slug, or finds nobody. Either way his count is what
    the team's leaderboard shows for that slug, and a player who is not on it scored none."""
    found, leaderboard = results
    slugs = {match["slug"] for match in found["matches"] if match["match"] == "exact"}
    return sum(
        row["touchdowns"]["regular_season"]
        for row in leaderboard["rows"]
        if row["player"]["slug"] in slugs
    )


def touchdowns_in_quarter(quarter: int) -> Solution:
    def count(results: Sequence[Result]) -> int:
        return sum(row["quarter"] == quarter for row in results[-1]["touchdowns"])

    return count


def more_in_2019_than_in_2007(results: Sequence[Result]) -> bool:
    """The player summary lists each season he scored in, with its regular-season count."""
    by_season = {
        season["season"]: season["touchdowns"]["regular_season"]
        for season in results[-1]["by_season"]
    }
    return by_season.get(2019, 0) > by_season.get(2007, 0)


def someone_ran_and_caught(results: Sequence[Result]) -> bool:
    kinds: dict[str, set[str]] = defaultdict(set)
    for row in results[-1]["touchdowns"]:
        kinds[row["scorer"]["slug"]].add(row["td_type"])
    return any({"rushing", "receiving"} <= scored for scored in kinds.values())


def quarterbacks_on_the_leaderboard(results: Sequence[Result]) -> set[str]:
    return {row["player"]["slug"] for row in results[-1]["rows"] if row["position"] == "QB"}


def quarterbacks_who_ran_in_both(results: Sequence[Result]) -> set[str]:
    """The playoff rushing leaderboard shows each scorer's position and both of his counts."""
    (leaderboard,) = results
    return {
        row["player"]["slug"]
        for row in leaderboard["rows"]
        if row["position"] == "QB"
        and row["touchdowns"]["regular_season"] > 0
        and row["touchdowns"]["postseason"] > 0
    }


# --- Games, where the tools count touchdowns --------------------------------------------------


def games_scored_in(results: Sequence[Result]) -> set[str]:
    """After the name search: the games of every row, each game once."""
    return {row["game_id"] for row in _touchdowns(results[1:])}


def number_of_games_scored_in(results: Sequence[Result]) -> int:
    """`matched` counts touchdowns. Two in one game are one game."""
    return len(games_scored_in(results))


def most_games_scored_in(results: Sequence[Result]) -> set[str]:
    games: dict[str, set[str]] = defaultdict(set)
    for row in _touchdowns(results):
        games[row["scorer"]["slug"]].add(row["game_id"])
    return _most({slug: len(scored_in) for slug, scored_in in games.items()})


def games_without_a_touchdown_of_its_own(results: Sequence[Result]) -> set[str]:
    """The team's games, less the games in which one of its touchdowns appears."""
    played = {game["game_id"] for game in results[0]["games"]}
    return played - {row["game_id"] for row in _touchdowns(results[1:])}


def teams_that_did_not_play(results: Sequence[Result]) -> set[str]:
    """Every team of a full week, less the teams of the week in question."""
    full_week, week = (
        {game[side]["abbr"] for game in result["games"] for side in ("away", "home")}
        for result in results
    )
    return full_week - week


def away_games_of_seattle_with_eight_or_more(results: Sequence[Result]) -> set[str]:
    (season,) = results
    return {
        game["game_id"]
        for game in season["games"]
        if game["away"]["abbr"] == "SEA" and game["touchdown_count"] >= 8
    }


def games_with_one_touchdown(results: Sequence[Result]) -> set[str]:
    """Every page of the season's games, keeping those with a count of exactly 1."""
    return {
        game["game_id"]
        for page in results
        for game in page["games"]
        if game["touchdown_count"] == 1
    }


def namesakes_who_scored(results: Sequence[Result]) -> set[str]:
    """After the name search: one search per namesake, keeping those that matched anything."""
    return {
        result["filters"]["player"]["slug"] for result in results[1:] if result["matched"]["total"]
    }


def opponents_scored_against(results: Sequence[Result]) -> set[str]:
    return {row["opponent"]["abbr"] for row in _touchdowns(results[1:])}


def scored_more_than_once(results: Sequence[Result]) -> set[str]:
    tally = Counter(row["scorer"]["slug"] for row in _touchdowns(results))
    return {slug for slug, count in tally.items() if count > 1 and slug is not None}


def scored_in_both_games(results: Sequence[Result]) -> set[str]:
    first, second = ({row["scorer"]["slug"] for row in game["touchdowns"]} for game in results[1:])
    return first & second


# --- Ties and rankings the tools do not make --------------------------------------------------


def teams_with_the_most(results: Sequence[Result]) -> set[str]:
    return _most(Counter(row["team"]["abbr"] for row in _touchdowns(results)))


def team_with_the_larger_count(results: Sequence[Result]) -> set[str]:
    """One search per team. Each result says which team it understood and how many matched."""
    return _most({r["filters"]["team"]["franchise"]: r["matched"]["total"] for r in results})


def games_with_the_most_touchdowns(results: Sequence[Result]) -> set[str]:
    games = [game for page in results for game in page["games"]]
    return _most({game["game_id"]: game["touchdown_count"] for game in games})


def players_tied_on_the_rows(results: Sequence[Result]) -> int:
    """A leaderboard asked for one rank returns every player tied for it: count them."""
    return len({row["player"]["slug"] for row in results[-1]["rows"]})


def games_of_the_longest(results: Sequence[Result]) -> set[str]:
    rows = _touchdowns(results)
    longest = max(row["yards"] for row in rows)
    return {row["game_id"] for row in rows if row["yards"] == longest}


def four_longest_with_ties(results: Sequence[Result]) -> list[str]:
    """Rows arrive longest first. Keep everything at least as long as the fourth."""
    rows = _touchdowns(results)
    fourth = rows[3]["yards"]
    return [row["scorer"]["slug"] for row in rows if row["yards"] >= fourth]


def top_two_places_across_two_types(results: Sequence[Result]) -> list[str]:
    """One leaderboard per type, added up per player, then cut after second place."""
    total: Counter[str] = Counter()
    for leaderboard in results:
        for row in leaderboard["rows"]:
            total[row["player"]["slug"]] += row["touchdowns"]["regular_season"]
    second_best = sorted(set(total.values()), reverse=True)[1]
    return [slug for slug, count in total.most_common() if count >= second_best]


SOLUTIONS: dict[str, Solution] = {
    "allowed-and-absent-01": given_up_in_a_season,
    "allowed-and-absent-02": scored_against("KC"),
    "allowed-and-absent-03": games_without_a_touchdown_of_its_own,
    "allowed-and-absent-04": teams_that_did_not_play,
    "ambiguous-cities-03": team_with_the_larger_count,
    "answerable-lookalikes-03": combined_length,
    "answerable-lookalikes-05": played_in_january_2020,
    "answerable-lookalikes-06": on_the_leaderboard_or_zero,
    "game-detail-04": touchdowns_in_quarter(2),
    "game-detail-07": scored_more_than_once,
    "game-lists-01": away_games_of_seattle_with_eight_or_more,
    "game-lists-03": games_with_the_most_touchdowns,
    "games-scored-in-01": number_of_games_scored_in,
    "games-scored-in-02": number_of_games_scored_in,
    "games-scored-in-03": games_scored_in,
    "games-scored-in-04": most_games_scored_in,
    "games-scored-in-05": scored_as_the_visitor,
    "leaderboard-ties-07": players_tied_on_the_rows,
    "namesakes-01": scored_against("MIN"),
    "namesakes-02": namesakes_who_scored,
    "namesakes-06": namesakes_who_scored,
    "namesakes-07": scored_against("GB"),
    "namesakes-08": both_namesakes,
    "overtime-quarters-03": touchdowns_in_quarter(4),
    "quarterback-scoring-04": quarterbacks_on_the_leaderboard,
    "quarterback-scoring-05": quarterbacks_who_ran_in_both,
    "tally-across-rows-01": four_longest_with_ties,
    "tally-across-rows-02": teams_with_the_most,
    "tally-across-rows-03": top_two_places_across_two_types,
    "tally-across-rows-04": games_of_the_longest,
    "tally-across-rows-05": exactly_one_yard,
    "tally-across-rows-06": scored_in_both_games,
    "two-season-totals-01": opponents_scored_against,
    "two-season-totals-02": more_in_2019_than_in_2007,
    "two-season-totals-04": games_scored_in,
    "yardage-paging-05": scored_more_than_once,
    "yardage-paging-06": games_with_one_touchdown,
    "zero-and-empty-06": someone_ran_and_caught,
}
