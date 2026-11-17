"""The eval grader: exact, but not picky about things that do not matter."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from evals import grade as grading
from evals import preflight
from evals import truth as truths
from evals.schema import Case, FinalAnswer
from tests.conftest import FIXTURE_DB

FRANCHISES = {
    "sd": "LAC",
    "lac": "LAC",
    "san diego": "LAC",
    "los angeles (lac)": "LAC",
    "la": "LA",
    "stl": "LA",
    "st louis": "LA",
    "kc": "KC",
    "kansas city": "KC",
}


def case(**fields: Any) -> Case:
    base: dict[str, Any] = {
        "id": "x-01",
        "category": "x",
        "split": "dev",
        "question": "?",
        "expect": "answer",
        "answer_kind": "number",
        "truth_sql": "SELECT 1",
        "reference_calls": [{"tool": "get_leaderboard", "args": {}}],
        "notes": "test",
    }
    return Case.model_validate({**base, **fields})


def decline_case(**fields: Any) -> Case:
    base = {"expect": "decline", "decline_reason": "passer_not_credited", "truth_sql": None,
            "reference_calls": []}  # fmt: skip
    return case(**{**base, **fields})


def answer(kind: str, **fields: Any) -> FinalAnswer:
    blank = {"number": None, "items": [], "boolean": None, "not_covered_reason": None, "note": ""}
    return FinalAnswer.model_validate({"kind": kind, **blank, **fields})


def check(c: Case, a: FinalAnswer | None, truth: truths.Truth | None, **kw: Any) -> grading.Grade:
    return grading.grade(c, a, truth, franchises=FRANCHISES, **kw)


# --- Numbers and yes/no ----------------------------------------------------------------------


def test_number_must_match_exactly() -> None:
    assert check(case(), answer("number", number=13), 13).correct
    assert check(case(), answer("number", number=12), 13).failure == "wrong_value"
    assert check(case(), answer("number", number=0), 0).correct  # zero is an answer


def test_boolean() -> None:
    yes_no = case(answer_kind="boolean")
    assert check(yes_no, answer("boolean", boolean=False), False).correct
    assert check(yes_no, answer("boolean", boolean=True), False).failure == "wrong_value"


def test_the_wrong_kind_of_answer_is_malformed() -> None:
    assert check(case(), answer("list", items=["13"]), 13).failure == "malformed"
    assert check(case(), answer("number"), 13).failure == "malformed"  # kind without a value
    assert (
        check(case(answer_kind="boolean"), answer("number", number=1), True).failure == "malformed"
    )


def test_no_answer_is_never_right_not_even_for_a_decline() -> None:
    assert check(case(), None, 13).failure == "no_answer"
    assert check(decline_case(), None, None).failure == "no_answer"


# --- Sets ------------------------------------------------------------------------------------


def test_set_ignores_order_case_and_duplicates() -> None:
    players = case(answer_kind="set", item_type="player_slug")
    truth = frozenset({"randy-moss", "aaron-jones-2"})

    assert check(players, answer("list", items=["Aaron-Jones-2 ", "randy-moss"]), truth).correct
    assert check(players, answer("list", items=["randy-moss"]), truth).failure == "wrong_identity"
    extra = answer("list", items=["randy-moss", "aaron-jones-2", "tom-brady"])
    assert check(players, extra, truth).failure == "wrong_identity"


def test_a_namesake_s_slug_is_a_different_player() -> None:
    players = case(answer_kind="set", item_type="player_slug")

    wrong = check(
        players, answer("list", items=["adrian-peterson"]), frozenset({"adrian-peterson-2"})
    )
    assert wrong.failure == "wrong_identity"


def test_teams_are_compared_by_franchise() -> None:
    teams = case(answer_kind="set", item_type="team_abbr")

    assert check(teams, answer("list", items=["sd"]), frozenset({"LAC"})).correct
    assert check(teams, answer("list", items=["STL"]), frozenset({"LA"})).correct
    assert (
        check(teams, answer("list", items=["KC"]), frozenset({"LAC"})).failure == "wrong_identity"
    )


def test_game_ids_ignore_case() -> None:
    games = case(answer_kind="set", item_type="game_id")

    assert check(
        games, answer("list", items=["2019-01-ind-lac"]), frozenset({"2019-01-IND-LAC"})
    ).correct


# --- Ranked lists ----------------------------------------------------------------------------

RANKED = [frozenset({"a"}), frozenset({"b", "c", "d"}), frozenset({"e"})]  # b, c, d are tied


def ranked(items: list[str]) -> grading.Grade:
    return check(
        case(answer_kind="ordered_list", item_type="player_slug"),
        answer("list", items=items),
        RANKED,
    )


def test_ranked_list_allows_any_order_inside_a_tie() -> None:
    assert ranked(["a", "b", "c", "d", "e"]).correct
    assert ranked(["a", "d", "b", "c", "e"]).correct


def test_ranked_list_must_keep_the_order_between_ranks() -> None:
    assert ranked(["b", "a", "c", "d", "e"]).failure == "wrong_order"
    assert ranked(["a", "b", "c", "e", "d"]).failure == "wrong_order"


def test_cutting_a_ranked_list_inside_a_tie_is_its_own_failure() -> None:
    assert ranked(["a", "b"]).failure == "tie_truncation"  # "top 2", ignoring the tie
    assert ranked(["a", "b", "c"]).failure == "tie_truncation"
    assert ranked(["a"]).failure == "wrong_identity"  # just incomplete
    assert ranked(["a", "b", "c", "d", "e", "f"]).failure == "wrong_identity"
    assert ranked(["a", "a", "b", "c", "d", "e"]).failure == "wrong_identity"  # a repeat


# --- Declines --------------------------------------------------------------------------------


def test_declining_is_right_only_when_the_case_calls_for_it() -> None:
    declined = answer("not_covered", not_covered_reason="passer_not_credited")

    assert check(decline_case(), declined, None).correct
    assert check(case(), declined, 13).failure == "false_decline"


def test_answering_a_decline_case_is_wrong_and_the_trap_is_noticed() -> None:
    trap_case = decline_case(trap_sql="SELECT 1")

    assert check(trap_case, answer("number", number=1), None, trap=1).failure == "fell_for_trap"
    assert check(trap_case, answer("number", number=30), None, trap=1).failure == "false_answer"
    assert check(decline_case(), answer("list", items=["x"]), None).failure == "false_answer"


def test_any_decline_reason_counts() -> None:
    assert check(decline_case(), answer("not_covered", not_covered_reason="other"), None).correct


# --- Grounding and tool use ------------------------------------------------------------------


def test_a_list_is_grounded_when_every_item_came_from_a_tool() -> None:
    players = case(answer_kind="set", item_type="player_slug")
    results = ['{"rows":[{"player":{"slug":"randy-moss"}},{"player":{"slug":"aaron-jones-2"}}]}']

    assert grading.grounded(players, answer("list", items=["Randy-Moss"]), results, FRANCHISES)
    assert not grading.grounded(players, answer("list", items=["tom-brady"]), results, FRANCHISES)
    assert not grading.grounded(players, answer("list", items=["randy-moss"]), [], FRANCHISES)
    # The whole slug must have been shown. A namesake's numbered slug does not show the plain
    # one, which is another player; and a longer slug is not shown by a shorter one.
    assert grading.grounded(players, answer("list", items=["aaron-jones-2"]), results, FRANCHISES)
    assert not grading.grounded(players, answer("list", items=["aaron-jones"]), results, FRANCHISES)
    assert not grading.grounded(players, answer("list", items=["moss"]), results, FRANCHISES)


def test_a_number_is_grounded_by_any_successful_tool_call() -> None:
    assert grading.grounded(
        case(), answer("number", number=3), ['{"matched":{"total":3}}'], FRANCHISES
    )
    assert not grading.grounded(
        case(), answer("number", number=3), [], FRANCHISES
    )  # answered from memory


def test_a_decline_needs_no_tool_call_but_silence_is_never_grounded() -> None:
    assert grading.grounded(decline_case(), answer("not_covered"), [], FRANCHISES)
    assert not grading.grounded(case(), None, ["anything"], FRANCHISES)


def test_a_list_chosen_by_an_attribute_needs_that_attribute_from_a_tool() -> None:
    """Asked which Miami rushing scorers were quarterbacks, three models listed the scorers
    with a tool that shows no position and picked the two quarterbacks from memory."""
    quarterbacks = case(answer_kind="set", item_type="player_slug", evidence_field="position")
    both = answer("list", items=["cleo-lemon", "john-beck"])
    scorers_only = [
        '{"touchdowns":[{"scorer":{"slug":"cleo-lemon","name":"Cleo Lemon"}},'
        '{"scorer":{"slug":"john-beck","name":"John Beck"}}]}'
    ]
    leaderboard = [
        '{"rows":[{"rank":1,"player":{"slug":"cleo-lemon"},"position":"QB"},'
        '{"rank":5,"player":{"slug":"john-beck"},"position":"QB"}]}'
    ]
    one_looked_up = [*scorers_only, '{"matches":[{"slug":"cleo-lemon","position":"QB"}]}']
    position_unknown = ['{"matches":[{"slug":"cleo-lemon","position":null}]}']

    assert not grading.grounded(quarterbacks, both, scorers_only, FRANCHISES)
    assert grading.grounded(quarterbacks, both, leaderboard, FRANCHISES)
    assert not grading.grounded(quarterbacks, both, one_looked_up, FRANCHISES)
    assert grading.grounded(
        quarterbacks, answer("list", items=["Cleo-Lemon"]), one_looked_up, FRANCHISES
    )
    assert not grading.grounded(
        quarterbacks, answer("list", items=["cleo-lemon"]), position_unknown, FRANCHISES
    )
    # Without the field named, appearing in a result is enough, as before.
    plain = case(answer_kind="set", item_type="player_slug")
    assert grading.grounded(plain, both, scorers_only, FRANCHISES)


def test_evidence_field_is_only_for_answers_that_list_players() -> None:
    with pytest.raises(ValueError, match="evidence_field"):
        case(answer_kind="set", item_type="game_id", evidence_field="position")
    with pytest.raises(ValueError, match="evidence_field"):
        decline_case(evidence_field="position")


def test_expected_tools_are_alternatives_within_a_group() -> None:
    c = case(expected_tools=[["find_players"], ["get_player", "search_touchdowns"]])

    assert grading.used_expected_tools(c, ["find_players", "search_touchdowns"])
    assert not grading.used_expected_tools(c, ["search_touchdowns"])
    assert grading.used_expected_tools(case(), [])  # nothing expected, nothing required


# --- Truth SQL -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM games WHERE stage = 'Championship'",
        "SELECT STAGE FROM games",  # SQLite does not care about case, so neither may the rule
        "SELECT td_regular FROM players WHERE slug = 'randy-moss'",
        "SELECT t.Label FROM teams AS t",
        "SELECT COUNT(*) FROM seasons",
        "SELECT * FROM games LIMIT 1",
        "SELECT COUNT(*) FROM (SELECT * FROM players) WHERE TD_REGULAR > 0",
    ],
)
def test_truth_sql_cannot_read_what_the_export_derives(sql: str) -> None:
    conn = truths.open_database(FIXTURE_DB)
    try:
        with pytest.raises(sqlite3.DatabaseError, match=r"prohibited|not authorized"):
            truths.run_independent(conn, sql)
        assert conn.execute(sql).fetchall()  # a valid query; only the rule stood in its way
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("sql", "problem"),
    [("DELETE FROM touchdowns", "single SELECT"), ("SELECT 1; SELECT 2", "one statement")],
)
def test_truth_sql_must_be_one_query(sql: str, problem: str) -> None:
    assert any(problem in p for p in truths.sql_problems(sql))


def test_an_answer_naming_one_franchise_twice_is_refused_as_ungradeable() -> None:
    """SD and LAC are one team to the grader, so an answer listing both cannot be graded."""
    conn = truths.open_database(FIXTURE_DB)
    try:
        both_eras = case(answer_kind="set", item_type="team_abbr", reference_path="rows[*].team",
                         truth_sql="SELECT 'SD' UNION SELECT 'LAC' UNION SELECT 'KC'")  # fmt: skip
        distinct = both_eras.model_copy(update={"truth_sql": "SELECT 'SD' UNION SELECT 'KC'"})

        assert any("LAC more than once" in p for p in preflight.validate([both_eras], conn))
        assert preflight.validate([distinct], conn) == []
    finally:
        conn.close()


def test_ordinary_truth_sql_is_accepted() -> None:
    sql = """WITH c AS (SELECT player_slug, COUNT(*) n FROM touchdowns
             WHERE season = 2007 AND postseason = 0 GROUP BY player_slug)
             SELECT player_slug, n FROM c ORDER BY n DESC LIMIT 3;"""
    assert truths.sql_problems(sql) == []


def test_truth_shapes() -> None:
    conn = truths.open_database(FIXTURE_DB)
    try:
        count = case(truth_sql="SELECT COUNT(*) FROM games WHERE season = 2007")
        assert truths.compute(conn, count) == 267

        yes_no = case(answer_kind="boolean", truth_sql="SELECT EXISTS (SELECT 1 FROM games)")
        assert truths.compute(conn, yes_no) is True

        teams = case(
            answer_kind="set",
            item_type="team_abbr",
            truth_sql="SELECT DISTINCT franchise FROM teams WHERE city = 'Los Angeles'",
        )
        assert truths.compute(conn, teams) == frozenset({"LA", "LAC"})

        leaders = case(answer_kind="ordered_list", item_type="player_slug", truth_sql="""
            SELECT player_slug, COUNT(*) n FROM touchdowns
            WHERE season = 2007 AND postseason = 0 AND player_slug IS NOT NULL
            GROUP BY player_slug HAVING n >= 15 ORDER BY n DESC, player_slug""")  # fmt: skip
        assert truths.compute(conn, leaders) == [
            frozenset({"randy-moss"}),
            frozenset({"ladainian-tomlinson"}),
            frozenset({"braylon-edwards"}),
            frozenset({"joseph-addai", "terrell-owens"}),  # tied on 15
        ]

        with pytest.raises(ValueError, match="one integer"):
            truths.compute(conn, case(truth_sql="SELECT 1, 2"))
        unsorted = case(
            answer_kind="ordered_list",
            item_type="player_slug",
            truth_sql="SELECT * FROM (VALUES ('a', 1), ('b', 2), ('c', 1))",
        )
        with pytest.raises(ValueError, match="adjacent"):
            truths.compute(conn, unsorted)
    finally:
        conn.close()


def test_a_team_may_be_named_by_abbreviation_city_or_label() -> None:
    teams = case(answer_kind="set", item_type="team_abbr")
    truth = frozenset({"LAC"})

    for spelling in ("LAC", "sd", "San Diego", "Los Angeles (LAC)"):
        assert check(teams, answer("list", items=[spelling]), truth).correct, spelling
    assert not check(teams, answer("list", items=["Los Angeles"]), truth).correct  # which one?


def test_a_team_is_grounded_under_any_of_its_names() -> None:
    teams = case(answer_kind="set", item_type="team_abbr")
    results = ['{"team":{"abbr":"SD","name":"San Diego"}}']

    assert grading.grounded(teams, answer("list", items=["LAC"]), results, FRANCHISES)
    assert not grading.grounded(teams, answer("list", items=["KC"]), results, FRANCHISES)
    # "LA" must not be found inside another word such as "Atlanta".
    atlanta = ['{"team":{"abbr":"ATL","name":"Atlanta"}}']
    assert not grading.grounded(teams, answer("list", items=["LA"]), atlanta, FRANCHISES)


def test_an_abbreviation_is_not_grounded_by_an_ordinary_word() -> None:
    """Folded, "NO", "WAS" and "TEN" are words. Only the capitals the tools write count."""
    teams = case(answer_kind="set", item_type="team_abbr")
    franchises = {"no": "NO", "new orleans": "NO", "was": "WAS", "washington": "WAS"}
    words_only = ['{"matches":[],"notes":["No touchdown scorer matches. It was not found."]}']
    by_abbreviation = ['{"game_id":"2019-17-NO-CAR","notes":["No game is incomplete."]}']
    by_city = ['{"team":{"abbr":"SEA","name":"Seattle"},"opponent":{"name":"Washington"}}']

    assert not grading.grounded(teams, answer("list", items=["NO"]), words_only, franchises)
    assert not grading.grounded(teams, answer("list", items=["WAS"]), words_only, franchises)
    assert grading.grounded(teams, answer("list", items=["no"]), by_abbreviation, franchises)
    assert grading.grounded(teams, answer("list", items=["WAS"]), by_city, franchises)


def test_touchdown_types_ignore_spacing_and_hyphens() -> None:
    types = case(answer_kind="set", item_type="td_type")
    truth = frozenset({"fumble_return", "int_return"})
    results = ['{"td_type":"fumble_return"},{"td_type":"int_return"}']
    given = answer("list", items=["Fumble return", "interception-return"])

    assert check(types, given, truth).correct
    assert grading.grounded(types, given, results, FRANCHISES)


def test_an_answer_shown_to_the_agent_before_any_call_is_a_leak() -> None:
    """An example slug in a tool description once gave away three cases."""
    players = case(answer_kind="set", item_type="player_slug")
    shown = "Player slug from find_players, e.g. 'adrian-peterson-2'."

    leaks = preflight._leaked(players, frozenset({"adrian-peterson-2", "randy-moss"}), shown)

    assert len(leaks) == 1 and "adrian-peterson-2" in leaks[0]
    assert preflight._leaked(players, [frozenset({"randy-moss"})], shown) == []
    assert preflight._leaked(case(), 2, "the number 2 appears here") == []  # numbers: not checked


def test_an_item_with_a_comment_attached_is_graded_on_its_identifier() -> None:
    """A small model in the pilot answered "torry-holt (WR, 7 TD)". The player is right; the
    form is not, and the two are reported separately."""
    players = case(answer_kind="ordered_list", item_type="player_slug")
    truth = [frozenset({"torry-holt"}), frozenset({"steven-jackson"})]
    results = ['{"rows":[{"player":{"slug":"torry-holt"}},{"player":{"slug":"steven-jackson"}}]}']
    commented = answer("list", items=["torry-holt (WR, 7 TD)", "steven-jackson (RB, 6 TD)"])
    bare = answer("list", items=["torry-holt", "steven-jackson"])

    assert check(players, commented, truth).correct
    assert grading.grounded(players, commented, results, FRANCHISES)
    assert grading.decorated(players, commented) and not grading.decorated(players, bare)
    # The comment cannot rescue a wrong identifier, or smuggle in a second one.
    assert not check(
        players, answer("list", items=["torry-holt-2 (WR)", "steven-jackson"]), truth
    ).correct
    assert not check(players, answer("list", items=["torry-holt steven-jackson"]), truth).correct

    games = case(answer_kind="set", item_type="game_id")
    dated = answer("list", items=["2019-08-car-sf (Week 8, vs Carolina)"])
    assert check(games, dated, frozenset({"2019-08-CAR-SF"})).correct
    assert grading.decorated(games, dated)


def test_a_team_named_by_its_label_is_not_called_decorated() -> None:
    teams = case(answer_kind="set", item_type="team_abbr")

    assert not grading.decorated(teams, answer("list", items=["Los Angeles (LAC)", "Kansas City"]))


def test_a_decline_is_checked_for_its_reason_without_being_graded_on_it() -> None:
    declined = decline_case()  # the right reason is passer_not_credited
    for_the_right_reason = answer("not_covered", not_covered_reason="passer_not_credited")
    for_another = answer("not_covered", not_covered_reason="other")

    assert check(declined, for_another, None).correct  # any decline passes
    assert grading.decline_reason_matches(declined, for_the_right_reason) is True
    assert grading.decline_reason_matches(declined, for_another) is False
    assert grading.decline_reason_matches(declined, answer("number", number=3)) is None
    assert grading.decline_reason_matches(case(), answer("number", number=3)) is None


def test_a_decline_case_can_accept_more_than_one_reason() -> None:
    """A question about another league in a covered year: the labels reach the agent with no
    explanation, and four of six trials called it "other" with the true reason in the note."""
    other_league = decline_case(decline_reason="outside_seasons", decline_reason_also=["other"])

    for reason in ("outside_seasons", "other"):
        given = answer("not_covered", not_covered_reason=reason)
        assert grading.decline_reason_matches(other_league, given) is True
    wrong_label = answer("not_covered", not_covered_reason="bulk_export")
    assert grading.decline_reason_matches(other_league, wrong_label) is False
    with pytest.raises(ValueError, match="decline"):
        case(decline_reason_also=["other"])  # a case that expects an answer has no reasons


def test_exact_repeats_of_a_call_are_counted() -> None:
    same = ("get_player", {"slug": "chester-taylor"})
    calls = [("find_players", {"query": "Chester Taylor"}), same, same, same,
             ("get_player", {"slug": "someone-else"})]  # fmt: skip

    assert grading.repeated_calls(calls) == 2
    assert grading.repeated_calls([]) == 0
    reordered = [("search", {"season": 2019, "week": 2}), ("search", {"week": 2, "season": 2019})]
    assert grading.repeated_calls(reordered) == 1  # the same call, arguments in another order
