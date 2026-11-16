-- The slim, public database the MCP server reads.
--
-- It holds structured facts only. Play descriptions, provider ids, surrogate ids and anything
-- from a source other than nflverse are left behind by the export.

CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
) STRICT, WITHOUT ROWID;

-- One row per abbreviation a team has used since 1999.
CREATE TABLE teams (
  abbr         TEXT PRIMARY KEY,  -- as used in that era: 'SD'
  franchise    TEXT NOT NULL,     -- the same franchise's current abbreviation: 'LAC'
  city         TEXT NOT NULL,     -- era city: 'San Diego'
  label        TEXT NOT NULL,     -- what to display: the city, plus '(ABBR)' when two
                                  -- franchises share it, e.g. 'Los Angeles (LAC)'
  first_season INTEGER NOT NULL,
  last_season  INTEGER            -- NULL = still in use
) STRICT, WITHOUT ROWID;

CREATE TABLE seasons (
  season     INTEGER PRIMARY KEY, -- named by the year it starts in
  final_week INTEGER,             -- the Championship week; NULL until that game is scheduled
  complete   INTEGER NOT NULL     -- 1 once the Championship game is final
) STRICT;

-- Only players who have scored at least one exported touchdown.
CREATE TABLE players (
  slug          TEXT PRIMARY KEY, -- stable public id: 'adrian-peterson-2'
  display_name  TEXT NOT NULL,
  search_name   TEXT NOT NULL,    -- normalised for matching: 'odell beckham'
  name_suffix   TEXT,             -- 'jr', 'iii', ... split off the search name
  position      TEXT,
  first_season  INTEGER NOT NULL, -- first and last season with a touchdown
  last_season   INTEGER NOT NULL,
  td_regular    INTEGER NOT NULL,
  td_postseason INTEGER NOT NULL,
  team_abbrs    TEXT NOT NULL     -- era abbreviations in order of first touchdown: 'SD,NYJ'
) STRICT, WITHOUT ROWID;

-- Finished games only.
CREATE TABLE games (
  game_id        TEXT PRIMARY KEY,  -- '{season}-{week:02}-{AWAY}-{HOME}'; sorts by season, week
  season         INTEGER NOT NULL,
  week           INTEGER NOT NULL,  -- keeps counting into the postseason
  postseason     INTEGER NOT NULL,  -- 0 or 1
  week_label     TEXT NOT NULL,     -- 'Week 7' or 'Championship'
  stage          TEXT NOT NULL,     -- 'Regular season', 'Playoffs' or 'Championship'
  date           TEXT NOT NULL,     -- 'YYYY-MM-DD'
  played_at      TEXT NOT NULL,     -- date and kick-off (US Eastern) as one sortable value:
                                    -- '2019-09-05T20:20'. Sorting by game_id alone would
                                    -- order a week's games by team name, not by time.
  away_abbr      TEXT NOT NULL REFERENCES teams (abbr),
  home_abbr      TEXT NOT NULL REFERENCES teams (abbr),
  away_franchise TEXT NOT NULL,
  home_franchise TEXT NOT NULL,
  away_score     INTEGER NOT NULL,
  home_score     INTEGER NOT NULL,
  touchdown_count INTEGER NOT NULL,
  -- 1 = the source has no touchdown rows for a game that must have had some.
  -- The game is real; its touchdown list is not trustworthy.
  touchdowns_incomplete INTEGER NOT NULL
) STRICT, WITHOUT ROWID;

CREATE INDEX games_season_week ON games (season, week);
CREATE INDEX games_played ON games (season, played_at, game_id);

CREATE TABLE touchdowns (
  game_id     TEXT NOT NULL REFERENCES games (game_id),
  game_seq    INTEGER NOT NULL,   -- 1..n in scoring order
  season      INTEGER NOT NULL,
  week        INTEGER NOT NULL,
  postseason  INTEGER NOT NULL,
  played_at   TEXT NOT NULL,      -- copied from the game, so touchdowns sort by time
  team_abbr   TEXT NOT NULL REFERENCES teams (abbr),  -- the scoring team, era abbreviation
  franchise   TEXT NOT NULL,
  player_slug TEXT REFERENCES players (slug),  -- NULL = scorer not linked to a player record
  scorer_name TEXT,               -- the player's full name when linked, else the name as
                                  -- written in the play-by-play; NULL = credited to the team
  td_type     TEXT NOT NULL CHECK (td_type IN (
                'receiving', 'rushing', 'int_return', 'fumble_return',
                'punt_return', 'kickoff_return', 'other')),
  quarter     INTEGER,            -- 5 and up = overtime
  clock       TEXT,               -- 'MM:SS' left in the quarter
  yards       INTEGER,
  away_score_after INTEGER,       -- score right after the touchdown, before the try;
  home_score_after INTEGER,       -- NULL in a few games whose running score was unusable
  PRIMARY KEY (game_id, game_seq)
) STRICT, WITHOUT ROWID;

CREATE INDEX touchdowns_player ON touchdowns (player_slug, game_id, game_seq);
CREATE INDEX touchdowns_played ON touchdowns (played_at, game_id, game_seq);
CREATE INDEX touchdowns_season ON touchdowns (season, td_type);
CREATE INDEX touchdowns_franchise ON touchdowns (franchise, season);
