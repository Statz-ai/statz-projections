"""
Repo for the two dataset tables that back the retraining pipeline:

  - projection_model_dataset: per-(fixture, team) training rows with
    actuals + history features. Written on every projection run via
    the dual-write wrapper below.
  - projection_accuracy_dataset: per-fixture projections vs actuals
    used to evaluate model accuracy over time.

Both tables have contamination risk: the existing parquet files that
feed these DataFrames have historical rows pooled in from other leagues
(comp_id labelled incorrectly). The transforms below filter by the
authoritative fixture.competition_id to keep only true-owned rows.

Invoked from projection_service / projection_all_teams_service etc.
alongside the existing `_write_df` parquet writes (dual-write phase
of the data-files-to-DB migration).
"""
import asyncio
import logging
import math

import pandas as pd

import app.database as _db
from app.database import get_connection
from app.repository.db_utils import execute_chunked

logger = logging.getLogger("projection_dataset_repo")


MODEL_DATASET_STATS = [
    "shots_total", "shots_on_target", "corners", "fouls", "yellowcards",
    "tackles", "passes", "successful_passes", "interceptions",
    "total_crosses", "offsides",
    # FPL bonus (BPS) inputs. PROJECTED for the PL only (get_stat_list(comp_id)),
    # but the columns are written for every league because the models train on
    # the full top-5 pool. For non-PL leagues the history features are simply
    # absent from the df and land as NULL; the actuals are filled by Laravel's
    # BackfillFixtureAccuracy::MODEL_ONLY_STAT_MAP regardless.
    "key_passes", "big_chances_created",
]

ACCURACY_DATASET_STATS = [
    "goals", "shots_total", "shots_on_target", "corners", "fouls",
    "yellowcards", "tackles", "passes", "successful_passes",
    "total_crosses", "interceptions", "offsides",
]


# ──────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────

def _val(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def _parse_percent(v):
    """'35.62%' → 35.62. NaN/None → None."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str):
        s = v.replace("%", "").strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return float(v)


def _parse_bool(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if v else 0
    if isinstance(v, str):
        # Outcome columns in the legacy parquet + in projection_service.py
        # gap-fill (line 326+) are stored as 'Y'/'N' strings, not booleans.
        # Prior version fell through to `return None` for all strings,
        # leaving every outcome flag NULL in the DB. Admin panel Accuracy
        # tab surfaced this 2026-04-23 (Real % + Brier all showing '—').
        s = v.strip().upper()
        if s in ('Y', 'YES', 'TRUE', '1'):
            return 1
        if s in ('N', 'NO', 'FALSE', '0'):
            return 0
    return None


def _resolve_scoped(name, teams, scoped_team_ids):
    """Prefer a team_id within the comp's registered pool; fall back to
    first global match. None if no match at all."""
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return None
    if len(scoped_team_ids) > 0:
        scoped = teams.loc[
            (teams["id"].isin(scoped_team_ids)) & (teams["name"] == name), "id"
        ]
        if not scoped.empty:
            return int(scoped.iloc[0])
    m = teams.loc[teams["name"] == name, "id"]
    if m.empty:
        return None
    return int(m.iloc[0])


def _title(stat_snake):
    return " ".join(w.capitalize() for w in stat_snake.split("_"))


def _title_parquet_col(stat_snake, history=False, opponent=False):
    t = _title(stat_snake)
    if history:
        return f"Opponent {t} History Against" if opponent else f"Team {t} History"
    return f"Team {t}"


# ──────────────────────────────────────────────────────────────────────────
#  projection_model_dataset
# ──────────────────────────────────────────────────────────────────────────

# 9 identity columns + 1 actual + 2 history per stat.
_MODEL_PLACEHOLDER_COUNT = 9 + 3 * len(MODEL_DATASET_STATS)
_MODEL_PLACEHOLDERS = ", ".join(["%s"] * _MODEL_PLACEHOLDER_COUNT)

MODEL_DATASET_SQL = """
INSERT INTO projection_model_dataset (
    fixture_id, competition_id, season_id, team_id, opponent_id,
    team_name, opponent_name, venue, kickoff_datetime,
    team_shots_total, team_shots_on_target, team_corners, team_fouls,
    team_yellowcards, team_tackles, team_passes, team_successful_passes,
    team_interceptions, team_total_crosses, team_offsides,
    team_key_passes, team_big_chances_created,
    team_shots_total_history, opponent_shots_total_history_against,
    team_shots_on_target_history, opponent_shots_on_target_history_against,
    team_corners_history, opponent_corners_history_against,
    team_fouls_history, opponent_fouls_history_against,
    team_yellowcards_history, opponent_yellowcards_history_against,
    team_tackles_history, opponent_tackles_history_against,
    team_passes_history, opponent_passes_history_against,
    team_successful_passes_history, opponent_successful_passes_history_against,
    team_interceptions_history, opponent_interceptions_history_against,
    team_total_crosses_history, opponent_total_crosses_history_against,
    team_offsides_history, opponent_offsides_history_against,
    team_key_passes_history, opponent_key_passes_history_against,
    team_big_chances_created_history, opponent_big_chances_created_history_against,
    created_at, updated_at
) VALUES (
    -- identity (9), then one actual per stat, then two history columns per
    -- stat — generated from MODEL_DATASET_STATS so the count cannot drift
    -- from the column list above when a stat is added.
    {model_placeholders},
    NOW(), NOW()
) ON DUPLICATE KEY UPDATE
    competition_id = VALUES(competition_id),
    season_id = VALUES(season_id),
    opponent_id = VALUES(opponent_id),
    team_name = VALUES(team_name),
    opponent_name = VALUES(opponent_name),
    venue = VALUES(venue),
    kickoff_datetime = VALUES(kickoff_datetime),
    -- team_{stat} columns intentionally omitted from UPDATE: those are
    -- exclusively owned by Laravel BackfillFixtureAccuracy. Projection
    -- only writes them on initial INSERT (typically NULL until the
    -- fixture is played + backfilled). See architecture note 2026-05-22.
    team_shots_total_history = VALUES(team_shots_total_history),
    opponent_shots_total_history_against = VALUES(opponent_shots_total_history_against),
    team_shots_on_target_history = VALUES(team_shots_on_target_history),
    opponent_shots_on_target_history_against = VALUES(opponent_shots_on_target_history_against),
    team_corners_history = VALUES(team_corners_history),
    opponent_corners_history_against = VALUES(opponent_corners_history_against),
    team_fouls_history = VALUES(team_fouls_history),
    opponent_fouls_history_against = VALUES(opponent_fouls_history_against),
    team_yellowcards_history = VALUES(team_yellowcards_history),
    opponent_yellowcards_history_against = VALUES(opponent_yellowcards_history_against),
    team_tackles_history = VALUES(team_tackles_history),
    opponent_tackles_history_against = VALUES(opponent_tackles_history_against),
    team_passes_history = VALUES(team_passes_history),
    opponent_passes_history_against = VALUES(opponent_passes_history_against),
    team_successful_passes_history = VALUES(team_successful_passes_history),
    opponent_successful_passes_history_against = VALUES(opponent_successful_passes_history_against),
    team_interceptions_history = VALUES(team_interceptions_history),
    opponent_interceptions_history_against = VALUES(opponent_interceptions_history_against),
    team_total_crosses_history = VALUES(team_total_crosses_history),
    opponent_total_crosses_history_against = VALUES(opponent_total_crosses_history_against),
    team_offsides_history = VALUES(team_offsides_history),
    opponent_offsides_history_against = VALUES(opponent_offsides_history_against),
    team_key_passes_history = VALUES(team_key_passes_history),
    opponent_key_passes_history_against = VALUES(opponent_key_passes_history_against),
    team_big_chances_created_history = VALUES(team_big_chances_created_history),
    opponent_big_chances_created_history_against = VALUES(opponent_big_chances_created_history_against),
    updated_at = NOW()
""".replace("{model_placeholders}", _MODEL_PLACEHOLDERS)


def _build_model_rows(df, league_id, teams, fixtures, comp_teams):
    """Transform in-memory model_dataset DataFrame → list of INSERT tuples.

    Contamination filter: skip rows where parquet's claimed comp_id doesn't
    match the fixture's actual competition_id (fixtures DB is source of
    truth). Historical pooled-in rows get dropped here.
    """
    fix_by_id = {
        int(r["id"]): (int(r["season_id"]), int(r["competition_id"]))
        for _, r in fixtures.iterrows()
    } if fixtures is not None else {}

    scoped_ids = comp_teams.loc[
        comp_teams["competition_id"] == int(league_id), "team_id"
    ].unique() if (comp_teams is not None and not comp_teams.empty) else []

    rows = []
    dropped = {"contamination": 0, "missing_fixture": 0, "unresolved": 0}

    for _, row in df.iterrows():
        fid = int(row["id"])
        fix_info = fix_by_id.get(fid)
        if fix_info is None:
            dropped["missing_fixture"] += 1
            continue
        actual_season_id, actual_comp_id = fix_info
        claimed = int(row["comp_id"]) if pd.notna(row.get("comp_id")) else None
        if claimed is not None and claimed != actual_comp_id:
            dropped["contamination"] += 1
            continue

        team_id = _resolve_scoped(row.get("Team"), teams, scoped_ids)
        opp_id = _resolve_scoped(row.get("Opponent"), teams, scoped_ids)
        if team_id is None or opp_id is None:
            dropped["unresolved"] += 1
            continue

        tup = [
            fid, actual_comp_id, actual_season_id, team_id, opp_id,
            _val(row.get("Team")), _val(row.get("Opponent")),
            _val(row.get("Venue")), _val(row.get("kickoff_datetime")),
        ]
        for stat in MODEL_DATASET_STATS:
            tup.append(_val(row.get(_title_parquet_col(stat, history=False))))
        for stat in MODEL_DATASET_STATS:
            tup.append(_val(row.get(_title_parquet_col(stat, history=True, opponent=False))))
            tup.append(_val(row.get(_title_parquet_col(stat, history=True, opponent=True))))
        rows.append(tuple(tup))

    return rows, dropped


async def insert_model_dataset_async(df, league_id, league_name, teams, fixtures, comp_teams):
    """Write one league's model_dataset DataFrame to DB. Safe to call
    alongside the existing parquet _write_df (dual-write)."""
    if df is None or len(df) == 0:
        return
    rows, dropped = _build_model_rows(df, league_id, teams, fixtures, comp_teams)
    logger.info(
        f"[model_dataset {league_name}] df_rows={len(df)} inserting={len(rows)} "
        f"dropped={dropped}"
    )
    if rows:
        await execute_chunked(MODEL_DATASET_SQL, rows, label=f"[model_dataset {league_name}]")


# ──────────────────────────────────────────────────────────────────────────
#  projection_accuracy_dataset
# ──────────────────────────────────────────────────────────────────────────

def _build_accuracy_insert_sql():
    # Actuals + outcome flags are written exclusively by the Laravel
    # BackfillFixtureAccuracy job (event-driven from ImportFixtureStatsJobV2).
    # Projection runs MUST NOT touch them on UPSERT — otherwise the next
    # projection would overwrite real backfilled values with NULL/NaN
    # from the in-memory DataFrame that no longer carries actuals.
    # See architecture note 2026-05-22.
    actuals_cols = set()
    for stat in ACCURACY_DATASET_STATS:
        for venue in ["total", "home", "away"]:
            actuals_cols.add(f"{venue}_{stat}")
    actuals_cols.update([
        "home_win", "draw", "away_win",
        "home_clean_sheet", "away_clean_sheet",
        "over_15", "over_25", "btts",
    ])

    cols = [
        "fixture_id", "competition_id", "home_team_id", "away_team_id",
        "home_team_name", "away_team_name", "kickoff_datetime",
    ]
    for stat in ACCURACY_DATASET_STATS:
        for venue in ["total", "home", "away"]:
            cols.append(f"{venue}_{stat}")
            cols.append(f"{venue}_projected_{stat}")
    cols.extend([
        "home_odds_percent", "draw_odds_percent", "away_odds_percent",
        "home_win_percent", "draw_percent", "away_win_percent",
        "home_clean_sheet_percent", "away_clean_sheet_percent",
        "over_15_goals_percent", "over_25_goals_percent", "both_teams_score_percent",
        "home_win", "draw", "away_win",
        "home_clean_sheet", "away_clean_sheet",
        "over_15", "over_25", "btts",
    ])
    placeholders = ", ".join(["%s"] * len(cols))
    col_list = ", ".join(cols)
    # ON DUPLICATE KEY UPDATE excludes fixture_id (the unique key) AND
    # the actuals/outcome columns owned by BackfillFixtureAccuracy.
    update_list = ",\n    ".join(
        f"{c} = VALUES({c})" for c in cols
        if c != "fixture_id" and c not in actuals_cols
    )
    return f"""
INSERT INTO projection_accuracy_dataset (
    {col_list},
    created_at, updated_at
) VALUES ({placeholders}, NOW(), NOW())
ON DUPLICATE KEY UPDATE
    {update_list},
    updated_at = NOW()
"""


ACCURACY_DATASET_SQL = _build_accuracy_insert_sql()


def _build_accuracy_rows(df, league_id, teams, fixtures, comp_teams):
    fix_by_id = {
        int(r["id"]): int(r["competition_id"])
        for _, r in fixtures.iterrows()
    } if fixtures is not None else {}

    scoped_ids = comp_teams.loc[
        comp_teams["competition_id"] == int(league_id), "team_id"
    ].unique() if (comp_teams is not None and not comp_teams.empty) else []

    rows = []
    dropped = {"contamination": 0, "missing_fixture": 0, "unresolved": 0}

    for _, row in df.iterrows():
        fid = int(row["fixture_id"])
        actual_comp_id = fix_by_id.get(fid)
        if actual_comp_id is None:
            dropped["missing_fixture"] += 1
            continue
        claimed = int(row["comp_id"]) if pd.notna(row.get("comp_id")) else None
        if claimed is not None and claimed != actual_comp_id:
            dropped["contamination"] += 1
            continue

        home_id = _resolve_scoped(row.get("Home Team"), teams, scoped_ids)
        away_id = _resolve_scoped(row.get("Away Team"), teams, scoped_ids)
        if home_id is None or away_id is None:
            dropped["unresolved"] += 1
            continue

        tup = [
            fid, actual_comp_id, home_id, away_id,
            _val(row.get("Home Team")), _val(row.get("Away Team")),
            _val(row.get("kickoff_datetime")),
        ]
        for stat in ACCURACY_DATASET_STATS:
            t = _title(stat)
            for venue in ["Total", "Home", "Away"]:
                tup.append(_val(row.get(f"{venue} {t}")))
                tup.append(_val(row.get(f"{venue} Projected {t}")))
        # Odds
        tup.append(_val(row.get("Home Odds %")))
        tup.append(_val(row.get("Draw Odds %")))
        tup.append(_val(row.get("Away Odds %")))
        # Projection %s (parquet stores as '35.62%' strings)
        tup.append(_parse_percent(row.get("Home Win %")))
        tup.append(_parse_percent(row.get("Draw %")))
        tup.append(_parse_percent(row.get("Away Win %")))
        tup.append(_parse_percent(row.get("Home Clean Sheet %")))
        tup.append(_parse_percent(row.get("Away Clean Sheet %")))
        tup.append(_parse_percent(row.get("Over 1.5 Goals %")))
        tup.append(_parse_percent(row.get("Over 2.5 Goals %")))
        tup.append(_parse_percent(row.get("Both Teams Score %")))
        # Outcome flags
        tup.append(_parse_bool(row.get("Home Win")))
        tup.append(_parse_bool(row.get("Draw")))
        tup.append(_parse_bool(row.get("Away Win")))
        tup.append(_parse_bool(row.get("Home Clean Sheet")))
        tup.append(_parse_bool(row.get("Away Clean Sheet")))
        tup.append(_parse_bool(row.get("Over 1.5")))
        tup.append(_parse_bool(row.get("Over 2.5")))
        tup.append(_parse_bool(row.get("BTTS")))
        rows.append(tuple(tup))

    return rows, dropped


async def insert_accuracy_dataset_async(df, league_id, league_name, teams, fixtures, comp_teams):
    if df is None or len(df) == 0:
        return
    rows, dropped = _build_accuracy_rows(df, league_id, teams, fixtures, comp_teams)
    logger.info(
        f"[accuracy_dataset {league_name}] df_rows={len(df)} inserting={len(rows)} "
        f"dropped={dropped}"
    )
    if rows:
        await execute_chunked(ACCURACY_DATASET_SQL, rows, label=f"[accuracy_dataset {league_name}]")


# ──────────────────────────────────────────────────────────────────────────
#  Phase 3: readers — DB → DataFrame (replaces parquet reads in services)
# ──────────────────────────────────────────────────────────────────────────
#
# Column aliasing: the DataFrame shape consumed by projection_service +
# projection_all_teams_service is derived from the legacy parquet format
# ("Team", "Team Shots Total", "Team Shots Total History", etc.). We alias
# the snake_case DB columns back to that format in-SQL, so downstream code
# keeps working unchanged.
#
# Why SELECT aliasing rather than a post-fetch pandas rename: fewer moving
# parts, faster (one iteration), and the SQL doubles as an explicit
# contract for which columns the DataFrame will have.


def _build_model_dataset_select() -> str:
    cols = [
        "fixture_id AS id",
        "competition_id AS comp_id",
        "season_id",
        "team_name AS `Team`",
        "opponent_name AS `Opponent`",
        "venue AS `Venue`",
        "kickoff_datetime",
    ]
    for stat in MODEL_DATASET_STATS:
        t = _title(stat)
        cols.append(f"team_{stat} AS `Team {t}`")
    for stat in MODEL_DATASET_STATS:
        t = _title(stat)
        cols.append(f"team_{stat}_history AS `Team {t} History`")
        cols.append(f"opponent_{stat}_history_against AS `Opponent {t} History Against`")
    return "SELECT " + ", ".join(cols) + " FROM projection_model_dataset"


def _build_accuracy_dataset_select() -> str:
    cols = [
        "fixture_id",
        "competition_id AS comp_id",
        "home_team_name AS `Home Team`",
        "away_team_name AS `Away Team`",
        "kickoff_datetime",
    ]
    for stat in ACCURACY_DATASET_STATS:
        t = _title(stat)
        for venue_db, venue_parq in [("total", "Total"), ("home", "Home"), ("away", "Away")]:
            cols.append(f"{venue_db}_{stat} AS `{venue_parq} {t}`")
            cols.append(f"{venue_db}_projected_{stat} AS `{venue_parq} Projected {t}`")
    # Odds % + outcome % columns. `%%` is the escaped form for aiomysql,
    # which runs SQL through Python %-format before sending; a literal `%`
    # in the column alias would otherwise be parsed as a format specifier
    # and crash with ValueError on the character after it. The DataFrame
    # column name still ends up as e.g. "Home Odds %" at runtime.
    cols.extend([
        "home_odds_percent AS `Home Odds %%`",
        "draw_odds_percent AS `Draw Odds %%`",
        "away_odds_percent AS `Away Odds %%`",
        "home_win_percent AS `Home Win %%`",
        "draw_percent AS `Draw %%`",
        "away_win_percent AS `Away Win %%`",
        "home_clean_sheet_percent AS `Home Clean Sheet %%`",
        "away_clean_sheet_percent AS `Away Clean Sheet %%`",
        "over_15_goals_percent AS `Over 1.5 Goals %%`",
        "over_25_goals_percent AS `Over 2.5 Goals %%`",
        "both_teams_score_percent AS `Both Teams Score %%`",
        # Outcome flags
        "home_win AS `Home Win`",
        "draw AS `Draw`",
        "away_win AS `Away Win`",
        "home_clean_sheet AS `Home Clean Sheet`",
        "away_clean_sheet AS `Away Clean Sheet`",
        "over_15 AS `Over 1.5`",
        "over_25 AS `Over 2.5`",
        "btts AS `BTTS`",
    ])
    return "SELECT " + ", ".join(cols) + " FROM projection_accuracy_dataset"


async def _fetch_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute SELECT and return results as a DataFrame. SQL with `%`
    characters in column aliases must escape them as `%%` because
    aiomysql always runs the query through Python's %-format step
    (regardless of whether params are present), and `%%` correctly
    becomes `%` in the final query either way.

    Casts any Decimal-typed columns to float64. aiomysql returns DECIMAL
    values as Python `Decimal` objects, which land in a pandas `object`
    column; downstream writers (pyarrow parquet in particular) can't
    handle a Decimal+None mix and raise ArrowTypeError. Float cast makes
    the DataFrame shape-compatible with the legacy parquet-read path.
    """
    from decimal import Decimal as _Decimal

    conn = None
    try:
        conn = await asyncio.wait_for(get_connection(), timeout=30)
        async with conn.cursor() as cursor:
            await cursor.execute(sql, params)
            rows = await cursor.fetchall()
            cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(rows, columns=cols)

        for col in df.columns:
            if df[col].dtype != object:
                continue
            non_null = df[col].dropna()
            if len(non_null) > 0 and isinstance(non_null.iloc[0], _Decimal):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)


async def load_model_dataset_async(competition_id: int = None) -> pd.DataFrame:
    """Load projection_model_dataset as DataFrame with parquet-compatible
    column names. Pass competition_id to filter to one league; None loads
    all leagues (used for the model_dataset_all pool)."""
    sql = _build_model_dataset_select()
    if competition_id is not None:
        df = await _fetch_df(sql + " WHERE competition_id = %s", (int(competition_id),))
    else:
        df = await _fetch_df(sql)
    logger.info(f"[model_dataset] loaded {len(df)} rows from DB (comp_id={competition_id})")
    return df


# Only the 8 outcome / projection % columns were stored as 'XX.XX%'
# strings in the legacy parquet format. The 3 odds columns (Home Odds %,
# Draw Odds %, Away Odds %) are numeric — score_preds writes them as
# floats at projection time (= 100 / bet365_decimal_odds), and
# downstream arithmetic (pd.isna check + odds_beta blending) assumes
# float. Leaving them as float on DB load keeps them compatible with
# both the concat of new projections and the parquet write.
_ACCURACY_PERCENT_COLUMNS = (
    "Home Win %", "Draw %", "Away Win %",
    "Home Clean Sheet %", "Away Clean Sheet %",
    "Over 1.5 Goals %", "Over 2.5 Goals %", "Both Teams Score %",
)


def _format_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric `%` columns back to 'XX.XX%' string form matching the
    legacy parquet shape. Downstream projection code writes NEW % values
    as formatted strings (e.g. score_preds['Home Win %'] = f'{pct:.2f}%'),
    so keeping loaded values as floats would produce a mixed-type column
    that breaks the belt-and-braces parquet writer with ArrowTypeError.

    Uses pandas nullable StringDtype so NULLs remain `pd.NA` rather than
    NaN (which is a float and re-introduces the mixed-type problem).
    """
    for col in _ACCURACY_PERCENT_COLUMNS:
        if col not in df.columns:
            continue
        formatted = df[col].apply(
            lambda v: f"{float(v):.2f}%" if v is not None and not (isinstance(v, float) and pd.isna(v)) else pd.NA
        )
        df[col] = formatted.astype("string")
    return df


async def load_accuracy_dataset_async(competition_id: int = None) -> pd.DataFrame:
    """Load projection_accuracy_dataset as DataFrame with parquet-compatible
    column names. Pass competition_id to filter to one league; None loads
    all leagues."""
    sql = _build_accuracy_dataset_select()
    if competition_id is not None:
        df = await _fetch_df(sql + " WHERE competition_id = %s", (int(competition_id),))
    else:
        df = await _fetch_df(sql)
    df = _format_percent_columns(df)
    logger.info(f"[accuracy_dataset] loaded {len(df)} rows from DB (comp_id={competition_id})")
    return df
