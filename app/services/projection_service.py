import asyncio
import logging
import time
from scipy.stats import poisson
import warnings
from app.repository.fixtures_repo import insert_fixtures_async
from app.repository.team_repo import insert_teams_async
from app.repository.predicted_table_repo import insert_predicted_table_async
from app.repository.league_position_repo import write_position_probabilities_async
from app.repository.player_stat_repo import insert_players_stats_async
from app.repository.player_repo import insert_player_async, get_players_from_league
from app.repository.fpl_repo import insert_fpl_projections_async
from app.repository.opta_repo import insert_opta_projections_async
from app.repository.fanteam_repo import insert_fanteam_projections_async
from app.repository.draftkings_repo import insert_draftkings_projections_async
from app.repository.dream11_repo import insert_dream11_projections_async
from app.data_loader import LeagueDataLoader
from app.repository.points_adjustments_repo import apply_points_adjustments
from app.services import unified_ratings
from app.source_database import get_source_connection, release_source_connection

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from .statz_functions import *
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
from fastapi import Response

logger = logging.getLogger("projection")

# Team-down defensive columns that must SURVIVE the explicit column filters in
# the FPL branch. distribute_team_predictions_to_players propagates them from
# team_projections via pivot; anything not named here is dropped before the
# team-down CBIT post-pass reads it, and the post-pass silently falls back to
# `Tackles + CBI(FPL)`.
#
# That fallback is why this is a shared constant rather than four copies of a
# literal. The combined CBIT column (999003) was added 2026-08-04 but not to
# the four allow-lists, so the whole "one combined CBIT" rework was inert —
# every player kept scoring on the old three-component path. Promoted clubs
# were the visible casualty: CBI(FPL) is Premier League only, so Coventry /
# Hull / Ipswich ran on TACKLES ALONE and projected ~0% DefCon (Bobby Thomas
# 1.6E-9 against a true ~8.5 CBIT per 90). George, 2026-08-04.
# Player-level stat columns persisted onto fpl_projections so the FPL detail
# tiles can read DIALLED values. frame column -> DB column. Team-level tiles
# (clean sheet %, xGC) come from fixture_projections and are not affected by a
# player dial, so they stay where they are.
# Share of a team's goals that come from the spot, measured over PL 2024/25 +
# 2025/26 (175 penalties taken, 146 scored, 29 missed). George reviewed it
# against his recollection of the long-run figure (8-9%) and chose the measured
# value, 2026-08-07. Named precisely because two seasons is a thin sample and
# it should be revisited once more history is available.
#
# The matching conversion rate (83.4%) lives in fpl_bps.PENALTY_CONVERSION_RATE
# — it is a scoring constant and is only needed on the points/BPS side.
PENALTY_GOAL_SHARE = 0.0676

FPL_STAT_COLUMNS = {
    'Goals': 'proj_goals',
    'Assists': 'proj_assists',
    'Saves': 'proj_saves',
    'Key Passes': 'proj_key_passes',
    'Shots On Target': 'proj_sot',
    'Tackles': 'proj_tackles',
}

FPL_DEF_EXTRA_COLS = [
    'Ball Recovery',
    'Clearances Blocks Interceptions (FPL)',
    'Clearances Blocks Interceptions Tackles (FPL)',
    'Tackles Won',
    # Penalty split — the player-level values distribute() derives from the
    # team columns above. Without them here the whole split is dropped by the
    # column filter, exactly as the combined CBIT column was.
    'Non-Penalty Goals',
    'Penalties Scored',
    'Big Chances Created',
]


class _NoPromotedRatings(Exception):
    """Sentinel: no promoted ratings configured for the league (no DB rows,
    no xlsx). Raised inside the promoted-blend try-block to skip the rest of
    it quietly — a normal state (e.g. MLS has no promotion at all), already
    logged at INFO where it's raised."""


class ProjectionService:
    CURRENT_DIR = Path(__file__).resolve().parent
    APP_DIR = CURRENT_DIR.parent

    DATA_FOLDER_PATH = APP_DIR / "data"
    MODEL_FILE_PATH = APP_DIR / "model-builds"
    SAVE_FILE_PATH = APP_DIR / "projection-outputs"
    DAYS = int(os.getenv("PROJECTION_DAYS", 5))

    # How many gameweeks ahead PL fantasy projections cover. The 5 fantasy
    # tables are capped to this many future-deadline gameweeks; player / team /
    # fixture projections get one extra buffer GW (see _filter_upcoming_fixtures)
    # so the fantasy filter still yields a full set after dropping a mid-flight
    # GW. Wide (16) so the FPL planner's rolling 6-GW window stays full as the
    # user navigates forward — the window stays 6 up to ~active GW 11, then
    # tapers toward the buffer's end (2026-07-09). Env-overridable for tuning.
    FANTASY_GAMEWEEKS = int(os.getenv("FANTASY_GAMEWEEKS", 19))

    # Per-league data source for the current run. Set in _setup_league to
    # the fresh LeagueDataLoader. Read elsewhere (transfermarkt mappings,
    # promoted ratings, FPL player mappings) for auxiliary tables that
    # don't already flow through ctx. Safe because projections are serialised
    # by the cross-worker file lock — only one runs at a time.
    _current_source = None
    # Per-run inputs for the Team Strength blend, stashed by _prepare_league
    # because they're intermediate state the returned ratings frame drops:
    #   pre_mv    ratings before the market-value adjustment (base component)
    #   mv_index  the MV Index per team (squad-value component)
    # Same lifecycle/safety as _current_source — runs are serialised by the
    # cross-worker projection lock, so only one league is ever in flight.
    _strength_inputs = {}

    @staticmethod
    def _filter_upcoming_fixtures(league: str, fixtures, date_from, date_to):
        """Slice fixtures to the projection scope for `league`.

        Premier League: project FANTASY_GAMEWEEKS+1 upcoming gameweeks
        (gameweek_id-based). Aligns with the FPL gameweek concept and feeds
        the FPL planning tools, whose rolling 6-GW window slides forward as the
        user navigates — so the buffer is wide (16) to keep that window full.
        gameweek_id survives postponements and double/blank gameweeks better
        than round_id or date-window.

        The window is FANTASY_GAMEWEEKS+1 (one extra) so the fantasy tables —
        which drop any gameweek whose deadline has already passed (see
        _fantasy_gw_filter) — still get the full FANTASY_GAMEWEEKS
        future-deadline gameweeks even when the soonest in-window gameweek is
        mid-flight. player / team / fixture projections simply get the one
        extra gameweek, harmlessly.

        All other leagues: stay on the date_from..date_to window
        (typically today + PROJECTION_DAYS=5). gameweek_id isn't reliably
        populated outside PL, so we don't risk an empty result.

        If PL upcoming fixtures don't have gameweek_id populated (rare —
        e.g. a fresh import that hasn't backfilled yet), falls back to
        the date window with a warning.
        """
        fixtures = fixtures.copy()
        fixtures['kickoff_datetime'] = pd.to_datetime(fixtures['kickoff_datetime'])
        if league == 'Premier League':
            future = fixtures[fixtures['kickoff_datetime'] >= pd.to_datetime('today')]
            if not future.empty and 'gameweek_id' in future.columns and pd.notna(future['gameweek_id'].min()):
                min_gw = future['gameweek_id'].min()
                # +1 buffer GW over the fantasy horizon so the fantasy filter
                # still yields a full FANTASY_GAMEWEEKS after dropping a
                # mid-flight GW.
                span = ProjectionService.FANTASY_GAMEWEEKS + 1
                next_fix = future[future['gameweek_id'] < min_gw + span]
                logger.info(f"[{league}] gameweek-based filter: GW {int(min_gw)}–{int(min_gw)+span-1} ({len(next_fix)} fixtures)")
                return next_fix
            logger.warning(f"[{league}] gameweek_id missing/null — falling back to date-window")
        return fixtures[(fixtures['kickoff_datetime'] >= date_from) & (fixtures['kickoff_datetime'] <= date_to)]

    @staticmethod
    def _fantasy_gw_filter(df, upcoming_gws):
        """Restrict a fantasy-projection DataFrame to gameweeks still open to
        plan for.

        `upcoming_gws` — the gameweek ids (gameweeks.id) whose deadline is in
        the future, capped to the FANTASY_GAMEWEEKS soonest. A fantasy
        projection for a gameweek whose deadline has already passed can't be
        acted on, so its rows are dropped before they reach the fantasy table.

        - `upcoming_gws is None` — deadline lookup failed / not a fantasy
          league: return `df` unchanged (fail-open — a transient lookup glitch
          must not blank the fantasy tables).
        - `upcoming_gws == []` — no gameweek has a future deadline (e.g. end
          of season): returns an empty frame, which is correct — there is
          nothing left to plan for.

        Only the five FANTASY frames go through this. player / team / fixture
        projections keep the current gameweek's unplayed fixtures.
        """
        if upcoming_gws is None:
            return df
        if df is None or df.empty or 'Gameweek' not in df.columns:
            return df
        return df[df['Gameweek'].isin(upcoming_gws)].copy()

    @staticmethod
    async def _resolve_league_id_db(league_name: str) -> int:
        """Direct DB lookup of competition_id by name.

        Used in DB-loader mode where we need league_id BEFORE the loader
        runs (loader scope is built around it). The hardcoded
        Brazil Serie A=648 mapping mirrors get_league_id's special case."""
        if league_name == "Brazil Serie A":
            return 648
        conn = await get_source_connection()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id FROM competitions WHERE name = %s", (league_name,)
                )
                row = await cur.fetchone()
                if row is None:
                    raise ValueError(f"League '{league_name}' not found in competitions table")
                return int(row[0])
        finally:
            release_source_connection(conn)

    @staticmethod
    async def _load_gameweek_deadlines(gameweek_ids) -> dict:
        """{gameweeks.id: deadline_time (pd.Timestamp)} for the given gameweek
        ids — a direct DB lookup, mirroring _resolve_league_id_db.

        Drives the fantasy-projection deadline filter: a fantasy table must
        only ever hold gameweeks whose deadline is still in the future.
        `deadline_time` is stored UTC; a NULL deadline maps to NaT and is
        treated as "not upcoming". gameweeks.id is the same id-space as the
        fixtures DataFrame's gameweek_id and the fantasy tables' Gameweek.
        """
        ids = sorted({int(g) for g in gameweek_ids if pd.notna(g)})
        if not ids:
            return {}
        conn = await get_source_connection()
        try:
            async with conn.cursor() as cur:
                placeholders = ",".join(["%s"] * len(ids))
                await cur.execute(
                    f"SELECT id, deadline_time FROM gameweeks WHERE id IN ({placeholders})",
                    tuple(ids),
                )
                rows = await cur.fetchall()
        finally:
            release_source_connection(conn)
        return {int(r[0]): pd.to_datetime(r[1]) for r in rows}

    @staticmethod
    def _read_df(path_no_ext: str) -> pd.DataFrame:
        """Read parquet file, falling back to xlsx if parquet doesn't exist yet (auto-migrates)."""
        parquet_path = f"{path_no_ext}.parquet"
        excel_path = f"{path_no_ext}.xlsx"
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        elif os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            ProjectionService._write_df(df, path_no_ext)
            logger.info(f"Migrated {os.path.basename(excel_path)} to parquet")
            return df
        raise FileNotFoundError(f"No data file found at {parquet_path} or {excel_path}")

    @staticmethod
    def _write_df(df: pd.DataFrame, path_no_ext: str) -> None:
        """Write DataFrame as parquet (fast, preserves dtypes)."""
        df = df.copy()
        for col in df.select_dtypes(["object"]).columns:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            inferred = pd.api.types.infer_dtype(non_null, skipna=True)
            if inferred in ("datetime", "datetime64", "date", "datetime with timezone"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif non_null.apply(lambda x: hasattr(x, "year")).any():
                df[col] = pd.to_datetime(df[col], errors="coerce")
        df.to_parquet(f"{path_no_ext}.parquet", index=False)

    @staticmethod
    def _read_df_with_fallback(path_no_ext: str, fallback_path_no_ext: str) -> pd.DataFrame:
        """Try to read league-specific file; fall back to all_leagues file if not found."""
        try:
            return ProjectionService._read_df(path_no_ext)
        except FileNotFoundError:
            logger.info(f"No data file for '{os.path.basename(path_no_ext)}', using all_leagues fallback")
            return ProjectionService._read_df(fallback_path_no_ext)


    async def _setup_league(self, league: str):
        """
        Shared setup for all projection methods. Returns a SimpleNamespace with all the
        league config, data, ratings, season IDs etc. that every method needs.
        """
        from types import SimpleNamespace
        ctx = SimpleNamespace()

        ctx.data_folder_path = ProjectionService.DATA_FOLDER_PATH
        ctx.model_file_path = ProjectionService.MODEL_FILE_PATH
        ctx.save_file_path = ProjectionService.SAVE_FILE_PATH
        ctx.league = league
        ctx.league_dashed = league.replace(' ', '-').replace('.', '').lower()
        ctx.date_from = pd.to_datetime('today')
        ctx.date_to = ctx.date_from + pd.DateOffset(days=ProjectionService.DAYS)

        # Data source: per-league LeagueDataLoader reads from the source DB
        # directly. Phase 7 cleanup (2026-05-11) flattened the previous
        # if Config.USE_DB_LOADER == "on" / else CSV+DataCache conditional —
        # USE_DB_LOADER has been the de-facto default since 2026-04-28 and
        # the off/shadow paths were dead code.
        ctx.league_id = await self._resolve_league_id_db(league)
        league_weightings_path = os.path.join(ctx.data_folder_path, "League Weightings.xlsx")
        loader = LeagueDataLoader(
            ctx.league_id,
            league_weightings_xlsx_path=league_weightings_path,
        )
        await loader.load()
        source = loader
        logger.info(f"[{league}] Data source: LeagueDataLoader")
        ProjectionService._current_source = source
        # Loader is per-call so mutation safety isn't a concern — no defensive
        # .copy() needed. Kept as a no-op shim so call sites don't churn.
        def _maybe_copy(df):
            return df

        db_config = source.projection_config
        db_row = db_config[db_config['league_name'] == league] if not db_config.empty else pd.DataFrame()

        if len(db_row) > 0:
            # DB-driven config (from admin panel)
            r = db_row.iloc[0]
            ctx.league_above = r.get('league_above_name') if pd.notna(r.get('league_above_name')) else None
            ctx.league_below = r.get('league_below_name') if pd.notna(r.get('league_below_name')) else None
            ctx.league_above_attack_weight = float(r.get('above_attack_weight', 1.0))
            ctx.league_above_defense_weight = float(r.get('above_defense_weight', 1.0))
            ctx.league_below_attack_weight = float(r.get('below_attack_weight', 1.0))
            ctx.league_below_defense_weight = float(r.get('below_defense_weight', 1.0))
            ctx.country_code = r.get('transfermarkt_code') if pd.notna(r.get('transfermarkt_code')) else None
            ctx.div = r.get('transfermarkt_div') if pd.notna(r.get('transfermarkt_div')) else None
            ctx.mv_beta = float(r.get('mv_beta', unified_ratings.W_MV_PRE))
            ctx.odds_beta = float(r.get('odds_beta', 0.3))
            # Dixon-Coles rho. NULL means the competition keeps the flat draw
            # boost, so this rolls out league by league rather than at once.
            _rho = r.get('dixon_coles_rho')
            ctx.dixon_coles_rho = float(_rho) if pd.notna(_rho) else 0.0
            ctx.fpl = (league == 'Premier League')  # FPL is always PL-only
            logger.info(f"[{league}] Config loaded from DB (projection_config.csv)")
        else:
            # Fallback to xlsx for leagues not yet in the DB config table.
            # Both `League Weightings.xlsx` and the DB are empty paths now
            # mostly defensive — competition_projection_config covers all 21
            # domestic projected leagues. If the xlsx file is missing
            # (post-2026-04-30 deletion), source.league_weightings is an
            # empty DataFrame and we drop straight into the defaults branch.
            league_weightings_df = source.league_weightings
            league_row = league_weightings_df[league_weightings_df['League'] == league] if (
                league_weightings_df is not None and not league_weightings_df.empty and 'League' in league_weightings_df.columns
            ) else pd.DataFrame()

            if len(league_row) > 0:
                ctx.league_below = league_row['League Below'].values[0]
                ctx.league_above = league_row['League Above'].values[0]
                ctx.league_below_attack_weight = league_row['League Below Attack Weight'].values[0]
                ctx.league_below_defense_weight = league_row['League Below Defense Weight'].values[0]
                ctx.league_above_attack_weight = league_row['League Above Attack Weight'].values[0]
                ctx.league_above_defense_weight = league_row['League Above Defense Weight'].values[0]
                ctx.country_code = league_row['code'].values[0]
                ctx.div = league_row['div'].values[0]
                ctx.mv_beta = league_row['mv_beta'].values[0]
                ctx.odds_beta = league_row['odds_beta'].values[0]
                ctx.dixon_coles_rho = 0.0   # xlsx fallback has no rho column
                logger.info(f"[{league}] Config loaded from League Weightings.xlsx (fallback)")
            else:
                ctx.league_below = None
                ctx.league_above = None
                ctx.league_below_attack_weight = 1.0
                ctx.league_below_defense_weight = 1.0
                ctx.league_above_attack_weight = 1.0
                ctx.league_above_defense_weight = 1.0
                ctx.country_code = None
                ctx.div = None
                # mv_beta = squad value's pre-season weight, so the
                # no-config fallback is the standard weight, not zero —
                # 0.0 under the old meaning was "apply no nudge", but here
                # it would mean "ignore squad value entirely".
                ctx.mv_beta = unified_ratings.W_MV_PRE
                ctx.odds_beta = 1.0
                ctx.dixon_coles_rho = 0.0
                logger.warning(f"[{league}] No config found in DB or xlsx — using defaults")

            ctx.fpl = (league == 'Premier League')

        ctx.weightings = [ctx.league_above_attack_weight, ctx.league_above_defense_weight,
                          ctx.league_below_attack_weight, ctx.league_below_defense_weight]

        # Load shared source data from cache. Everything is now .copy()ed to
        # prevent any in-place mutation inside a projection run from polluting
        # the shared cache for subsequent runs. Previously only the "big 4"
        # (player_stats, team_stats, standings, fixtures_df) were copied and
        # the others were passed by reference — which matched an observed
        # warm-cache vs fresh-cache drift of ~12 extra qualified players.
        ctx.player_stats = _maybe_copy(source.player_stats)
        ctx.team_stats = _maybe_copy(source.team_stats)
        ctx.standings_all = _maybe_copy(source.standings)
        ctx.seasons = _maybe_copy(source.seasons)
        ctx.comps = _maybe_copy(source.comps)
        ctx.comp_teams = _maybe_copy(source.comp_teams)
        ctx.teams = _maybe_copy(source.teams)
        # Players from LeagueDataLoader (DB-direct, scoped to teams in this
        # run's current squads). display_name already stripped upstream.
        ctx.players = source.players
        ctx.fixtures_df = _maybe_copy(source.fixtures_df)
        ctx.b365_odds = _maybe_copy(source.b365_odds)
        ctx.stats_types = _maybe_copy(source.stats_types)

        # League / season IDs — ctx.league_id was already resolved via
        # _resolve_league_id_db before the loader scope ran.

        # Model and accuracy datasets — Phase 3: read from DB (projection_model_dataset
        # + projection_accuracy_dataset) instead of parquet files. Eliminates the
        # pooled all_leagues parquet contamination (Scottish Prem's 15,440 cross-league
        # rows) that Phase 1+2 seeded out. The DB is now the source of truth.
        from app.repository.projection_dataset_repo import (
            load_model_dataset_async, load_accuracy_dataset_async,
        )
        ctx.model_dataset_all = await load_model_dataset_async()
        ctx.model_dataset_league = await load_model_dataset_async(competition_id=ctx.league_id)
        ctx.projection_accuracy_dataset_league = await load_accuracy_dataset_async(competition_id=ctx.league_id)
        ctx.projection_accuracy_dataset_all = await load_accuracy_dataset_async()

        # Team ratings — sourced from current data source (cache or loader).
        ctx.all_team_ratings = _maybe_copy(source.team_ratings)

        ctx.fixtures = ctx.fixtures_df[ctx.fixtures_df['competition_id'] == ctx.league_id]
        ctx.league_standings = ctx.standings_all[ctx.standings_all['competition_id'] == ctx.league_id]

        ctx.league_above_id = get_league_id(ctx.league_above, ctx.comps) if pd.notna(ctx.league_above) else None
        ctx.league_below_id = get_league_id(ctx.league_below, ctx.comps) if pd.notna(ctx.league_below) else None

        ctx.previous_season_id = get_season_id(ctx.league_id, ctx.seasons, True)
        ctx.current_season_id = get_season_id(ctx.league_id, ctx.seasons, False)
        if ctx.current_season_id is None:
            # No is_current=1 season row — typically a league between
            # seasons (old one ended + new one not yet created in
            # Sportmonks). Skip cleanly rather than crashing later.
            raise RuntimeError(
                f"no current season in seasons table for competition_id={ctx.league_id} — skipping"
            )

        ctx.standings = ctx.standings_all[ctx.standings_all['season_id'] == ctx.current_season_id]
        ctx.matches_played = ctx.standings['played'].mode().values[0]

        ctx.season_fixtures = ctx.fixtures[ctx.fixtures['season_id'] == ctx.current_season_id]
        ctx.total_matches = (ctx.season_fixtures['home_team_id'].value_counts() +
                             ctx.season_fixtures['away_team_id'].value_counts()).mean().round(0)

        # League Two used to pin this to 23846 (National League 2024/25).
        # That was correct while 2025/26 was live and then silently went
        # stale at the rollover: this season's promotees (Rochdale, York)
        # were being rated on National League form from TWO seasons ago,
        # while the teams the pinned season did cover (Barnet, Oldham) had
        # a full League Two season of their own by then. Resolve it like
        # every other league — verified 2026-08-02 to return 25752
        # (National League 2025/26), which is the season the current
        # promotees actually played in.
        ctx.previous_season_id_below = get_season_id(ctx.league_below_id, ctx.seasons, True) if ctx.league_below_id else None
        ctx.previous_season_id_above = get_season_id(ctx.league_above_id, ctx.seasons, True) if ctx.league_above_id else None

        ctx.stat_list = get_stat_list(ctx.league_id)

        # Auto-detect xG availability by checking if any player xG stats
        # exist for this league's current-season fixtures. No manual config
        # needed — if the data exists, we use it.
        xg_stat_id = get_stat_id('Expected Goals (xG)', ctx.stats_types)
        season_fixture_ids = set(ctx.season_fixtures['id'].values)
        has_xg = ctx.player_stats[
            (ctx.player_stats['stats_type_id'] == xg_stat_id) &
            (ctx.player_stats['fixture_id'].isin(season_fixture_ids))
        ]
        ctx.xG = len(has_xg) > 0
        logger.info(f"[{league}] xG auto-detected: {'enabled' if ctx.xG else 'disabled'} ({len(has_xg)} xG rows found)")

        return ctx

    async def _prepare_league(self, league, data_folder_path, model_file_path, save_file_path,
                        league_id, league_dashed, model_dataset_all, model_dataset_league,
                        projection_accuracy_dataset_all, projection_accuracy_dataset_league,
                        all_team_ratings, team_stats, player_stats, teams, stats_types, stat_list,
                        comp_teams, fixtures_df, fixtures, seasons, comps,
                        current_season_id, previous_season_id, previous_season_id_above,
                        previous_season_id_below, weightings, mv_beta, odds_beta,
                        country_code, div, matches_played, standings,
                        league_above, league_below, league_standings,
                        league_below_attack_weight, league_below_defense_weight,
                        league_above_id, league_below_id, xG, fpl, b365_odds,
                        season_fixtures, total_matches, players, mode="full"):
        """
        Shared preparation: gap-fill model/accuracy datasets, retrain models,
        calculate accuracy, build ratings with MV adjustment.
        Returns the computed ratings DataFrame.

        mode="refresh" skips the historical accuracy dataset gap-fill and the
        aggregated accuracy metrics calculation. These blocks are expensive
        (looping past fixtures + merging team stats) and aren't meaningfully
        different from what the 2am full run already computed a few hours
        earlier, so the 1:35pm refresh run skips them entirely. The
        projections() method still appends NEW projected values for upcoming
        fixtures to the accuracy dataset after _prepare_league() returns —
        that append path is NOT skipped, so historical tracking stays intact.
        """
        skip_accuracy = (mode == "refresh")
        logger.info(f"[{league}] _prepare_league mode={mode} skip_accuracy={skip_accuracy}")
        model_dataset_league['comp_id'] = league_id
        # In-projection gap-fill removed 2026-05-22. Actuals + outcome
        # flags now flow into projection_accuracy_dataset +
        # projection_model_dataset via the Laravel BackfillFixtureAccuracy
        # job — dispatched from ImportFixtureStatsJobV2 the moment team
        # stats land in fixture_team_stats. Sweep command
        # `projections:sweep-accuracy-backfill` (06:00 UTC daily) catches
        # any rows the event-driven path missed (e.g. stats arrived
        # before projection created the row).
        #
        # Projection no longer writes anything to the actual-stat /
        # outcome columns; the matching `ON DUPLICATE KEY UPDATE` clauses
        # in projection_dataset_repo.py now exclude those columns so
        # re-projecting a played fixture can't overwrite real values.

        # ## **Re-Train Models**

        # In[ ]:

        # Per-league + All-Leagues model training/load block removed
        # 2026-05-22 as Phase 2 of the per-league model retirement.
        # Projection runs no longer train models — retrain_service.py
        # owns all training. The `model` / `model_all` locals built
        # here were never referenced outside the for-loop; the real
        # model load for projection happens via load_all_models() later
        # in the function. See Phase 1 in projection_all_teams_service.

        # ## **Re-Calculate Accuracy**

        # ## Team Stat Accuracy

        # In[ ]:

        ## THIS IS ALL NEW - CALCULATE AND SAVE PROJECTION ACCURACY
        ## Skipped on refresh runs — the 2am full run already computed and
        ## saved these CSVs a few hours earlier. Rebuilding them at 1:35pm is
        ## wasted work since the underlying historical data hasn't changed.

        if skip_accuracy:
            logger.info(f"[{league}] skipping accuracy metrics + save (refresh mode)")
        else:
            logger.info(f"[{league}] Step: calculating projection accuracy")

            cols = ['Home {}', 'Away {}', 'Total {}', 'Total Projected {}', 'Home Projected {}', 'Away Projected {}']
            metrics = [
                ('Fixture Error', lambda df, s: df[f'Total Projected {s}'] - df[f'Total {s}']),
                ('Home Team Error', lambda df, s: df[f'Home Projected {s}'] - df[f'Home {s}']),
                ('Away Team Error', lambda df, s: df[f'Away Projected {s}'] - df[f'Away {s}']),
            ]
            abs_metrics = [
                ('Fixture Abs Error', 'Fixture Error'),
                ('Home Team Abs Error', 'Home Team Error'),
                ('Away Team Abs Error', 'Away Team Error'),
            ]

            def calc_errors(df, stat):
                d = {name: func(df, stat) for name, func in metrics}
                for name, base in abs_metrics:
                    d[name] = d[base].abs()
                return d

            def summarize(df, stat):
                d = calc_errors(df, stat)
                return {
                    'Stat': stat,
                    'Fixture Error': d['Fixture Error'].mean(),
                    'Home Team Error': d['Home Team Error'].mean(),
                    'Away Team Error': d['Away Team Error'].mean(),
                    'Fixture Abs Error': d['Fixture Abs Error'].mean(),
                    'Home Team Abs Error': d['Home Team Abs Error'].mean(),
                    'Away Team Abs Error': d['Away Team Abs Error'].mean(),
                }

            projection_accuracy_dataset_all_copy = projection_accuracy_dataset_all.dropna().copy()
            projection_accuracy_dataset_all_copy = projection_accuracy_dataset_all_copy[
                projection_accuracy_dataset_all_copy['Total Passes'] > 0]
            projection_accuracy_dataset_all_copy.reset_index(drop=True, inplace=True)
            projection_accuracy_dataset_league_copy = projection_accuracy_dataset_league.dropna().copy()
            projection_accuracy_dataset_league_copy = projection_accuracy_dataset_league_copy[
                projection_accuracy_dataset_league_copy['Total Passes'] > 0]
            projection_accuracy_dataset_league_copy.reset_index(drop=True, inplace=True)
            accuracy_df_league = pd.DataFrame(
                [summarize(projection_accuracy_dataset_league_copy, stat) for stat in accuracy_stat_list(stat_list)])
            accuracy_df_all = pd.DataFrame([summarize(projection_accuracy_dataset_all_copy, stat) for stat in accuracy_stat_list(stat_list)])
            accuracy_df_league = accuracy_df_league.round(2)
            accuracy_df_all = accuracy_df_all.round(2)

            # Za league
            file_path_league = os.path.join(data_folder_path, f"{league} Projection Accuracy.csv")

            # Za sve lige
            file_path_all = os.path.join(data_folder_path, "All Leagues Projection Accuracy.csv")

            logger.info(f"[{league}] Step: projection accuracy saved")
            ## THIS IS ALL NEW - ADD ABSOLUTE ERROR COLUMNS TO ACCURACY DATASET

            for stat in accuracy_stat_list(stat_list):
                # Calculate absolute errors
                for prefix in ['Total', 'Home', 'Away']:
                    abs_err_col = f"{prefix} {stat} Absolute Error"
                    proj_col = f"{prefix} Projected {stat}"
                    actual_col = f"{prefix} {stat}"
                    projection_accuracy_dataset_all_copy[abs_err_col] = (
                                projection_accuracy_dataset_all_copy[proj_col] - projection_accuracy_dataset_all_copy[
                            actual_col]).abs()
                    # Move the absolute error column next to projected column
                    cols = list(projection_accuracy_dataset_all_copy.columns)
                    if abs_err_col in cols and proj_col in cols:
                        idx = cols.index(proj_col) + 1
                        cols.remove(abs_err_col)
                        cols.insert(idx, abs_err_col)
                        projection_accuracy_dataset_all_copy = projection_accuracy_dataset_all_copy[cols]

            ProjectionService._write_df(projection_accuracy_dataset_all_copy, os.path.join(data_folder_path, "Accuracy Dataset with Errors"))

        # ## **Team Ratings**
        #
        # Team Ratings are calculated by combining a weighted average of Actual Goals (30%) and Expected Goals (70%) over the last 50 games.

        # In[ ]:

        ## UPDATED - Added new input: previous_team_rating (using the team_ratings dataset)
        ## UPDATED - Change weight to 0.95 and games to 30

        ratings = get_ratings(league_id=league_id, previous_team_ratings=all_team_ratings,
                              current_season_id=current_season_id,
                              all_season_ids=[current_season_id, previous_season_id, previous_season_id_above,
                                              previous_season_id_below],
                              comp_teams=comp_teams, teams_df=teams, fixtures_df=fixtures_df, team_stats=team_stats,
                              stats_types=stats_types, weight=0.96, games=30, weightings=weightings,
                              league_above_id=league_above_id, league_below_id=league_below_id)
        # In[12]:

        # Team-name mapping: all mappings live in transfermarkt_team_mappings DB table.
        # Read from the current run's data source (cache or loader) — set in _setup_league.
        db_mappings = ProjectionService._current_source.transfermarkt_team_mappings
        if not db_mappings.empty:
            team_mapping = dict(zip(db_mappings['from_name'], db_mappings['to_name']))
            logger.info(f"[{league}] Team mappings: {len(team_mapping)} entries (DB)")
        else:
            team_mapping = {}
            logger.warning(f"[{league}] Team mappings: DB empty — MV adjustment will run unmapped")

        # In[13]:

        try:
            # Try DB-driven promoted ratings first (from admin panel),
            # fall back to the per-league xlsx file.
            db_promoted = ProjectionService._current_source.promoted_team_ratings
            db_promoted_rows = db_promoted[db_promoted['league_name'] == league] if not db_promoted.empty else pd.DataFrame()

            if len(db_promoted_rows) > 0:
                second_ratings = db_promoted_rows[['team_name', 'attack', 'defense']].copy()
                second_ratings.columns = ['Team', 'Attack', 'Defense']
                # DB DECIMAL columns arrive as decimal.Decimal — Decimal*float
                # raises TypeError, so the whole blend crashed for every
                # DB-config league (silently, until the except below logged).
                # The xlsx path never hit this (floats all the way).
                second_ratings['Attack'] = second_ratings['Attack'].astype(float)
                second_ratings['Defense'] = second_ratings['Defense'].astype(float)
                logger.info(f"[{league}] Promoted team ratings loaded from DB ({len(second_ratings)} teams)")
            else:
                xlsx_path = f"{data_folder_path}/{league} Promoted Team Ratings.xlsx"
                if not os.path.exists(xlsx_path):
                    # Nothing configured anywhere — a NORMAL state, not a
                    # fault: leagues with no promotion at all (MLS), or
                    # nothing entered yet (admin badge prompts when a
                    # newcomer actually needs a prior). The FileNotFound
                    # from read_excel used to land as a daily WARNING.
                    logger.info(f"[{league}] No promoted ratings configured (no DB rows, no xlsx) — skipping blend")
                    raise _NoPromotedRatings()
                second_ratings = pd.read_excel(xlsx_path)
                second_ratings = second_ratings[['Team', 'Attack', 'Defense']]
                logger.info(f"[{league}] Promoted team ratings loaded from xlsx")
            second_ratings['Attack'] = (second_ratings['Attack']) * float(league_below_attack_weight)
            second_ratings['Defense'] = (second_ratings[
                'Defense']) / float(league_below_defense_weight)  # UPDATED - divide instead of multiply
            promoted_teams = second_ratings['Team'].unique()
            old_weight = 0.85 ** matches_played  # NEW - uses matches played so far in season
            new_weight = 1 - old_weight  # NEW - opposite of old weight
            ratings_copy = ratings.copy()  # NEW - This was just to stop warnings in my program so not necessary for functionality
            second_ratings['New Attack'] = second_ratings['Team'].map(ratings_copy.set_index('Team')[
                                                                          'Attack'])  # NEW - This maps the new attack rating from get_ratings function
            second_ratings['New Defense'] = second_ratings['Team'].map(ratings_copy.set_index('Team')[
                                                                           'Defense'])  # NEW - This maps the new defense rating from get_ratings function
            # A promoted team usually has NO computed rating (that's why it
            # has a prior at all) — its mapped New Attack/Defense is NaN, and
            # NaN * new_weight is NaN even at new_weight=0, so the blend
            # produced NaN and dropna() below deleted exactly the teams this
            # block exists to protect. Fall back to the scaled prior when
            # there's no computed rating to blend against.
            second_ratings['New Attack'] = second_ratings['New Attack'].fillna(second_ratings['Attack'])
            second_ratings['New Defense'] = second_ratings['New Defense'].fillna(second_ratings['Defense'])
            second_ratings['Attack'] = (second_ratings['Attack'] * old_weight) + (
                        second_ratings['New Attack'] * new_weight)  # NEW - This calculates the updated attack rating
            second_ratings['Defense'] = (second_ratings['Defense'] * old_weight) + (
                        second_ratings['New Defense'] * new_weight)  # NEW - This calculates the updated defense rating
            second_ratings = second_ratings[['Team', 'Attack', 'Defense']]  # NEW - This drops the temporary columns
            ratings = ratings[~ratings['Team'].isin(promoted_teams)]
            ratings = pd.concat([ratings, second_ratings], ignore_index=True)
            ratings.dropna(inplace=True)
            ratings.reset_index(drop=True, inplace=True)
        except _NoPromotedRatings:
            pass  # already logged at INFO — nothing configured, nothing to blend
        except Exception as promoted_err:
            # Was a bare `except: pass` — which is how the NaN blend above
            # went unnoticed for a full season-turnover cycle.
            logger.warning(f"[{league}] Promoted-ratings blend failed (continuing without): {promoted_err}")

        # In[ ]:

        # ratings['Attack'] = (ratings['Attack'] / ratings['Attack'].mean()) * 100
        # ratings['Defense'] = (ratings['Defense'] / ratings['Defense'].mean()) * 100
        # ratings['Overall'] = (ratings['Attack'] + ratings['Defense']) / 2
        # ratings.sort_values('Overall', ascending=False, inplace=True)
        # ratings.reset_index(drop=True, inplace=True)

        # In[15]:

        ## NEW - Function to rescale market values

        def rescale_to_range(series, new_min=0.5, new_max=2.0):
            old_min = series.min()
            old_max = series.max()
            return new_min + (series - old_min) * (new_max - new_min) / (old_max - old_min)

        # In[ ]:

        # Snapshot the ratings BEFORE the market-value adjustment: the
        # unified blend takes squad value in as its own weighted component,
        # so its base must be the pre-MV rating or money is counted twice.
        ProjectionService._strength_inputs = {
            'pre_mv': ratings[['Team', 'Attack', 'Defense']].copy(),
            'mv_index': {},
        }

        try:
            market_values = await get_market_value_with_cache(league_dashed, div, country_code)
            market_values['MV Index'] = market_values['Market Value'].astype(float) / market_values['Market Value'].astype(
                float).median()
            market_values['MV Index'] = np.log1p(market_values['MV Index'])
            market_values['MV Index'] = market_values['MV Index'] / market_values['MV Index'].mean()
            max = market_values['MV Index'].max() if market_values['MV Index'].max() < 2.0 else 2.0  # NEW - Cap max at 2.0
            min = market_values['MV Index'].min() if market_values[
                                                         'MV Index'].min() > 0.5 else 0.5  # NEW - Floor min at 0.5
            market_values['MV Index'] = rescale_to_range(market_values['MV Index'], min,
                                                         max)  # NEW - Rescale to new range to avoid outliers
            market_values['MV Index'] = market_values['MV Index'] / market_values['MV Index'].mean()  # NEW - Re-normalize
            market_values['Team'] = market_values['Team'].replace(team_mapping)
            market_values['Team'] = market_values['Team'].str.strip()

            ratings['Team'] = ratings['Team'].str.strip()
            ratings['MV Index'] = ratings['Team'].map(market_values.set_index('Team')['MV Index'])
            ratings['MV Index Reverse'] = (
                        ratings['MV Index'].mean() / ratings['MV Index'])  # NEW - Inverse MV Index (for defence)
            ratings['MV Index Reverse'] = ratings['MV Index Reverse'] / ratings[
                'MV Index Reverse'].mean()  # NEW - Normalize

            teams_to_map = ratings.loc[ratings['MV Index'].isna(), 'Team']  # NEW - Identify any teams not mapped

            if len(teams_to_map) > 0:
                market_values_not_mapped = market_values[~market_values['Team'].isin(ratings['Team'])]
                unmapped_names = market_values_not_mapped['Team'].tolist()
                logger.warning(f"[{league}] {len(unmapped_names)} unmapped Transfermarkt teams: {unmapped_names}")

                # Save unmapped teams to DB as pending mappings (to_name=NULL)
                # so the admin panel can show them for resolution.
                # Plain synchronous pymysql on purpose: this code runs inside
                # the already-running event loop's thread, where
                # run_until_complete() always raises "this event loop is
                # already running" — the save silently failed for every league
                # until 2026-07-18, which is why the admin "unmapped" badge
                # never lit up. A tiny 2-row insert doesn't need the pool.
                try:
                    import pymysql
                    from app.config import Config

                    sync_conn = pymysql.connect(
                        host=Config.DB_HOST, port=Config.DB_PORT,
                        user=Config.DB_USER, password=Config.DB_PASSWORD,
                        database=Config.DB_NAME, connect_timeout=10,
                    )
                    try:
                        with sync_conn.cursor() as cur:
                            for name in unmapped_names:
                                cur.execute(
                                    "INSERT IGNORE INTO transfermarkt_team_mappings "
                                    "(competition_id, from_name, to_name, created_at, updated_at) "
                                    "VALUES ((SELECT id FROM competitions WHERE name = %s LIMIT 1), %s, NULL, NOW(), NOW())",
                                    (league, name)
                                )
                        sync_conn.commit()
                        logger.info(f"[{league}] Saved {len(unmapped_names)} unmapped teams to DB for admin resolution")
                    finally:
                        sync_conn.close()
                except Exception as save_err:
                    logger.warning(f"[{league}] Could not save unmapped teams to DB: {save_err}")

                # Fill unmapped teams with neutral MV Index (1.0) instead of crashing
                ratings['MV Index'] = ratings['MV Index'].fillna(1.0)
                ratings['MV Index Reverse'] = ratings['MV Index Reverse'].fillna(1.0)

            # Hand the finished MV Index to the unified blend as its
            # squad-value component (the same index, not a second derivation).
            ProjectionService._strength_inputs['mv_index'] = dict(
                zip(ratings['Team'], pd.to_numeric(ratings['MV Index'], errors='coerce'))
            )

            ratings.drop(columns=['MV Index', 'MV Index Reverse'], inplace=True)
            logger.info(f"[{league}] Step: market value index built")
        except Exception as _mv_err:
            logger.warning(f"[{league}] Market value block failed for {league}: {_mv_err} — skipping MV adjustment")

        # --- UNIFIED RATING -------------------------------------------------
        # One rating for fixtures AND the season simulation. Replaced the
        # mv_beta nudge that used to sit here, which tilted only the ratings
        # driving FIXTURE predictions while a separate team_strength blend
        # drove the SEASON SIMULATION — so the model could project Liverpool
        # 3rd and Brentford 10th, then make Brentford favourites when they
        # met (George, 2026-07-31).
        #
        # mv_beta is still read, but it now means squad value's PRE-SEASON
        # WEIGHT rather than the old nudge coefficient — money predicts the
        # table far better in some leagues than others, so it is set per
        # competition. odds_beta is untouched and still drives the
        # fixture-level goals blend, which is a different thing entirely.
        try:
            _uni_ids = {}
            for _name in ratings['Team']:
                try:
                    _uni_ids[_name] = int(get_team_id(_name, teams, league_id, comp_teams))
                except Exception:
                    continue
            _uni_gpg = (get_home_goal_avg(league_id, team_stats, fixtures, stats_types)
                        + get_away_goal_avg(league_id, team_stats, fixtures, stats_types)) / 2
            _uni_conn = await get_source_connection()
            try:
                ratings, _uni_audit = await unified_ratings.apply_unified_ratings(
                    _uni_conn,
                    ratings,
                    competition_id=league_id,
                    season_id=current_season_id,
                    team_ids_by_name=_uni_ids,
                    mv_index=(ProjectionService._strength_inputs or {}).get('mv_index') or {},
                    matches_played=matches_played,
                    # NB: `max`/`min` are shadowed by numpy floats in the
                    # market-value block above, so the builtins are not
                    # callable here. Written without them on purpose.
                    games_in_season=((len(ratings) - 1) * 2) or 1,
                    goals_per_game=_uni_gpg,
                    mv_weight_pre=mv_beta,
                )
            finally:
                release_source_connection(_uni_conn)
            logger.info(f"[{league}] Step: unified rating applied")
        except Exception as _uni_err:
            # Never fail a projection run over the blend — fall through on
            # the form ratings, which is the pre-blend behaviour.
            logger.warning(
                f"[{league}] Unified rating failed ({_uni_err}) — continuing on form ratings only"
            )
        # the rescale-to-mean=100, so dialled teams shift the league mean
        # and other teams' indexed values drift naturally — exactly what
        # an operator expects when they say "boost Arsenal by 20%".
        # Also runs before the xG snapshot below, so the override flows
        # through to the xG-per-game column as well.
        try:
            from app.repository.team_dials_repo import apply_team_dials_to_ratings
            await apply_team_dials_to_ratings(ratings, league_id, teams, league)
        except Exception as _dial_err:
            logger.warning(f"[{league}] team dials block failed: {_dial_err} — skipping overrides")

        # Snapshot the post-MV, pre-rescale ratings in xG/game units. These
        # ride through to the writer alongside the indexed columns so the UI
        # can display "xGF per game" / "xGA per game" directly.
        ratings['Attack_xG'] = ratings['Attack']
        ratings['Defense_xG'] = ratings['Defense']
        ratings['Overall_xG'] = ratings['Attack'] - ratings['Defense']

        # In[17]:

        # Readjust so that 100 is the mean for Attack, Defense, and Overall
        for col in ['Attack', 'Defense']:
            ratings[col] = ratings[col] / ratings[col].mean() * 100
        ratings['Overall'] = ratings['Attack'] - ratings['Defense']  # UPDATED - Overall is now Attack minus Defense
        ratings.sort_values('Overall', ascending=False, inplace=True)
        ratings.reset_index(drop=True, inplace=True)
        # Indexed columns stay at 1dp (legacy precision). xG/game columns
        # go to 2dp so values like 1.85 don't flatten to 1.9.
        ratings[['Attack', 'Defense', 'Overall']] = ratings[['Attack', 'Defense', 'Overall']].round(1)
        for _xg_col in ('Attack_xG', 'Defense_xG', 'Overall_xG'):
            if _xg_col in ratings.columns:
                ratings[_xg_col] = ratings[_xg_col].round(2)
        ratings['Rank'] = ratings.index + 1
        # Movement = rank change vs most recent snapshot at least 7 days old.
        # Rationale: matches football's natural matchday cadence. Looking only
        # at yesterday's snapshot (the prior default) produced noisy day-over-
        # day movement; a 7-day window captures "since last week's matchday"
        # across every league + euro comp we project. Falls back to 0 when
        # there's no snapshot that old (new league / first run).
        from datetime import timedelta
        cutoff = pd.to_datetime('today').date() - timedelta(days=7)
        old_league = all_team_ratings[all_team_ratings['League'] == league]
        old_week_ago = old_league[old_league['Date'] <= cutoff]

        if len(old_week_ago) > 0:
            old_ratings = old_week_ago[old_week_ago['Date'] == old_week_ago['Date'].max()].copy()
            old_ratings.reset_index(drop=True, inplace=True)
            old_ratings['Rank'] = old_ratings.index + 1
            for i in range(len(ratings)):
                team = ratings.loc[i, 'Team']
                match = old_ratings.loc[old_ratings['Team'] == team, 'Rank']
                old_rank = match.values[0] if len(match) > 0 else ratings.loc[i, 'Rank']
                new_rank = ratings.loc[i, 'Rank']
                ratings.loc[i, 'Movement'] = old_rank - new_rank
        else:
            # Not enough history (new league or <7 days since start) — skip movement.
            ratings['Movement'] = 0
            logger.info(f"[{league}] No ratings snapshot older than 7 days — movement set to 0")
        ratings = ratings[['Team', 'Attack', 'Defense', 'Overall', 'Attack_xG', 'Defense_xG', 'Overall_xG', 'Movement']]

        # In[ ]:

        ## NEW - Save ratings to the team_ratings DB table (was parquet).
        ratings['Date'] = pd.to_datetime('today').date()
        ratings['League'] = league
        from app.repository.team_ratings_repo import insert_team_ratings_async
        await insert_team_ratings_async(
            ratings, league, league_id, teams,
            comp_teams=comp_teams,
        )

        logger.info(f"[{league}] Step: team ratings calculated + saved to DB")
        


        return ratings

    async def projections(self, league_request):
        league = league_request.league or 'Championship'
        _start_time = time.time()
        logger.info(f'[{league}] START projections')
        # Run start on the SOURCE DB's clock, for the post-run dial true-up
        # skip. Must not use the container clock: the two differ by an hour, so
        # comparing a local timestamp against fpl_player_dials.updated_at would
        # skip a true-up that was needed (or run one that wasn't).
        _run_started_at = None
        try:
            _rs_conn = await get_source_connection()
            try:
                async with _rs_conn.cursor() as _rc:
                    await _rc.execute("SELECT NOW()")
                    _rs_row = await _rc.fetchone()
                    _run_started_at = _rs_row[0] if _rs_row else None
            finally:
                release_source_connection(_rs_conn)
        except Exception as _rs_err:
            logger.warning(f"[{league}] could not read run-start time: {_rs_err}")


        ctx = await self._setup_league(league)

        # Unpack shared context into local variables so downstream code is unchanged
        data_folder_path = ctx.data_folder_path
        model_file_path = ctx.model_file_path
        save_file_path = ctx.save_file_path
        league_dashed = ctx.league_dashed
        date_from = ctx.date_from
        date_to = ctx.date_to
        league_below = ctx.league_below
        league_above = ctx.league_above
        league_below_attack_weight = ctx.league_below_attack_weight
        league_below_defense_weight = ctx.league_below_defense_weight
        league_above_attack_weight = ctx.league_above_attack_weight
        league_above_defense_weight = ctx.league_above_defense_weight
        country_code = ctx.country_code
        div = ctx.div
        weightings = ctx.weightings
        mv_beta = ctx.mv_beta
        odds_beta = ctx.odds_beta
        xG = ctx.xG
        fpl = ctx.fpl
        player_stats = ctx.player_stats
        team_stats = ctx.team_stats
        standings = ctx.standings
        seasons = ctx.seasons
        comps = ctx.comps
        comp_teams = ctx.comp_teams
        teams = ctx.teams
        players = ctx.players
        fixtures_df = ctx.fixtures_df
        b365_odds = ctx.b365_odds
        stats_types = ctx.stats_types
        model_dataset_all = ctx.model_dataset_all
        model_dataset_league = ctx.model_dataset_league
        projection_accuracy_dataset_league = ctx.projection_accuracy_dataset_league
        projection_accuracy_dataset_all = ctx.projection_accuracy_dataset_all
        all_team_ratings = ctx.all_team_ratings
        league_id = ctx.league_id
        fixtures = ctx.fixtures
        league_standings = ctx.league_standings
        league_above_id = ctx.league_above_id
        league_below_id = ctx.league_below_id
        previous_season_id = ctx.previous_season_id
        current_season_id = ctx.current_season_id
        matches_played = ctx.matches_played
        season_fixtures = ctx.season_fixtures
        total_matches = ctx.total_matches
        previous_season_id_below = ctx.previous_season_id_below
        previous_season_id_above = ctx.previous_season_id_above
        stat_list = ctx.stat_list

        ratings = await self._prepare_league(
            league=league, data_folder_path=data_folder_path, model_file_path=model_file_path,
            save_file_path=save_file_path, league_id=league_id, league_dashed=league_dashed,
            model_dataset_all=model_dataset_all, model_dataset_league=model_dataset_league,
            projection_accuracy_dataset_all=projection_accuracy_dataset_all,
            projection_accuracy_dataset_league=projection_accuracy_dataset_league,
            all_team_ratings=all_team_ratings, team_stats=team_stats, player_stats=player_stats,
            teams=teams, stats_types=stats_types, stat_list=stat_list,
            comp_teams=comp_teams, fixtures_df=fixtures_df, fixtures=fixtures, seasons=seasons, comps=comps,
            current_season_id=current_season_id, previous_season_id=previous_season_id,
            previous_season_id_above=previous_season_id_above,
            previous_season_id_below=previous_season_id_below,
            weightings=weightings, mv_beta=mv_beta, odds_beta=odds_beta,
            country_code=country_code, div=div, matches_played=matches_played, standings=standings,
            league_above=league_above, league_below=league_below, league_standings=league_standings,
            league_below_attack_weight=league_below_attack_weight,
            league_below_defense_weight=league_below_defense_weight,
            league_above_id=league_above_id, league_below_id=league_below_id,
            xG=xG, fpl=fpl, b365_odds=b365_odds,
            season_fixtures=season_fixtures, total_matches=total_matches, players=players,
            mode=(league_request.mode if hasattr(league_request, 'mode') and league_request.mode else "full"),
        )

        # ## **Make Predictions for Next Fixture Round**
        #
        # Result, Score, Clean Sheets, Over 1.5, Over 2.5 and BTTS all calculated here using Poisson Distribution.

        # In[18]:

        next_fix = ProjectionService._filter_upcoming_fixtures(league, fixtures, date_from, date_to)
        fixtures['kickoff_datetime'] = pd.to_datetime(fixtures['kickoff_datetime'])
        if hasattr(league_request, 'fixture_ids') and league_request.fixture_ids:
            next_fix = next_fix[next_fix['id'].isin(league_request.fixture_ids)]
            logger.info(f'[{league}] Filtered to {len(next_fix)} of {len(fixtures[(fixtures["kickoff_datetime"] >= date_from) & (fixtures["kickoff_datetime"] <= date_to)])} fixtures')
        next_fix = next_fix[
            ['id', 'kickoff_datetime', 'name', 'home_team_id', 'away_team_id', 'bet365_home_odds_decimal',
             'bet365_draw_odds_decimal', 'bet365_away_odds_decimal']]
        next_fix['home_team'] = next_fix['home_team_id'].apply(lambda x: get_team(x, teams))
        next_fix['away_team'] = next_fix['away_team_id'].apply(lambda x: get_team(x, teams))
        next_fix = next_fix.drop(columns=['home_team_id', 'away_team_id'])
        next_fix = drop_placeholder_fixtures(next_fix, league)
        next_fix.sort_values(by=['kickoff_datetime', 'home_team'], inplace=True)
        next_fix.reset_index(drop=True, inplace=True)

        # In[ ]:

        avg_home_goals = get_home_goal_avg(league_id, team_stats, fixtures, stats_types)
        avg_away_goals = get_away_goal_avg(league_id, team_stats, fixtures, stats_types)

        logger.info(f"[{league}] avg_home_goals={avg_home_goals:.3f}, avg_away_goals={avg_away_goals:.3f}")
        

        logger.info(f"[{league}] Predicting fixtures ({len(next_fix)} matches)...")
        _t = time.time()
        score_preds = make_round_goal_prediction(next_fix, ratings, avg_home_goals, avg_away_goals)
        logger.info(f"[{league}] Fixtures predicted ({time.time()-_t:.1f}s)")
        # boost = get_draw_boost(ratings, avg_home_goals, avg_away_goals, get_draw_perc(league_id, fixtures))
        # Dixon-Coles replaces the flat draw boost where a league has rho
        # configured; get_result_probs applies one or the other, never both.
        dixon_coles_rho = getattr(ctx, 'dixon_coles_rho', 0.0) or 0.0
        boost = 1.0 if dixon_coles_rho else 1.1
        score_preds['Home Odds %'] = ((1 / next_fix['bet365_home_odds_decimal']) * 100)
        score_preds['Draw Odds %'] = ((1 / next_fix['bet365_draw_odds_decimal']) * 100)
        score_preds['Away Odds %'] = ((1 / next_fix['bet365_away_odds_decimal']) * 100)

        home_win = []
        draw = []
        away_win = []
        home_clean = []
        away_clean = []
        over_1 = []
        over_2 = []
        btts = []

        # Pre-load bet365 goals over/under for the upcoming fixtures.
        # The cascade in compute_final_goals_and_probs (paths 1-3) uses
        # per-team + match-total ladders directly; path 4 (legacy 1X2
        # reverse-solve) is the fallback when those markets are absent.
        from app.services.odds_blend import (
            load_goals_odds_for_fixtures,
            compute_final_goals_and_probs,
        )
        _odds_conn = await get_source_connection()
        try:
            goals_odds_map = await load_goals_odds_for_fixtures(
                _odds_conn, next_fix['id'].tolist(),
            )
        finally:
            release_source_connection(_odds_conn)

        for i in range(len(score_preds)):
            bookie_margin = 1 + (
                        score_preds.loc[i, 'Home Odds %'] + score_preds.loc[i, 'Draw Odds %'] + score_preds.loc[
                    i, 'Away Odds %'] - 100) / 100
            score_preds.loc[i, 'Home Odds %'] = (score_preds.loc[i, 'Home Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Draw Odds %'] = (score_preds.loc[i, 'Draw Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Away Odds %'] = (score_preds.loc[i, 'Away Odds %'] / bookie_margin).round(2)
            home_goals = score_preds['Home Goals'][i]
            away_goals = score_preds['Away Goals'][i]
            bookie_1x2_pct = None
            if not pd.isna(score_preds['Home Odds %'][i]):
                bookie_1x2_pct = (
                    float(score_preds['Home Odds %'][i]) / 100.0,
                    float(score_preds['Draw Odds %'][i]) / 100.0,
                    float(score_preds['Away Odds %'][i]) / 100.0,
                )
            fixture_id = int(next_fix['id'].iloc[i])
            new_home_goals, new_away_goals, adjusted_home_win_prob, adjusted_draw_prob, adjusted_away_win_prob = (
                compute_final_goals_and_probs(
                    fixture_id,
                    float(home_goals), float(away_goals),
                    bookie_1x2_pct,
                    goals_odds_map.get(fixture_id, {}),
                    odds_beta,
                    boost,
                    dixon_coles_rho,
                )
            )
            score_preds.loc[i, 'Home Goals'] = round(new_home_goals, 2)
            score_preds.loc[i, 'Away Goals'] = round(new_away_goals, 2)
            home_clean_sheet = poisson.pmf(0, new_away_goals)
            away_clean_sheet = poisson.pmf(0, new_home_goals)
            x = np.arange(0, 9)
            y = np.arange(0, 9)
            X, Y = np.meshgrid(x, y)
            Z = poisson.pmf(X, new_home_goals) * poisson.pmf(Y, new_away_goals)
            home_win.append(f"{adjusted_home_win_prob:.2f}%")
            draw.append(f"{adjusted_draw_prob:.2f}%")
            away_win.append(f"{adjusted_away_win_prob:.2f}%")
            home_clean.append(f"{home_clean_sheet * 100:.2f}%")
            away_clean.append(f"{away_clean_sheet * 100:.2f}%")
            over_1_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1]) * 100
            over_2_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1] - Z[2, 0] - Z[0, 2] - Z[1, 1]) * 100
            both_teams_score_prob = (1 - Z[0, :].sum() - Z[:, 0].sum() + Z[0, 0]) * 100
            over_1.append(f"{over_1_goals:.2f}%")
            over_2.append(f"{over_2_goals:.2f}%")
            btts.append(f"{both_teams_score_prob:.2f}%")

        # score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'], inplace=True)
        score_preds['Home Win %'] = home_win
        score_preds['Draw %'] = draw
        score_preds['Away Win %'] = away_win
        score_preds['Home Clean Sheet %'] = home_clean
        score_preds['Away Clean Sheet %'] = away_clean
        score_preds['Over 1.5 Goals %'] = over_1
        score_preds['Over 2.5 Goals %'] = over_2
        score_preds['Both Teams Score %'] = btts
        score_preds['Home Goals'] = score_preds['Home Goals'].round(2)
        score_preds['Away Goals'] = score_preds['Away Goals'].round(2)
        score_preds_with_odds = score_preds.copy()  # NEW - Create a copy with odds included
        score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'],
                         inplace=True)  # NEW - Drop odds from main predictions dataframe

        # score_preds.to_csv(rf"{save_file_path}\{league} Fixtures.csv", index=False)

        logger.info(f"[{league}] Inserting fixtures into DB...")
        _t = time.time()
        await insert_fixtures_async(score_preds, teams=teams, competition_id=league_id, comp_teams=comp_teams)
        logger.info(f"[{league}] Fixtures inserted ({time.time()-_t:.1f}s)")

        # In[ ]:

        ## NEW - Update accuracy dataset with new predictions

        score_preds_with_odds.rename(
            columns={'id': 'fixture_id', 'Home Goals': 'Home Projected Goals', 'Away Goals': 'Away Projected Goals'},
            inplace=True)
        score_preds_with_odds['Total Projected Goals'] = score_preds_with_odds['Home Projected Goals'] + \
                                                         score_preds_with_odds['Away Projected Goals']
        score_preds_with_odds['comp_id'] = league_id
        projection_accuracy_dataset_league = pd.concat([projection_accuracy_dataset_league, score_preds_with_odds],
                                                       ignore_index=True)
        score_preds_with_odds.rename(
            columns={'fixture_id': 'id', 'Home Projected Goals': 'Home Goals', 'Away Projected Goals': 'Away Goals'},
            inplace=True)
        score_preds_with_odds.drop(columns=['comp_id', 'Total Projected Goals'], inplace=True)

        # In[ ]:

        ## NEW - 4+ STAR BETS SECTION

        # ## **4+ Star Bets**

        # In[ ]:

        # NEW - Load previous best bets file and append new best bets

        # best_bets = pd.read_excel(rf"{data_folder_path}\Best Bets.xlsx")
        best_bets = ProjectionService._read_df(f"{data_folder_path}/Best Bets")

        new_best_bets = pd.DataFrame()
        for i in range(len(score_preds)):
            fix_id = score_preds.loc[i, 'id']
            date = score_preds.loc[i, 'kickoff_datetime']
            date = date.strftime('%d-%m')
            fix = fixtures_df[fixtures_df['id'] == fix_id]
            home_win = float(score_preds.loc[i, 'Home Win %'].strip('%')) / 100
            draw = float(score_preds.loc[i, 'Draw %'].strip('%')) / 100
            away_win = float(score_preds.loc[i, 'Away Win %'].strip('%')) / 100
            over_1_5_goals = float(score_preds.loc[i, 'Over 1.5 Goals %'].strip('%')) / 100
            over_2_5_goals = float(score_preds.loc[i, 'Over 2.5 Goals %'].strip('%')) / 100
            btts = float(score_preds.loc[i, 'Both Teams Score %'].strip('%')) / 100

            # Skip best-bet eval if any bet365 odd is missing (None/NaN).
            # bet365_totals_odds in particular is sparse for some leagues
            # (e.g. Belgian Pro League playoff fixtures not yet priced).
            _odds_vals = [
                fix['bet365_home_odds_decimal'].values[0],
                fix['bet365_draw_odds_decimal'].values[0],
                fix['bet365_away_odds_decimal'].values[0],
                fix['over_1_5_odds_decimal'].values[0],
                fix['over_2_5_odds_decimal'].values[0],
                fix['bet365_btts_yes_odds_decimal'].values[0],
            ]
            if any(v is None or pd.isna(v) for v in _odds_vals):
                continue
            home_win_odds = 1 / _odds_vals[0]
            draw_odds = 1 / _odds_vals[1]
            away_win_odds = 1 / _odds_vals[2]
            over_1_5_goals_odds = 1 / _odds_vals[3]
            over_2_5_goals_odds = 1 / _odds_vals[4]
            btts_odds = 1 / _odds_vals[5]

            home_win_edge = home_win - home_win_odds
            draw_edge = draw - draw_odds
            away_win_edge = away_win - away_win_odds
            over_1_5_goals_edge = over_1_5_goals - over_1_5_goals_odds
            over_2_5_goals_edge = over_2_5_goals - over_2_5_goals_odds
            btts_edge = btts - btts_odds

            home_win_edge_rating = (home_win_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            draw_edge_rating = (draw_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            away_win_edge_rating = (away_win_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            over_1_5_goals_edge_rating = (over_1_5_goals_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            over_2_5_goals_edge_rating = (over_2_5_goals_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            btts_edge_rating = (btts_edge - (-0.1)) * 5 / (0.1 - (-0.1))

            home_win_prob_rating = (home_win) * 5 / (0.9)
            draw_prob_rating = (draw) * 5 / (0.9)
            away_win_prob_rating = (away_win) * 5 / (0.9)
            over_1_5_goals_prob_rating = (over_1_5_goals) * 5 / (0.9)
            over_2_5_goals_prob_rating = (over_2_5_goals) * 5 / (0.9)
            btts_prob_rating = (btts) * 5 / (0.9)

            home_win_total_rating = (home_win_edge_rating * 0.7 if home_win_edge_rating > 0 else 0) + (
                home_win_prob_rating * 0.3 if home_win_prob_rating < 5 else 5 * 0.3)
            draw_total_rating = (draw_edge_rating * 0.7 if draw_edge_rating > 0 else 0) + (
                draw_prob_rating * 0.3 if draw_prob_rating < 5 else 5 * 0.3)
            away_win_total_rating = (away_win_edge_rating * 0.7 if away_win_edge_rating > 0 else 0) + (
                away_win_prob_rating * 0.3 if away_win_prob_rating < 5 else 5 * 0.3)
            over_1_5_goals_total_rating = (
                                              over_1_5_goals_edge_rating * 0.7 if over_1_5_goals_edge_rating > 0 else 0) + (
                                              over_1_5_goals_prob_rating * 0.3 if over_1_5_goals_prob_rating < 5 else 5 * 0.3)
            over_2_5_goals_total_rating = (
                                              over_2_5_goals_edge_rating * 0.7 if over_2_5_goals_edge_rating > 0 else 0) + (
                                              over_2_5_goals_prob_rating * 0.3 if over_2_5_goals_prob_rating < 5 else 5 * 0.3)
            btts_total_rating = (btts_edge_rating * 0.7 if btts_edge_rating > 0 else 0) + (
                btts_prob_rating * 0.3 if btts_prob_rating < 5 else 5 * 0.3)

            for bet_type in ['Home Win', 'Draw', 'Away Win', 'Over 1.5 Goals', 'Over 2.5 Goals', 'BTTS']:
                edge = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_edge']
                edge_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_edge_rating']
                prob_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_prob_rating']
                total_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_total_rating']
                if total_rating >= 4.0:
                    new_best_bets = pd.concat([new_best_bets, pd.DataFrame({
                        'Date': [date],
                        'Competition': [league],
                        'Home Team': [score_preds.loc[i, 'Home Team']],
                        'Away Team': [score_preds.loc[i, 'Away Team']],
                        'Bet Type': [bet_type],
                        'Rating': [round(total_rating, 1) if total_rating < 5 else 5.0],
                        'Edge %': [round(edge * 100, 2)],
                        'Price': [
                            round(1 / locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_odds'], 2)]
                    })], ignore_index=True)

        best_bets = pd.concat([best_bets, new_best_bets], ignore_index=True)
        best_bets.drop_duplicates(subset=['Date', 'Competition', 'Home Team', 'Away Team', 'Bet Type'], keep='last',
                                  inplace=True)
        # best_bets.to_excel(rf"{data_folder_path}\Best Bets.xlsx", index=False)
        ProjectionService._write_df(best_bets, f"{data_folder_path}/Best Bets")

        # # **League Projections**
        logger.info(f"[{league}] Step: predicted table simulation complete")
        # In[ ]:

        if league != 'Major League Soccer':
            season_fixtures = fixtures.copy()
            today = pd.to_datetime('today')
            season_fixtures['kickoff_datetime'] = pd.to_datetime(season_fixtures['kickoff_datetime'])
            season_fixtures = season_fixtures[season_fixtures['kickoff_datetime'] >= today]
            season_fixtures.loc[:, 'home_team'] = season_fixtures['home_team_id'].map(teams.set_index('id')['name'])
            season_fixtures.loc[:, 'away_team'] = season_fixtures['away_team_id'].map(teams.set_index('id')['name'])
            season_fixtures.sort_values(by='kickoff_datetime', inplace=True)
            season_fixtures.reset_index(drop=True, inplace=True)
            season_fixtures = drop_placeholder_fixtures(season_fixtures, league)

            # The season simulation uses the SAME rating as the fixture
            # projections. There used to be a swap here to a separate
            # team_strength rating, which is how the model could project a
            # team 3rd over the season yet make them underdogs in the match.
            _sim_ratings = ratings

            season_score_preds = make_round_goal_prediction(season_fixtures, _sim_ratings, avg_home_goals, avg_away_goals)

            for i in range(len(season_score_preds)):
                home_goals = season_score_preds['Home Goals'][i]
                away_goals = season_score_preds['Away Goals'][i]

            season_score_preds['Home Goals'] = season_score_preds['Home Goals'].round(2)
            season_score_preds['Away Goals'] = season_score_preds['Away Goals'].round(2)

            current_standings = standings.copy()
            current_standings['Team'] = current_standings['team_id'].map(teams.set_index('id')['name'])
            current_standings.rename(
                columns={'goals_for': 'Goals For', 'goals_against': 'Goals Against', 'points': 'Points'}, inplace=True)
            current_standings['Goal Difference'] = current_standings['Goals For'] - current_standings['Goals Against']
            current_standings = current_standings[['Team', 'Points', 'Goals For', 'Goals Against', 'Goal Difference']]
            current_standings.reset_index(drop=True, inplace=True)
            current_standings = current_standings.astype(
                {'Points': 'int', 'Goals For': 'int', 'Goals Against': 'int', 'Goal Difference': 'int'})
            current_league_table = {
                team: {'Points': points, 'Goals For': gf, 'Goals Against': ga, 'Goal Difference': gd} for
                team, points, gf, ga, gd in current_standings.values}

            # Manual points adjustments — a deduction announced before
            # Sportmonks folds it into `points`. Skipped automatically once
            # the standings already show it, so it cannot double-count.
            current_league_table = await apply_points_adjustments(
                current_league_table, standings, league_id, current_season_id, teams, league)

            avg_table, all_tables = sim_multiple_seasons(season_score_preds, current_league_table, num_sims=10000)

            avg_table_with_probs_and_point_limits = get_avg_table_with_probs_and_point_limits(avg_table,
                                                                                              all_tables)
            # avg_table_with_probs_and_point_limits.to_csv(rf"{save_file_path}\{league} Predicted Table.csv", index=False)
            await insert_predicted_table_async(avg_table_with_probs_and_point_limits, teams, comps, league)
            # Per-team / per-position finishing distribution — every positional
            # market on the read side is a range-sum over this. Non-fatal: must
            # not break the run.
            try:
                await write_position_probabilities_async(all_tables, teams, comps, league)
            except Exception as e:
                logger.error(f"[{league}] league position probabilities write failed (non-fatal): {e}", exc_info=True)

        # # **Team Projections**
        #
        # Getting each Teams stat projections using the models

        # In[20]:

        stat_list = get_stat_list(league_id)

        # In[21]:

        models = load_all_models(stat_list, model_file_path)

        # In[22]:

        if next_fix.empty:
            return Response(status_code=204)

        todays_date = pd.to_datetime(next_fix['kickoff_datetime'].iloc[0]).date()

        # In[ ]:

        team_projections = get_team_round_predictions(next_fix, stat_list, fixtures_df, team_stats, teams, stats_types,
                                                      models, ratings=ratings,
                                                      league_weightings=[league_above_attack_weight,
                                                                         league_above_defense_weight,
                                                                         league_below_attack_weight,
                                                                         league_below_defense_weight],
                                                      season_id=[current_season_id, previous_season_id,
                                                                 previous_season_id_above, previous_season_id_below],
                                                      games=50,
                                                      comp_teams=comp_teams[comp_teams['competition_id'] == league_id])
        # In[ ]:

        ## NEW - Add historical stats to the model dataset and drop them from team projections afterwards

        new_rows = []

        for i in range(len(team_projections)):
            team_df = team_projections.iloc[[i]]
            new_row = {}
            new_row['id'] = team_df['fixture_id'].values[0]
            new_row['kickoff_datetime'] = team_df['kickoff_datetime'].values[0]
            new_row['comp_id'] = league_id
            new_row['Team'] = team_df['Team'].values[0]
            new_row['Opponent'] = team_df['Opponent'].values[0]
            new_row['Venue'] = team_df['Venue'].values[0]
            for stat in stat_list:
                new_row['Team ' + stat + ' History'] = team_df['Team ' + stat + ' History'].values[0]
                new_row['Opponent ' + stat + ' History Against'] = \
                team_df['Opponent ' + stat + ' History Against'].values[0]
            new_rows.append(new_row)

        model_dataset_league = pd.concat([model_dataset_league, pd.DataFrame(new_rows)], ignore_index=True)
        model_dataset_all = pd.concat([model_dataset_all, pd.DataFrame(new_rows)], ignore_index=True)
        model_dataset_league.drop_duplicates(subset=['id', 'Team', 'Opponent', 'Venue'], keep='last', inplace=True)
        model_dataset_all.drop_duplicates(subset=['id', 'Team', 'Opponent', 'Venue'], keep='last', inplace=True)

        ProjectionService._write_df(model_dataset_league, f"{data_folder_path}/{league}_model_dataset_with_history")
        ProjectionService._write_df(model_dataset_all, f"{data_folder_path}/all_leagues_model_dataset_with_history")

        # Dual-write to DB (Phase 2 of the data-files-to-DB migration). Only
        # the per-league df — the all_leagues table is implicit in the DB
        # ("SELECT WHERE competition_id = X" or no filter = all pool). Wrapped
        # in try/except so a DB failure doesn't break the parquet-based flow
        # during the dual-write validation window.
        try:
            from app.repository.projection_dataset_repo import insert_model_dataset_async
            await insert_model_dataset_async(
                model_dataset_league, league_id, league,
                teams, fixtures_df, comp_teams,
            )
        except Exception as _db_err:
            logger.warning(f"[{league}] model_dataset DB dual-write failed: {_db_err}")

        # model_dataset_league.to_excel(rf"{data_folder_path}\{league}_model_dataset_with_history.xlsx", index=False)
        # model_dataset_all.to_excel(rf"{data_folder_path}\all_leagues_model_dataset_with_history.xlsx", index=False)

        team_projections.drop(
            columns=['Team ' + stat + ' History' for stat in stat_list] + ['Opponent ' + stat + ' History Against' for
                                                                           stat in stat_list], inplace=True)

        # In[ ]:

        avg_goals = (avg_home_goals + avg_away_goals) / 2

        league_team_stats = team_stats[
            team_stats['fixture_id'].isin(fixtures_df[fixtures_df['competition_id'] == league_id]['id'])]

        league_shots = league_team_stats[league_team_stats['stats_type_id'] == get_stat_id('Shots Total',
                                                                                           stats_types)].copy()  # NEW - all team shots for specific league
        league_shots['Date'] = league_shots['fixture_id'].map(
            fixtures_df.set_index('id')['kickoff_datetime'])  # NEW - map fixture dates
        league_shots['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(
            league_shots['Date'])).dt.days // 7  # NEW - calculate weeks since kickoff
        league_shots['Weight'] = 0.9 ** (
                    league_shots['Weeks Since Kickoff'] - 5)  # NEW - apply weighting to more recent matches
        league_shots.loc[league_shots['Weeks Since Kickoff'] < 6, 'Weight'] = 1  # NEW - full weight for last 5 weeks
        league_shots['Weighted Shots'] = league_shots['Weight'] * league_shots[
            'value']  # NEW - calculate weighted shots
        avg_shots = league_shots['Weighted Shots'].sum() / league_shots[
            'Weight'].sum()  # UPDATED - new formula for average shots

        league_shots_on_target = league_team_stats[league_team_stats['stats_type_id'] == get_stat_id('Shots On Target',
                                                                                                     stats_types)].copy()  # NEW - all team shots on target for specific league
        league_shots_on_target['Date'] = league_shots_on_target['fixture_id'].map(
            fixtures_df.set_index('id')['kickoff_datetime'])  # NEW - map fixture dates
        league_shots_on_target['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(
            league_shots_on_target['Date'])).dt.days // 7  # NEW - calculate weeks since kickoff
        league_shots_on_target['Weight'] = 0.9 ** (
                    league_shots_on_target['Weeks Since Kickoff'] - 5)  # NEW - apply weighting to more recent matches
        league_shots_on_target.loc[
            league_shots_on_target['Weeks Since Kickoff'] < 6, 'Weight'] = 1  # NEW - full weight for last 5 weeks
        league_shots_on_target['Weighted Shots On Target'] = league_shots_on_target['Weight'] * league_shots_on_target[
            'value']  # NEW - calculate weighted shots on target
        avg_shots_on_target = league_shots_on_target['Weighted Shots On Target'].sum() / league_shots_on_target[
            'Weight'].sum()  # UPDATED - new formula for average shots on target

        avg_shots_per_goal = avg_shots / avg_goals
        avg_shots_on_target_per_goal = avg_shots_on_target / avg_goals

        # In[ ]:

        # if 'team_projections' in globals():
        goals = []
        assists = []
        for i in range(len(team_projections)):
            team = team_projections['Team'].iloc[i]
            opp = team_projections['Opponent'].iloc[i]
            # try:
            #    team_pred = score_preds[score_preds['Home Team'] == team]['Home Goals'].values[0]
            # except:
            #    team_pred = score_preds[score_preds['Away Team'] == team]['Away Goals'].values[0]
            fixture = score_preds[score_preds['id'] == team_projections['fixture_id'].iloc[
                i]]  # NEW - Get the fixture from score_preds
            team_pred = fixture['Home Goals'].values[0] if fixture['Home Team'].values[0] == team else \
            fixture['Away Goals'].values[
                0]  # UPDATED - new way to get team prediction that handles teams having multiple matches in a round
            opp_pred = fixture['Away Goals'].values[0] if fixture['Home Team'].values[0] == opp else \
            fixture['Home Goals'].values[
                0]  # UPDATED - new way to get opponent prediction that handles teams having multiple matches in a round
            goals.append(team_pred)
            assists.append((team_pred * 0.82).round(2))
            projected_shots = team_projections['Shots Total'].iloc[i]
            projected_shots_on_target = team_projections['Shots On Target'].iloc[i]

            adjusted_shots, adjusted_shots_on_target = adjust_shots_projection(
                team_pred,
                projected_shots,
                projected_shots_on_target,
                avg_shots_per_goal,
                avg_shots_on_target_per_goal
            )
            team_projections.at[i, 'Shots Total'] = adjusted_shots
            team_projections.at[i, 'Shots On Target'] = adjusted_shots_on_target

        team_projections['Goals'] = goals
        team_projections['Assists'] = assists

        # PL only: project team-level Ball Recovery + CBI(FPL) per fixture.
        # No PoissonRegressor exists for these stats (Sportmonks contributes
        # zero team-level rows); use get_simple_team_stat_prediction's
        # closed-form opponent-adjusted weighted average.
        # distribute_team_predictions_to_players auto-projects per-player
        # values from any column on team_projections, so adding these here
        # gives us per-player Recoveries + CBI for the team-down CBIT calc.
        if fpl:
            _lw_def = [league_above_attack_weight, league_above_defense_weight,
                       league_below_attack_weight, league_below_defense_weight]
            _sid_def = [current_season_id, previous_season_id,
                        previous_season_id_above, previous_season_id_below]
            _cpl_def = comp_teams[comp_teams['competition_id'] == league_id]
            _rec_col = []
            _cbi_col = []
            _cbit_col = []
            for i in range(len(team_projections)):
                _row = team_projections.iloc[i]
                try:
                    rec_v, _, _ = get_simple_team_stat_prediction(
                        _row['Team'], _row['Opponent'], fixtures_df, 'Ball Recovery',
                        team_stats, teams, stats_types,
                        ratings=ratings, venue=_row['Venue'], comp_id=league_id,
                        league_weightings=_lw_def, season_id=_sid_def, games=50,
                        comp_teams=_cpl_def,
                    )
                except Exception:
                    rec_v = 0
                try:
                    cbi_v, _, _ = get_simple_team_stat_prediction(
                        _row['Team'], _row['Opponent'], fixtures_df,
                        'Clearances Blocks Interceptions (FPL)',
                        team_stats, teams, stats_types,
                        ratings=ratings, venue=_row['Venue'], comp_id=league_id,
                        league_weightings=_lw_def, season_id=_sid_def, games=50,
                        comp_teams=_cpl_def,
                    )
                except Exception:
                    cbi_v = 0
                # Combined CBIT — the quantity FPL scores the threshold
                # against. Projected as ONE stat rather than assembling a
                # modelled Tackles with a blended CBI lump: FPL and Sportmonks
                # agree on the total to 0.3%, so one definition costs nothing
                # and removes the three inconsistent notions the pipeline
                # carried. George, 2026-08-04.
                try:
                    cbit_v, _cbit_th, _cbit_oh = get_simple_team_stat_prediction(
                        _row['Team'], _row['Opponent'], fixtures_df,
                        'Clearances Blocks Interceptions Tackles (FPL)',
                        team_stats, teams, stats_types,
                        ratings=ratings, venue=_row['Venue'], comp_id=league_id,
                        league_weightings=_lw_def, season_id=_sid_def, games=50,
                        comp_teams=_cpl_def,
                    )
                except Exception:
                    cbit_v = 0
                    _cbit_th = _cbit_oh = None
                # Diagnostic for the ~5% under-projection measured 2026-08-05
                # (projected 51.88 CBIT/match against an actual 55.59; 13 of 15
                # clubs low, and NOT explained by opponent mix or missing
                # history). The blend is
                #     alpha * team_history + (1-alpha) * opponent_history
                # and both terms are returned but discarded, so log them for a
                # handful of fixtures to see which side is light before
                # guessing at the cause.
                if i < 6:
                    logger.info(
                        f"[{league}] CBIT blend probe: {_row['Team']} vs {_row['Opponent']} "
                        f"({_row['Venue']}) team_hist={_cbit_th} opp_hist={_cbit_oh} "
                        f"-> blended={cbit_v}"
                    )
                _rec_col.append(rec_v)
                _cbi_col.append(cbi_v)
                _cbit_col.append(cbit_v)
            team_projections['Ball Recovery'] = _rec_col
            team_projections['Clearances Blocks Interceptions (FPL)'] = _cbi_col
            team_projections['Clearances Blocks Interceptions Tackles (FPL)'] = _cbit_col
            # Tackles Won rides a team TACKLES total, and that total is
            # CBIT - CBI rather than the modelled 'Tackles' column. Measured
            # against 760 real PL team-matches: the model averages 14.37 against
            # an actual 16.69 (-13.9%), while CBIT - CBI gives 15.74 (-5.7%) —
            # the two blended quantities are uniformly ~7% light, so their
            # DIFFERENCE tracks tackles better than the dedicated model does.
            # George called it before it was measured, 2026-08-05.
            #
            # It also makes BPS reconcile with DefCon by construction: BPS scores
            # CBI + tackles = CBI + (CBIT - CBI) = CBIT, the exact quantity the
            # DefCon threshold uses. Projecting the two separately had them
            # disagreeing by 10-14% on promoted clubs (Coventry 58.05 vs 51.00).
            #
            # The SHARE is unchanged and still measured from history — his
            # tackles won over his team's actual tackles (_TEAM_DENOMINATOR_ALIAS
            # maps the denominator to 'Tackles'), so a player's personal success
            # rate is folded in automatically.
            _tkl_from_cbit = (team_projections['Clearances Blocks Interceptions Tackles (FPL)']
                              - team_projections['Clearances Blocks Interceptions (FPL)'])
            # A fixture where the CBI blend outruns the CBIT blend would give a
            # negative tackle count; fall back to the modelled column there.
            team_projections['Tackles Won'] = _tkl_from_cbit.where(
                _tkl_from_cbit > 0, team_projections['Tackles'])
            # Split the goal projection into penalty and non-penalty halves.
            #
            # Penalties are taken as a fixed PROPORTION of projected goals, not
            # a flat per-match constant, so an attacking team is projected more
            # of them (George's call, 2026-08-07). Note the measured
            # correlation between a team's goals and its penalties across PL
            # 25/26 was 0.029 — i.e. none — with Brentford winning 10 on 55
            # goals against Man City's 4 on 77. George's position is that the
            # mechanism is real and one season of 2-10 counts is too noisy to
            # disprove it; the risk is City runs high and Brentford low.
            #
            # Constants measured over PL 24/25 + 25/26 (175 penalties):
            # penalties are 6.76% of goals, converted at 83.4%.
            # A proportional split cannot leak or invent goals — the two halves
            # always sum back to the original projection.
            _pg = team_projections['Goals'] * PENALTY_GOAL_SHARE
            team_projections['Non-Penalty Goals'] = team_projections['Goals'] - _pg
            # Named 'Penalties Scored' rather than 'Penalty Goals' so it matches
            # the PLAYER stat of the same name (111). distribute() resolves a
            # share for every team column by looking up the identically-named
            # player stat, so a column with no player counterpart hands every
            # player a zero share. Attempts and misses derive from this
            # downstream (attempts = scored / conversion), avoiding a second
            # distributed column that would need its own share.
            team_projections['Penalties Scored'] = _pg

        saves = []
        for i in range(len(team_projections)):
            # opp = team_projections['Opponent'].iloc[i]
            # try:
            #    opp_pred = score_preds[score_preds['Home Team'] == opp]['Home Goals'].values[0]
            # except:
            #    opp_pred = score_preds[score_preds['Away Team'] == opp]['Away Goals'].values[0]
            # saves.append(team_projections[team_projections['Team'] == opp]['Shots On Target'].values[0] - opp_pred)
            fixture_id = team_projections['fixture_id'].iloc[i]  # NEW - Get fixture ID
            fixture_team_projections = team_projections[
                team_projections['fixture_id'] == fixture_id]  # NEW - Get both teams' projections for the fixture
            fixture_team_projections = fixture_team_projections.drop(
                i)  # NEW - Drop the current team to get the opponent projections
            saves.append(
                fixture_team_projections['Shots On Target'].values[0] - fixture_team_projections['Goals'].values[
                    0])  # UPDATED - New way to calculate saves based on opponent projections that handles teams having multiple matches in a round

        team_projections['Saves'] = saves
        team_projections['Saves'] = team_projections['Saves'].round(2)  # NEW - Round saves to 2 decimal places
        # PL projects Key Passes properly (get_stat_list). Everywhere else it
        # stays derived — measured 0.72-0.74 across the top 5, so 0.75 runs a
        # little high. George, 2026-08-02.
        if 'Key Passes' not in team_projections.columns:
            team_projections['Key Passes'] = (team_projections['Shots Total'] * 0.75).round(2)
        # Retain Ball Recovery + CBI(FPL) columns when present (added by the
        # PL-only block above). Other leagues skip these columns.
        _extra_def_cols = [c for c in ['Ball Recovery', 'Clearances Blocks Interceptions (FPL)', 'Clearances Blocks Interceptions Tackles (FPL)', 'Tackles Won', 'Non-Penalty Goals', 'Penalties Scored']
                           if c in team_projections.columns]
        team_projections = team_projections[
            ['fixture_id', 'kickoff_datetime', 'Team', 'Opponent', 'Venue', 'Goals', 'Assists',
             'Key Passes'] + [c for c in stat_list if c != 'Key Passes'] + ['Fouls Drawn', 'Saves'] + _extra_def_cols]
        team_projections.rename(columns={'Successful Passes': 'Accurate Passes'}, inplace=True)
        logger.debug(f"[{league}] team_projections columns ready")

        # ── Team-stat odds-blend ──
        # Reels each team's projected stats (corners/cards/shots/SoT/
        # fouls/tackles) toward bookmaker expected via the cascade
        # (Path 1 per-team ladder → Path 1.5 partial+match → Path 2
        # match-split via model ratio). Per-stat bookmaker priority
        # lists in TEAM_STAT_BOOKIE_PRIORITY. Falls back to model
        # unchanged for any (stat, fixture) with no usable book data.
        from app.services.odds_blend import (
            load_team_stat_odds, blend_team_stat,
            TEAM_STAT_BOOKIE_PRIORITY, STAT_COLUMN_TO_MARKET,
        )
        _fix_ids = team_projections['fixture_id'].astype(int).unique().tolist()
        _odds_per_market = {}
        _odds_conn = await get_source_connection()
        try:
            for _market, _books in TEAM_STAT_BOOKIE_PRIORITY.items():
                _odds_per_market[_market] = await load_team_stat_odds(
                    _odds_conn, _fix_ids, _market, _books,
                )
        finally:
            release_source_connection(_odds_conn)

        _fid_to_home_team = {}
        for _fid in _fix_ids:
            _row = next_fix[next_fix['id'] == _fid]
            if not _row.empty:
                _fid_to_home_team[_fid] = _row['home_team'].iloc[0]

        _seen_fixtures = set()
        for _i in range(len(team_projections)):
            fid = int(team_projections['fixture_id'].iloc[_i])
            if fid in _seen_fixtures:
                continue
            _seen_fixtures.add(fid)
            pair = team_projections[team_projections['fixture_id'] == fid]
            if len(pair) != 2:
                continue
            home_team_name = _fid_to_home_team.get(fid)
            if not home_team_name:
                continue
            home_mask = (team_projections['fixture_id'] == fid) & (team_projections['Team'] == home_team_name)
            away_mask = (team_projections['fixture_id'] == fid) & (team_projections['Team'] != home_team_name)

            for stat_col, market in STAT_COLUMN_TO_MARKET.items():
                if stat_col not in team_projections.columns:
                    continue
                try:
                    mh = float(team_projections.loc[home_mask, stat_col].iloc[0])
                    ma = float(team_projections.loc[away_mask, stat_col].iloc[0])
                except (IndexError, KeyError, ValueError):
                    continue
                fh, fa = blend_team_stat(
                    mh, ma,
                    _odds_per_market.get(market, {}).get(fid, {}),
                    market, odds_beta,
                )
                team_projections.loc[home_mask, stat_col] = round(fh, 2)
                team_projections.loc[away_mask, stat_col] = round(fa, 2)
        
        # print(team_projections['Assists', 'Key Passes'])
        # In[ ]:

        # team_projections_save = team_projections.copy()
        # team_projections_save.drop(['Assists', 'Fouls Drawn', 'Saves', 'Key Passes'], axis=1,
        #                            inplace=True)  # UPDATED - No longer dropping interceptions and accurate passes

        team_projections_save = team_projections.copy()
        
        team_projections_save.drop(
            ['Assists', 'Fouls Drawn', 'Saves', 'Key Passes'],
            axis=1,
            inplace=True,
            errors='ignore'  # <- ovo sprečava KeyError ako kolona ne postoji
        )

        team_projections_save = team_projections_save.round(2)

        # team_projections_save.to_csv(rf"{save_file_path}\{league} Team.csv", index=False)
        await insert_teams_async(team_projections_save, teams=teams, competition_id=league_id, comp_teams=comp_teams)

        team_projections_save.rename(columns={'Accurate Passes': 'Successful Passes'},
                                     inplace=True)  # NEW - Rename back for consistency with other datasets

        # In[ ]:

        ## NEW - Update projection accuracy dataset

        for fixture_id in team_projections_save['fixture_id'].unique():
            fixture_projections = team_projections_save[team_projections_save['fixture_id'] == fixture_id]
            # accuracy dataset has no columns for the PL-only stats
            for stat in accuracy_stat_list(stat_list):
                projection_accuracy_dataset_league.loc[
                    projection_accuracy_dataset_league['fixture_id'] == fixture_id, 'Home Projected ' + stat] = \
                fixture_projections.loc[fixture_projections['Venue'] == 'H', stat].values[0]
                projection_accuracy_dataset_league.loc[
                    projection_accuracy_dataset_league['fixture_id'] == fixture_id, 'Away Projected ' + stat] = \
                fixture_projections.loc[fixture_projections['Venue'] == 'A', stat].values[0]
                projection_accuracy_dataset_league.loc[
                    projection_accuracy_dataset_league['fixture_id'] == fixture_id, 'Total Projected ' + stat] = \
                fixture_projections[stat].sum()

        projection_accuracy_dataset_league.drop_duplicates(subset=['fixture_id'], keep='last', inplace=True)
        projection_accuracy_dataset_league.reset_index(drop=True, inplace=True)
        # projection_accuracy_dataset_league.to_excel(rf"{data_folder_path}\{league}_accuracy_dataset.xlsx", index=False)
        ProjectionService._write_df(projection_accuracy_dataset_league, f"{data_folder_path}/{league}_accuracy_dataset")

        # Dual-write to DB (Phase 2 of data-files-to-DB migration).
        try:
            from app.repository.projection_dataset_repo import insert_accuracy_dataset_async
            await insert_accuracy_dataset_async(
                projection_accuracy_dataset_league, league_id, league,
                teams, fixtures_df, comp_teams,
            )
        except Exception as _db_err:
            logger.warning(f"[{league}] accuracy_dataset DB dual-write failed: {_db_err}")

        projection_accuracy_dataset_all = pd.concat(
            [projection_accuracy_dataset_all, projection_accuracy_dataset_league], ignore_index=True)
        projection_accuracy_dataset_all.drop_duplicates(subset=['fixture_id'], keep='last', inplace=True)
        projection_accuracy_dataset_all.reset_index(drop=True, inplace=True)
        # projection_accuracy_dataset_all.to_excel(rf"{data_folder_path}\all_leagues_accuracy_dataset.xlsx", index=False)
        ProjectionService._write_df(projection_accuracy_dataset_all, f"{data_folder_path}/all_leagues_accuracy_dataset")

        #
        # # **Player Projections**
        #
        # Distributing the above dataframe's values to each player based on the % of teams total

        # In[ ]:

        # UPDATED: Removed xG parameter, added comps parameter and added season_id paramter

        logger.debug(f"[{league}] season_ids: {[current_season_id, previous_season_id, previous_season_id_above, previous_season_id_below]}")

        logger.info(f"[{league}] Starting player projections...")
        _t = time.time()
        # Pre-load confirmed XI per (fixture, team). distribute drops
        # bench rows when a key matches — fires for per-fixture lineup
        # reruns AND for league runs where one upcoming fixture has
        # had its XI confirmed.
        #
        # Also pre-load player-prop odds for the same fixture batch.
        # distribute blends model λ toward bookie λ for v1 stats
        # (Goals / Shots Total / Shots On Target) at α=odds_beta.
        from app.services.odds_blend import (
            load_confirmed_lineups, load_player_odds,
            PLAYER_BLEND_BOOKS, PLAYER_BLEND_STAT_IDS,
        )
        _pl_fix_ids = next_fix['id'].astype(int).unique().tolist()
        _ll_conn = await get_source_connection()
        try:
            _confirmed_lineups = await load_confirmed_lineups(_ll_conn, _pl_fix_ids)
            _odds_for_fixture_players = await load_player_odds(
                _ll_conn, _pl_fix_ids, PLAYER_BLEND_STAT_IDS, PLAYER_BLEND_BOOKS,
            )
        finally:
            release_source_connection(_ll_conn)
        # Per-90 capture (xmins-methodology §11 Task 1): PL full runs only.
        # FPL_PER90_WRITE / FPL_PER90_POINTS both removed 2026-08-03.
        # FPL_PER90_POINTS=0 selected the legacy exposure scaling as a
        # rollback, but the bonus simulator reads the "{stat} per90" columns
        # that only apply_per90_scaling stamps — so setting it to 0 would have
        # produced silently meaningless bonus rather than the old behaviour.
        # A kill switch that no longer kills cleanly is worse than none.
        _per90_collector = (
            [] if league_id == 8 else None
        )
        pl_projections = distribute_team_predictions_to_players(player_stats, team_stats, team_projections, stats_types,
                                                                fixtures_df, players, teams, comps, 0.97,
                                                                season_id=[current_season_id, previous_season_id,
                                                                           previous_season_id_above,
                                                                           previous_season_id_below],
                                                                competition_id=league_id, comp_teams=comp_teams,
                                                                confirmed_lineups=_confirmed_lineups,
                                                                odds_for_fixture_players=_odds_for_fixture_players,
                                                                odds_blend_weight=odds_beta,
                                                                per90_collector=_per90_collector)
        logger.info(f"[{league}] Player projections computed - {len(pl_projections)} players ({time.time()-_t:.1f}s)")
        if _per90_collector:
            from app.repository.fpl_per90_repo import insert_per90_shares_async
            try:
                await insert_per90_shares_async(_per90_collector, league_id)
            except Exception as _p90_err:
                # Isolated by design — a per-90 write failure must never
                # damage the live projection run.
                logger.warning(f"[{league}] per90 share write failed (non-fatal): {_p90_err}")

        # Vectorized: build player lookup, merge, derive Position/Saves AND Start? in one pass
        _team_names = teams[['id', 'name']].rename(columns={'id': '_team_id', 'name': 'Team'})
        _player_lookup = players.merge(
            _team_names, left_on='current_team_id', right_on='_team_id', how='left'
        )[['display_name', 'Team', 'id', '_team_id', 'position']].rename(
            columns={'display_name': 'Player', 'id': '_player_id'}
        ).drop_duplicates(subset=['Player', 'Team'])

        pl_projections = pl_projections.merge(_player_lookup, on=['Player', 'Team'], how='left')

        _pos_map = {'goalkeeper': 'GK', 'defender': 'DEF', 'midfielder': 'MID', 'attacker': 'FWD'}
        # Final .fillna('Unknown') catches players whose Sportmonks row has
        # NULL position (3 Allsvenskan players hit this 2026-05-28). The
        # downstream player_prop_projections.position column is NOT NULL,
        # so leaving NaN here propagates through to the SQL insert and
        # kills the whole league's projection run.
        pl_projections['Position'] = pl_projections['position'].map(_pos_map).fillna(pl_projections['position']).fillna('Unknown')
        pl_projections.loc[pl_projections['Player'] == 'Caoimhin Kelleher', 'Position'] = 'GK'

        pl_projections['Saves'] = 0
        _team_saves = team_projections[['fixture_id', 'Team', 'Saves']].rename(columns={'Saves': '_gk_saves'})
        pl_projections = pl_projections.merge(_team_saves, on=['fixture_id', 'Team'], how='left')
        _gk_mask = pl_projections['Position'] == 'GK'
        pl_projections.loc[_gk_mask, 'Saves'] = pl_projections.loc[_gk_mask, '_gk_saves'].fillna(0)
        pl_projections.drop(columns=['_gk_saves'], inplace=True)

        # Predicted starters (was a separate row-by-row loop further down — moved here so it runs
        # before the column reorder strips _team_id and _player_id).
        # Old loop also had a bug: get_player_id was called with 3 args instead of 4, raising
        # TypeError silently swallowed by bare except — every player got 'No'. Now fixed.
        _pred_starters = player_stats[player_stats['fixture_id'].isin(next_fix['id'])]
        _pred_starters = _pred_starters[_pred_starters['stats_type_id'] == 11]
        _starter_pairs = set(zip(
            _pred_starters['team_id'].astype('Int64'),
            _pred_starters['player_id'].astype('Int64')
        ))
        pl_projections['Start?'] = [
            'Yes' if (pd.notna(t) and pd.notna(p) and (int(t), int(p)) in _starter_pairs) else 'No'
            for t, p in zip(pl_projections['_team_id'], pl_projections['_player_id'])
        ]
        pl_projections.drop(columns=['_player_id', '_team_id', 'position'], inplace=True, errors='ignore')

        # PL only: retain Ball Recovery + CBI(FPL) team-down columns through
        # the explicit column filter so the team-down CBIT post-pass below
        # can read them. distribute_team_predictions_to_players propagated
        # them from team_projections via pivot; without this they'd be
        # dropped here and the post-pass would compute hit rate on Tackles
        # alone (giving ~0% for everyone).
        _def_extra = [c for c in FPL_DEF_EXTRA_COLS if c in pl_projections.columns]
        pl_projections = pl_projections[
            ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue',
             'Start?',
             'Assists', 'Key Passes', 'Accurate Passes', 'Goals',
             'Shots Total',
             'Shots On Target',  'Passes',  'Interceptions', 'Tackles', 'Total Crosses',
             'Yellowcards', 'Offsides', 'Fouls', 'Fouls Drawn', 'Saves'] + _def_extra]

        pl_projections.rename(columns={'Yellowcards': 'Yellow Cards'}, inplace=True)

        # ## **Predicted Lineups**
        #
        # Which players are predicted to play?

        # In[ ]:

        logger.info(f"[{league}] Player projections: {len(pl_projections)} rows")
        _def_extra2 = [c for c in FPL_DEF_EXTRA_COLS if c in pl_projections.columns]
        pl_projections = pl_projections[
            ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue', 'Start?', 'Shots Total',
              'Goals', 'Assists', 'Key Passes', 'Accurate Passes',
             'Shots On Target', 'Passes', 'Interceptions', 'Tackles', 'Total Crosses',
             'Yellow Cards', 'Offsides', 'Fouls', 'Fouls Drawn', 'Saves'] + _def_extra2]
        pl_projections = pl_projections.round(2)

        # In[ ]:

        # pl_projections.sort_values(by='Goals', ascending=False, inplace=True)
        pl_projections.reset_index(drop=True, inplace=True)
        pl_projections = pl_projections.round(2)
        # pl_projections.to_csv(rf"{save_file_path}\{league} Player.csv", index=False)
        logger.info(f"[{league}] Inserting player projections into DB ({len(pl_projections)} rows)...")
        _t = time.time()
        await insert_player_async(pl_projections, teams=teams, competition_id=league_id, comp_teams=comp_teams)
        logger.info(f"[{league}] Player projections inserted ({time.time()-_t:.1f}s)")

        # ## **FPL / OPTA / FanTeam Points** (Premier League only)
        # Mirrors the block in projection_all_teams_service.py so daily
        # scheduled PL projections (which go through this single-league
        # path via /api/projections) write fresh fpl_projections /
        # opta_projections / fanteam_projections rows. Previously these
        # tables only updated when someone manually clicked "Run All
        # Leagues" — silently broken on the daily schedule for months.

        # Fantasy projections must only cover gameweeks still open to plan
        # for — those whose deadline is in the future. Load each in-window
        # gameweek's deadline up front, derive the 6 soonest future-deadline
        # gameweeks, and filter the five fantasy frames to them before insert
        # (see _fantasy_gw_filter). player / team / fixture projections are
        # deliberately unaffected — they keep the current GW's unplayed
        # fixtures. _fantasy_upcoming_gws stays None on lookup failure so the
        # filter fails open rather than blanking the fantasy tables.
        _fantasy_now = pd.Timestamp.utcnow().tz_localize(None)
        _fantasy_upcoming_gws = None
        if fpl:
            try:
                _gw_deadlines = await ProjectionService._load_gameweek_deadlines(
                    fixtures.loc[fixtures['id'].isin(next_fix['id']), 'gameweek_id']
                )
                _fantasy_upcoming_gws = sorted(
                    gw for gw, dl in _gw_deadlines.items()
                    if pd.notna(dl) and dl > _fantasy_now
                )[:ProjectionService.FANTASY_GAMEWEEKS]
                logger.info(f"[{league}] fantasy projections scoped to gameweeks "
                            f"{_fantasy_upcoming_gws} (future-deadline only)")
            except Exception as e:
                logger.warning(f"[{league}] gameweek-deadline load failed — fantasy "
                               f"frames left unfiltered: {e}", exc_info=True)

        if fpl:
            try:
                # FPL position now sourced from fpl_player_mappings table
                # (Laravel DB) instead of PL Fantasy Players.xlsx. Joining
                # by player_id is FAR more reliable than name-matching —
                # no more fragile string matches on accents/initials/etc.
                # The xlsx is still used by the FanTeam block below for
                # FanTeam Position (which isn't in fpl_player_mappings).
                pl_projections['Player'] = pl_projections['Player'].str.strip()
                fpl_mappings = ProjectionService._current_source.fpl_player_mappings
                if fpl_mappings is None or fpl_mappings.empty:
                    raise RuntimeError("fpl_player_mappings reference table empty — check loader")
                _pos_by_pid = (
                    fpl_mappings
                    .drop_duplicates(subset=['player_id'])
                    .set_index('player_id')['fpl_element_type']
                    .map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'})
                )
                pl_projections['FPL Position'] = pl_projections['player_id'].map(_pos_by_pid)

                # FPL gate = FPL membership (George, 2026-07-30): every
                # FPL-mapped squad player gets FPL rows even when the
                # appearance gate (player_criteria) excluded him — backup
                # GKs, deep bench. Second distribute pass for the missing
                # ids only; xMins prices their minutes so per-start shares
                # are safe. FPL-LOCAL: extras join _fpl_frame below, never
                # pl_projections — props/Opta/site paths keep the old gate.
                _fpl_extras = None
                _p90_mark = len(_per90_collector) if _per90_collector is not None else 0
                try:
                    _mapped_ids = set(fpl_mappings['player_id'].dropna().astype(int))
                    _existing_ids = set(pl_projections['player_id'].dropna().astype(int))
                    _missing_ids = _mapped_ids - _existing_ids
                    if _missing_ids:
                        # FPL's club assignment is authoritative for the
                        # extras pass: players.current_team_id goes stale on
                        # transfers/promotions, and a wrong club means the
                        # team loop never reaches the player (40 dialed
                        # players had no bundles, 2026-07-30).
                        _players_fpl = players.copy()
                        if 'fpl_club_team_id' in fpl_mappings.columns:
                            _club_by_pid = (fpl_mappings.dropna(subset=['fpl_club_team_id'])
                                            .drop_duplicates('player_id')
                                            .set_index('player_id')['fpl_club_team_id'])
                            _fix_mask = _players_fpl['id'].isin(_missing_ids)
                            _players_fpl.loc[_fix_mask, 'current_team_id'] = (
                                _players_fpl.loc[_fix_mask, 'id'].map(_club_by_pid)
                                .fillna(_players_fpl.loc[_fix_mask, 'current_team_id'])
                            )
                        _fpl_extras = distribute_team_predictions_to_players(
                            player_stats, team_stats, team_projections, stats_types,
                            fixtures_df, _players_fpl, teams, comps, 0.97,
                            season_id=[current_season_id, previous_season_id,
                                       previous_season_id_above, previous_season_id_below],
                            competition_id=league_id, comp_teams=comp_teams,
                            per90_collector=_per90_collector,
                            only_player_ids=_missing_ids)
                        if _fpl_extras is not None and len(_fpl_extras):
                            _fpl_extras['FPL Position'] = _fpl_extras['player_id'].map(_pos_by_pid)
                            _fpl_extras = _fpl_extras[_fpl_extras['FPL Position'].notna()]
                            # GK extras take the team's saves projection,
                            # mirroring the main path's GK assignment.
                            _ts = team_projections[['fixture_id', 'Team', 'Saves']].rename(columns={'Saves': '_tgs'})
                            _fpl_extras = _fpl_extras.merge(_ts, on=['fixture_id', 'Team'], how='left')
                            if 'Saves' not in _fpl_extras.columns:
                                _fpl_extras['Saves'] = 0.0
                            _gk_m = _fpl_extras['FPL Position'] == 'GK'
                            _fpl_extras.loc[_gk_m, 'Saves'] = _fpl_extras.loc[_gk_m, '_tgs'].fillna(0)
                            _fpl_extras.drop(columns=['_tgs'], inplace=True)
                            for _c in pl_projections.columns:
                                if _c not in _fpl_extras.columns:
                                    _fpl_extras[_c] = 0
                            _fpl_extras = _fpl_extras[list(pl_projections.columns)]
                            logger.info(f"[{league}] FPL gate-bypass: {_fpl_extras['player_id'].nunique()} "
                                        f"FPL-mapped players added to the FPL frame")
                except Exception as _extras_err:
                    logger.warning(f"[{league}] FPL extras pass failed (non-fatal): {_extras_err}")
                    _fpl_extras = None
                # (_fpl_base built AFTER the enrichment below — 2026-07-30 fix:
                # building it here missed the extra-stats columns and crashed
                # the whole FPL stage with KeyError 'Clearances Average'.)

                # Compute extra stats per player (Clearances, Blocked Shots, Ball Recovery averages)
                for _col in ['CBIT Hit Rate', 'CBIT Average', 'Clearances Average', 'Blocked Shots Average',
                             'Ball Recovery Average', 'Tackles Won Average', 'Full Match Hit Rate']:
                    if _col not in pl_projections.columns:
                        pl_projections.loc[:, _col] = 0
                for _player in pl_projections['Player'].unique():
                    _team = pl_projections[pl_projections['Player'] == _player]['Team'].values[0]
                    _pos = pl_projections[pl_projections['Player'] == _player]['FPL Position'].values[0]
                    try:
                        _cbit, _cbit_avg, _clr, _blk, _rec, _twon, _fmhr = get_extra_stats(
                            _player, _pos, _team, teams, players, player_stats, fixtures_df, stats_types,
                            weight=0.96, mins=50, games=50,
                            competition_id=league_id, comp_teams=comp_teams)
                        _mask = (pl_projections['Player'] == _player) & (pl_projections['Team'] == _team)
                        pl_projections.loc[_mask, 'CBIT Hit Rate'] = _cbit
                        pl_projections.loc[_mask, 'CBIT Average'] = _cbit_avg
                        pl_projections.loc[_mask, 'Clearances Average'] = _clr
                        pl_projections.loc[_mask, 'Blocked Shots Average'] = _blk
                        pl_projections.loc[_mask, 'Ball Recovery Average'] = _rec
                        pl_projections.loc[_mask, 'Tackles Won Average'] = _twon
                        pl_projections.loc[_mask, 'Full Match Hit Rate'] = _fmhr
                    except Exception:
                        continue
                logger.info(f"[{league}] FPL: extra stats computed for {len(pl_projections['Player'].unique())} players")

                # Team-down CBIT Hit Rate — overrides the empirical hit rate
                # stamped above. Uses per-player Tackles + Ball Recovery +
                # CBI(FPL) projections that distribute_team_predictions_to_players
                # auto-projected from the team_projections columns we added
                # above. Sum per FPL position, apply Poisson SF for hit%.
                # PL-only by construction (we're inside `if fpl:`).
                from scipy.stats import poisson as _td_poisson, nbinom as _td_nbinom
                # CBIT counts are mildly OVERDISPERSED relative to Poisson:
                # measured variance/mean across 312 players with 10+ starts in
                # 25/26 is median 1.27, mean 1.34 (passes, by contrast, are
                # 4.15). Poisson therefore has too thin a tail and understates
                # hit rates near the threshold — the calibration check showed
                # predicted 24.6% against 30.2% actual in that bucket.
                # Negative binomial with the same mean and var = 1.27 x mean.
                # George, 2026-08-04.
                # Poisson retained (George, 2026-08-04). NegBinom at 1.27 IS
                # measurably better — mean predicted 19.4% vs actual 19.9% and
                # MAE 0.0392 against Poisson's 18.3% / 0.0409 across 312
                # players — but it fattens BOTH tails, so it pulls the top
                # DefCon players DOWN ~2.5pp (Hill 68.1 -> 65.4) while lifting
                # the low band. Set to 1.0 to keep Poisson; raise to 1.27 to
                # switch.
                _TD_DISPERSION = 1.0
                # Imported, not re-spelled: these columns only exist because
                # apply_per90_scaling stamps them, so the suffix must track it.
                from app.services.xminutes import (
                    PER90_SUFFIX as _TD_P90_SUFFIX,
                    DC_DIAL_RATE_COL as _TD_DC_DIAL_COL,
                    DC_RATE_COL as _TD_DC_RATE_COL,
                    defcon_band_hit_rate as _td_band_hit_rate,
                )
                def _td_safe(v):
                    if v is None or pd.isna(v):
                        return 0.0
                    return float(v)
                def _td_tail(total, threshold):
                    """P(X >= threshold) for a single minutes outcome."""
                    if total <= 0:
                        return 0.0
                    if _TD_DISPERSION <= 1.0001:
                        return float(_td_poisson.sf(threshold - 1, total))
                    # var = mean x d  ->  r = mean/(d-1), p = r/(r+mean)
                    _r = total / (_TD_DISPERSION - 1.0)
                    _p = _r / (_r + total)
                    return float(_td_nbinom.sf(threshold - 1, _r, _p))

                def _td_cbit_total(row, pos, suffix=''):
                    """CBIT (+ recoveries for MID/FWD) off the frame.

                    suffix='' reads lambda at expected minutes; ' per90' reads
                    the per-90 companion apply_per90_scaling stamps.

                    ONE combined CBIT quantity (999003) rather than adding a
                    modelled Tackles to a blended CBI lump — the pipeline used
                    to carry three inconsistent notions of the same thing.
                    Falls back to the old assembly when the combined column is
                    absent (non-PL, or a frame built before the overlay).
                    """
                    cbit = row.get('Clearances Blocks Interceptions Tackles (FPL)' + suffix)
                    if cbit is None or (isinstance(cbit, float) and pd.isna(cbit)):
                        cbit = _td_safe(row.get('Tackles' + suffix)) + _td_safe(
                            row.get('Clearances Blocks Interceptions (FPL)' + suffix))
                    else:
                        cbit = _td_safe(cbit)
                    if pos != 'DEF':
                        cbit += _td_safe(row.get('Ball Recovery' + suffix))
                    return cbit

                def _td_dc_rate90(row):
                    """Defensive-contribution rate per 90 — the quantity the
                    threshold is scored against, before minutes.

                    A defcon_share dial replaces it outright (replacement, not
                    delta — same contract as goal_share). apply_share_dials
                    writes that as team defensive-contribution total x dial, so
                    it already varies by fixture; minutes are applied by the
                    banding, not here.
                    """
                    pos = row.get('FPL Position')
                    if pos == 'GK' or pos is None or (isinstance(pos, float) and pd.isna(pos)):
                        return 0.0
                    _dialled = row.get(_TD_DC_DIAL_COL)
                    if _dialled is not None and not (isinstance(_dialled, float) and pd.isna(_dialled)):
                        return float(_dialled)
                    rate = _td_cbit_total(row, pos, _TD_P90_SUFFIX)
                    if rate > 0:
                        return rate
                    # Provisional pre-xMinutes pass: no per-90 companions exist
                    # yet, so fall back to lambda at expected minutes. Overwritten
                    # by the post-scaling recompute, which is what ships.
                    return _td_cbit_total(row, pos)

                def _td_cbit_hit_rate(row):
                    pos = row.get('FPL Position')
                    if pos == 'GK' or pos is None or (isinstance(pos, float) and pd.isna(pos)):
                        return 0.0
                    threshold = 10 if pos == 'DEF' else 12
                    # Evaluate the threshold PER MINUTES BAND and weight by the
                    # band probabilities, rather than collapsing minutes into a
                    # single lambda first.
                    #
                    # Reaching 10 CBIT is a tail event, and the tail is convex
                    # in lambda, so E[P(hit | minutes)] != P(hit | E[minutes]) —
                    # averaging first invents a "53-minute Greaves" who never
                    # takes the pitch. He either starts and has a real chance,
                    # or he doesn't play and has none. Measured on the
                    # non-dialled rows of the 2026-08-04 run, mean DefCon rose
                    # 9.9% -> 17.7%, concentrated entirely on rotation risks:
                    # Bentancur (p90 0.19) 1.2% -> 25.6%, while a nailed starter
                    # like Thiaw (p90 0.91) moved 30.1% -> 34.7%. That split is
                    # the check the maths is behaving — with one outcome to
                    # average, both methods must agree.
                    #
                    # Same treatment fpl_bonus_sim already uses (it draws a band
                    # then draws actions conditional on it); DefCon was the odd
                    # one out. Bands are read post-dial and post-availability
                    # flag, so a dialled player bands on the value George set.
                    # George, 2026-08-04.
                    # A defcon_share dial replaces the assembled rate outright
                    # (replacement, not delta — same contract as goal_share).
                    # apply_share_dials writes it as team defensive-contribution
                    # total x dial, so it already varies by fixture; minutes are
                    # still applied by the banding below.
                    p_play = row.get('xmin_p_play')
                    p60 = row.get('xmin_p60')
                    p90 = row.get('xmin_p90')
                    if any(v is None or (isinstance(v, float) and pd.isna(v))
                           for v in (p_play, p60, p90)):
                        # Provisional pre-xMinutes pass: no bands on the frame
                        # yet, so score the single-point lambda. Overwritten by
                        # the post-scaling recompute, which is what ships.
                        return _td_tail(_td_cbit_total(row, pos), threshold)
                    return _td_band_hit_rate(_td_dc_rate90(row), p_play, p60, p90,
                                             threshold, _TD_DISPERSION)
                pl_projections['CBIT Hit Rate'] = pl_projections.apply(_td_cbit_hit_rate, axis=1)
                # Phase 2 persistence: store the percentage form of CBIT
                # Hit Rate so the fantasy API can surface it as
                # `def_con_pct`. CBIT Hit Rate itself stays 0-1 because
                # downstream FPL scoring (statz_functions.py line 2078)
                # multiplies it by 2 to award the +2 defensive-contribution
                # bonus.
                pl_projections['def_con_pct'] = (pl_projections['CBIT Hit Rate'] * 100).round(2)
                logger.info(f"[{league}] FPL: CBIT Hit Rate replaced with team-down projection")

                fpl_points_dict_gk = {'Goals': 10, 'Assists': 3, 'Clean Sheet': 4, 'Saves': 1, 'Penalties Saved': 5, 'Goals Conceded': -1, 'Yellow Card': -1}
                fpl_points_dict_def = {'Goals': 6, 'Assists': 3, 'Clean Sheet': 4, 'Goals Conceded': -1, 'Yellow Card': -1}
                fpl_points_dict_mid = {'Goals': 5, 'Assists': 3, 'Clean Sheet': 1, 'Yellow Card': -1}
                fpl_points_dict_fwd = {'Goals': 4, 'Assists': 3, 'Yellow Card': -1}

                fpl_bonus_dict_gk = {'Goals': 12, 'Winning Goal': 3, 'Assists': 9, 'Clean Sheet': 12, 'Saves': 2.85, 'Penalties Saved': 7, 'Key Passes': 1, 'Big Chances Created': 3, 'Successful Dribbles': 1, 'Clearance Offline': 9, 'Big Chances Missed': -3, 'Clearances, Blocks & Interceptions': 0.333, 'Recoveries': 0.33, 'Tackles Won': 2, 'Fouls Drawn': 1, 'Shots On Target': 2, 'Shots Off Target': -1, 'Offsides': -1, 'Fouls': -1, '70-79% Passes Completed': 2, '80-89% Passes Completed': 4, '90%+ Passes Completed': 6, 'Goals Conceded': -4, 'Yellow Card': -3}
                fpl_bonus_dict_def = {'Goals': 12, 'Winning Goal': 3, 'Assists': 9, 'Clean Sheet': 12, 'Clearances, Blocks & Interceptions': 0.333, 'Recoveries': 0.33, 'Tackles Won': 2, 'Fouls Drawn': 1, 'Shots On Target': 2, 'Shots Off Target': -1, 'Offsides': -1, 'Fouls': -1, '70-79% Passes Completed': 2, '80-89% Passes Completed': 4, '90%+ Passes Completed': 6, 'Key Passes': 1, 'Big Chances Created': 3, 'Successful Dribbles': 1, 'Clearance Offline': 9, 'Big Chances Missed': -3, 'Goals Conceded': -4, 'Yellow Card': -3}
                fpl_bonus_dict_mid = {'Goals': 18, 'Winning Goal': 3, 'Assists': 9, 'Clearances, Blocks & Interceptions': 0.333, 'Recoveries': 0.33, 'Tackles Won': 2, 'Fouls Drawn': 1, 'Shots On Target': 2, 'Shots Off Target': -1, 'Offsides': -1, 'Fouls': -1, '70-79% Passes Completed': 2, '80-89% Passes Completed': 4, '90%+ Passes Completed': 6, 'Key Passes': 1, 'Big Chances Created': 3, 'Successful Dribbles': 1, 'Clearance Offline': 9, 'Big Chances Missed': -3, 'Yellow Card': -3}
                fpl_bonus_dict_fwd = {'Goals': 24, 'Winning Goal': 3, 'Assists': 9, 'Key Passes': 1, 'Big Chances Created': 3, 'Successful Dribbles': 1, 'Clearance Offline': 9, 'Big Chances Missed': -3, 'Clearances, Blocks & Interceptions': 0.333, 'Recoveries': 0.33, 'Tackles Won': 2, 'Fouls Drawn': 1, 'Shots On Target': 2, 'Shots Off Target': -1, 'Offsides': -1, 'Fouls': -1, '70-79% Passes Completed': 2, '80-89% Passes Completed': 4, '90%+ Passes Completed': 6, 'Yellow Card': -3}

                for _col in ['CBIT Hit Rate', 'Clearances Average', 'Blocked Shots Average', 'Interceptions', 'Ball Recovery Average']:
                    if _col not in pl_projections.columns:
                        pl_projections.loc[:, _col] = 0

                # per90 rows for the EXTRAS were collected AFTER the early insert
                # ran (post-main-distribute) — Kudus-class players computed shares
                # that never persisted (found 2026-07-30). Upsert the new slice.
                if _per90_collector is not None and len(_per90_collector) > _p90_mark:
                    try:
                        from app.repository.fpl_per90_repo import insert_per90_shares_async
                        await insert_per90_shares_async(_per90_collector[_p90_mark:], league_id)
                    except Exception as _p90x_err:
                        logger.warning(f"[{league}] extras per90 write failed (non-fatal): {_p90x_err}")
                _fpl_base = (pd.concat([pl_projections, _fpl_extras], ignore_index=True)
                             if _fpl_extras is not None and len(_fpl_extras) else pl_projections)
                for _c in ['CBIT Hit Rate', 'CBIT Average', 'Clearances Average', 'Blocked Shots Average', 'Ball Recovery Average', 'Tackles Won Average', 'Full Match Hit Rate', 'def_con_pct']:
                    if _c in _fpl_base.columns:
                        _fpl_base[_c] = _fpl_base[_c].fillna(0)
                    else:
                        _fpl_base[_c] = 0
                # ---- xMinutes (George, 2026-07-24) ----
                # Expected-minutes profile per player (p_play / p60 /
                # exposure vs the per-start sample his rates came from),
                # applied on an FPL-LOCAL copy: pl_projections itself must
                # stay untouched because Opta / FanTeam / Dream11 /
                # DraftKings below read its unscaled columns. With
                # FPL_XMINUTES=0 no columns are stamped and get_fpl_points /
                # bonus_points_score reproduce the legacy 90-minute output.
                from app.services.xminutes import (
                    XMIN_ENABLED as _xmin_enabled,
                    eligible_past_fixtures, build_minutes_frame,
                    get_expected_minutes, stamp_xmin_columns,
                    apply_per90_scaling,
                    apply_band_dials, apply_share_dials,
                    apply_availability_flags,
                )
                # Admin dials (§12 Phase 5) — standing per-player overrides,
                # loaded via the same DB-source pattern as promoted ratings.
                _fpl_dials = getattr(ProjectionService._current_source, 'fpl_player_dials', None)
                _fpl_frame = _fpl_base
                if _xmin_enabled:
                    try:
                        _t_xm = time.time()
                        _xm_past_fx = eligible_past_fixtures(fixtures_df, comps)
                        _xm_minutes = build_minutes_frame(player_stats, _xm_past_fx)
                        _xm_team_ids = {
                            _tn: get_team_id(_tn, teams, league_id, comp_teams)
                            for _tn in _fpl_base['Team'].unique()
                        }
                        _xm_profiles = {}
                        for _xm_pid, _xm_tname, _xm_pos in _fpl_base[
                                ['player_id', 'Team', 'FPL Position']
                        ].drop_duplicates('player_id').itertuples(index=False):
                            if pd.isna(_xm_pid):
                                continue
                            _xm_profiles[int(_xm_pid)] = get_expected_minutes(
                                int(_xm_pid), _xm_team_ids.get(_xm_tname),
                                _xm_minutes, _xm_past_fx, position=_xm_pos,
                            )
                        # FPL availability flags shape the bands (George,
                        # 2026-08-03: "keep them in but give them xMins
                        # relating to their flag"). BEFORE the bands persist,
                        # so the panel shows an honest model value, and BEFORE
                        # dials, so a manual override still wins.
                        try:
                            _flag_conn = await get_source_connection()
                            try:
                                async with _flag_conn.cursor() as _fc:
                                    await _fc.execute("""
                                        SELECT m.player_id, s.status, s.chance_of_playing_next_round
                                        FROM fpl_player_mappings m
                                        JOIN fpl_player_snapshots s ON s.fpl_id = m.fpl_id
                                         AND s.snapshot_date = (SELECT MAX(snapshot_date) FROM fpl_player_snapshots)
                                        WHERE m.player_id IS NOT NULL AND m.fpl_id IS NOT NULL
                                    """)
                                    _flags = {int(r[0]): (r[1], r[2]) for r in await _fc.fetchall()}
                            finally:
                                release_source_connection(_flag_conn)
                            _fz, _fs = apply_availability_flags(_xm_profiles, _flags)
                            logger.info(f"[{league}] FPL availability flags: {_fz} zeroed, {_fs} scaled "
                                        f"(from {len(_flags)} flagged records)")
                        except Exception as _flag_err:
                            logger.warning(f"[{league}] availability flags skipped: {_flag_err}")
                        # §12 Phase 0: persist standing bands (PRE confirmed-XI —
                        # _xm_profiles is built before any starter_override).
                        # Panel reads fpl_player_bands, never recomputes in PHP.
                        try:
                            from app.repository.fpl_per90_repo import insert_player_bands_async
                            await insert_player_bands_async(_xm_profiles, league_id)
                        except Exception as _band_err:
                            logger.warning(f"[{league}] player-bands write failed (non-fatal): {_band_err}")
                        # Pure-model profile copy for the assembly-bundle
                        # snapshot below: the recalc path applies CURRENT dial
                        # state from scratch on a MODEL bundle, so Reset-to-
                        # model restores instantly with post-edit model values
                        # (George's invariant, 2026-07-30). Copy BEFORE dials
                        # mutate the profiles.
                        _model_profiles = {pid: dict(p) for pid, p in _xm_profiles.items()}
                        # §12 Phase 5: band dials replace standing model bands
                        # AFTER the model-values persist above, BEFORE stamping.
                        # Confirmed-XI snap still wins at fixture level.
                        _n_dials = apply_band_dials(_xm_profiles, _fpl_dials)
                        if _n_dials:
                            logger.info(f"[{league}] FPL dials: bands replaced for {_n_dials} players")
                        _fpl_frame = _fpl_base.copy()
                        _fpl_frame = stamp_xmin_columns(_fpl_frame, _xm_profiles,
                                                        confirmed_xi=_confirmed_lineups)

                        # BPS static rates (George, 2026-08-03): successful
                        # dribbles + big chances missed are player traits, not
                        # a share of a team total, so they are rated per-90
                        # from the last two seasons rather than projected.
                        # Stamped AFTER stamp_xmin_columns because lambda =
                        # per90 x xmin_bands / 90 — which is also why these two
                        # must NOT be added to XMIN_SCALED_STAT_COLS, or the
                        # minutes term would be applied twice.
                        try:
                            from app.services.fpl_static_rates import (
                                compute_per90_rates, stamp_rate_columns,
                            )
                            _rate_seasons = [s for s in (current_season_id, previous_season_id) if s]
                            _rate_fx = set(
                                fixtures_df.loc[
                                    fixtures_df['season_id'].isin(_rate_seasons), 'id'
                                ].astype(int).tolist()
                            ) if _rate_seasons else None
                            _rate_specs = {
                                'Successful Dribbles': 'Successful Dribbles',
                                'Big Chances Missed': 'Big Chances Missed',
                            }
                            _rates = {}
                            for _col, _sname in _rate_specs.items():
                                # get_stat_id raises on an unknown name rather
                                # than returning None, so guard per-stat: one
                                # missing stat must not take the other with it.
                                try:
                                    _sid = get_stat_id(_sname, stats_types)
                                except Exception:
                                    logger.warning(f"[{league}] no stats_type for '{_sname}' — rate skipped")
                                    continue
                                _rates[_col] = compute_per90_rates(
                                    player_stats, _sid, _pos_by_pid, fixture_ids=_rate_fx,
                                )
                            if _rates:
                                _fpl_frame = stamp_rate_columns(_fpl_frame, _rates)
                        except Exception as _rate_err:
                            # Non-fatal: nothing consumes these columns until the
                            # BPS rebuild lands, and a run must not die for them.
                            logger.warning(f"[{league}] static rates skipped: {_rate_err}")
                        if _per90_collector:
                            # Per-90 path (George, 2026-07-29): λ = column ×
                            # xmin_bands ÷ that stat's own m̄ — i.e. team_proj
                            # × share90 × xMins/90, xMins from the three
                            # bands (§1-2). Supersedes exposure scaling.
                            # Collector stat names → frame column aliases.
                            _P90_ALIASES = {'Yellowcards': 'Yellow Cards'}
                            _m_bar_lookup = {
                                (r['player_id'], _P90_ALIASES.get(r['stat_name'], r['stat_name'])): r['m_bar']
                                for r in _per90_collector if r.get('m_bar')
                            }
                            _fpl_frame = apply_per90_scaling(_fpl_frame, _m_bar_lookup)
                            logger.info(f"[{league}] FPL per-90 points path ON — "
                                        f"{len(_m_bar_lookup)} (player, stat) m̄ terms")
                            # Prove the per-90 companions are on the LIVE frame
                            # rather than inferring it from a unit test. The
                            # bonus simulator reads these; a stat missing one
                            # contributes 0 to BPS. Identity checked on a real
                            # row: col == per90 x xmin_bands / 90.
                            _p90_cols = [c for c in _fpl_frame.columns if c.endswith(' per90')]
                            _p90_chk = 'n/a'
                            if _p90_cols and 'xmin_bands' in _fpl_frame.columns:
                                _c0 = _p90_cols[0][:-len(' per90')]
                                if _c0 in _fpl_frame.columns:
                                    _lhs = pd.to_numeric(_fpl_frame[_c0], errors='coerce')
                                    _rhs = (pd.to_numeric(_fpl_frame[_p90_cols[0]], errors='coerce')
                                            * pd.to_numeric(_fpl_frame['xmin_bands'], errors='coerce') / 90.0)
                                    _p90_chk = 'OK' if bool((_lhs - _rhs).abs().max() < 1e-6) else 'MISMATCH'
                            logger.info(f"[{league}] per-90 companions on frame: {len(_p90_cols)} "
                                        f"(identity {_p90_chk}) — {sorted(_p90_cols)[:4]}")

                        # Penalty-taker cascade from FPL's designated order,
                        # BEFORE the dials so a panel override still wins. Must
                        # run after per-90 scaling: the cascade already carries
                        # the minutes term (xMins/90), so scaling it again would
                        # apply minutes twice. George, 2026-08-07.
                        try:
                            from app.services.fpl_penalties import apply_penalty_order_shares
                            _pen_map = {}
                            _fm = ProjectionService._current_source.fpl_player_mappings
                            if _fm is not None and 'penalties_order' in getattr(_fm, 'columns', []):
                                for _r in _fm.itertuples(index=False):
                                    _o = getattr(_r, 'penalties_order', None)
                                    if _o is None or pd.isna(_o):
                                        continue
                                    # (rank, weight). Weight is only ever set by an admin
                                    # override; FPL never shares a rank, so a NULL weight
                                    # just means "this tier holds one player".
                                    _w = getattr(_r, 'penalty_weight', None)
                                    _pen_map[int(_r.player_id)] = (
                                        int(_o),
                                        None if (_w is None or pd.isna(_w)) else float(_w),
                                    )
                            if _pen_map:
                                _fpl_frame = apply_penalty_order_shares(
                                    _fpl_frame, _pen_map, team_projections)
                            else:
                                logger.warning(f"[{league}] no penalties_order rows — "
                                               "penalty shares left on history")
                        except Exception as _pen_err:
                            # Never fatal: falling back to historical penalty shares is
                            # the old behaviour, not a broken run.
                            logger.warning(f"[{league}] penalty order shares skipped: {_pen_err}")
                        # §12 Phase 5: share/defcon dials — replacement at
                        # assembly, after per-90 scaling. Runs BEFORE the DC
                        # recompute (it used to run after): defcon_share is a
                        # share of the team's defensive-contribution total, so
                        # it feeds the hit rate as a RATE and must be in place
                        # before the threshold maths runs. George, 2026-08-04.
                        _fpl_frame = apply_share_dials(_fpl_frame, _fpl_dials, team_projections)
                        # Team-down DC hit rate on the exposure-scaled inputs so
                        # def_con_pct is minutes-aware, and on the dialled rate
                        # where a dial is set. The rate is banked as its own
                        # column first — .apply(axis=1) hands the function a COPY
                        # of each row, so a function that tried to stash it back
                        # onto `row` would silently write nothing.
                        _fpl_frame[_TD_DC_RATE_COL] = _fpl_frame.apply(_td_dc_rate90, axis=1)
                        _fpl_frame['CBIT Hit Rate'] = _fpl_frame.apply(_td_cbit_hit_rate, axis=1)
                        _fpl_frame['def_con_pct'] = (_fpl_frame['CBIT Hit Rate'] * 100).round(2)
                        # Persist the model's DC share so the dials panel has a
                        # slider baseline for EVERY position. Computed here
                        # because it needs the assembled rate and the team
                        # totals together — the panel can't derive it (MID/FWD
                        # are scored on CBIT + recoveries, so the stored CBIT
                        # share alone understates them).
                        try:
                            from app.services.xminutes import build_dc_share_rows
                            from app.repository.fpl_per90_repo import (
                                insert_per90_shares_async as _dc_insert,
                            )
                            _dc_rows = build_dc_share_rows(
                                _fpl_frame, team_projections, _per90_collector)
                            if _dc_rows:
                                await _dc_insert(_dc_rows, league_id)
                                logger.info(f"[{league}] FPL: DC share baseline "
                                            f"written for {len(_dc_rows)} players")
                        except Exception as _dcs_err:
                            logger.warning(f"[{league}] DC share baseline write failed "
                                           f"(non-fatal): {_dcs_err}")
                        logger.info(f"[{league}] FPL xMinutes: profiles for {len(_xm_profiles)} "
                                    f"players ({time.time()-_t_xm:.1f}s)")
                    except Exception as _xm_err:
                        # Never let xMinutes kill the FPL insert — fall back
                        # to the legacy unscaled frame and flag it loudly.
                        logger.warning(f"[{league}] FPL xMinutes failed — falling back to "
                                       f"unscaled points: {_xm_err}", exc_info=True)
                        _fpl_frame = _fpl_base

                # Persist a PURE-MODEL scoring frame per (player, fixture) —
                # model bands, model shares, NO dials, standing (no XI snap).
                # The recalc path layers current dials on top from scratch,
                # so Reset-to-model is instant and consecutive edits always
                # compose from fresh model values (fpl_recalc_service).
                try:
                    from app.repository.fpl_recalc_repo import save_assembly_bundles
                    if _xmin_enabled and '_model_profiles' in dir():
                        _bundle_frame = _fpl_base.copy()
                        _bundle_frame = stamp_xmin_columns(_bundle_frame, _model_profiles, confirmed_xi=None)
                        if _per90_collector:
                            _bundle_frame = apply_per90_scaling(_bundle_frame, _m_bar_lookup)

                        _bundle_frame['CBIT Hit Rate'] = _bundle_frame.apply(_td_cbit_hit_rate, axis=1)
                        _bundle_frame['def_con_pct'] = (_bundle_frame['CBIT Hit Rate'] * 100).round(2)
                        # Penalty cascade on the SNAPSHOT too. Recalc re-derives it on load,
                        # so this is idempotent and cannot double-count — but without it the
                        # bundle carries raw HISTORICAL penalty shares while the live points
                        # carry cascade ones, and the two silently disagree. That gap cost an
                        # evening: measuring the bundle read as "penalties 25% light" when the
                        # live numbers were correct all along.
                        try:
                            from app.services.fpl_penalties import apply_penalty_order_shares
                            if _pen_map:
                                _bundle_frame = apply_penalty_order_shares(
                                    _bundle_frame, _pen_map, team_projections)
                        except Exception as _bpen_err:
                            logger.warning(f"[{league}] bundle penalty shares skipped: {_bpen_err}")
                        await save_assembly_bundles(_bundle_frame, score_preds, team_projections)
                except Exception as _bundle_err:
                    logger.warning(f"[{league}] assembly-bundle snapshot failed (non-fatal): {_bundle_err}")
                fpl_point_df = get_fpl_points(_fpl_frame, score_preds, fpl_points_dict_gk, fpl_points_dict_def, fpl_points_dict_mid, fpl_points_dict_fwd)
                # Bonus: Monte Carlo the fixture and award 3/2/1 on ranked BPS
                # (fpl_bonus_sim), replacing the softmax over EXPECTED BPS.
                # The old allocator sized its pool as 0.5 x count(BPS >= 7.5),
                # which measured 3.44 per fixture on prod against FPL's 6, and
                # gave a full unit of weight to players with xMins = 0.
                # FPL_BONUS_SIM=0 falls back to the softmax.
                _bonus_sim_on = os.getenv('FPL_BONUS_SIM', '1') != '0'
                bonus = None
                if _bonus_sim_on:
                    try:
                        from app.services.fpl_bonus_sim import simulate_bonus_for_frame
                        _t_sim = time.time()
                        bonus = simulate_bonus_for_frame(_fpl_frame, score_preds)
                        if bonus is None or bonus.empty:
                            raise RuntimeError("simulator returned no rows")
                        logger.info(f"[{league}] bonus simulator: {len(bonus)} rows "
                                    f"in {time.time() - _t_sim:.1f}s")
                    except Exception as _sim_err:
                        logger.warning(f"[{league}] bonus simulator failed, falling back "
                                       f"to softmax: {_sim_err}")
                        bonus = None
                if bonus is None:
                    bps_df = bonus_points_score(_fpl_frame, score_preds, fpl_bonus_dict_gk, fpl_bonus_dict_def, fpl_bonus_dict_mid, fpl_bonus_dict_fwd)
                    from app.services.statz_functions import get_bonus_points_by_fixture
                    bonus = get_bonus_points_by_fixture(bps_df, score_preds, expo_factor=0.1)

                fpl_df = fpl_point_df.merge(bonus, on=['fixture_id', 'player_id'], how='left')
                fpl_df['FPL Points'] = fpl_df['PTS'] + fpl_df['Bonus Points'].fillna(0)
                # Phase 2: pull def_con_pct off pl_projections so it lands
                # in fpl_df for persistence. Keyed on (fixture_id, player_id)
                # — the natural per-row identifier — instead of the
                # (Player, Team, Opponent) triple the bonus merge above
                # uses. Player triples can be ambiguous in pathological
                # data (same matchup repeating across GWs); fixture_id +
                # player_id is unambiguous. drop_duplicates is a
                # defensive guard against any stray data dupes on those
                # keys; the underlying intent is "one def_con_pct per
                # (fixture, player)".
                # _fpl_frame (not pl_projections): when xMinutes is on its
                # def_con_pct is the minutes-aware recompute; when off it IS
                # pl_projections.
                _def_con = (
                    _fpl_frame[['fixture_id', 'player_id', 'def_con_pct']]
                    .drop_duplicates(['fixture_id', 'player_id'])
                )
                fpl_df = fpl_df.merge(_def_con, on=['fixture_id', 'player_id'], how='left')
                # Expected minutes = xmin_bands, the PUBLISHED xMins
                # (90*p90 + 75*(p60-p90) + 30*(p_play-p60), methodology §2) and
                # the figure that drives scoring. Was xmin_expected — the
                # legacy exposure field superseded on 2026-07-29 — which left
                # 1,995 rows across 105 players showing minutes that had
                # nothing to do with the points beside them: Rodri read 51.3
                # xMins against 0 points because the flag zeroed his BANDS.
                if 'xmin_bands' in _fpl_frame.columns:
                    _xm_exp = (
                        _fpl_frame[['fixture_id', 'player_id', 'xmin_bands']]
                        .drop_duplicates(['fixture_id', 'player_id'])
                        .rename(columns={'xmin_bands': 'expected_minutes'})
                    )
                    fpl_df = fpl_df.merge(_xm_exp, on=['fixture_id', 'player_id'], how='left')
                else:
                    fpl_df['expected_minutes'] = None
                # Per-stat DIALLED projections. The FPL detail tiles used to read
                # player_projections, which is written by the non-FPL pipeline
                # and knows nothing about dials — so a dialled player's tiles
                # disagreed with his points. Bruno Fernandes read xA 0.77 on the
                # site against 0.55 in the panel (assist_share dialled 42%->30%),
                # and because BAND dials scale every player-level stat, his goals
                # / saves / key passes were all ~8% out too (xMins 81.4 -> 88.2).
                # Carried from _fpl_frame, which is post-dial. George, 2026-08-05.
                for _src, _dst in FPL_STAT_COLUMNS.items():
                    if _src in _fpl_frame.columns:
                        _stat = (
                            _fpl_frame[['fixture_id', 'player_id', _src]]
                            .drop_duplicates(['fixture_id', 'player_id'])
                            .rename(columns={_src: _dst})
                        )
                        fpl_df = fpl_df.merge(_stat, on=['fixture_id', 'player_id'], how='left')
                    else:
                        fpl_df[_dst] = None
                fpl_df = fpl_df[['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue', 'FPL Points', 'Bonus Points', 'def_con_pct', 'expected_minutes'] + list(FPL_STAT_COLUMNS.values())].copy()
                # Stamp gameweek_id + team_id + opponent_id from the source
                # fixtures table. gameweek_id makes the 6-GW horizon
                # queryable; team_id / opponent_id let consumers filter
                # without a JOIN through fixtures + venue CASE.
                _fix_idx = fixtures.set_index('id')
                _home_id = fpl_df['fixture_id'].map(_fix_idx['home_team_id'])
                _away_id = fpl_df['fixture_id'].map(_fix_idx['away_team_id'])
                fpl_df['Gameweek'] = fpl_df['fixture_id'].map(_fix_idx['gameweek_id'])
                fpl_df['team_id'] = np.where(fpl_df['Venue'] == 'H', _home_id, _away_id)
                fpl_df['opponent_id'] = np.where(fpl_df['Venue'] == 'H', _away_id, _home_id)
                fpl_df = fpl_df.round(2)
                # Fantasy-only: keep just gameweeks still open to plan for.
                fpl_df = ProjectionService._fantasy_gw_filter(fpl_df, _fantasy_upcoming_gws)

                # FPL membership + availability guard (George, 2026-07-23).
                # ALLOW-list: a player only gets FPL fantasy rows if he is
                #   (1) IN the current FPL game — active mapping, fpl_id NOT
                #       NULL. The stat model rightly projects Sportmonks-world
                #       players who aren't FPL assets at all (departed like
                #       Trossard, parent-club loanees like Antoñito Cordero);
                #       none of them belong in fpl_projections.
                #   (2) not red-flagged — 'i' injured / 's' suspended /
                #       'u' unavailable-loaned / 'n' ineligible; and
                #   (3) not a ≤25%-chance doubtful (50/75% 'd' still project —
                #       George's call).
                # Status is read live at insert time and bootstraps land twice
                # daily, so players re-enter automatically as flags lift.
                try:
                    _u_conn = await get_source_connection()
                    try:
                        async with _u_conn.cursor() as _cur:
                            await _cur.execute("""
                                SELECT m.player_id FROM fpl_player_mappings m
                                JOIN fpl_player_snapshots s ON s.fpl_id = m.fpl_id
                                 AND s.snapshot_date = (SELECT MAX(snapshot_date) FROM fpl_player_snapshots)
                                WHERE m.player_id IS NOT NULL AND m.fpl_id IS NOT NULL
                                  -- Status is NO LONGER a filter (George,
                                  -- 2026-08-03). A flagged player is a real
                                  -- FPL asset people own; dropping his rows
                                  -- meant he vanished from the dials panel and
                                  -- could not be adjusted. Rodri was flagged
                                  -- 'i' while playing 99 minutes for Spain
                                  -- three weeks earlier. The flag now shapes
                                  -- his BANDS (apply_availability_flags), so
                                  -- he projects at ~0 xMins and stays visible
                                  -- and dialable. Membership in the current
                                  -- bootstrap is handled by the loader gate.
                            """)
                            _allowed = {int(r[0]) for r in await _cur.fetchall()}
                    finally:
                        release_source_connection(_u_conn)
                    if _allowed and 'player_id' in fpl_df.columns:
                        _before = len(fpl_df)
                        fpl_df = fpl_df[fpl_df['player_id'].astype('Int64').isin(_allowed)]
                        logger.info(f"[{league}] FPL membership guard: {_before - len(fpl_df)} rows dropped (not in current FPL game / red-flagged), {len(fpl_df)} kept")
                    else:
                        # Never no-op silently — a missing column or empty allow
                        # set means ghosts (departed/loaned players) ship.
                        logger.warning(f"[{league}] FPL membership guard NOT applied: allowed={len(_allowed)}, has player_id col={'player_id' in fpl_df.columns}")
                except Exception as _u_err:
                    logger.warning(f"[{league}] FPL membership guard skipped: {_u_err}")

                logger.info(f"[{league}] Inserting FPL projections into DB ({len(fpl_df)} rows)...")
                _t = time.time()
                await insert_fpl_projections_async(fpl_df)
                from app.repository.fpl_repo import prune_stale_fpl_rows
                await prune_stale_fpl_rows()
                # Post-run dial true-up (2026-07-30): the run read dials at
                # DATA-LOAD time, so a dial saved mid-run gets clobbered by
                # the insert above. Re-read dials fresh and recalc all dialed
                # players from the just-written bundles — a run can then never
                # change a dialed player's points except through his dials.
                #
                # SKIP when no dial changed during the run. The true-up exists
                # only to catch a mid-run edit, but it re-scores every player in
                # any fixture containing a dialled player — 14,260 rows — and
                # since the bonus simulator went in that costs ~5.5 minutes,
                # taking runs from ~21 to ~28 min. Nothing to true up in the
                # common case. George, 2026-08-03.
                try:
                    from app.repository.fpl_recalc_repo import load_all_dials_and_bands
                    from app.services.fpl_recalc_service import recalc_fpl_players
                    _dials_now, _ = await load_all_dials_and_bands()
                    _dials_touched = None
                    try:
                        _tu_conn = await get_source_connection()
                        try:
                            async with _tu_conn.cursor() as _tc:
                                await _tc.execute("SELECT MAX(updated_at) FROM fpl_player_dials")
                                _row = await _tc.fetchone()
                                _dials_touched = _row[0] if _row else None
                        finally:
                            release_source_connection(_tu_conn)
                    except Exception as _tu_probe_err:
                        # Can't tell -> run it. Correctness over speed.
                        logger.warning(f"[{league}] dial-change probe failed, running true-up: {_tu_probe_err}")
                    _stale = (
                        _dials_touched is not None
                        and _run_started_at is not None
                        and pd.to_datetime(_dials_touched) < pd.to_datetime(_run_started_at)
                    )
                    if _dials_now and _stale:
                        logger.info(f"[{league}] post-run dial true-up SKIPPED — no dial edited since "
                                    f"{_run_started_at} (last edit {_dials_touched})")
                    elif _dials_now:
                        _tu = await recalc_fpl_players(list(_dials_now.keys()))
                        logger.info(f"[{league}] post-run dial true-up: "
                                    f"{_tu.get('rows_updated')} rows in {_tu.get('duration_seconds')}s")
                except Exception as _tu_err:
                    logger.warning(f"[{league}] post-run dial true-up failed (non-fatal): {_tu_err}")
                logger.info(f"[{league}] FPL projections inserted ({time.time()-_t:.1f}s)")

                # Stale-row cleanup: the insert is an upsert, so it can't
                # REMOVE players — one the membership guard newly excluded,
                # or who fell out of the pool, keeps his previous run's rows
                # (J.Timber kept run-7 rows after his 'i' flag landed). Delete
                # rows in the covered gameweeks for players not in this frame.
                try:
                    from app.repository.fpl_repo import cleanup_fpl_projections_async
                    _cl_gws = [int(g) for g in fpl_df['Gameweek'].dropna().unique().tolist()]
                    # (fixture, player) pairs, not player ids — a transferred
                    # player is still "kept" but his OLD club's rows hang off
                    # the old club's fixtures and no upsert ever touches them.
                    _cl_pairs = [
                        (int(f), int(p)) for f, p in
                        fpl_df[['fixture_id', 'player_id']].dropna().itertuples(index=False, name=None)
                    ]
                    _stale = await cleanup_fpl_projections_async(_cl_gws, _cl_pairs)
                    if _stale:
                        logger.info(f"[{league}] FPL: {_stale} stale rows removed (not produced by this run)")
                except Exception as _cl_err:
                    logger.warning(f"[{league}] FPL stale-row cleanup skipped: {_cl_err}")
            except Exception as e:
                logger.warning(f"[{league}] FPL computation failed (skipping): {e}", exc_info=True)

        # OPTA Points
        if fpl:
            try:
                opta_points_dict = {
                    'Goals': 10, 'Assists': 6, 'Shots Off': 2, 'Shots On Target': 4,
                    'Passes': 0.2, 'Interceptions': 2, 'Tackles': 2, 'Blocked Shots': 2,
                    'Total Crosses': 0.2, 'Yellow Cards': -2, 'Fouls': -1, 'Fouls Drawn': 1,
                    'Saves': 5, 'Offsides': -1, 'Goals Conceded': -1, 'Penalties Saved': 5
                }
                for _col in ['Blocked Shots Average']:
                    if _col not in pl_projections.columns:
                        pl_projections.loc[:, _col] = 0
                opta_df = get_opta_points(pl_projections, score_preds, opta_points_dict)
                opta_df = opta_df[['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue', 'PTS', 'Floor PTS']].copy()
                # Stamp gameweek_id + team_id + opponent_id (parity with FPL/FanTeam).
                _fix_idx_op = fixtures.set_index('id')
                _home_id_op = opta_df['fixture_id'].map(_fix_idx_op['home_team_id'])
                _away_id_op = opta_df['fixture_id'].map(_fix_idx_op['away_team_id'])
                opta_df['Gameweek'] = opta_df['fixture_id'].map(_fix_idx_op['gameweek_id'])
                opta_df['team_id'] = np.where(opta_df['Venue'] == 'H', _home_id_op, _away_id_op)
                opta_df['opponent_id'] = np.where(opta_df['Venue'] == 'H', _away_id_op, _home_id_op)
                # Fantasy-only: keep just gameweeks still open to plan for.
                opta_df = ProjectionService._fantasy_gw_filter(opta_df, _fantasy_upcoming_gws)
                logger.info(f"[{league}] Inserting OPTA projections into DB ({len(opta_df)} rows)...")
                _t = time.time()
                await insert_opta_projections_async(opta_df)
                logger.info(f"[{league}] OPTA projections inserted ({time.time()-_t:.1f}s)")
            except Exception as e:
                logger.warning(f"[{league}] OPTA computation failed (skipping): {e}", exc_info=True)

        # FanTeam Points
        # Same approach as FPL: 6-GW horizon, every PL player in scope.
        # FanTeam uses identical GK/DEF/MID/FWD groupings as FPL, so we
        # reuse the FPL Position from fpl_player_mappings (already set on
        # pl_projections by the FPL block above) — no separate xlsx /
        # mapping table needed. Price + Lineup CSV import dropped per
        # 2026-04-29: we project every player rather than gating on
        # FanTeam's "expected/possible" lineup status.
        if fpl:
            try:
                fanteam_points_dict_gk = {
                    'Goals': 8, 'Assists': 3, 'Shots On Target': 1, 'Saves': 0.5,
                    'Penalties Saved': 5, 'Clean Sheet': 4, 'Win': 0.3, 'Lose': -0.3,
                    'Goals Conceded': -1, 'Yellow Card': -1
                }
                fanteam_points_dict_def = {
                    'Goals': 6, 'Assists': 3, 'Shots On Target': 0.6, 'Clean Sheet': 4,
                    'Win': 0.3, 'Lose': -0.3, 'Goals Conceded': -1, 'Yellow Card': -1
                }
                fanteam_points_dict_mid = {
                    'Goals': 5, 'Assists': 3, 'Shots On Target': 0.4, 'Clean Sheet': 1,
                    'Win': 0.3, 'Lose': -0.3, 'Yellow Card': -1, 'Full Match': 1
                }
                fanteam_points_dict_fwd = {
                    'Goals': 4, 'Assists': 3, 'Shots On Target': 0.4,
                    'Win': 0.3, 'Lose': -0.3, 'Yellow Card': -1, 'Full Match': 1
                }
                # Reuse FPL Position (already mapped from fpl_player_mappings
                # in the FPL block above). FanTeam Position column kept for
                # backward-compat with get_fanteam_points internals.
                pl_projections['FanTeam Position'] = pl_projections['FPL Position']
                ft_temp = pl_projections[pl_projections['FanTeam Position'].notna()].reset_index(drop=True)
                fanteam_df = get_fanteam_points(ft_temp, score_preds, fanteam_points_dict_gk,
                                                fanteam_points_dict_def, fanteam_points_dict_mid, fanteam_points_dict_fwd)
                # Defensive: drop only rows missing the join keys (player_id /
                # fixture_id). Wholesale dropna() killed everything once we
                # stopped sourcing Price from the CSV (NaN price → row drop).
                fanteam_df = fanteam_df.dropna(subset=['player_id', 'fixture_id'])
                # Stamp gameweek_id + team_id + opponent_id (parity with FPL).
                _fix_idx_ft = fixtures.set_index('id')
                _home_id_ft = fanteam_df['fixture_id'].map(_fix_idx_ft['home_team_id'])
                _away_id_ft = fanteam_df['fixture_id'].map(_fix_idx_ft['away_team_id'])
                fanteam_df['Gameweek'] = fanteam_df['fixture_id'].map(_fix_idx_ft['gameweek_id'])
                fanteam_df['team_id'] = np.where(fanteam_df['Venue'] == 'H', _home_id_ft, _away_id_ft)
                fanteam_df['opponent_id'] = np.where(fanteam_df['Venue'] == 'H', _away_id_ft, _home_id_ft)
                # Fantasy-only: keep just gameweeks still open to plan for.
                fanteam_df = ProjectionService._fantasy_gw_filter(fanteam_df, _fantasy_upcoming_gws)
                logger.info(f"[{league}] Inserting FanTeam projections into DB ({len(fanteam_df)} rows)...")
                _t = time.time()
                await insert_fanteam_projections_async(fanteam_df)
                logger.info(f"[{league}] FanTeam projections inserted ({time.time()-_t:.1f}s)")
            except Exception as e:
                logger.warning(f"[{league}] FanTeam computation failed (skipping): {e}", exc_info=True)

        # DraftKings Points
        # Same approach as FPL/FanTeam: 6-GW horizon, every PL player with
        # an FPL position. DraftKings positions (GK/DEF/MID/FWD) match FPL
        # exactly so we reuse FPL Position from fpl_player_mappings — no
        # separate mapping needed. Drops the legacy Draftkings Position
        # column from PL Fantasy Players.xlsx.
        if fpl:
            try:
                draftkings_points_dict_gk = {
                    'Goals': 10, 'Assists': 6, 'Shots Total': 1, 'Shots On Target': 1,
                    'Total Crosses': 0.7, 'Key Passes': 1, 'Successful Passes': 0.02,
                    'Fouls Drawn': 1, 'Fouls Committed': -0.5, 'Tackles Won': 1,
                    'Saves': 2, 'Penalties Saved': 5, 'Clean Sheet': 5, 'Win': 5,
                    'Goals Conceded': -2, 'Yellow Card': -1.5,
                }
                draftkings_points_dict_def = {
                    'Goals': 10, 'Assists': 6, 'Shots Total': 1, 'Shots On Target': 1,
                    'Total Crosses': 0.7, 'Key Passes': 1, 'Successful Passes': 0.02,
                    'Fouls Drawn': 1, 'Fouls Committed': -0.5, 'Tackles Won': 1,
                    'Interceptions': 0.5, 'Clean Sheet': 3, 'Yellow Card': -1.5,
                }
                draftkings_points_dict_mid = {
                    'Goals': 10, 'Assists': 6, 'Shots Total': 1, 'Shots On Target': 1,
                    'Total Crosses': 0.7, 'Key Passes': 1, 'Successful Passes': 0.02,
                    'Fouls Drawn': 1, 'Fouls Committed': -0.5, 'Tackles Won': 1,
                    'Interceptions': 0.5, 'Yellow Card': -1.5,
                }
                draftkings_points_dict_fwd = {
                    'Goals': 10, 'Assists': 6, 'Shots Total': 1, 'Shots On Target': 1,
                    'Total Crosses': 0.7, 'Key Passes': 1, 'Successful Passes': 0.02,
                    'Fouls Drawn': 1, 'Fouls Committed': -0.5, 'Tackles Won': 1,
                    'Interceptions': 0.5, 'Yellow Card': -1.5,
                }
                # get_draftkings_points reads pl_projections['Draftkings Position'].
                # Reuse FPL Position (already mapped from fpl_player_mappings
                # in the FPL block above).
                pl_projections['Draftkings Position'] = pl_projections['FPL Position']
                dk_temp = pl_projections[pl_projections['Draftkings Position'].notna()].reset_index(drop=True)
                dk_df = get_draftkings_points(dk_temp, score_preds, draftkings_points_dict_gk,
                                              draftkings_points_dict_def, draftkings_points_dict_mid,
                                              draftkings_points_dict_fwd)
                dk_df = dk_df.dropna(subset=['player_id', 'fixture_id'])
                # Stamp gameweek_id + team_id + opponent_id (parity with FPL/FanTeam).
                _fix_idx_dk = fixtures.set_index('id')
                _home_id_dk = dk_df['fixture_id'].map(_fix_idx_dk['home_team_id'])
                _away_id_dk = dk_df['fixture_id'].map(_fix_idx_dk['away_team_id'])
                dk_df['Gameweek'] = dk_df['fixture_id'].map(_fix_idx_dk['gameweek_id'])
                dk_df['team_id'] = np.where(dk_df['Venue'] == 'H', _home_id_dk, _away_id_dk)
                dk_df['opponent_id'] = np.where(dk_df['Venue'] == 'H', _away_id_dk, _home_id_dk)
                # Fantasy-only: keep just gameweeks still open to plan for.
                dk_df = ProjectionService._fantasy_gw_filter(dk_df, _fantasy_upcoming_gws)
                logger.info(f"[{league}] Inserting DraftKings projections into DB ({len(dk_df)} rows)...")
                _t = time.time()
                await insert_draftkings_projections_async(dk_df)
                logger.info(f"[{league}] DraftKings projections inserted ({time.time()-_t:.1f}s)")
            except Exception as e:
                logger.warning(f"[{league}] DraftKings computation failed (skipping): {e}", exc_info=True)

        # Dream11 Points
        # Same approach as DraftKings: 6-GW horizon, FPL Position reused as
        # Dream11 Position (GK/DEF/MID/FWD taxonomy is identical).
        if fpl:
            try:
                dream11_points_dict_gk = {
                    'Goals': 60, 'Assists': 20, 'Key Passes': 3, 'Shots On Target': 6,
                    'Successful Passes': 0.2, 'Tackles Won': 4, 'Interceptions': 4,
                    'Clean Sheet': 20, 'Saves': 6, 'Penalties Saved': 50,
                    'Goals Conceded': -2, 'Yellow Card': -4,
                }
                dream11_points_dict_def = {
                    'Goals': 60, 'Assists': 20, 'Key Passes': 3, 'Shots On Target': 6,
                    'Successful Passes': 0.2, 'Tackles Won': 4, 'Interceptions': 4,
                    'Clean Sheet': 20, 'Goals Conceded': -2, 'Yellow Card': -4,
                }
                dream11_points_dict_mid = {
                    'Goals': 50, 'Assists': 20, 'Key Passes': 3, 'Shots On Target': 6,
                    'Successful Passes': 0.2, 'Tackles Won': 4, 'Interceptions': 4,
                    'Yellow Card': -4,
                }
                dream11_points_dict_fwd = {
                    'Goals': 40, 'Assists': 20, 'Key Passes': 3, 'Shots On Target': 6,
                    'Successful Passes': 0.2, 'Tackles Won': 4, 'Interceptions': 4,
                    'Yellow Card': -4,
                }
                pl_projections['Dream11 Position'] = pl_projections['FPL Position']
                d11_temp = pl_projections[pl_projections['Dream11 Position'].notna()].reset_index(drop=True)
                d11_df = get_dream11_points(d11_temp, score_preds, dream11_points_dict_gk,
                                            dream11_points_dict_def, dream11_points_dict_mid,
                                            dream11_points_dict_fwd)
                d11_df = d11_df.dropna(subset=['player_id', 'fixture_id'])
                _fix_idx_d11 = fixtures.set_index('id')
                _home_id_d11 = d11_df['fixture_id'].map(_fix_idx_d11['home_team_id'])
                _away_id_d11 = d11_df['fixture_id'].map(_fix_idx_d11['away_team_id'])
                d11_df['Gameweek'] = d11_df['fixture_id'].map(_fix_idx_d11['gameweek_id'])
                d11_df['team_id'] = np.where(d11_df['Venue'] == 'H', _home_id_d11, _away_id_d11)
                d11_df['opponent_id'] = np.where(d11_df['Venue'] == 'H', _away_id_d11, _home_id_d11)
                # Fantasy-only: keep just gameweeks still open to plan for.
                d11_df = ProjectionService._fantasy_gw_filter(d11_df, _fantasy_upcoming_gws)
                logger.info(f"[{league}] Inserting Dream11 projections into DB ({len(d11_df)} rows)...")
                _t = time.time()
                await insert_dream11_projections_async(d11_df)
                logger.info(f"[{league}] Dream11 projections inserted ({time.time()-_t:.1f}s)")
            except Exception as e:
                logger.warning(f"[{league}] Dream11 computation failed (skipping): {e}", exc_info=True)

        # ## **Player Stat Probabilities**
        #
        # Using Poisson Distribution to get the likelihood of players acheiving certain statistics.

        # In[ ]:

        pl_projections.rename(columns={'Fouls': 'Fouls Committed'}, inplace=True)

        # In[ ]:

        # Multi-line markets (1+, 2+, 3+). Yellowcards handled separately below
        # because 2+ yellows = red card (probability ~0, not a useful market).
        perc_stats = ['Shots On Target', 'Fouls Committed', 'Fouls Drawn',
                      'Goals', 'Tackles', 'Shots Total', 'Offsides']
        lines = [1, 2, 3]

        # In[ ]:

        logger.info(f"[{league}] Computing player stat probabilities...")
        _t = time.time()
        player_stat_probs = get_poisson_probs(pl_projections, perc_stats, lines)
        # Yellowcards: single threshold (1+ only).
        # Note: 'Yellowcards' is renamed to 'Yellow Cards' upstream of this point.
        if 'Yellow Cards' in pl_projections.columns:
            yellow_probs = get_poisson_probs(pl_projections, ['Yellow Cards'], [1])
            player_stat_probs = pd.concat([player_stat_probs, yellow_probs], ignore_index=True)
        logger.info(f"[{league}] Player stat probabilities done ({time.time()-_t:.1f}s)")
        player_stat_probs = player_stat_probs.round(2)
        # player_stat_probs.to_csv(rf"{save_file_path}\{league} Player Stat Probabilities.csv", index=False)
        # await insert_players_stats_async(pl_projections)
        logger.info(f"[{league}] Inserting player stat probabilities into DB...")
        _t = time.time()
        await insert_players_stats_async(player_stat_probs, teams=teams, competition_id=league_id, comp_teams=comp_teams)
        logger.info(f"[{league}] Player stat probs inserted ({time.time()-_t:.1f}s)")
        logger.info(f"[{league}] COMPLETE - total time: {(time.time()-_start_time)/60:.1f} min")


    async def fixtures(self, league_request):
        league = league_request.league or 'Championship'

        ctx = await self._setup_league(league)

        # Unpack shared context into local variables so downstream code is unchanged
        data_folder_path = ctx.data_folder_path
        model_file_path = ctx.model_file_path
        save_file_path = ctx.save_file_path
        league_dashed = ctx.league_dashed
        date_from = ctx.date_from
        date_to = ctx.date_to
        league_below = ctx.league_below
        league_above = ctx.league_above
        league_below_attack_weight = ctx.league_below_attack_weight
        league_below_defense_weight = ctx.league_below_defense_weight
        league_above_attack_weight = ctx.league_above_attack_weight
        league_above_defense_weight = ctx.league_above_defense_weight
        country_code = ctx.country_code
        div = ctx.div
        weightings = ctx.weightings
        mv_beta = ctx.mv_beta
        odds_beta = ctx.odds_beta
        xG = ctx.xG
        fpl = ctx.fpl
        player_stats = ctx.player_stats
        team_stats = ctx.team_stats
        standings = ctx.standings
        seasons = ctx.seasons
        comps = ctx.comps
        comp_teams = ctx.comp_teams
        teams = ctx.teams
        players = ctx.players
        fixtures_df = ctx.fixtures_df
        b365_odds = ctx.b365_odds
        stats_types = ctx.stats_types
        model_dataset_all = ctx.model_dataset_all
        model_dataset_league = ctx.model_dataset_league
        projection_accuracy_dataset_league = ctx.projection_accuracy_dataset_league
        projection_accuracy_dataset_all = ctx.projection_accuracy_dataset_all
        all_team_ratings = ctx.all_team_ratings
        league_id = ctx.league_id
        fixtures = ctx.fixtures
        league_standings = ctx.league_standings
        league_above_id = ctx.league_above_id
        league_below_id = ctx.league_below_id
        previous_season_id = ctx.previous_season_id
        current_season_id = ctx.current_season_id
        matches_played = ctx.matches_played
        season_fixtures = ctx.season_fixtures
        total_matches = ctx.total_matches
        previous_season_id_below = ctx.previous_season_id_below
        previous_season_id_above = ctx.previous_season_id_above
        stat_list = ctx.stat_list

        ratings = await self._prepare_league(
            league=league, data_folder_path=data_folder_path, model_file_path=model_file_path,
            save_file_path=save_file_path, league_id=league_id, league_dashed=league_dashed,
            model_dataset_all=model_dataset_all, model_dataset_league=model_dataset_league,
            projection_accuracy_dataset_all=projection_accuracy_dataset_all,
            projection_accuracy_dataset_league=projection_accuracy_dataset_league,
            all_team_ratings=all_team_ratings, team_stats=team_stats, player_stats=player_stats,
            teams=teams, stats_types=stats_types, stat_list=stat_list,
            comp_teams=comp_teams, fixtures_df=fixtures_df, fixtures=fixtures, seasons=seasons, comps=comps,
            current_season_id=current_season_id, previous_season_id=previous_season_id,
            previous_season_id_above=previous_season_id_above,
            previous_season_id_below=previous_season_id_below,
            weightings=weightings, mv_beta=mv_beta, odds_beta=odds_beta,
            country_code=country_code, div=div, matches_played=matches_played, standings=standings,
            league_above=league_above, league_below=league_below, league_standings=league_standings,
            league_below_attack_weight=league_below_attack_weight,
            league_below_defense_weight=league_below_defense_weight,
            league_above_id=league_above_id, league_below_id=league_below_id,
            xG=xG, fpl=fpl, b365_odds=b365_odds,
            season_fixtures=season_fixtures, total_matches=total_matches, players=players,
            mode=(league_request.mode if hasattr(league_request, 'mode') and league_request.mode else "full"),
        )

        # ## **Make Predictions for Next Fixture Round**
        #
        # Result, Score, Clean Sheets, Over 1.5, Over 2.5 and BTTS all calculated here using Poisson Distribution.

        # In[18]:

        next_fix = ProjectionService._filter_upcoming_fixtures(league, fixtures, date_from, date_to)
        fixtures['kickoff_datetime'] = pd.to_datetime(fixtures['kickoff_datetime'])
        if hasattr(league_request, 'fixture_ids') and league_request.fixture_ids:
            next_fix = next_fix[next_fix['id'].isin(league_request.fixture_ids)]
            logger.info(f'[{league}] Filtered to {len(next_fix)} of {len(fixtures[(fixtures["kickoff_datetime"] >= date_from) & (fixtures["kickoff_datetime"] <= date_to)])} fixtures')
        next_fix = next_fix[
            ['id', 'kickoff_datetime', 'name', 'home_team_id', 'away_team_id', 'bet365_home_odds_decimal',
             'bet365_draw_odds_decimal', 'bet365_away_odds_decimal']]
        next_fix['home_team'] = next_fix['home_team_id'].apply(lambda x: get_team(x, teams))
        next_fix['away_team'] = next_fix['away_team_id'].apply(lambda x: get_team(x, teams))
        next_fix = next_fix.drop(columns=['home_team_id', 'away_team_id'])
        next_fix = drop_placeholder_fixtures(next_fix, league)
        next_fix.sort_values(by=['kickoff_datetime', 'home_team'], inplace=True)
        next_fix.reset_index(drop=True, inplace=True)

        # In[ ]:

        avg_home_goals = get_home_goal_avg(league_id, team_stats, fixtures, stats_types)
        avg_away_goals = get_away_goal_avg(league_id, team_stats, fixtures, stats_types)

        logger.info(f"[{league}] avg_home_goals={avg_home_goals:.3f}, avg_away_goals={avg_away_goals:.3f}")
        

        score_preds = make_round_goal_prediction(next_fix, ratings, avg_home_goals, avg_away_goals)
        # debug prints removed

        # boost = get_draw_boost(ratings, avg_home_goals, avg_away_goals, get_draw_perc(league_id, fixtures))
        # Dixon-Coles replaces the flat draw boost where a league has rho
        # configured; get_result_probs applies one or the other, never both.
        dixon_coles_rho = getattr(ctx, 'dixon_coles_rho', 0.0) or 0.0
        boost = 1.0 if dixon_coles_rho else 1.1
        score_preds['Home Odds %'] = ((1 / next_fix['bet365_home_odds_decimal']) * 100)
        score_preds['Draw Odds %'] = ((1 / next_fix['bet365_draw_odds_decimal']) * 100)
        score_preds['Away Odds %'] = ((1 / next_fix['bet365_away_odds_decimal']) * 100)

        home_win = []
        draw = []
        away_win = []
        home_clean = []
        away_clean = []
        over_1 = []
        over_2 = []
        btts = []

        # Pre-load bet365 goals over/under for the upcoming fixtures.
        # The cascade in compute_final_goals_and_probs (paths 1-3) uses
        # per-team + match-total ladders directly; path 4 (legacy 1X2
        # reverse-solve) is the fallback when those markets are absent.
        from app.services.odds_blend import (
            load_goals_odds_for_fixtures,
            compute_final_goals_and_probs,
        )
        _odds_conn = await get_source_connection()
        try:
            goals_odds_map = await load_goals_odds_for_fixtures(
                _odds_conn, next_fix['id'].tolist(),
            )
        finally:
            release_source_connection(_odds_conn)

        for i in range(len(score_preds)):
            bookie_margin = 1 + (
                    score_preds.loc[i, 'Home Odds %'] + score_preds.loc[i, 'Draw Odds %'] + score_preds.loc[
                i, 'Away Odds %'] - 100) / 100
            score_preds.loc[i, 'Home Odds %'] = (score_preds.loc[i, 'Home Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Draw Odds %'] = (score_preds.loc[i, 'Draw Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Away Odds %'] = (score_preds.loc[i, 'Away Odds %'] / bookie_margin).round(2)
            home_goals = score_preds['Home Goals'][i]
            away_goals = score_preds['Away Goals'][i]
            bookie_1x2_pct = None
            if not pd.isna(score_preds['Home Odds %'][i]):
                bookie_1x2_pct = (
                    float(score_preds['Home Odds %'][i]) / 100.0,
                    float(score_preds['Draw Odds %'][i]) / 100.0,
                    float(score_preds['Away Odds %'][i]) / 100.0,
                )
            fixture_id = int(next_fix['id'].iloc[i])
            new_home_goals, new_away_goals, adjusted_home_win_prob, adjusted_draw_prob, adjusted_away_win_prob = (
                compute_final_goals_and_probs(
                    fixture_id,
                    float(home_goals), float(away_goals),
                    bookie_1x2_pct,
                    goals_odds_map.get(fixture_id, {}),
                    odds_beta,
                    boost,
                    dixon_coles_rho,
                )
            )
            score_preds.loc[i, 'Home Goals'] = round(new_home_goals, 2)
            score_preds.loc[i, 'Away Goals'] = round(new_away_goals, 2)
            home_clean_sheet = poisson.pmf(0, new_away_goals)
            away_clean_sheet = poisson.pmf(0, new_home_goals)
            x = np.arange(0, 9)
            y = np.arange(0, 9)
            X, Y = np.meshgrid(x, y)
            Z = poisson.pmf(X, new_home_goals) * poisson.pmf(Y, new_away_goals)
            home_win.append(f"{adjusted_home_win_prob:.2f}%")
            draw.append(f"{adjusted_draw_prob:.2f}%")
            away_win.append(f"{adjusted_away_win_prob:.2f}%")
            home_clean.append(f"{home_clean_sheet * 100:.2f}%")
            away_clean.append(f"{away_clean_sheet * 100:.2f}%")
            over_1_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1]) * 100
            over_2_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1] - Z[2, 0] - Z[0, 2] - Z[1, 1]) * 100
            both_teams_score_prob = (1 - Z[0, :].sum() - Z[:, 0].sum() + Z[0, 0]) * 100
            over_1.append(f"{over_1_goals:.2f}%")
            over_2.append(f"{over_2_goals:.2f}%")
            btts.append(f"{both_teams_score_prob:.2f}%")

        # score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'], inplace=True)
        score_preds['Home Win %'] = home_win
        score_preds['Draw %'] = draw
        score_preds['Away Win %'] = away_win
        score_preds['Home Clean Sheet %'] = home_clean
        score_preds['Away Clean Sheet %'] = away_clean
        score_preds['Over 1.5 Goals %'] = over_1
        score_preds['Over 2.5 Goals %'] = over_2
        score_preds['Both Teams Score %'] = btts
        score_preds['Home Goals'] = score_preds['Home Goals'].round(2)
        score_preds['Away Goals'] = score_preds['Away Goals'].round(2)
        score_preds_with_odds = score_preds.copy()  # NEW - Create a copy with odds included
        score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'],
                         inplace=True)  # NEW - Drop odds from main predictions dataframe

        # score_preds.to_csv(rf"{save_file_path}\{league} Fixtures.csv", index=False)
        # debug print removed
        # debug print removed
        await insert_fixtures_async(score_preds, teams=teams, competition_id=league_id, comp_teams=comp_teams)

    async def predicted_table(self, league_request):
        league = league_request.league or 'Championship'

        ctx = await self._setup_league(league)

        # Unpack shared context into local variables so downstream code is unchanged
        data_folder_path = ctx.data_folder_path
        model_file_path = ctx.model_file_path
        save_file_path = ctx.save_file_path
        league_dashed = ctx.league_dashed
        date_from = ctx.date_from
        date_to = ctx.date_to
        league_below = ctx.league_below
        league_above = ctx.league_above
        league_below_attack_weight = ctx.league_below_attack_weight
        league_below_defense_weight = ctx.league_below_defense_weight
        league_above_attack_weight = ctx.league_above_attack_weight
        league_above_defense_weight = ctx.league_above_defense_weight
        country_code = ctx.country_code
        div = ctx.div
        weightings = ctx.weightings
        mv_beta = ctx.mv_beta
        odds_beta = ctx.odds_beta
        xG = ctx.xG
        fpl = ctx.fpl
        player_stats = ctx.player_stats
        team_stats = ctx.team_stats
        standings = ctx.standings
        seasons = ctx.seasons
        comps = ctx.comps
        comp_teams = ctx.comp_teams
        teams = ctx.teams
        players = ctx.players
        fixtures_df = ctx.fixtures_df
        b365_odds = ctx.b365_odds
        stats_types = ctx.stats_types
        model_dataset_all = ctx.model_dataset_all
        model_dataset_league = ctx.model_dataset_league
        projection_accuracy_dataset_league = ctx.projection_accuracy_dataset_league
        projection_accuracy_dataset_all = ctx.projection_accuracy_dataset_all
        all_team_ratings = ctx.all_team_ratings
        league_id = ctx.league_id
        fixtures = ctx.fixtures
        league_standings = ctx.league_standings
        league_above_id = ctx.league_above_id
        league_below_id = ctx.league_below_id
        previous_season_id = ctx.previous_season_id
        current_season_id = ctx.current_season_id
        matches_played = ctx.matches_played
        season_fixtures = ctx.season_fixtures
        total_matches = ctx.total_matches
        previous_season_id_below = ctx.previous_season_id_below
        previous_season_id_above = ctx.previous_season_id_above
        stat_list = ctx.stat_list

        ratings = await self._prepare_league(
            league=league, data_folder_path=data_folder_path, model_file_path=model_file_path,
            save_file_path=save_file_path, league_id=league_id, league_dashed=league_dashed,
            model_dataset_all=model_dataset_all, model_dataset_league=model_dataset_league,
            projection_accuracy_dataset_all=projection_accuracy_dataset_all,
            projection_accuracy_dataset_league=projection_accuracy_dataset_league,
            all_team_ratings=all_team_ratings, team_stats=team_stats, player_stats=player_stats,
            teams=teams, stats_types=stats_types, stat_list=stat_list,
            comp_teams=comp_teams, fixtures_df=fixtures_df, fixtures=fixtures, seasons=seasons, comps=comps,
            current_season_id=current_season_id, previous_season_id=previous_season_id,
            previous_season_id_above=previous_season_id_above,
            previous_season_id_below=previous_season_id_below,
            weightings=weightings, mv_beta=mv_beta, odds_beta=odds_beta,
            country_code=country_code, div=div, matches_played=matches_played, standings=standings,
            league_above=league_above, league_below=league_below, league_standings=league_standings,
            league_below_attack_weight=league_below_attack_weight,
            league_below_defense_weight=league_below_defense_weight,
            league_above_id=league_above_id, league_below_id=league_below_id,
            xG=xG, fpl=fpl, b365_odds=b365_odds,
            season_fixtures=season_fixtures, total_matches=total_matches, players=players,
            mode=(league_request.mode if hasattr(league_request, 'mode') and league_request.mode else "full"),
        )

        # ## **Make Predictions for Next Fixture Round**
        #
        # Result, Score, Clean Sheets, Over 1.5, Over 2.5 and BTTS all calculated here using Poisson Distribution.

        # In[18]:

        next_fix = ProjectionService._filter_upcoming_fixtures(league, fixtures, date_from, date_to)
        fixtures['kickoff_datetime'] = pd.to_datetime(fixtures['kickoff_datetime'])
        if hasattr(league_request, 'fixture_ids') and league_request.fixture_ids:
            next_fix = next_fix[next_fix['id'].isin(league_request.fixture_ids)]
            logger.info(f'[{league}] Filtered to {len(next_fix)} of {len(fixtures[(fixtures["kickoff_datetime"] >= date_from) & (fixtures["kickoff_datetime"] <= date_to)])} fixtures')
        next_fix = next_fix[
            ['id', 'kickoff_datetime', 'name', 'home_team_id', 'away_team_id', 'bet365_home_odds_decimal',
             'bet365_draw_odds_decimal', 'bet365_away_odds_decimal']]
        next_fix['home_team'] = next_fix['home_team_id'].apply(lambda x: get_team(x, teams))
        next_fix['away_team'] = next_fix['away_team_id'].apply(lambda x: get_team(x, teams))
        next_fix = next_fix.drop(columns=['home_team_id', 'away_team_id'])
        next_fix = drop_placeholder_fixtures(next_fix, league)
        next_fix.sort_values(by=['kickoff_datetime', 'home_team'], inplace=True)
        next_fix.reset_index(drop=True, inplace=True)
        # In[ ]:

        avg_home_goals = get_home_goal_avg(league_id, team_stats, fixtures, stats_types)
        avg_away_goals = get_away_goal_avg(league_id, team_stats, fixtures, stats_types)
        score_preds = make_round_goal_prediction(next_fix, ratings, avg_home_goals, avg_away_goals)
        # boost = get_draw_boost(ratings, avg_home_goals, avg_away_goals, get_draw_perc(league_id, fixtures))
        # Dixon-Coles replaces the flat draw boost where a league has rho
        # configured; get_result_probs applies one or the other, never both.
        dixon_coles_rho = getattr(ctx, 'dixon_coles_rho', 0.0) or 0.0
        boost = 1.0 if dixon_coles_rho else 1.1
        score_preds['Home Odds %'] = ((1 / next_fix['bet365_home_odds_decimal']) * 100)
        score_preds['Draw Odds %'] = ((1 / next_fix['bet365_draw_odds_decimal']) * 100)
        score_preds['Away Odds %'] = ((1 / next_fix['bet365_away_odds_decimal']) * 100)



        home_win = []
        draw = []
        away_win = []
        home_clean = []
        away_clean = []
        over_1 = []
        over_2 = []
        btts = []

        # Pre-load bet365 goals over/under for the upcoming fixtures.
        # The cascade in compute_final_goals_and_probs (paths 1-3) uses
        # per-team + match-total ladders directly; path 4 (legacy 1X2
        # reverse-solve) is the fallback when those markets are absent.
        from app.services.odds_blend import (
            load_goals_odds_for_fixtures,
            compute_final_goals_and_probs,
        )
        _odds_conn = await get_source_connection()
        try:
            goals_odds_map = await load_goals_odds_for_fixtures(
                _odds_conn, next_fix['id'].tolist(),
            )
        finally:
            release_source_connection(_odds_conn)

        for i in range(len(score_preds)):
            bookie_margin = 1 + (
                    score_preds.loc[i, 'Home Odds %'] + score_preds.loc[i, 'Draw Odds %'] + score_preds.loc[
                i, 'Away Odds %'] - 100) / 100
            score_preds.loc[i, 'Home Odds %'] = (score_preds.loc[i, 'Home Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Draw Odds %'] = (score_preds.loc[i, 'Draw Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Away Odds %'] = (score_preds.loc[i, 'Away Odds %'] / bookie_margin).round(2)
            home_goals = score_preds['Home Goals'][i]
            away_goals = score_preds['Away Goals'][i]
            bookie_1x2_pct = None
            if not pd.isna(score_preds['Home Odds %'][i]):
                bookie_1x2_pct = (
                    float(score_preds['Home Odds %'][i]) / 100.0,
                    float(score_preds['Draw Odds %'][i]) / 100.0,
                    float(score_preds['Away Odds %'][i]) / 100.0,
                )
            fixture_id = int(next_fix['id'].iloc[i])
            new_home_goals, new_away_goals, adjusted_home_win_prob, adjusted_draw_prob, adjusted_away_win_prob = (
                compute_final_goals_and_probs(
                    fixture_id,
                    float(home_goals), float(away_goals),
                    bookie_1x2_pct,
                    goals_odds_map.get(fixture_id, {}),
                    odds_beta,
                    boost,
                    dixon_coles_rho,
                )
            )
            score_preds.loc[i, 'Home Goals'] = round(new_home_goals, 2)
            score_preds.loc[i, 'Away Goals'] = round(new_away_goals, 2)
            home_clean_sheet = poisson.pmf(0, new_away_goals)
            away_clean_sheet = poisson.pmf(0, new_home_goals)
            x = np.arange(0, 9)
            y = np.arange(0, 9)
            X, Y = np.meshgrid(x, y)
            Z = poisson.pmf(X, new_home_goals) * poisson.pmf(Y, new_away_goals)
            home_win.append(f"{adjusted_home_win_prob:.2f}%")
            draw.append(f"{adjusted_draw_prob:.2f}%")
            away_win.append(f"{adjusted_away_win_prob:.2f}%")
            home_clean.append(f"{home_clean_sheet * 100:.2f}%")
            away_clean.append(f"{away_clean_sheet * 100:.2f}%")
            over_1_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1]) * 100
            over_2_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1] - Z[2, 0] - Z[0, 2] - Z[1, 1]) * 100
            both_teams_score_prob = (1 - Z[0, :].sum() - Z[:, 0].sum() + Z[0, 0]) * 100
            over_1.append(f"{over_1_goals:.2f}%")
            over_2.append(f"{over_2_goals:.2f}%")
            btts.append(f"{both_teams_score_prob:.2f}%")

        # score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'], inplace=True)
        score_preds['Home Win %'] = home_win
        score_preds['Draw %'] = draw
        score_preds['Away Win %'] = away_win
        score_preds['Home Clean Sheet %'] = home_clean
        score_preds['Away Clean Sheet %'] = away_clean
        score_preds['Over 1.5 Goals %'] = over_1
        score_preds['Over 2.5 Goals %'] = over_2
        score_preds['Both Teams Score %'] = btts
        score_preds['Home Goals'] = score_preds['Home Goals'].round(2)
        score_preds['Away Goals'] = score_preds['Away Goals'].round(2)
        score_preds_with_odds = score_preds.copy()  # NEW - Create a copy with odds included
        score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'],
                         inplace=True)  # NEW - Drop odds from main predictions dataframe



        # In[ ]:

        ## NEW - Update accuracy dataset with new predictions

        score_preds_with_odds.rename(
            columns={'id': 'fixture_id', 'Home Goals': 'Home Projected Goals', 'Away Goals': 'Away Projected Goals'},
            inplace=True)
        score_preds_with_odds['Total Projected Goals'] = score_preds_with_odds['Home Projected Goals'] + \
                                                         score_preds_with_odds['Away Projected Goals']
        score_preds_with_odds['comp_id'] = league_id
        projection_accuracy_dataset_league = pd.concat([projection_accuracy_dataset_league, score_preds_with_odds],
                                                       ignore_index=True)
        score_preds_with_odds.rename(
            columns={'fixture_id': 'id', 'Home Projected Goals': 'Home Goals', 'Away Projected Goals': 'Away Goals'},
            inplace=True)
        score_preds_with_odds.drop(columns=['comp_id', 'Total Projected Goals'], inplace=True)

        # In[ ]:

        ## NEW - 4+ STAR BETS SECTION

        # ## **4+ Star Bets**

        # In[ ]:

        # NEW - Load previous best bets dat and append new best bets

        # best_bets = pd.read_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\Best Bets.xlsx")
        best_bets = ProjectionService._read_df(f"{ProjectionService.DATA_FOLDER_PATH}/Best Bets")

        new_best_bets = pd.DataFrame()
        for i in range(len(score_preds)):
            fix_id = score_preds.loc[i, 'id']
            date = score_preds.loc[i, 'kickoff_datetime']
            date = date.strftime('%d-%m')
            fix = fixtures_df[fixtures_df['id'] == fix_id]
            home_win = float(score_preds.loc[i, 'Home Win %'].strip('%')) / 100
            draw = float(score_preds.loc[i, 'Draw %'].strip('%')) / 100
            away_win = float(score_preds.loc[i, 'Away Win %'].strip('%')) / 100
            over_1_5_goals = float(score_preds.loc[i, 'Over 1.5 Goals %'].strip('%')) / 100
            over_2_5_goals = float(score_preds.loc[i, 'Over 2.5 Goals %'].strip('%')) / 100
            btts = float(score_preds.loc[i, 'Both Teams Score %'].strip('%')) / 100

            # Skip best-bet eval if any bet365 odd is missing (None/NaN).
            # bet365_totals_odds in particular is sparse for some leagues
            # (e.g. Belgian Pro League playoff fixtures not yet priced).
            _odds_vals = [
                fix['bet365_home_odds_decimal'].values[0],
                fix['bet365_draw_odds_decimal'].values[0],
                fix['bet365_away_odds_decimal'].values[0],
                fix['over_1_5_odds_decimal'].values[0],
                fix['over_2_5_odds_decimal'].values[0],
                fix['bet365_btts_yes_odds_decimal'].values[0],
            ]
            if any(v is None or pd.isna(v) for v in _odds_vals):
                continue
            home_win_odds = 1 / _odds_vals[0]
            draw_odds = 1 / _odds_vals[1]
            away_win_odds = 1 / _odds_vals[2]
            over_1_5_goals_odds = 1 / _odds_vals[3]
            over_2_5_goals_odds = 1 / _odds_vals[4]
            btts_odds = 1 / _odds_vals[5]

            home_win_edge = home_win - home_win_odds
            draw_edge = draw - draw_odds
            away_win_edge = away_win - away_win_odds
            over_1_5_goals_edge = over_1_5_goals - over_1_5_goals_odds
            over_2_5_goals_edge = over_2_5_goals - over_2_5_goals_odds
            btts_edge = btts - btts_odds

            home_win_edge_rating = (home_win_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            draw_edge_rating = (draw_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            away_win_edge_rating = (away_win_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            over_1_5_goals_edge_rating = (over_1_5_goals_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            over_2_5_goals_edge_rating = (over_2_5_goals_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            btts_edge_rating = (btts_edge - (-0.1)) * 5 / (0.1 - (-0.1))

            home_win_prob_rating = (home_win) * 5 / (0.9)
            draw_prob_rating = (draw) * 5 / (0.9)
            away_win_prob_rating = (away_win) * 5 / (0.9)
            over_1_5_goals_prob_rating = (over_1_5_goals) * 5 / (0.9)
            over_2_5_goals_prob_rating = (over_2_5_goals) * 5 / (0.9)
            btts_prob_rating = (btts) * 5 / (0.9)

            home_win_total_rating = (home_win_edge_rating * 0.7 if home_win_edge_rating > 0 else 0) + (
                home_win_prob_rating * 0.3 if home_win_prob_rating < 5 else 5 * 0.3)
            draw_total_rating = (draw_edge_rating * 0.7 if draw_edge_rating > 0 else 0) + (
                draw_prob_rating * 0.3 if draw_prob_rating < 5 else 5 * 0.3)
            away_win_total_rating = (away_win_edge_rating * 0.7 if away_win_edge_rating > 0 else 0) + (
                away_win_prob_rating * 0.3 if away_win_prob_rating < 5 else 5 * 0.3)
            over_1_5_goals_total_rating = (
                                              over_1_5_goals_edge_rating * 0.7 if over_1_5_goals_edge_rating > 0 else 0) + (
                                              over_1_5_goals_prob_rating * 0.3 if over_1_5_goals_prob_rating < 5 else 5 * 0.3)
            over_2_5_goals_total_rating = (
                                              over_2_5_goals_edge_rating * 0.7 if over_2_5_goals_edge_rating > 0 else 0) + (
                                              over_2_5_goals_prob_rating * 0.3 if over_2_5_goals_prob_rating < 5 else 5 * 0.3)
            btts_total_rating = (btts_edge_rating * 0.7 if btts_edge_rating > 0 else 0) + (
                btts_prob_rating * 0.3 if btts_prob_rating < 5 else 5 * 0.3)

            for bet_type in ['Home Win', 'Draw', 'Away Win', 'Over 1.5 Goals', 'Over 2.5 Goals', 'BTTS']:
                edge = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_edge']
                edge_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_edge_rating']
                prob_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_prob_rating']
                total_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_total_rating']
                if total_rating >= 4.0:
                    new_best_bets = pd.concat([new_best_bets, pd.DataFrame({
                        'Date': [date],
                        'Competition': [league],
                        'Home Team': [score_preds.loc[i, 'Home Team']],
                        'Away Team': [score_preds.loc[i, 'Away Team']],
                        'Bet Type': [bet_type],
                        'Rating': [round(total_rating, 1) if total_rating < 5 else 5.0],
                        'Edge %': [round(edge * 100, 2)],
                        'Price': [
                            round(1 / locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_odds'], 2)]
                    })], ignore_index=True)

        best_bets = pd.concat([best_bets, new_best_bets], ignore_index=True)
        best_bets.drop_duplicates(subset=['Date', 'Competition', 'Home Team', 'Away Team', 'Bet Type'], keep='last',
                                  inplace=True)
        # best_bets.to_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\Best Bets.xlsx", index=False)
        ProjectionService._write_df(best_bets, f"{ProjectionService.DATA_FOLDER_PATH}/Best Bets")

        # # **League Projections**
        logger.info(f"[{league}] Step: predicted table simulation complete")
        # In[ ]:

        if league != 'Major League Soccer':
            season_fixtures = fixtures.copy()
            today = pd.to_datetime('today')
            season_fixtures['kickoff_datetime'] = pd.to_datetime(season_fixtures['kickoff_datetime'])
            season_fixtures = season_fixtures[season_fixtures['kickoff_datetime'] >= today]
            season_fixtures.loc[:, 'home_team'] = season_fixtures['home_team_id'].map(teams.set_index('id')['name'])
            season_fixtures.loc[:, 'away_team'] = season_fixtures['away_team_id'].map(teams.set_index('id')['name'])
            season_fixtures.sort_values(by='kickoff_datetime', inplace=True)
            season_fixtures.reset_index(drop=True, inplace=True)
            season_fixtures = drop_placeholder_fixtures(season_fixtures, league)

            # The season simulation uses the SAME rating as the fixture
            # projections. There used to be a swap here to a separate
            # team_strength rating, which is how the model could project a
            # team 3rd over the season yet make them underdogs in the match.
            _sim_ratings = ratings

            season_score_preds = make_round_goal_prediction(season_fixtures, _sim_ratings, avg_home_goals, avg_away_goals)

            for i in range(len(season_score_preds)):
                home_goals = season_score_preds['Home Goals'][i]
                away_goals = season_score_preds['Away Goals'][i]

            season_score_preds['Home Goals'] = season_score_preds['Home Goals'].round(2)
            season_score_preds['Away Goals'] = season_score_preds['Away Goals'].round(2)

            current_standings = standings.copy()
            current_standings['Team'] = current_standings['team_id'].map(teams.set_index('id')['name'])
            current_standings.rename(
                columns={'goals_for': 'Goals For', 'goals_against': 'Goals Against', 'points': 'Points'}, inplace=True)
            current_standings['Goal Difference'] = current_standings['Goals For'] - current_standings['Goals Against']
            current_standings = current_standings[['Team', 'Points', 'Goals For', 'Goals Against', 'Goal Difference']]
            current_standings.reset_index(drop=True, inplace=True)
            current_standings = current_standings.astype(
                {'Points': 'int', 'Goals For': 'int', 'Goals Against': 'int', 'Goal Difference': 'int'})
            current_league_table = {
                team: {'Points': points, 'Goals For': gf, 'Goals Against': ga, 'Goal Difference': gd} for
                team, points, gf, ga, gd in current_standings.values}

            # Manual points adjustments — a deduction announced before
            # Sportmonks folds it into `points`. Skipped automatically once
            # the standings already show it, so it cannot double-count.
            current_league_table = await apply_points_adjustments(
                current_league_table, standings, league_id, current_season_id, teams, league)

            avg_table, all_tables = sim_multiple_seasons(season_score_preds, current_league_table, num_sims=10000)

            avg_table_with_probs_and_point_limits = get_avg_table_with_probs_and_point_limits(avg_table,
                                                                                              all_tables)
            # avg_table_with_probs_and_point_limits.to_csv(rf"{save_file_path}\{league} Predicted Table.csv", index=False)
            await insert_predicted_table_async(avg_table_with_probs_and_point_limits, teams, comps, league)
            # Per-team / per-position finishing distribution — every positional
            # market on the read side is a range-sum over this. Non-fatal: must
            # not break the run.
            try:
                await write_position_probabilities_async(all_tables, teams, comps, league)
            except Exception as e:
                logger.error(f"[{league}] league position probabilities write failed (non-fatal): {e}", exc_info=True)

    async def teams(self, league_request):
        league = league_request.league or 'Championship'

        ctx = await self._setup_league(league)

        # Unpack shared context into local variables so downstream code is unchanged
        data_folder_path = ctx.data_folder_path
        model_file_path = ctx.model_file_path
        save_file_path = ctx.save_file_path
        league_dashed = ctx.league_dashed
        date_from = ctx.date_from
        date_to = ctx.date_to
        league_below = ctx.league_below
        league_above = ctx.league_above
        league_below_attack_weight = ctx.league_below_attack_weight
        league_below_defense_weight = ctx.league_below_defense_weight
        league_above_attack_weight = ctx.league_above_attack_weight
        league_above_defense_weight = ctx.league_above_defense_weight
        country_code = ctx.country_code
        div = ctx.div
        weightings = ctx.weightings
        mv_beta = ctx.mv_beta
        odds_beta = ctx.odds_beta
        xG = ctx.xG
        fpl = ctx.fpl
        player_stats = ctx.player_stats
        team_stats = ctx.team_stats
        standings = ctx.standings
        seasons = ctx.seasons
        comps = ctx.comps
        comp_teams = ctx.comp_teams
        teams = ctx.teams
        players = ctx.players
        fixtures_df = ctx.fixtures_df
        b365_odds = ctx.b365_odds
        stats_types = ctx.stats_types
        model_dataset_all = ctx.model_dataset_all
        model_dataset_league = ctx.model_dataset_league
        projection_accuracy_dataset_league = ctx.projection_accuracy_dataset_league
        projection_accuracy_dataset_all = ctx.projection_accuracy_dataset_all
        all_team_ratings = ctx.all_team_ratings
        league_id = ctx.league_id
        fixtures = ctx.fixtures
        league_standings = ctx.league_standings
        league_above_id = ctx.league_above_id
        league_below_id = ctx.league_below_id
        previous_season_id = ctx.previous_season_id
        current_season_id = ctx.current_season_id
        matches_played = ctx.matches_played
        season_fixtures = ctx.season_fixtures
        total_matches = ctx.total_matches
        previous_season_id_below = ctx.previous_season_id_below
        previous_season_id_above = ctx.previous_season_id_above
        stat_list = ctx.stat_list

        ratings = await self._prepare_league(
            league=league, data_folder_path=data_folder_path, model_file_path=model_file_path,
            save_file_path=save_file_path, league_id=league_id, league_dashed=league_dashed,
            model_dataset_all=model_dataset_all, model_dataset_league=model_dataset_league,
            projection_accuracy_dataset_all=projection_accuracy_dataset_all,
            projection_accuracy_dataset_league=projection_accuracy_dataset_league,
            all_team_ratings=all_team_ratings, team_stats=team_stats, player_stats=player_stats,
            teams=teams, stats_types=stats_types, stat_list=stat_list,
            comp_teams=comp_teams, fixtures_df=fixtures_df, fixtures=fixtures, seasons=seasons, comps=comps,
            current_season_id=current_season_id, previous_season_id=previous_season_id,
            previous_season_id_above=previous_season_id_above,
            previous_season_id_below=previous_season_id_below,
            weightings=weightings, mv_beta=mv_beta, odds_beta=odds_beta,
            country_code=country_code, div=div, matches_played=matches_played, standings=standings,
            league_above=league_above, league_below=league_below, league_standings=league_standings,
            league_below_attack_weight=league_below_attack_weight,
            league_below_defense_weight=league_below_defense_weight,
            league_above_id=league_above_id, league_below_id=league_below_id,
            xG=xG, fpl=fpl, b365_odds=b365_odds,
            season_fixtures=season_fixtures, total_matches=total_matches, players=players,
            mode=(league_request.mode if hasattr(league_request, 'mode') and league_request.mode else "full"),
        )

        # ## **Make Predictions for Next Fixture Round**
        #
        # Result, Score, Clean Sheets, Over 1.5, Over 2.5 and BTTS all calculated here using Poisson Distribution.

        # In[18]:

        next_fix = ProjectionService._filter_upcoming_fixtures(league, fixtures, date_from, date_to)
        fixtures['kickoff_datetime'] = pd.to_datetime(fixtures['kickoff_datetime'])
        if hasattr(league_request, 'fixture_ids') and league_request.fixture_ids:
            next_fix = next_fix[next_fix['id'].isin(league_request.fixture_ids)]
            logger.info(f'[{league}] Filtered to {len(next_fix)} of {len(fixtures[(fixtures["kickoff_datetime"] >= date_from) & (fixtures["kickoff_datetime"] <= date_to)])} fixtures')
        next_fix = next_fix[
            ['id', 'kickoff_datetime', 'name', 'home_team_id', 'away_team_id', 'bet365_home_odds_decimal',
             'bet365_draw_odds_decimal', 'bet365_away_odds_decimal']]
        next_fix['home_team'] = next_fix['home_team_id'].apply(lambda x: get_team(x, teams))
        next_fix['away_team'] = next_fix['away_team_id'].apply(lambda x: get_team(x, teams))
        next_fix = next_fix.drop(columns=['home_team_id', 'away_team_id'])
        next_fix = drop_placeholder_fixtures(next_fix, league)
        next_fix.sort_values(by=['kickoff_datetime', 'home_team'], inplace=True)
        next_fix.reset_index(drop=True, inplace=True)

        # In[ ]:

        avg_home_goals = get_home_goal_avg(league_id, team_stats, fixtures, stats_types)
        avg_away_goals = get_away_goal_avg(league_id, team_stats, fixtures, stats_types)
        score_preds = make_round_goal_prediction(next_fix, ratings, avg_home_goals, avg_away_goals)
        # boost = get_draw_boost(ratings, avg_home_goals, avg_away_goals, get_draw_perc(league_id, fixtures))
        # Dixon-Coles replaces the flat draw boost where a league has rho
        # configured; get_result_probs applies one or the other, never both.
        dixon_coles_rho = getattr(ctx, 'dixon_coles_rho', 0.0) or 0.0
        boost = 1.0 if dixon_coles_rho else 1.1
        score_preds['Home Odds %'] = ((1 / next_fix['bet365_home_odds_decimal']) * 100)
        score_preds['Draw Odds %'] = ((1 / next_fix['bet365_draw_odds_decimal']) * 100)
        score_preds['Away Odds %'] = ((1 / next_fix['bet365_away_odds_decimal']) * 100)

        home_win = []
        draw = []
        away_win = []
        home_clean = []
        away_clean = []
        over_1 = []
        over_2 = []
        btts = []

        # Pre-load bet365 goals over/under for the upcoming fixtures.
        # The cascade in compute_final_goals_and_probs (paths 1-3) uses
        # per-team + match-total ladders directly; path 4 (legacy 1X2
        # reverse-solve) is the fallback when those markets are absent.
        from app.services.odds_blend import (
            load_goals_odds_for_fixtures,
            compute_final_goals_and_probs,
        )
        _odds_conn = await get_source_connection()
        try:
            goals_odds_map = await load_goals_odds_for_fixtures(
                _odds_conn, next_fix['id'].tolist(),
            )
        finally:
            release_source_connection(_odds_conn)

        for i in range(len(score_preds)):
            bookie_margin = 1 + (
                    score_preds.loc[i, 'Home Odds %'] + score_preds.loc[i, 'Draw Odds %'] + score_preds.loc[
                i, 'Away Odds %'] - 100) / 100
            score_preds.loc[i, 'Home Odds %'] = (score_preds.loc[i, 'Home Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Draw Odds %'] = (score_preds.loc[i, 'Draw Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Away Odds %'] = (score_preds.loc[i, 'Away Odds %'] / bookie_margin).round(2)
            home_goals = score_preds['Home Goals'][i]
            away_goals = score_preds['Away Goals'][i]
            bookie_1x2_pct = None
            if not pd.isna(score_preds['Home Odds %'][i]):
                bookie_1x2_pct = (
                    float(score_preds['Home Odds %'][i]) / 100.0,
                    float(score_preds['Draw Odds %'][i]) / 100.0,
                    float(score_preds['Away Odds %'][i]) / 100.0,
                )
            fixture_id = int(next_fix['id'].iloc[i])
            new_home_goals, new_away_goals, adjusted_home_win_prob, adjusted_draw_prob, adjusted_away_win_prob = (
                compute_final_goals_and_probs(
                    fixture_id,
                    float(home_goals), float(away_goals),
                    bookie_1x2_pct,
                    goals_odds_map.get(fixture_id, {}),
                    odds_beta,
                    boost,
                    dixon_coles_rho,
                )
            )
            score_preds.loc[i, 'Home Goals'] = round(new_home_goals, 2)
            score_preds.loc[i, 'Away Goals'] = round(new_away_goals, 2)
            home_clean_sheet = poisson.pmf(0, new_away_goals)
            away_clean_sheet = poisson.pmf(0, new_home_goals)
            x = np.arange(0, 9)
            y = np.arange(0, 9)
            X, Y = np.meshgrid(x, y)
            Z = poisson.pmf(X, new_home_goals) * poisson.pmf(Y, new_away_goals)
            home_win.append(f"{adjusted_home_win_prob:.2f}%")
            draw.append(f"{adjusted_draw_prob:.2f}%")
            away_win.append(f"{adjusted_away_win_prob:.2f}%")
            home_clean.append(f"{home_clean_sheet * 100:.2f}%")
            away_clean.append(f"{away_clean_sheet * 100:.2f}%")
            over_1_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1]) * 100
            over_2_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1] - Z[2, 0] - Z[0, 2] - Z[1, 1]) * 100
            both_teams_score_prob = (1 - Z[0, :].sum() - Z[:, 0].sum() + Z[0, 0]) * 100
            over_1.append(f"{over_1_goals:.2f}%")
            over_2.append(f"{over_2_goals:.2f}%")
            btts.append(f"{both_teams_score_prob:.2f}%")

        # score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'], inplace=True)
        score_preds['Home Win %'] = home_win
        score_preds['Draw %'] = draw
        score_preds['Away Win %'] = away_win
        score_preds['Home Clean Sheet %'] = home_clean
        score_preds['Away Clean Sheet %'] = away_clean
        score_preds['Over 1.5 Goals %'] = over_1
        score_preds['Over 2.5 Goals %'] = over_2
        score_preds['Both Teams Score %'] = btts
        score_preds['Home Goals'] = score_preds['Home Goals'].round(2)
        score_preds['Away Goals'] = score_preds['Away Goals'].round(2)
        score_preds_with_odds = score_preds.copy()  # NEW - Create a copy with odds included
        score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'],
                         inplace=True)


        # In[ ]:

        ## NEW - Update accuracy dataset with new predictions

        score_preds_with_odds.rename(
            columns={'id': 'fixture_id', 'Home Goals': 'Home Projected Goals', 'Away Goals': 'Away Projected Goals'},
            inplace=True)
        score_preds_with_odds['Total Projected Goals'] = score_preds_with_odds['Home Projected Goals'] + \
                                                         score_preds_with_odds['Away Projected Goals']
        score_preds_with_odds['comp_id'] = league_id
        projection_accuracy_dataset_league = pd.concat([projection_accuracy_dataset_league, score_preds_with_odds],
                                                       ignore_index=True)
        score_preds_with_odds.rename(
            columns={'fixture_id': 'id', 'Home Projected Goals': 'Home Goals', 'Away Projected Goals': 'Away Goals'},
            inplace=True)
        score_preds_with_odds.drop(columns=['comp_id', 'Total Projected Goals'], inplace=True)

        # In[ ]:

        ## NEW - 4+ STAR BETS SECTION

        # ## **4+ Star Bets**

        # In[ ]:

        # NEW - Load previous best bets file and append new best bets

        # best_bets = pd.read_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\Best Bets.xlsx")
        best_bets = ProjectionService._read_df(f"{ProjectionService.DATA_FOLDER_PATH}/Best Bets")

        new_best_bets = pd.DataFrame()
        for i in range(len(score_preds)):
            fix_id = score_preds.loc[i, 'id']
            date = score_preds.loc[i, 'kickoff_datetime']
            date = date.strftime('%d-%m')
            fix = fixtures_df[fixtures_df['id'] == fix_id]
            home_win = float(score_preds.loc[i, 'Home Win %'].strip('%')) / 100
            draw = float(score_preds.loc[i, 'Draw %'].strip('%')) / 100
            away_win = float(score_preds.loc[i, 'Away Win %'].strip('%')) / 100
            over_1_5_goals = float(score_preds.loc[i, 'Over 1.5 Goals %'].strip('%')) / 100
            over_2_5_goals = float(score_preds.loc[i, 'Over 2.5 Goals %'].strip('%')) / 100
            btts = float(score_preds.loc[i, 'Both Teams Score %'].strip('%')) / 100

            # Skip best-bet eval if any bet365 odd is missing (None/NaN).
            # bet365_totals_odds in particular is sparse for some leagues
            # (e.g. Belgian Pro League playoff fixtures not yet priced).
            _odds_vals = [
                fix['bet365_home_odds_decimal'].values[0],
                fix['bet365_draw_odds_decimal'].values[0],
                fix['bet365_away_odds_decimal'].values[0],
                fix['over_1_5_odds_decimal'].values[0],
                fix['over_2_5_odds_decimal'].values[0],
                fix['bet365_btts_yes_odds_decimal'].values[0],
            ]
            if any(v is None or pd.isna(v) for v in _odds_vals):
                continue
            home_win_odds = 1 / _odds_vals[0]
            draw_odds = 1 / _odds_vals[1]
            away_win_odds = 1 / _odds_vals[2]
            over_1_5_goals_odds = 1 / _odds_vals[3]
            over_2_5_goals_odds = 1 / _odds_vals[4]
            btts_odds = 1 / _odds_vals[5]

            home_win_edge = home_win - home_win_odds
            draw_edge = draw - draw_odds
            away_win_edge = away_win - away_win_odds
            over_1_5_goals_edge = over_1_5_goals - over_1_5_goals_odds
            over_2_5_goals_edge = over_2_5_goals - over_2_5_goals_odds
            btts_edge = btts - btts_odds

            home_win_edge_rating = (home_win_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            draw_edge_rating = (draw_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            away_win_edge_rating = (away_win_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            over_1_5_goals_edge_rating = (over_1_5_goals_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            over_2_5_goals_edge_rating = (over_2_5_goals_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            btts_edge_rating = (btts_edge - (-0.1)) * 5 / (0.1 - (-0.1))

            home_win_prob_rating = (home_win) * 5 / (0.9)
            draw_prob_rating = (draw) * 5 / (0.9)
            away_win_prob_rating = (away_win) * 5 / (0.9)
            over_1_5_goals_prob_rating = (over_1_5_goals) * 5 / (0.9)
            over_2_5_goals_prob_rating = (over_2_5_goals) * 5 / (0.9)
            btts_prob_rating = (btts) * 5 / (0.9)

            home_win_total_rating = (home_win_edge_rating * 0.7 if home_win_edge_rating > 0 else 0) + (
                home_win_prob_rating * 0.3 if home_win_prob_rating < 5 else 5 * 0.3)
            draw_total_rating = (draw_edge_rating * 0.7 if draw_edge_rating > 0 else 0) + (
                draw_prob_rating * 0.3 if draw_prob_rating < 5 else 5 * 0.3)
            away_win_total_rating = (away_win_edge_rating * 0.7 if away_win_edge_rating > 0 else 0) + (
                away_win_prob_rating * 0.3 if away_win_prob_rating < 5 else 5 * 0.3)
            over_1_5_goals_total_rating = (
                                              over_1_5_goals_edge_rating * 0.7 if over_1_5_goals_edge_rating > 0 else 0) + (
                                              over_1_5_goals_prob_rating * 0.3 if over_1_5_goals_prob_rating < 5 else 5 * 0.3)
            over_2_5_goals_total_rating = (
                                              over_2_5_goals_edge_rating * 0.7 if over_2_5_goals_edge_rating > 0 else 0) + (
                                              over_2_5_goals_prob_rating * 0.3 if over_2_5_goals_prob_rating < 5 else 5 * 0.3)
            btts_total_rating = (btts_edge_rating * 0.7 if btts_edge_rating > 0 else 0) + (
                btts_prob_rating * 0.3 if btts_prob_rating < 5 else 5 * 0.3)

            for bet_type in ['Home Win', 'Draw', 'Away Win', 'Over 1.5 Goals', 'Over 2.5 Goals', 'BTTS']:
                edge = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_edge']
                edge_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_edge_rating']
                prob_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_prob_rating']
                total_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_total_rating']
                if total_rating >= 4.0:
                    new_best_bets = pd.concat([new_best_bets, pd.DataFrame({
                        'Date': [date],
                        'Competition': [league],
                        'Home Team': [score_preds.loc[i, 'Home Team']],
                        'Away Team': [score_preds.loc[i, 'Away Team']],
                        'Bet Type': [bet_type],
                        'Rating': [round(total_rating, 1) if total_rating < 5 else 5.0],
                        'Edge %': [round(edge * 100, 2)],
                        'Price': [
                            round(1 / locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_odds'], 2)]
                    })], ignore_index=True)

        best_bets = pd.concat([best_bets, new_best_bets], ignore_index=True)
        best_bets.drop_duplicates(subset=['Date', 'Competition', 'Home Team', 'Away Team', 'Bet Type'], keep='last',
                                  inplace=True)
        # best_bets.to_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\Best Bets.xlsx", index=False)
        ProjectionService._write_df(best_bets, f"{ProjectionService.DATA_FOLDER_PATH}/Best Bets")

        # # **League Projections**
        logger.info(f"[{league}] Step: predicted table simulation complete")
        # In[ ]:


        stat_list = get_stat_list(league_id)

        # In[21]:

        models = load_all_models(stat_list, ProjectionService.MODEL_FILE_PATH)

        # In[22]:

        if next_fix.empty:
            return Response(status_code=204)

        todays_date = pd.to_datetime(next_fix['kickoff_datetime'].iloc[0]).date()

        # In[ ]:

        team_projections = get_team_round_predictions(next_fix, stat_list, fixtures_df, team_stats, teams, stats_types,
                                                      models, ratings=ratings,
                                                      league_weightings=[league_above_attack_weight,
                                                                         league_above_defense_weight,
                                                                         league_below_attack_weight,
                                                                         league_below_defense_weight],
                                                      season_id=[current_season_id, previous_season_id,
                                                                 previous_season_id_above, previous_season_id_below],
                                                      games=50,
                                                      comp_teams=comp_teams[comp_teams['competition_id'] == league_id])

        # In[ ]:

        ## NEW - Add historical stats to the model dataset and drop them from team projections afterwards

        new_rows = []

        for i in range(len(team_projections)):
            team_df = team_projections.iloc[[i]]
            new_row = {}
            new_row['id'] = team_df['fixture_id'].values[0]
            new_row['kickoff_datetime'] = team_df['kickoff_datetime'].values[0]
            new_row['comp_id'] = league_id
            new_row['Team'] = team_df['Team'].values[0]
            new_row['Opponent'] = team_df['Opponent'].values[0]
            new_row['Venue'] = team_df['Venue'].values[0]
            for stat in stat_list:
                new_row['Team ' + stat + ' History'] = team_df['Team ' + stat + ' History'].values[0]
                new_row['Opponent ' + stat + ' History Against'] = \
                    team_df['Opponent ' + stat + ' History Against'].values[0]
            new_rows.append(new_row)

        model_dataset_league = pd.concat([model_dataset_league, pd.DataFrame(new_rows)], ignore_index=True)
        model_dataset_all = pd.concat([model_dataset_all, pd.DataFrame(new_rows)], ignore_index=True)
        model_dataset_league.drop_duplicates(subset=['id', 'Team', 'Opponent', 'Venue'], keep='last', inplace=True)
        model_dataset_all.drop_duplicates(subset=['id', 'Team', 'Opponent', 'Venue'], keep='last', inplace=True)

        ProjectionService._write_df(model_dataset_league, f"{ProjectionService.DATA_FOLDER_PATH}/{league}_model_dataset_with_history")
        ProjectionService._write_df(model_dataset_all, f"{ProjectionService.DATA_FOLDER_PATH}/all_leagues_model_dataset_with_history")
        # Dual-write to DB (see projections() for rationale).
        try:
            from app.repository.projection_dataset_repo import insert_model_dataset_async
            await insert_model_dataset_async(model_dataset_league, league_id, league, teams, fixtures_df, comp_teams)
        except Exception as _db_err:
            logger.warning(f"[{league}] model_dataset DB dual-write failed: {_db_err}")

        # model_dataset_league.to_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\{league}_model_dataset_with_history.xlsx", index=False)
        # model_dataset_all.to_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\all_leagues_model_dataset_with_history.xlsx", index=False)

        team_projections.drop(
            columns=['Team ' + stat + ' History' for stat in stat_list] + ['Opponent ' + stat + ' History Against' for
                                                                           stat in stat_list], inplace=True)

        # In[ ]:

        avg_goals = (avg_home_goals + avg_away_goals) / 2

        league_team_stats = team_stats[
            team_stats['fixture_id'].isin(fixtures_df[fixtures_df['competition_id'] == league_id]['id'])]

        league_shots = league_team_stats[league_team_stats['stats_type_id'] == get_stat_id('Shots Total',
                                                                                           stats_types)].copy()  # NEW - all team shots for specific league
        league_shots['Date'] = league_shots['fixture_id'].map(
            fixtures_df.set_index('id')['kickoff_datetime'])  # NEW - map fixture dates
        league_shots['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(
            league_shots['Date'])).dt.days // 7  # NEW - calculate weeks since kickoff
        league_shots['Weight'] = 0.9 ** (
                league_shots['Weeks Since Kickoff'] - 5)  # NEW - apply weighting to more recent matches
        league_shots.loc[league_shots['Weeks Since Kickoff'] < 6, 'Weight'] = 1  # NEW - full weight for last 5 weeks
        league_shots['Weighted Shots'] = league_shots['Weight'] * league_shots[
            'value']  # NEW - calculate weighted shots
        avg_shots = league_shots['Weighted Shots'].sum() / league_shots[
            'Weight'].sum()  # UPDATED - new formula for average shots

        league_shots_on_target = league_team_stats[league_team_stats['stats_type_id'] == get_stat_id('Shots On Target',
                                                                                                     stats_types)].copy()  # NEW - all team shots on target for specific league
        league_shots_on_target['Date'] = league_shots_on_target['fixture_id'].map(
            fixtures_df.set_index('id')['kickoff_datetime'])  # NEW - map fixture dates
        league_shots_on_target['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(
            league_shots_on_target['Date'])).dt.days // 7  # NEW - calculate weeks since kickoff
        league_shots_on_target['Weight'] = 0.9 ** (
                league_shots_on_target['Weeks Since Kickoff'] - 5)  # NEW - apply weighting to more recent matches
        league_shots_on_target.loc[
            league_shots_on_target['Weeks Since Kickoff'] < 6, 'Weight'] = 1  # NEW - full weight for last 5 weeks
        league_shots_on_target['Weighted Shots On Target'] = league_shots_on_target['Weight'] * league_shots_on_target[
            'value']
        avg_shots_on_target = league_shots_on_target['Weighted Shots On Target'].sum() / league_shots_on_target[
            'Weight'].sum()

        avg_shots_per_goal = avg_shots / avg_goals
        avg_shots_on_target_per_goal = avg_shots_on_target / avg_goals
        goals = []
        assists = []
        for i in range(len(team_projections)):
            team = team_projections['Team'].iloc[i]
            opp = team_projections['Opponent'].iloc[i]
            fixture = score_preds[score_preds['id'] == team_projections['fixture_id'].iloc[i]]
            team_pred = fixture['Home Goals'].values[0] if fixture['Home Team'].values[0] == team else \
                fixture['Away Goals'].values[0]
            opp_pred = fixture['Away Goals'].values[0] if fixture['Home Team'].values[0] == opp else \
                fixture['Home Goals'].values[0]
            goals.append(team_pred)
            assists.append((team_pred * 0.82).round(2))
            projected_shots = team_projections['Shots Total'].iloc[i]
            projected_shots_on_target = team_projections['Shots On Target'].iloc[i]

            adjusted_shots, adjusted_shots_on_target = adjust_shots_projection(
                team_pred,
                projected_shots,
                projected_shots_on_target,
                avg_shots_per_goal,
                avg_shots_on_target_per_goal
            )
            team_projections.at[i, 'Shots Total'] = adjusted_shots
            team_projections.at[i, 'Shots On Target'] = adjusted_shots_on_target

        team_projections['Goals'] = goals
        team_projections['Assists'] = assists

        # PL only: project team-level Ball Recovery + CBI(FPL) per fixture.
        # No PoissonRegressor exists for these stats (Sportmonks contributes
        # zero team-level rows); use get_simple_team_stat_prediction's
        # closed-form opponent-adjusted weighted average.
        # distribute_team_predictions_to_players auto-projects per-player
        # values from any column on team_projections, so adding these here
        # gives us per-player Recoveries + CBI for the team-down CBIT calc.
        if fpl:
            _lw_def = [league_above_attack_weight, league_above_defense_weight,
                       league_below_attack_weight, league_below_defense_weight]
            _sid_def = [current_season_id, previous_season_id,
                        previous_season_id_above, previous_season_id_below]
            _cpl_def = comp_teams[comp_teams['competition_id'] == league_id]
            _rec_col = []
            _cbi_col = []
            _cbit_col = []
            for i in range(len(team_projections)):
                _row = team_projections.iloc[i]
                try:
                    rec_v, _, _ = get_simple_team_stat_prediction(
                        _row['Team'], _row['Opponent'], fixtures_df, 'Ball Recovery',
                        team_stats, teams, stats_types,
                        ratings=ratings, venue=_row['Venue'], comp_id=league_id,
                        league_weightings=_lw_def, season_id=_sid_def, games=50,
                        comp_teams=_cpl_def,
                    )
                except Exception:
                    rec_v = 0
                try:
                    cbi_v, _, _ = get_simple_team_stat_prediction(
                        _row['Team'], _row['Opponent'], fixtures_df,
                        'Clearances Blocks Interceptions (FPL)',
                        team_stats, teams, stats_types,
                        ratings=ratings, venue=_row['Venue'], comp_id=league_id,
                        league_weightings=_lw_def, season_id=_sid_def, games=50,
                        comp_teams=_cpl_def,
                    )
                except Exception:
                    cbi_v = 0
                # Combined CBIT — the quantity FPL scores the threshold
                # against. Projected as ONE stat rather than assembling a
                # modelled Tackles with a blended CBI lump: FPL and Sportmonks
                # agree on the total to 0.3%, so one definition costs nothing
                # and removes the three inconsistent notions the pipeline
                # carried. George, 2026-08-04.
                try:
                    cbit_v, _cbit_th, _cbit_oh = get_simple_team_stat_prediction(
                        _row['Team'], _row['Opponent'], fixtures_df,
                        'Clearances Blocks Interceptions Tackles (FPL)',
                        team_stats, teams, stats_types,
                        ratings=ratings, venue=_row['Venue'], comp_id=league_id,
                        league_weightings=_lw_def, season_id=_sid_def, games=50,
                        comp_teams=_cpl_def,
                    )
                except Exception:
                    cbit_v = 0
                    _cbit_th = _cbit_oh = None
                # Diagnostic for the ~5% under-projection measured 2026-08-05
                # (projected 51.88 CBIT/match against an actual 55.59; 13 of 15
                # clubs low, and NOT explained by opponent mix or missing
                # history). The blend is
                #     alpha * team_history + (1-alpha) * opponent_history
                # and both terms are returned but discarded, so log them for a
                # handful of fixtures to see which side is light before
                # guessing at the cause.
                if i < 6:
                    logger.info(
                        f"[{league}] CBIT blend probe: {_row['Team']} vs {_row['Opponent']} "
                        f"({_row['Venue']}) team_hist={_cbit_th} opp_hist={_cbit_oh} "
                        f"-> blended={cbit_v}"
                    )
                _rec_col.append(rec_v)
                _cbi_col.append(cbi_v)
                _cbit_col.append(cbit_v)
            team_projections['Ball Recovery'] = _rec_col
            team_projections['Clearances Blocks Interceptions (FPL)'] = _cbi_col
            team_projections['Clearances Blocks Interceptions Tackles (FPL)'] = _cbit_col
            # Tackles Won rides a team TACKLES total, and that total is
            # CBIT - CBI rather than the modelled 'Tackles' column. Measured
            # against 760 real PL team-matches: the model averages 14.37 against
            # an actual 16.69 (-13.9%), while CBIT - CBI gives 15.74 (-5.7%) —
            # the two blended quantities are uniformly ~7% light, so their
            # DIFFERENCE tracks tackles better than the dedicated model does.
            # George called it before it was measured, 2026-08-05.
            #
            # It also makes BPS reconcile with DefCon by construction: BPS scores
            # CBI + tackles = CBI + (CBIT - CBI) = CBIT, the exact quantity the
            # DefCon threshold uses. Projecting the two separately had them
            # disagreeing by 10-14% on promoted clubs (Coventry 58.05 vs 51.00).
            #
            # The SHARE is unchanged and still measured from history — his
            # tackles won over his team's actual tackles (_TEAM_DENOMINATOR_ALIAS
            # maps the denominator to 'Tackles'), so a player's personal success
            # rate is folded in automatically.
            _tkl_from_cbit = (team_projections['Clearances Blocks Interceptions Tackles (FPL)']
                              - team_projections['Clearances Blocks Interceptions (FPL)'])
            # A fixture where the CBI blend outruns the CBIT blend would give a
            # negative tackle count; fall back to the modelled column there.
            team_projections['Tackles Won'] = _tkl_from_cbit.where(
                _tkl_from_cbit > 0, team_projections['Tackles'])
            # Split the goal projection into penalty and non-penalty halves.
            #
            # Penalties are taken as a fixed PROPORTION of projected goals, not
            # a flat per-match constant, so an attacking team is projected more
            # of them (George's call, 2026-08-07). Note the measured
            # correlation between a team's goals and its penalties across PL
            # 25/26 was 0.029 — i.e. none — with Brentford winning 10 on 55
            # goals against Man City's 4 on 77. George's position is that the
            # mechanism is real and one season of 2-10 counts is too noisy to
            # disprove it; the risk is City runs high and Brentford low.
            #
            # Constants measured over PL 24/25 + 25/26 (175 penalties):
            # penalties are 6.76% of goals, converted at 83.4%.
            # A proportional split cannot leak or invent goals — the two halves
            # always sum back to the original projection.
            _pg = team_projections['Goals'] * PENALTY_GOAL_SHARE
            team_projections['Non-Penalty Goals'] = team_projections['Goals'] - _pg
            # Named 'Penalties Scored' rather than 'Penalty Goals' so it matches
            # the PLAYER stat of the same name (111). distribute() resolves a
            # share for every team column by looking up the identically-named
            # player stat, so a column with no player counterpart hands every
            # player a zero share. Attempts and misses derive from this
            # downstream (attempts = scored / conversion), avoiding a second
            # distributed column that would need its own share.
            team_projections['Penalties Scored'] = _pg

        saves = []
        for i in range(len(team_projections)):
            fixture_id = team_projections['fixture_id'].iloc[i]
            fixture_team_projections = team_projections[
                team_projections['fixture_id'] == fixture_id]
            fixture_team_projections = fixture_team_projections.drop(
                i)
            saves.append(
                fixture_team_projections['Shots On Target'].values[0] - fixture_team_projections['Goals'].values[
                    0])

        team_projections['Saves'] = saves
        team_projections['Saves'] = team_projections['Saves'].round(2)  # NEW - Round saves to 2 decimal places
        # PL projects Key Passes properly (get_stat_list). Everywhere else it
        # stays derived — measured 0.72-0.74 across the top 5, so 0.75 runs a
        # little high. George, 2026-08-02.
        if 'Key Passes' not in team_projections.columns:
            team_projections['Key Passes'] = (team_projections['Shots Total'] * 0.75).round(2)
        # Retain Ball Recovery + CBI(FPL) columns when present (added by the
        # PL-only block above). Other leagues skip these columns.
        _extra_def_cols = [c for c in ['Ball Recovery', 'Clearances Blocks Interceptions (FPL)', 'Clearances Blocks Interceptions Tackles (FPL)', 'Tackles Won', 'Non-Penalty Goals', 'Penalties Scored']
                           if c in team_projections.columns]
        team_projections = team_projections[
            ['fixture_id', 'kickoff_datetime', 'Team', 'Opponent', 'Venue', 'Goals', 'Assists',
             'Key Passes'] + [c for c in stat_list if c != 'Key Passes'] + ['Fouls Drawn', 'Saves'] + _extra_def_cols]
        team_projections.rename(columns={'Successful Passes': 'Accurate Passes'}, inplace=True)
        logger.debug(f"[{league}] team_projections columns ready")

        # ── Team-stat odds-blend ──
        # Reels each team's projected stats (corners/cards/shots/SoT/
        # fouls/tackles) toward bookmaker expected via the cascade
        # (Path 1 per-team ladder → Path 1.5 partial+match → Path 2
        # match-split via model ratio). Per-stat bookmaker priority
        # lists in TEAM_STAT_BOOKIE_PRIORITY. Falls back to model
        # unchanged for any (stat, fixture) with no usable book data.
        from app.services.odds_blend import (
            load_team_stat_odds, blend_team_stat,
            TEAM_STAT_BOOKIE_PRIORITY, STAT_COLUMN_TO_MARKET,
        )
        _fix_ids = team_projections['fixture_id'].astype(int).unique().tolist()
        _odds_per_market = {}
        _odds_conn = await get_source_connection()
        try:
            for _market, _books in TEAM_STAT_BOOKIE_PRIORITY.items():
                _odds_per_market[_market] = await load_team_stat_odds(
                    _odds_conn, _fix_ids, _market, _books,
                )
        finally:
            release_source_connection(_odds_conn)

        _fid_to_home_team = {}
        for _fid in _fix_ids:
            _row = next_fix[next_fix['id'] == _fid]
            if not _row.empty:
                _fid_to_home_team[_fid] = _row['home_team'].iloc[0]

        _seen_fixtures = set()
        for _i in range(len(team_projections)):
            fid = int(team_projections['fixture_id'].iloc[_i])
            if fid in _seen_fixtures:
                continue
            _seen_fixtures.add(fid)
            pair = team_projections[team_projections['fixture_id'] == fid]
            if len(pair) != 2:
                continue
            home_team_name = _fid_to_home_team.get(fid)
            if not home_team_name:
                continue
            home_mask = (team_projections['fixture_id'] == fid) & (team_projections['Team'] == home_team_name)
            away_mask = (team_projections['fixture_id'] == fid) & (team_projections['Team'] != home_team_name)

            for stat_col, market in STAT_COLUMN_TO_MARKET.items():
                if stat_col not in team_projections.columns:
                    continue
                try:
                    mh = float(team_projections.loc[home_mask, stat_col].iloc[0])
                    ma = float(team_projections.loc[away_mask, stat_col].iloc[0])
                except (IndexError, KeyError, ValueError):
                    continue
                fh, fa = blend_team_stat(
                    mh, ma,
                    _odds_per_market.get(market, {}).get(fid, {}),
                    market, odds_beta,
                )
                team_projections.loc[home_mask, stat_col] = round(fh, 2)
                team_projections.loc[away_mask, stat_col] = round(fa, 2)
        
        team_projections_save = team_projections.copy()
        
        team_projections_save.drop(
            ['Assists', 'Fouls Drawn', 'Saves', 'Key Passes'],
            axis=1,
            inplace=True,
            errors='ignore'
        )

        team_projections_save = team_projections_save.round(2)

        await insert_teams_async(team_projections_save, teams=teams, competition_id=league_id, comp_teams=comp_teams)


    async def players(self, league_request):
        league = league_request.league or 'Championship'

        ctx = await self._setup_league(league)

        # Unpack shared context into local variables so downstream code is unchanged
        data_folder_path = ctx.data_folder_path
        model_file_path = ctx.model_file_path
        save_file_path = ctx.save_file_path
        league_dashed = ctx.league_dashed
        date_from = ctx.date_from
        date_to = ctx.date_to
        league_below = ctx.league_below
        league_above = ctx.league_above
        league_below_attack_weight = ctx.league_below_attack_weight
        league_below_defense_weight = ctx.league_below_defense_weight
        league_above_attack_weight = ctx.league_above_attack_weight
        league_above_defense_weight = ctx.league_above_defense_weight
        country_code = ctx.country_code
        div = ctx.div
        weightings = ctx.weightings
        mv_beta = ctx.mv_beta
        odds_beta = ctx.odds_beta
        xG = ctx.xG
        fpl = ctx.fpl
        player_stats = ctx.player_stats
        team_stats = ctx.team_stats
        standings = ctx.standings
        seasons = ctx.seasons
        comps = ctx.comps
        comp_teams = ctx.comp_teams
        teams = ctx.teams
        players = ctx.players
        fixtures_df = ctx.fixtures_df
        b365_odds = ctx.b365_odds
        stats_types = ctx.stats_types
        model_dataset_all = ctx.model_dataset_all
        model_dataset_league = ctx.model_dataset_league
        projection_accuracy_dataset_league = ctx.projection_accuracy_dataset_league
        projection_accuracy_dataset_all = ctx.projection_accuracy_dataset_all
        all_team_ratings = ctx.all_team_ratings
        league_id = ctx.league_id
        fixtures = ctx.fixtures
        league_standings = ctx.league_standings
        league_above_id = ctx.league_above_id
        league_below_id = ctx.league_below_id
        previous_season_id = ctx.previous_season_id
        current_season_id = ctx.current_season_id
        matches_played = ctx.matches_played
        season_fixtures = ctx.season_fixtures
        total_matches = ctx.total_matches
        previous_season_id_below = ctx.previous_season_id_below
        previous_season_id_above = ctx.previous_season_id_above
        stat_list = ctx.stat_list

        ratings = await self._prepare_league(
            league=league, data_folder_path=data_folder_path, model_file_path=model_file_path,
            save_file_path=save_file_path, league_id=league_id, league_dashed=league_dashed,
            model_dataset_all=model_dataset_all, model_dataset_league=model_dataset_league,
            projection_accuracy_dataset_all=projection_accuracy_dataset_all,
            projection_accuracy_dataset_league=projection_accuracy_dataset_league,
            all_team_ratings=all_team_ratings, team_stats=team_stats, player_stats=player_stats,
            teams=teams, stats_types=stats_types, stat_list=stat_list,
            comp_teams=comp_teams, fixtures_df=fixtures_df, fixtures=fixtures, seasons=seasons, comps=comps,
            current_season_id=current_season_id, previous_season_id=previous_season_id,
            previous_season_id_above=previous_season_id_above,
            previous_season_id_below=previous_season_id_below,
            weightings=weightings, mv_beta=mv_beta, odds_beta=odds_beta,
            country_code=country_code, div=div, matches_played=matches_played, standings=standings,
            league_above=league_above, league_below=league_below, league_standings=league_standings,
            league_below_attack_weight=league_below_attack_weight,
            league_below_defense_weight=league_below_defense_weight,
            league_above_id=league_above_id, league_below_id=league_below_id,
            xG=xG, fpl=fpl, b365_odds=b365_odds,
            season_fixtures=season_fixtures, total_matches=total_matches, players=players,
            mode=(league_request.mode if hasattr(league_request, 'mode') and league_request.mode else "full"),
        )

        # ## **Make Predictions for Next Fixture Round**
        #
        # Result, Score, Clean Sheets, Over 1.5, Over 2.5 and BTTS all calculated here using Poisson Distribution.

        # In[18]:

        next_fix = ProjectionService._filter_upcoming_fixtures(league, fixtures, date_from, date_to)
        fixtures['kickoff_datetime'] = pd.to_datetime(fixtures['kickoff_datetime'])
        if hasattr(league_request, 'fixture_ids') and league_request.fixture_ids:
            next_fix = next_fix[next_fix['id'].isin(league_request.fixture_ids)]
            logger.info(f'[{league}] Filtered to {len(next_fix)} of {len(fixtures[(fixtures["kickoff_datetime"] >= date_from) & (fixtures["kickoff_datetime"] <= date_to)])} fixtures')
        next_fix = next_fix[
            ['id', 'kickoff_datetime', 'name', 'home_team_id', 'away_team_id', 'bet365_home_odds_decimal',
             'bet365_draw_odds_decimal', 'bet365_away_odds_decimal']]
        next_fix['home_team'] = next_fix['home_team_id'].apply(lambda x: get_team(x, teams))
        next_fix['away_team'] = next_fix['away_team_id'].apply(lambda x: get_team(x, teams))
        next_fix = next_fix.drop(columns=['home_team_id', 'away_team_id'])
        next_fix = drop_placeholder_fixtures(next_fix, league)
        next_fix.sort_values(by=['kickoff_datetime', 'home_team'], inplace=True)
        next_fix.reset_index(drop=True, inplace=True)

        # In[ ]:

        avg_home_goals = get_home_goal_avg(league_id, team_stats, fixtures, stats_types)
        avg_away_goals = get_away_goal_avg(league_id, team_stats, fixtures, stats_types)
        score_preds = make_round_goal_prediction(next_fix, ratings, avg_home_goals, avg_away_goals)
        # boost = get_draw_boost(ratings, avg_home_goals, avg_away_goals, get_draw_perc(league_id, fixtures))
        # Dixon-Coles replaces the flat draw boost where a league has rho
        # configured; get_result_probs applies one or the other, never both.
        dixon_coles_rho = getattr(ctx, 'dixon_coles_rho', 0.0) or 0.0
        boost = 1.0 if dixon_coles_rho else 1.1
        score_preds['Home Odds %'] = ((1 / next_fix['bet365_home_odds_decimal']) * 100)
        score_preds['Draw Odds %'] = ((1 / next_fix['bet365_draw_odds_decimal']) * 100)
        score_preds['Away Odds %'] = ((1 / next_fix['bet365_away_odds_decimal']) * 100)

        home_win = []
        draw = []
        away_win = []
        home_clean = []
        away_clean = []
        over_1 = []
        over_2 = []
        btts = []

        # Pre-load bet365 goals over/under for the upcoming fixtures.
        # The cascade in compute_final_goals_and_probs (paths 1-3) uses
        # per-team + match-total ladders directly; path 4 (legacy 1X2
        # reverse-solve) is the fallback when those markets are absent.
        from app.services.odds_blend import (
            load_goals_odds_for_fixtures,
            compute_final_goals_and_probs,
        )
        _odds_conn = await get_source_connection()
        try:
            goals_odds_map = await load_goals_odds_for_fixtures(
                _odds_conn, next_fix['id'].tolist(),
            )
        finally:
            release_source_connection(_odds_conn)

        for i in range(len(score_preds)):
            bookie_margin = 1 + (
                        score_preds.loc[i, 'Home Odds %'] + score_preds.loc[i, 'Draw Odds %'] + score_preds.loc[
                    i, 'Away Odds %'] - 100) / 100
            score_preds.loc[i, 'Home Odds %'] = (score_preds.loc[i, 'Home Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Draw Odds %'] = (score_preds.loc[i, 'Draw Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Away Odds %'] = (score_preds.loc[i, 'Away Odds %'] / bookie_margin).round(2)
            home_goals = score_preds['Home Goals'][i]
            away_goals = score_preds['Away Goals'][i]
            bookie_1x2_pct = None
            if not pd.isna(score_preds['Home Odds %'][i]):
                bookie_1x2_pct = (
                    float(score_preds['Home Odds %'][i]) / 100.0,
                    float(score_preds['Draw Odds %'][i]) / 100.0,
                    float(score_preds['Away Odds %'][i]) / 100.0,
                )
            fixture_id = int(next_fix['id'].iloc[i])
            new_home_goals, new_away_goals, adjusted_home_win_prob, adjusted_draw_prob, adjusted_away_win_prob = (
                compute_final_goals_and_probs(
                    fixture_id,
                    float(home_goals), float(away_goals),
                    bookie_1x2_pct,
                    goals_odds_map.get(fixture_id, {}),
                    odds_beta,
                    boost,
                    dixon_coles_rho,
                )
            )
            score_preds.loc[i, 'Home Goals'] = round(new_home_goals, 2)
            score_preds.loc[i, 'Away Goals'] = round(new_away_goals, 2)
            home_clean_sheet = poisson.pmf(0, new_away_goals)
            away_clean_sheet = poisson.pmf(0, new_home_goals)
            x = np.arange(0, 9)
            y = np.arange(0, 9)
            X, Y = np.meshgrid(x, y)
            Z = poisson.pmf(X, new_home_goals) * poisson.pmf(Y, new_away_goals)
            home_win.append(f"{adjusted_home_win_prob:.2f}%")
            draw.append(f"{adjusted_draw_prob:.2f}%")
            away_win.append(f"{adjusted_away_win_prob:.2f}%")
            home_clean.append(f"{home_clean_sheet * 100:.2f}%")
            away_clean.append(f"{away_clean_sheet * 100:.2f}%")
            over_1_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1]) * 100
            over_2_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1] - Z[2, 0] - Z[0, 2] - Z[1, 1]) * 100
            both_teams_score_prob = (1 - Z[0, :].sum() - Z[:, 0].sum() + Z[0, 0]) * 100
            over_1.append(f"{over_1_goals:.2f}%")
            over_2.append(f"{over_2_goals:.2f}%")
            btts.append(f"{both_teams_score_prob:.2f}%")

        # score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'], inplace=True)
        score_preds['Home Win %'] = home_win
        score_preds['Draw %'] = draw
        score_preds['Away Win %'] = away_win
        score_preds['Home Clean Sheet %'] = home_clean
        score_preds['Away Clean Sheet %'] = away_clean
        score_preds['Over 1.5 Goals %'] = over_1
        score_preds['Over 2.5 Goals %'] = over_2
        score_preds['Both Teams Score %'] = btts
        score_preds['Home Goals'] = score_preds['Home Goals'].round(2)
        score_preds['Away Goals'] = score_preds['Away Goals'].round(2)
        score_preds_with_odds = score_preds.copy()  # NEW - Create a copy with odds included
        score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'],
                         inplace=True)  # NEW - Drop odds from main predictions dataframe

        # In[ ]:

        ## NEW - Update accuracy dataset with new predictions

        score_preds_with_odds.rename(
            columns={'id': 'fixture_id', 'Home Goals': 'Home Projected Goals', 'Away Goals': 'Away Projected Goals'},
            inplace=True)
        score_preds_with_odds['Total Projected Goals'] = score_preds_with_odds['Home Projected Goals'] + \
                                                         score_preds_with_odds['Away Projected Goals']
        score_preds_with_odds['comp_id'] = league_id
        projection_accuracy_dataset_league = pd.concat([projection_accuracy_dataset_league, score_preds_with_odds],
                                                       ignore_index=True)
        score_preds_with_odds.rename(
            columns={'fixture_id': 'id', 'Home Projected Goals': 'Home Goals', 'Away Projected Goals': 'Away Goals'},
            inplace=True)
        score_preds_with_odds.drop(columns=['comp_id', 'Total Projected Goals'], inplace=True)

        # In[ ]:

        ## NEW - 4+ STAR BETS SECTION

        # ## **4+ Star Bets**

        # In[ ]:

        # NEW - Load previous best bets file and append new best bets

        # best_bets = pd.read_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\Best Bets.xlsx")
        best_bets = ProjectionService._read_df(f"{ProjectionService.DATA_FOLDER_PATH}/Best Bets")

        new_best_bets = pd.DataFrame()
        for i in range(len(score_preds)):
            fix_id = score_preds.loc[i, 'id']
            date = score_preds.loc[i, 'kickoff_datetime']
            date = date.strftime('%d-%m')
            fix = fixtures_df[fixtures_df['id'] == fix_id]
            home_win = float(score_preds.loc[i, 'Home Win %'].strip('%')) / 100
            draw = float(score_preds.loc[i, 'Draw %'].strip('%')) / 100
            away_win = float(score_preds.loc[i, 'Away Win %'].strip('%')) / 100
            over_1_5_goals = float(score_preds.loc[i, 'Over 1.5 Goals %'].strip('%')) / 100
            over_2_5_goals = float(score_preds.loc[i, 'Over 2.5 Goals %'].strip('%')) / 100
            btts = float(score_preds.loc[i, 'Both Teams Score %'].strip('%')) / 100

            # Skip best-bet eval if any bet365 odd is missing (None/NaN).
            # bet365_totals_odds in particular is sparse for some leagues
            # (e.g. Belgian Pro League playoff fixtures not yet priced).
            _odds_vals = [
                fix['bet365_home_odds_decimal'].values[0],
                fix['bet365_draw_odds_decimal'].values[0],
                fix['bet365_away_odds_decimal'].values[0],
                fix['over_1_5_odds_decimal'].values[0],
                fix['over_2_5_odds_decimal'].values[0],
                fix['bet365_btts_yes_odds_decimal'].values[0],
            ]
            if any(v is None or pd.isna(v) for v in _odds_vals):
                continue
            home_win_odds = 1 / _odds_vals[0]
            draw_odds = 1 / _odds_vals[1]
            away_win_odds = 1 / _odds_vals[2]
            over_1_5_goals_odds = 1 / _odds_vals[3]
            over_2_5_goals_odds = 1 / _odds_vals[4]
            btts_odds = 1 / _odds_vals[5]

            home_win_edge = home_win - home_win_odds
            draw_edge = draw - draw_odds
            away_win_edge = away_win - away_win_odds
            over_1_5_goals_edge = over_1_5_goals - over_1_5_goals_odds
            over_2_5_goals_edge = over_2_5_goals - over_2_5_goals_odds
            btts_edge = btts - btts_odds

            home_win_edge_rating = (home_win_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            draw_edge_rating = (draw_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            away_win_edge_rating = (away_win_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            over_1_5_goals_edge_rating = (over_1_5_goals_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            over_2_5_goals_edge_rating = (over_2_5_goals_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            btts_edge_rating = (btts_edge - (-0.1)) * 5 / (0.1 - (-0.1))

            home_win_prob_rating = (home_win) * 5 / (0.9)
            draw_prob_rating = (draw) * 5 / (0.9)
            away_win_prob_rating = (away_win) * 5 / (0.9)
            over_1_5_goals_prob_rating = (over_1_5_goals) * 5 / (0.9)
            over_2_5_goals_prob_rating = (over_2_5_goals) * 5 / (0.9)
            btts_prob_rating = (btts) * 5 / (0.9)

            home_win_total_rating = (home_win_edge_rating * 0.7 if home_win_edge_rating > 0 else 0) + (
                home_win_prob_rating * 0.3 if home_win_prob_rating < 5 else 5 * 0.3)
            draw_total_rating = (draw_edge_rating * 0.7 if draw_edge_rating > 0 else 0) + (
                draw_prob_rating * 0.3 if draw_prob_rating < 5 else 5 * 0.3)
            away_win_total_rating = (away_win_edge_rating * 0.7 if away_win_edge_rating > 0 else 0) + (
                away_win_prob_rating * 0.3 if away_win_prob_rating < 5 else 5 * 0.3)
            over_1_5_goals_total_rating = (
                                              over_1_5_goals_edge_rating * 0.7 if over_1_5_goals_edge_rating > 0 else 0) + (
                                              over_1_5_goals_prob_rating * 0.3 if over_1_5_goals_prob_rating < 5 else 5 * 0.3)
            over_2_5_goals_total_rating = (
                                              over_2_5_goals_edge_rating * 0.7 if over_2_5_goals_edge_rating > 0 else 0) + (
                                              over_2_5_goals_prob_rating * 0.3 if over_2_5_goals_prob_rating < 5 else 5 * 0.3)
            btts_total_rating = (btts_edge_rating * 0.7 if btts_edge_rating > 0 else 0) + (
                btts_prob_rating * 0.3 if btts_prob_rating < 5 else 5 * 0.3)

            for bet_type in ['Home Win', 'Draw', 'Away Win', 'Over 1.5 Goals', 'Over 2.5 Goals', 'BTTS']:
                edge = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_edge']
                edge_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_edge_rating']
                prob_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_prob_rating']
                total_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_total_rating']
                if total_rating >= 4.0:
                    new_best_bets = pd.concat([new_best_bets, pd.DataFrame({
                        'Date': [date],
                        'Competition': [league],
                        'Home Team': [score_preds.loc[i, 'Home Team']],
                        'Away Team': [score_preds.loc[i, 'Away Team']],
                        'Bet Type': [bet_type],
                        'Rating': [round(total_rating, 1) if total_rating < 5 else 5.0],
                        'Edge %': [round(edge * 100, 2)],
                        'Price': [
                            round(1 / locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_odds'], 2)]
                    })], ignore_index=True)

        best_bets = pd.concat([best_bets, new_best_bets], ignore_index=True)
        best_bets.drop_duplicates(subset=['Date', 'Competition', 'Home Team', 'Away Team', 'Bet Type'], keep='last',
                                  inplace=True)
        # best_bets.to_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\Best Bets.xlsx", index=False)
        ProjectionService._write_df(best_bets, f"{ProjectionService.DATA_FOLDER_PATH}/Best Bets")

        # # **League Projections**
        logger.info(f"[{league}] Step: predicted table simulation complete")
        # In[ ]:

        if league != 'Major League Soccer':
            season_fixtures = fixtures.copy()
            today = pd.to_datetime('today')
            season_fixtures['kickoff_datetime'] = pd.to_datetime(season_fixtures['kickoff_datetime'])
            season_fixtures = season_fixtures[season_fixtures['kickoff_datetime'] >= today]
            season_fixtures.loc[:, 'home_team'] = season_fixtures['home_team_id'].map(teams.set_index('id')['name'])
            season_fixtures.loc[:, 'away_team'] = season_fixtures['away_team_id'].map(teams.set_index('id')['name'])
            season_fixtures.sort_values(by='kickoff_datetime', inplace=True)
            season_fixtures.reset_index(drop=True, inplace=True)
            season_fixtures = drop_placeholder_fixtures(season_fixtures, league)

            season_score_preds = make_round_goal_prediction(season_fixtures, ratings, avg_home_goals, avg_away_goals)

            for i in range(len(season_score_preds)):
                home_goals = season_score_preds['Home Goals'][i]
                away_goals = season_score_preds['Away Goals'][i]

            season_score_preds['Home Goals'] = season_score_preds['Home Goals'].round(2)
            season_score_preds['Away Goals'] = season_score_preds['Away Goals'].round(2)

            current_standings = standings.copy()
            current_standings['Team'] = current_standings['team_id'].map(teams.set_index('id')['name'])
            current_standings.rename(
                columns={'goals_for': 'Goals For', 'goals_against': 'Goals Against', 'points': 'Points'}, inplace=True)
            current_standings['Goal Difference'] = current_standings['Goals For'] - current_standings['Goals Against']
            current_standings = current_standings[['Team', 'Points', 'Goals For', 'Goals Against', 'Goal Difference']]
            current_standings.reset_index(drop=True, inplace=True)
            current_standings = current_standings.astype(
                {'Points': 'int', 'Goals For': 'int', 'Goals Against': 'int', 'Goal Difference': 'int'})
            current_league_table = {
                team: {'Points': points, 'Goals For': gf, 'Goals Against': ga, 'Goal Difference': gd} for
                team, points, gf, ga, gd in current_standings.values}

            # Manual points adjustments — a deduction announced before
            # Sportmonks folds it into `points`. Skipped automatically once
            # the standings already show it, so it cannot double-count.
            current_league_table = await apply_points_adjustments(
                current_league_table, standings, league_id, current_season_id, teams, league)

            avg_table, all_tables = sim_multiple_seasons(season_score_preds, current_league_table, num_sims=10000)

        # # **Team Projections**
        #
        # Getting each Teams stat projections using the models

        # In[20]:

        stat_list = get_stat_list(league_id)

        # In[21]:

        models = load_all_models(stat_list, ProjectionService.MODEL_FILE_PATH)

        # In[22]:

        if next_fix.empty:
            return Response(status_code=204)

        todays_date = pd.to_datetime(next_fix['kickoff_datetime'].iloc[0]).date()

        # In[ ]:

        team_projections = get_team_round_predictions(next_fix, stat_list, fixtures_df, team_stats, teams, stats_types,
                                                      models, ratings=ratings,
                                                      league_weightings=[league_above_attack_weight,
                                                                         league_above_defense_weight,
                                                                         league_below_attack_weight,
                                                                         league_below_defense_weight],
                                                      season_id=[current_season_id, previous_season_id,
                                                                 previous_season_id_above, previous_season_id_below],
                                                      games=50,
                                                      comp_teams=comp_teams[comp_teams['competition_id'] == league_id])

        # In[ ]:

        ## NEW - Add historical stats to the model dataset and drop them from team projections afterwards

        new_rows = []

        for i in range(len(team_projections)):
            team_df = team_projections.iloc[[i]]
            new_row = {}
            new_row['id'] = team_df['fixture_id'].values[0]
            new_row['kickoff_datetime'] = team_df['kickoff_datetime'].values[0]
            new_row['comp_id'] = league_id
            new_row['Team'] = team_df['Team'].values[0]
            new_row['Opponent'] = team_df['Opponent'].values[0]
            new_row['Venue'] = team_df['Venue'].values[0]
            for stat in stat_list:
                new_row['Team ' + stat + ' History'] = team_df['Team ' + stat + ' History'].values[0]
                new_row['Opponent ' + stat + ' History Against'] = \
                team_df['Opponent ' + stat + ' History Against'].values[0]
            new_rows.append(new_row)

        model_dataset_league = pd.concat([model_dataset_league, pd.DataFrame(new_rows)], ignore_index=True)
        model_dataset_all = pd.concat([model_dataset_all, pd.DataFrame(new_rows)], ignore_index=True)
        model_dataset_league.drop_duplicates(subset=['id', 'Team', 'Opponent', 'Venue'], keep='last', inplace=True)
        model_dataset_all.drop_duplicates(subset=['id', 'Team', 'Opponent', 'Venue'], keep='last', inplace=True)

        ProjectionService._write_df(model_dataset_league, f"{ProjectionService.DATA_FOLDER_PATH}/{league}_model_dataset_with_history")
        ProjectionService._write_df(model_dataset_all, f"{ProjectionService.DATA_FOLDER_PATH}/all_leagues_model_dataset_with_history")
        # Dual-write to DB (see projections() for rationale).
        try:
            from app.repository.projection_dataset_repo import insert_model_dataset_async
            await insert_model_dataset_async(model_dataset_league, league_id, league, teams, fixtures_df, comp_teams)
        except Exception as _db_err:
            logger.warning(f"[{league}] model_dataset DB dual-write failed: {_db_err}")

        # model_dataset_league.to_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\{league}_model_dataset_with_history.xlsx", index=False)
        # model_dataset_all.to_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\all_leagues_model_dataset_with_history.xlsx", index=False)

        team_projections.drop(
            columns=['Team ' + stat + ' History' for stat in stat_list] + ['Opponent ' + stat + ' History Against' for
                                                                           stat in stat_list], inplace=True)

        # In[ ]:

        avg_goals = (avg_home_goals + avg_away_goals) / 2

        league_team_stats = team_stats[
            team_stats['fixture_id'].isin(fixtures_df[fixtures_df['competition_id'] == league_id]['id'])]

        league_shots = league_team_stats[league_team_stats['stats_type_id'] == get_stat_id('Shots Total',
                                                                                           stats_types)].copy()  # NEW - all team shots for specific league
        league_shots['Date'] = league_shots['fixture_id'].map(
            fixtures_df.set_index('id')['kickoff_datetime'])  # NEW - map fixture dates
        league_shots['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(
            league_shots['Date'])).dt.days // 7  # NEW - calculate weeks since kickoff
        league_shots['Weight'] = 0.9 ** (
                    league_shots['Weeks Since Kickoff'] - 5)  # NEW - apply weighting to more recent matches
        league_shots.loc[league_shots['Weeks Since Kickoff'] < 6, 'Weight'] = 1  # NEW - full weight for last 5 weeks
        league_shots['Weighted Shots'] = league_shots['Weight'] * league_shots[
            'value']  # NEW - calculate weighted shots
        avg_shots = league_shots['Weighted Shots'].sum() / league_shots[
            'Weight'].sum()  # UPDATED - new formula for average shots

        league_shots_on_target = league_team_stats[league_team_stats['stats_type_id'] == get_stat_id('Shots On Target',
                                                                                                     stats_types)].copy()  # NEW - all team shots on target for specific league
        league_shots_on_target['Date'] = league_shots_on_target['fixture_id'].map(
            fixtures_df.set_index('id')['kickoff_datetime'])  # NEW - map fixture dates
        league_shots_on_target['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(
            league_shots_on_target['Date'])).dt.days // 7  # NEW - calculate weeks since kickoff
        league_shots_on_target['Weight'] = 0.9 ** (
                    league_shots_on_target['Weeks Since Kickoff'] - 5)  # NEW - apply weighting to more recent matches
        league_shots_on_target.loc[
            league_shots_on_target['Weeks Since Kickoff'] < 6, 'Weight'] = 1  # NEW - full weight for last 5 weeks
        league_shots_on_target['Weighted Shots On Target'] = league_shots_on_target['Weight'] * league_shots_on_target[
            'value']  # NEW - calculate weighted shots on target
        avg_shots_on_target = league_shots_on_target['Weighted Shots On Target'].sum() / league_shots_on_target[
            'Weight'].sum()  # UPDATED - new formula for average shots on target

        avg_shots_per_goal = avg_shots / avg_goals
        avg_shots_on_target_per_goal = avg_shots_on_target / avg_goals

        # In[ ]:

        # if 'team_projections' in globals():
        goals = []
        assists = []
        for i in range(len(team_projections)):
            team = team_projections['Team'].iloc[i]
            opp = team_projections['Opponent'].iloc[i]
            # try:
            #    team_pred = score_preds[score_preds['Home Team'] == team]['Home Goals'].values[0]
            # except:
            #    team_pred = score_preds[score_preds['Away Team'] == team]['Away Goals'].values[0]
            fixture = score_preds[score_preds['id'] == team_projections['fixture_id'].iloc[
                i]]  # NEW - Get the fixture from score_preds
            team_pred = fixture['Home Goals'].values[0] if fixture['Home Team'].values[0] == team else \
            fixture['Away Goals'].values[
                0]  # UPDATED - new way to get team prediction that handles teams having multiple matches in a round
            opp_pred = fixture['Away Goals'].values[0] if fixture['Home Team'].values[0] == opp else \
            fixture['Home Goals'].values[
                0]  # UPDATED - new way to get opponent prediction that handles teams having multiple matches in a round
            goals.append(team_pred)
            assists.append((team_pred * 0.82).round(2))
            projected_shots = team_projections['Shots Total'].iloc[i]
            projected_shots_on_target = team_projections['Shots On Target'].iloc[i]

            adjusted_shots, adjusted_shots_on_target = adjust_shots_projection(
                team_pred,
                projected_shots,
                projected_shots_on_target,
                avg_shots_per_goal,
                avg_shots_on_target_per_goal
            )
            team_projections.at[i, 'Shots Total'] = adjusted_shots
            team_projections.at[i, 'Shots On Target'] = adjusted_shots_on_target

        team_projections['Goals'] = goals
        team_projections['Assists'] = assists

        # PL only: project team-level Ball Recovery + CBI(FPL) per fixture.
        # No PoissonRegressor exists for these stats (Sportmonks contributes
        # zero team-level rows); use get_simple_team_stat_prediction's
        # closed-form opponent-adjusted weighted average.
        # distribute_team_predictions_to_players auto-projects per-player
        # values from any column on team_projections, so adding these here
        # gives us per-player Recoveries + CBI for the team-down CBIT calc.
        if fpl:
            _lw_def = [league_above_attack_weight, league_above_defense_weight,
                       league_below_attack_weight, league_below_defense_weight]
            _sid_def = [current_season_id, previous_season_id,
                        previous_season_id_above, previous_season_id_below]
            _cpl_def = comp_teams[comp_teams['competition_id'] == league_id]
            _rec_col = []
            _cbi_col = []
            _cbit_col = []
            for i in range(len(team_projections)):
                _row = team_projections.iloc[i]
                try:
                    rec_v, _, _ = get_simple_team_stat_prediction(
                        _row['Team'], _row['Opponent'], fixtures_df, 'Ball Recovery',
                        team_stats, teams, stats_types,
                        ratings=ratings, venue=_row['Venue'], comp_id=league_id,
                        league_weightings=_lw_def, season_id=_sid_def, games=50,
                        comp_teams=_cpl_def,
                    )
                except Exception:
                    rec_v = 0
                try:
                    cbi_v, _, _ = get_simple_team_stat_prediction(
                        _row['Team'], _row['Opponent'], fixtures_df,
                        'Clearances Blocks Interceptions (FPL)',
                        team_stats, teams, stats_types,
                        ratings=ratings, venue=_row['Venue'], comp_id=league_id,
                        league_weightings=_lw_def, season_id=_sid_def, games=50,
                        comp_teams=_cpl_def,
                    )
                except Exception:
                    cbi_v = 0
                # Combined CBIT — the quantity FPL scores the threshold
                # against. Projected as ONE stat rather than assembling a
                # modelled Tackles with a blended CBI lump: FPL and Sportmonks
                # agree on the total to 0.3%, so one definition costs nothing
                # and removes the three inconsistent notions the pipeline
                # carried. George, 2026-08-04.
                try:
                    cbit_v, _cbit_th, _cbit_oh = get_simple_team_stat_prediction(
                        _row['Team'], _row['Opponent'], fixtures_df,
                        'Clearances Blocks Interceptions Tackles (FPL)',
                        team_stats, teams, stats_types,
                        ratings=ratings, venue=_row['Venue'], comp_id=league_id,
                        league_weightings=_lw_def, season_id=_sid_def, games=50,
                        comp_teams=_cpl_def,
                    )
                except Exception:
                    cbit_v = 0
                    _cbit_th = _cbit_oh = None
                # Diagnostic for the ~5% under-projection measured 2026-08-05
                # (projected 51.88 CBIT/match against an actual 55.59; 13 of 15
                # clubs low, and NOT explained by opponent mix or missing
                # history). The blend is
                #     alpha * team_history + (1-alpha) * opponent_history
                # and both terms are returned but discarded, so log them for a
                # handful of fixtures to see which side is light before
                # guessing at the cause.
                if i < 6:
                    logger.info(
                        f"[{league}] CBIT blend probe: {_row['Team']} vs {_row['Opponent']} "
                        f"({_row['Venue']}) team_hist={_cbit_th} opp_hist={_cbit_oh} "
                        f"-> blended={cbit_v}"
                    )
                _rec_col.append(rec_v)
                _cbi_col.append(cbi_v)
                _cbit_col.append(cbit_v)
            team_projections['Ball Recovery'] = _rec_col
            team_projections['Clearances Blocks Interceptions (FPL)'] = _cbi_col
            team_projections['Clearances Blocks Interceptions Tackles (FPL)'] = _cbit_col
            # Tackles Won rides a team TACKLES total, and that total is
            # CBIT - CBI rather than the modelled 'Tackles' column. Measured
            # against 760 real PL team-matches: the model averages 14.37 against
            # an actual 16.69 (-13.9%), while CBIT - CBI gives 15.74 (-5.7%) —
            # the two blended quantities are uniformly ~7% light, so their
            # DIFFERENCE tracks tackles better than the dedicated model does.
            # George called it before it was measured, 2026-08-05.
            #
            # It also makes BPS reconcile with DefCon by construction: BPS scores
            # CBI + tackles = CBI + (CBIT - CBI) = CBIT, the exact quantity the
            # DefCon threshold uses. Projecting the two separately had them
            # disagreeing by 10-14% on promoted clubs (Coventry 58.05 vs 51.00).
            #
            # The SHARE is unchanged and still measured from history — his
            # tackles won over his team's actual tackles (_TEAM_DENOMINATOR_ALIAS
            # maps the denominator to 'Tackles'), so a player's personal success
            # rate is folded in automatically.
            _tkl_from_cbit = (team_projections['Clearances Blocks Interceptions Tackles (FPL)']
                              - team_projections['Clearances Blocks Interceptions (FPL)'])
            # A fixture where the CBI blend outruns the CBIT blend would give a
            # negative tackle count; fall back to the modelled column there.
            team_projections['Tackles Won'] = _tkl_from_cbit.where(
                _tkl_from_cbit > 0, team_projections['Tackles'])
            # Split the goal projection into penalty and non-penalty halves.
            #
            # Penalties are taken as a fixed PROPORTION of projected goals, not
            # a flat per-match constant, so an attacking team is projected more
            # of them (George's call, 2026-08-07). Note the measured
            # correlation between a team's goals and its penalties across PL
            # 25/26 was 0.029 — i.e. none — with Brentford winning 10 on 55
            # goals against Man City's 4 on 77. George's position is that the
            # mechanism is real and one season of 2-10 counts is too noisy to
            # disprove it; the risk is City runs high and Brentford low.
            #
            # Constants measured over PL 24/25 + 25/26 (175 penalties):
            # penalties are 6.76% of goals, converted at 83.4%.
            # A proportional split cannot leak or invent goals — the two halves
            # always sum back to the original projection.
            _pg = team_projections['Goals'] * PENALTY_GOAL_SHARE
            team_projections['Non-Penalty Goals'] = team_projections['Goals'] - _pg
            # Named 'Penalties Scored' rather than 'Penalty Goals' so it matches
            # the PLAYER stat of the same name (111). distribute() resolves a
            # share for every team column by looking up the identically-named
            # player stat, so a column with no player counterpart hands every
            # player a zero share. Attempts and misses derive from this
            # downstream (attempts = scored / conversion), avoiding a second
            # distributed column that would need its own share.
            team_projections['Penalties Scored'] = _pg

        saves = []
        for i in range(len(team_projections)):
            # opp = team_projections['Opponent'].iloc[i]
            # try:
            #    opp_pred = score_preds[score_preds['Home Team'] == opp]['Home Goals'].values[0]
            # except:
            #    opp_pred = score_preds[score_preds['Away Team'] == opp]['Away Goals'].values[0]
            # saves.append(team_projections[team_projections['Team'] == opp]['Shots On Target'].values[0] - opp_pred)
            fixture_id = team_projections['fixture_id'].iloc[i]  # NEW - Get fixture ID
            fixture_team_projections = team_projections[
                team_projections['fixture_id'] == fixture_id]  # NEW - Get both teams' projections for the fixture
            fixture_team_projections = fixture_team_projections.drop(
                i)  # NEW - Drop the current team to get the opponent projections
            saves.append(
                fixture_team_projections['Shots On Target'].values[0] - fixture_team_projections['Goals'].values[
                    0])  # UPDATED - New way to calculate saves based on opponent projections that handles teams having multiple matches in a round

        team_projections['Saves'] = saves
        team_projections['Saves'] = team_projections['Saves'].round(2)  # NEW - Round saves to 2 decimal places
        # PL projects Key Passes properly (get_stat_list). Everywhere else it
        # stays derived — measured 0.72-0.74 across the top 5, so 0.75 runs a
        # little high. George, 2026-08-02.
        if 'Key Passes' not in team_projections.columns:
            team_projections['Key Passes'] = (team_projections['Shots Total'] * 0.75).round(2)
        # Retain Ball Recovery + CBI(FPL) columns when present (added by the
        # PL-only block above). Other leagues skip these columns.
        _extra_def_cols = [c for c in ['Ball Recovery', 'Clearances Blocks Interceptions (FPL)', 'Clearances Blocks Interceptions Tackles (FPL)', 'Tackles Won', 'Non-Penalty Goals', 'Penalties Scored']
                           if c in team_projections.columns]
        team_projections = team_projections[
            ['fixture_id', 'kickoff_datetime', 'Team', 'Opponent', 'Venue', 'Goals', 'Assists',
             'Key Passes'] + [c for c in stat_list if c != 'Key Passes'] + ['Fouls Drawn', 'Saves'] + _extra_def_cols]
        team_projections.rename(columns={'Successful Passes': 'Accurate Passes'}, inplace=True)
        logger.debug(f"[{league}] team_projections columns ready")

        # ── Team-stat odds-blend ──
        # Reels each team's projected stats (corners/cards/shots/SoT/
        # fouls/tackles) toward bookmaker expected via the cascade
        # (Path 1 per-team ladder → Path 1.5 partial+match → Path 2
        # match-split via model ratio). Per-stat bookmaker priority
        # lists in TEAM_STAT_BOOKIE_PRIORITY. Falls back to model
        # unchanged for any (stat, fixture) with no usable book data.
        from app.services.odds_blend import (
            load_team_stat_odds, blend_team_stat,
            TEAM_STAT_BOOKIE_PRIORITY, STAT_COLUMN_TO_MARKET,
        )
        _fix_ids = team_projections['fixture_id'].astype(int).unique().tolist()
        _odds_per_market = {}
        _odds_conn = await get_source_connection()
        try:
            for _market, _books in TEAM_STAT_BOOKIE_PRIORITY.items():
                _odds_per_market[_market] = await load_team_stat_odds(
                    _odds_conn, _fix_ids, _market, _books,
                )
        finally:
            release_source_connection(_odds_conn)

        _fid_to_home_team = {}
        for _fid in _fix_ids:
            _row = next_fix[next_fix['id'] == _fid]
            if not _row.empty:
                _fid_to_home_team[_fid] = _row['home_team'].iloc[0]

        _seen_fixtures = set()
        for _i in range(len(team_projections)):
            fid = int(team_projections['fixture_id'].iloc[_i])
            if fid in _seen_fixtures:
                continue
            _seen_fixtures.add(fid)
            pair = team_projections[team_projections['fixture_id'] == fid]
            if len(pair) != 2:
                continue
            home_team_name = _fid_to_home_team.get(fid)
            if not home_team_name:
                continue
            home_mask = (team_projections['fixture_id'] == fid) & (team_projections['Team'] == home_team_name)
            away_mask = (team_projections['fixture_id'] == fid) & (team_projections['Team'] != home_team_name)

            for stat_col, market in STAT_COLUMN_TO_MARKET.items():
                if stat_col not in team_projections.columns:
                    continue
                try:
                    mh = float(team_projections.loc[home_mask, stat_col].iloc[0])
                    ma = float(team_projections.loc[away_mask, stat_col].iloc[0])
                except (IndexError, KeyError, ValueError):
                    continue
                fh, fa = blend_team_stat(
                    mh, ma,
                    _odds_per_market.get(market, {}).get(fid, {}),
                    market, odds_beta,
                )
                team_projections.loc[home_mask, stat_col] = round(fh, 2)
                team_projections.loc[away_mask, stat_col] = round(fa, 2)
        
        # print(team_projections['Assists', 'Key Passes'])
        # In[ ]:

        # team_projections_save = team_projections.copy()
        # team_projections_save.drop(['Assists', 'Fouls Drawn', 'Saves', 'Key Passes'], axis=1,
        #                            inplace=True)  # UPDATED - No longer dropping interceptions and accurate passes

        team_projections_save = team_projections.copy()
        
        team_projections_save.drop(
            ['Assists', 'Fouls Drawn', 'Saves', 'Key Passes'],
            axis=1,
            inplace=True,
            errors='ignore'  # <- ovo sprečava KeyError ako kolona ne postoji
        )

        team_projections_save = team_projections_save.round(2)

        team_projections_save.rename(columns={'Accurate Passes': 'Successful Passes'},
                                     inplace=True)  # NEW - Rename back for consistency with other datasets

        # In[ ]:

        ## NEW - Update projection accuracy dataset

        for fixture_id in team_projections_save['fixture_id'].unique():
            fixture_projections = team_projections_save[team_projections_save['fixture_id'] == fixture_id]
            # accuracy dataset has no columns for the PL-only stats
            for stat in accuracy_stat_list(stat_list):
                projection_accuracy_dataset_league.loc[
                    projection_accuracy_dataset_league['fixture_id'] == fixture_id, 'Home Projected ' + stat] = \
                fixture_projections.loc[fixture_projections['Venue'] == 'H', stat].values[0]
                projection_accuracy_dataset_league.loc[
                    projection_accuracy_dataset_league['fixture_id'] == fixture_id, 'Away Projected ' + stat] = \
                fixture_projections.loc[fixture_projections['Venue'] == 'A', stat].values[0]
                projection_accuracy_dataset_league.loc[
                    projection_accuracy_dataset_league['fixture_id'] == fixture_id, 'Total Projected ' + stat] = \
                fixture_projections[stat].sum()

        projection_accuracy_dataset_league.drop_duplicates(subset=['fixture_id'], keep='last', inplace=True)
        projection_accuracy_dataset_league.reset_index(drop=True, inplace=True)
        # projection_accuracy_dataset_league.to_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\{league}_accuracy_dataset.xlsx", index=False)
        ProjectionService._write_df(projection_accuracy_dataset_league, f"{ProjectionService.DATA_FOLDER_PATH}/{league}_accuracy_dataset")
        # Dual-write to DB (see projections() for rationale).
        try:
            from app.repository.projection_dataset_repo import insert_accuracy_dataset_async
            await insert_accuracy_dataset_async(projection_accuracy_dataset_league, league_id, league, teams, fixtures_df, comp_teams)
        except Exception as _db_err:
            logger.warning(f"[{league}] accuracy_dataset DB dual-write failed: {_db_err}")

        projection_accuracy_dataset_all = pd.concat(
            [projection_accuracy_dataset_all, projection_accuracy_dataset_league], ignore_index=True)
        projection_accuracy_dataset_all.drop_duplicates(subset=['fixture_id'], keep='last', inplace=True)
        projection_accuracy_dataset_all.reset_index(drop=True, inplace=True)
        # projection_accuracy_dataset_all.to_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\all_leagues_accuracy_dataset.xlsx", index=False)
        ProjectionService._write_df(projection_accuracy_dataset_all, f"{ProjectionService.DATA_FOLDER_PATH}/all_leagues_accuracy_dataset")

        #
        # # **Player Projections**
        #
        # Distributing the above dataframe's values to each player based on the % of teams total

        # In[ ]:

        # UPDATED: Removed xG parameter, added comps parameter and added season_id paramter
        # Pre-load confirmed XI + player-prop odds for the same fixture
        # batch (see the projections() site for the canonical comment).
        from app.services.odds_blend import (
            load_confirmed_lineups, load_player_odds,
            PLAYER_BLEND_BOOKS, PLAYER_BLEND_STAT_IDS,
        )
        _pl_fix_ids = next_fix['id'].astype(int).unique().tolist()
        _ll_conn = await get_source_connection()
        try:
            _confirmed_lineups = await load_confirmed_lineups(_ll_conn, _pl_fix_ids)
            _odds_for_fixture_players = await load_player_odds(
                _ll_conn, _pl_fix_ids, PLAYER_BLEND_STAT_IDS, PLAYER_BLEND_BOOKS,
            )
        finally:
            release_source_connection(_ll_conn)
        pl_projections = distribute_team_predictions_to_players(player_stats, team_stats, team_projections, stats_types,
                                                                fixtures_df, players, teams, comps, 0.97,
                                                                season_id=[current_season_id, previous_season_id,
                                                                           previous_season_id_above,
                                                                           previous_season_id_below],
                                                                competition_id=league_id, comp_teams=comp_teams,
                                                                confirmed_lineups=_confirmed_lineups,
                                                                odds_for_fixture_players=_odds_for_fixture_players,
                                                                odds_blend_weight=odds_beta)

        # Vectorized: build player lookup, merge, derive Position/Saves AND Start? in one pass
        _team_names = teams[['id', 'name']].rename(columns={'id': '_team_id', 'name': 'Team'})
        _player_lookup = players.merge(
            _team_names, left_on='current_team_id', right_on='_team_id', how='left'
        )[['display_name', 'Team', 'id', '_team_id', 'position']].rename(
            columns={'display_name': 'Player', 'id': '_player_id'}
        ).drop_duplicates(subset=['Player', 'Team'])

        pl_projections = pl_projections.merge(_player_lookup, on=['Player', 'Team'], how='left')

        _pos_map = {'goalkeeper': 'GK', 'defender': 'DEF', 'midfielder': 'MID', 'attacker': 'FWD'}
        # Final .fillna('Unknown') catches players whose Sportmonks row has
        # NULL position (3 Allsvenskan players hit this 2026-05-28). The
        # downstream player_prop_projections.position column is NOT NULL,
        # so leaving NaN here propagates through to the SQL insert and
        # kills the whole league's projection run.
        pl_projections['Position'] = pl_projections['position'].map(_pos_map).fillna(pl_projections['position']).fillna('Unknown')
        pl_projections.loc[pl_projections['Player'] == 'Caoimhin Kelleher', 'Position'] = 'GK'

        pl_projections['Saves'] = 0
        _team_saves = team_projections[['fixture_id', 'Team', 'Saves']].rename(columns={'Saves': '_gk_saves'})
        pl_projections = pl_projections.merge(_team_saves, on=['fixture_id', 'Team'], how='left')
        _gk_mask = pl_projections['Position'] == 'GK'
        pl_projections.loc[_gk_mask, 'Saves'] = pl_projections.loc[_gk_mask, '_gk_saves'].fillna(0)
        pl_projections.drop(columns=['_gk_saves'], inplace=True)

        # Predicted starters (was a separate row-by-row loop further down — moved here so it runs
        # before the column reorder strips _team_id and _player_id).
        # Old loop also had a bug: get_player_id was called with 3 args instead of 4, raising
        # TypeError silently swallowed by bare except — every player got 'No'. Now fixed.
        _pred_starters = player_stats[player_stats['fixture_id'].isin(next_fix['id'])]
        _pred_starters = _pred_starters[_pred_starters['stats_type_id'] == 11]
        _starter_pairs = set(zip(
            _pred_starters['team_id'].astype('Int64'),
            _pred_starters['player_id'].astype('Int64')
        ))
        pl_projections['Start?'] = [
            'Yes' if (pd.notna(t) and pd.notna(p) and (int(t), int(p)) in _starter_pairs) else 'No'
            for t, p in zip(pl_projections['_team_id'], pl_projections['_player_id'])
        ]
        pl_projections.drop(columns=['_player_id', '_team_id', 'position'], inplace=True, errors='ignore')

        # PL only: retain Ball Recovery + CBI(FPL) team-down columns through
        # the explicit column filter so the team-down CBIT post-pass below
        # can read them. distribute_team_predictions_to_players propagated
        # them from team_projections via pivot; without this they'd be
        # dropped here and the post-pass would compute hit rate on Tackles
        # alone (giving ~0% for everyone).
        _def_extra = [c for c in FPL_DEF_EXTRA_COLS if c in pl_projections.columns]
        pl_projections = pl_projections[
            ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue',
             'Start?',
             'Assists', 'Key Passes', 'Accurate Passes', 'Goals',
             'Shots Total',
             'Shots On Target',  'Passes',  'Interceptions', 'Tackles', 'Total Crosses',
             'Yellowcards', 'Offsides', 'Fouls', 'Fouls Drawn', 'Saves'] + _def_extra]

        pl_projections.rename(columns={'Yellowcards': 'Yellow Cards'}, inplace=True)

        # ## **Predicted Lineups**
        #
        # Which players are predicted to play?

        # In[ ]:

        logger.info(f"[{league}] Player projections: {len(pl_projections)} rows")
        _def_extra2 = [c for c in FPL_DEF_EXTRA_COLS if c in pl_projections.columns]
        pl_projections = pl_projections[
            ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue', 'Start?', 'Shots Total',
              'Goals', 'Assists', 'Key Passes', 'Accurate Passes',
             'Shots On Target', 'Passes', 'Interceptions', 'Tackles', 'Total Crosses',
             'Yellow Cards', 'Offsides', 'Fouls', 'Fouls Drawn', 'Saves'] + _def_extra2]
        pl_projections = pl_projections.round(2)

        # In[ ]:

        # pl_projections.sort_values(by='Goals', ascending=False, inplace=True)
        pl_projections.reset_index(drop=True, inplace=True)
        pl_projections = pl_projections.round(2)
        # pl_projections.to_csv(rf"{save_file_path}\{league} Player.csv", index=False)
        await insert_player_async(pl_projections, teams=teams, competition_id=league_id, comp_teams=comp_teams)

    async def player_props(self, league_request):
        league = league_request or 'Championship'

        ctx = await self._setup_league(league)

        # Unpack shared context into local variables so downstream code is unchanged
        data_folder_path = ctx.data_folder_path
        model_file_path = ctx.model_file_path
        save_file_path = ctx.save_file_path
        league_dashed = ctx.league_dashed
        date_from = ctx.date_from
        date_to = ctx.date_to
        league_below = ctx.league_below
        league_above = ctx.league_above
        league_below_attack_weight = ctx.league_below_attack_weight
        league_below_defense_weight = ctx.league_below_defense_weight
        league_above_attack_weight = ctx.league_above_attack_weight
        league_above_defense_weight = ctx.league_above_defense_weight
        country_code = ctx.country_code
        div = ctx.div
        weightings = ctx.weightings
        mv_beta = ctx.mv_beta
        odds_beta = ctx.odds_beta
        xG = ctx.xG
        fpl = ctx.fpl
        player_stats = ctx.player_stats
        team_stats = ctx.team_stats
        standings = ctx.standings
        seasons = ctx.seasons
        comps = ctx.comps
        comp_teams = ctx.comp_teams
        teams = ctx.teams
        players = ctx.players
        fixtures_df = ctx.fixtures_df
        b365_odds = ctx.b365_odds
        stats_types = ctx.stats_types
        model_dataset_all = ctx.model_dataset_all
        model_dataset_league = ctx.model_dataset_league
        projection_accuracy_dataset_league = ctx.projection_accuracy_dataset_league
        projection_accuracy_dataset_all = ctx.projection_accuracy_dataset_all
        all_team_ratings = ctx.all_team_ratings
        league_id = ctx.league_id
        fixtures = ctx.fixtures
        league_standings = ctx.league_standings
        league_above_id = ctx.league_above_id
        league_below_id = ctx.league_below_id
        previous_season_id = ctx.previous_season_id
        current_season_id = ctx.current_season_id
        matches_played = ctx.matches_played
        season_fixtures = ctx.season_fixtures
        total_matches = ctx.total_matches
        previous_season_id_below = ctx.previous_season_id_below
        previous_season_id_above = ctx.previous_season_id_above
        stat_list = ctx.stat_list

        ratings = await self._prepare_league(
            league=league, data_folder_path=data_folder_path, model_file_path=model_file_path,
            save_file_path=save_file_path, league_id=league_id, league_dashed=league_dashed,
            model_dataset_all=model_dataset_all, model_dataset_league=model_dataset_league,
            projection_accuracy_dataset_all=projection_accuracy_dataset_all,
            projection_accuracy_dataset_league=projection_accuracy_dataset_league,
            all_team_ratings=all_team_ratings, team_stats=team_stats, player_stats=player_stats,
            teams=teams, stats_types=stats_types, stat_list=stat_list,
            comp_teams=comp_teams, fixtures_df=fixtures_df, fixtures=fixtures, seasons=seasons, comps=comps,
            current_season_id=current_season_id, previous_season_id=previous_season_id,
            previous_season_id_above=previous_season_id_above,
            previous_season_id_below=previous_season_id_below,
            weightings=weightings, mv_beta=mv_beta, odds_beta=odds_beta,
            country_code=country_code, div=div, matches_played=matches_played, standings=standings,
            league_above=league_above, league_below=league_below, league_standings=league_standings,
            league_below_attack_weight=league_below_attack_weight,
            league_below_defense_weight=league_below_defense_weight,
            league_above_id=league_above_id, league_below_id=league_below_id,
            xG=xG, fpl=fpl, b365_odds=b365_odds,
            season_fixtures=season_fixtures, total_matches=total_matches, players=players,
            mode=(league_request.mode if hasattr(league_request, 'mode') and league_request.mode else "full"),
        )

        # ## **Make Predictions for Next Fixture Round**
        #
        # Result, Score, Clean Sheets, Over 1.5, Over 2.5 and BTTS all calculated here using Poisson Distribution.

        # In[18]:

        next_fix = ProjectionService._filter_upcoming_fixtures(league, fixtures, date_from, date_to)
        fixtures['kickoff_datetime'] = pd.to_datetime(fixtures['kickoff_datetime'])
        if hasattr(league_request, 'fixture_ids') and league_request.fixture_ids:
            next_fix = next_fix[next_fix['id'].isin(league_request.fixture_ids)]
            logger.info(f'[{league}] Filtered to {len(next_fix)} of {len(fixtures[(fixtures["kickoff_datetime"] >= date_from) & (fixtures["kickoff_datetime"] <= date_to)])} fixtures')
        next_fix = next_fix[
            ['id', 'kickoff_datetime', 'name', 'home_team_id', 'away_team_id', 'bet365_home_odds_decimal',
             'bet365_draw_odds_decimal', 'bet365_away_odds_decimal']]
        next_fix['home_team'] = next_fix['home_team_id'].apply(lambda x: get_team(x, teams))
        next_fix['away_team'] = next_fix['away_team_id'].apply(lambda x: get_team(x, teams))
        next_fix = next_fix.drop(columns=['home_team_id', 'away_team_id'])
        next_fix = drop_placeholder_fixtures(next_fix, league)
        next_fix.sort_values(by=['kickoff_datetime', 'home_team'], inplace=True)
        next_fix.reset_index(drop=True, inplace=True)

        # In[ ]:

        avg_home_goals = get_home_goal_avg(league_id, team_stats, fixtures, stats_types)
        avg_away_goals = get_away_goal_avg(league_id, team_stats, fixtures, stats_types)
        score_preds = make_round_goal_prediction(next_fix, ratings, avg_home_goals, avg_away_goals)
        # boost = get_draw_boost(ratings, avg_home_goals, avg_away_goals, get_draw_perc(league_id, fixtures))
        # Dixon-Coles replaces the flat draw boost where a league has rho
        # configured; get_result_probs applies one or the other, never both.
        dixon_coles_rho = getattr(ctx, 'dixon_coles_rho', 0.0) or 0.0
        boost = 1.0 if dixon_coles_rho else 1.1
        score_preds['Home Odds %'] = ((1 / next_fix['bet365_home_odds_decimal']) * 100)
        score_preds['Draw Odds %'] = ((1 / next_fix['bet365_draw_odds_decimal']) * 100)
        score_preds['Away Odds %'] = ((1 / next_fix['bet365_away_odds_decimal']) * 100)

        home_win = []
        draw = []
        away_win = []
        home_clean = []
        away_clean = []
        over_1 = []
        over_2 = []
        btts = []

        # Pre-load bet365 goals over/under for the upcoming fixtures.
        # The cascade in compute_final_goals_and_probs (paths 1-3) uses
        # per-team + match-total ladders directly; path 4 (legacy 1X2
        # reverse-solve) is the fallback when those markets are absent.
        from app.services.odds_blend import (
            load_goals_odds_for_fixtures,
            compute_final_goals_and_probs,
        )
        _odds_conn = await get_source_connection()
        try:
            goals_odds_map = await load_goals_odds_for_fixtures(
                _odds_conn, next_fix['id'].tolist(),
            )
        finally:
            release_source_connection(_odds_conn)

        for i in range(len(score_preds)):
            bookie_margin = 1 + (
                    score_preds.loc[i, 'Home Odds %'] + score_preds.loc[i, 'Draw Odds %'] + score_preds.loc[
                i, 'Away Odds %'] - 100) / 100
            score_preds.loc[i, 'Home Odds %'] = (score_preds.loc[i, 'Home Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Draw Odds %'] = (score_preds.loc[i, 'Draw Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Away Odds %'] = (score_preds.loc[i, 'Away Odds %'] / bookie_margin).round(2)
            home_goals = score_preds['Home Goals'][i]
            away_goals = score_preds['Away Goals'][i]
            bookie_1x2_pct = None
            if not pd.isna(score_preds['Home Odds %'][i]):
                bookie_1x2_pct = (
                    float(score_preds['Home Odds %'][i]) / 100.0,
                    float(score_preds['Draw Odds %'][i]) / 100.0,
                    float(score_preds['Away Odds %'][i]) / 100.0,
                )
            fixture_id = int(next_fix['id'].iloc[i])
            new_home_goals, new_away_goals, adjusted_home_win_prob, adjusted_draw_prob, adjusted_away_win_prob = (
                compute_final_goals_and_probs(
                    fixture_id,
                    float(home_goals), float(away_goals),
                    bookie_1x2_pct,
                    goals_odds_map.get(fixture_id, {}),
                    odds_beta,
                    boost,
                    dixon_coles_rho,
                )
            )
            score_preds.loc[i, 'Home Goals'] = round(new_home_goals, 2)
            score_preds.loc[i, 'Away Goals'] = round(new_away_goals, 2)
            home_clean_sheet = poisson.pmf(0, new_away_goals)
            away_clean_sheet = poisson.pmf(0, new_home_goals)
            x = np.arange(0, 9)
            y = np.arange(0, 9)
            X, Y = np.meshgrid(x, y)
            Z = poisson.pmf(X, new_home_goals) * poisson.pmf(Y, new_away_goals)
            home_win.append(f"{adjusted_home_win_prob:.2f}%")
            draw.append(f"{adjusted_draw_prob:.2f}%")
            away_win.append(f"{adjusted_away_win_prob:.2f}%")
            home_clean.append(f"{home_clean_sheet * 100:.2f}%")
            away_clean.append(f"{away_clean_sheet * 100:.2f}%")
            over_1_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1]) * 100
            over_2_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1] - Z[2, 0] - Z[0, 2] - Z[1, 1]) * 100
            both_teams_score_prob = (1 - Z[0, :].sum() - Z[:, 0].sum() + Z[0, 0]) * 100
            over_1.append(f"{over_1_goals:.2f}%")
            over_2.append(f"{over_2_goals:.2f}%")
            btts.append(f"{both_teams_score_prob:.2f}%")

        # score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'], inplace=True)
        score_preds['Home Win %'] = home_win
        score_preds['Draw %'] = draw
        score_preds['Away Win %'] = away_win
        score_preds['Home Clean Sheet %'] = home_clean
        score_preds['Away Clean Sheet %'] = away_clean
        score_preds['Over 1.5 Goals %'] = over_1
        score_preds['Over 2.5 Goals %'] = over_2
        score_preds['Both Teams Score %'] = btts
        score_preds['Home Goals'] = score_preds['Home Goals'].round(2)
        score_preds['Away Goals'] = score_preds['Away Goals'].round(2)
        score_preds_with_odds = score_preds.copy()  # NEW - Create a copy with odds included
        score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'],
                         inplace=True)  # NEW - Drop odds from main predictions dataframe

        # In[ ]:

        ## NEW - Update accuracy dataset with new predictions

        score_preds_with_odds.rename(
            columns={'id': 'fixture_id', 'Home Goals': 'Home Projected Goals', 'Away Goals': 'Away Projected Goals'},
            inplace=True)
        score_preds_with_odds['Total Projected Goals'] = score_preds_with_odds['Home Projected Goals'] + \
                                                         score_preds_with_odds['Away Projected Goals']
        score_preds_with_odds['comp_id'] = league_id
        projection_accuracy_dataset_league = pd.concat([projection_accuracy_dataset_league, score_preds_with_odds],
                                                       ignore_index=True)
        score_preds_with_odds.rename(
            columns={'fixture_id': 'id', 'Home Projected Goals': 'Home Goals', 'Away Projected Goals': 'Away Goals'},
            inplace=True)
        score_preds_with_odds.drop(columns=['comp_id', 'Total Projected Goals'], inplace=True)

        # In[ ]:

        ## NEW - 4+ STAR BETS SECTION

        # ## **4+ Star Bets**

        # In[ ]:

        # NEW - Load previous best bets file and append new best bets

        # best_bets = pd.read_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\Best Bets.xlsx")
        best_bets = ProjectionService._read_df(f"{ProjectionService.DATA_FOLDER_PATH}/Best Bets")

        new_best_bets = pd.DataFrame()
        for i in range(len(score_preds)):
            fix_id = score_preds.loc[i, 'id']
            date = score_preds.loc[i, 'kickoff_datetime']
            date = date.strftime('%d-%m')
            fix = fixtures_df[fixtures_df['id'] == fix_id]
            home_win = float(score_preds.loc[i, 'Home Win %'].strip('%')) / 100
            draw = float(score_preds.loc[i, 'Draw %'].strip('%')) / 100
            away_win = float(score_preds.loc[i, 'Away Win %'].strip('%')) / 100
            over_1_5_goals = float(score_preds.loc[i, 'Over 1.5 Goals %'].strip('%')) / 100
            over_2_5_goals = float(score_preds.loc[i, 'Over 2.5 Goals %'].strip('%')) / 100
            btts = float(score_preds.loc[i, 'Both Teams Score %'].strip('%')) / 100

            # Skip best-bet eval if any bet365 odd is missing (None/NaN).
            # bet365_totals_odds in particular is sparse for some leagues
            # (e.g. Belgian Pro League playoff fixtures not yet priced).
            _odds_vals = [
                fix['bet365_home_odds_decimal'].values[0],
                fix['bet365_draw_odds_decimal'].values[0],
                fix['bet365_away_odds_decimal'].values[0],
                fix['over_1_5_odds_decimal'].values[0],
                fix['over_2_5_odds_decimal'].values[0],
                fix['bet365_btts_yes_odds_decimal'].values[0],
            ]
            if any(v is None or pd.isna(v) for v in _odds_vals):
                continue
            home_win_odds = 1 / _odds_vals[0]
            draw_odds = 1 / _odds_vals[1]
            away_win_odds = 1 / _odds_vals[2]
            over_1_5_goals_odds = 1 / _odds_vals[3]
            over_2_5_goals_odds = 1 / _odds_vals[4]
            btts_odds = 1 / _odds_vals[5]

            home_win_edge = home_win - home_win_odds
            draw_edge = draw - draw_odds
            away_win_edge = away_win - away_win_odds
            over_1_5_goals_edge = over_1_5_goals - over_1_5_goals_odds
            over_2_5_goals_edge = over_2_5_goals - over_2_5_goals_odds
            btts_edge = btts - btts_odds

            home_win_edge_rating = (home_win_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            draw_edge_rating = (draw_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            away_win_edge_rating = (away_win_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            over_1_5_goals_edge_rating = (over_1_5_goals_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            over_2_5_goals_edge_rating = (over_2_5_goals_edge - (-0.1)) * 5 / (0.1 - (-0.1))
            btts_edge_rating = (btts_edge - (-0.1)) * 5 / (0.1 - (-0.1))

            home_win_prob_rating = (home_win) * 5 / (0.9)
            draw_prob_rating = (draw) * 5 / (0.9)
            away_win_prob_rating = (away_win) * 5 / (0.9)
            over_1_5_goals_prob_rating = (over_1_5_goals) * 5 / (0.9)
            over_2_5_goals_prob_rating = (over_2_5_goals) * 5 / (0.9)
            btts_prob_rating = (btts) * 5 / (0.9)

            home_win_total_rating = (home_win_edge_rating * 0.7 if home_win_edge_rating > 0 else 0) + (
                home_win_prob_rating * 0.3 if home_win_prob_rating < 5 else 5 * 0.3)
            draw_total_rating = (draw_edge_rating * 0.7 if draw_edge_rating > 0 else 0) + (
                draw_prob_rating * 0.3 if draw_prob_rating < 5 else 5 * 0.3)
            away_win_total_rating = (away_win_edge_rating * 0.7 if away_win_edge_rating > 0 else 0) + (
                away_win_prob_rating * 0.3 if away_win_prob_rating < 5 else 5 * 0.3)
            over_1_5_goals_total_rating = (
                                              over_1_5_goals_edge_rating * 0.7 if over_1_5_goals_edge_rating > 0 else 0) + (
                                              over_1_5_goals_prob_rating * 0.3 if over_1_5_goals_prob_rating < 5 else 5 * 0.3)
            over_2_5_goals_total_rating = (
                                              over_2_5_goals_edge_rating * 0.7 if over_2_5_goals_edge_rating > 0 else 0) + (
                                              over_2_5_goals_prob_rating * 0.3 if over_2_5_goals_prob_rating < 5 else 5 * 0.3)
            btts_total_rating = (btts_edge_rating * 0.7 if btts_edge_rating > 0 else 0) + (
                btts_prob_rating * 0.3 if btts_prob_rating < 5 else 5 * 0.3)

            for bet_type in ['Home Win', 'Draw', 'Away Win', 'Over 1.5 Goals', 'Over 2.5 Goals', 'BTTS']:
                edge = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_edge']
                edge_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_edge_rating']
                prob_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_prob_rating']
                total_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_total_rating']
                if total_rating >= 4.0:
                    new_best_bets = pd.concat([new_best_bets, pd.DataFrame({
                        'Date': [date],
                        'Competition': [league],
                        'Home Team': [score_preds.loc[i, 'Home Team']],
                        'Away Team': [score_preds.loc[i, 'Away Team']],
                        'Bet Type': [bet_type],
                        'Rating': [round(total_rating, 1) if total_rating < 5 else 5.0],
                        'Edge %': [round(edge * 100, 2)],
                        'Price': [
                            round(1 / locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_odds'], 2)]
                    })], ignore_index=True)

        best_bets = pd.concat([best_bets, new_best_bets], ignore_index=True)
        best_bets.drop_duplicates(subset=['Date', 'Competition', 'Home Team', 'Away Team', 'Bet Type'], keep='last',
                                  inplace=True)
        # best_bets.to_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\Best Bets.xlsx", index=False)
        ProjectionService._write_df(best_bets, f"{ProjectionService.DATA_FOLDER_PATH}/Best Bets")

        # # **League Projections**
        logger.info(f"[{league}] Step: predicted table simulation complete")
        # In[ ]:

        if league != 'Major League Soccer':
            season_fixtures = fixtures.copy()
            today = pd.to_datetime('today')
            season_fixtures['kickoff_datetime'] = pd.to_datetime(season_fixtures['kickoff_datetime'])
            season_fixtures = season_fixtures[season_fixtures['kickoff_datetime'] >= today]
            season_fixtures.loc[:, 'home_team'] = season_fixtures['home_team_id'].map(teams.set_index('id')['name'])
            season_fixtures.loc[:, 'away_team'] = season_fixtures['away_team_id'].map(teams.set_index('id')['name'])
            season_fixtures.sort_values(by='kickoff_datetime', inplace=True)
            season_fixtures.reset_index(drop=True, inplace=True)
            season_fixtures = drop_placeholder_fixtures(season_fixtures, league)

            season_score_preds = make_round_goal_prediction(season_fixtures, ratings, avg_home_goals, avg_away_goals)

            for i in range(len(season_score_preds)):
                home_goals = season_score_preds['Home Goals'][i]
                away_goals = season_score_preds['Away Goals'][i]

            season_score_preds['Home Goals'] = season_score_preds['Home Goals'].round(2)
            season_score_preds['Away Goals'] = season_score_preds['Away Goals'].round(2)

            current_standings = standings.copy()
            current_standings['Team'] = current_standings['team_id'].map(teams.set_index('id')['name'])
            current_standings.rename(
                columns={'goals_for': 'Goals For', 'goals_against': 'Goals Against', 'points': 'Points'}, inplace=True)
            current_standings['Goal Difference'] = current_standings['Goals For'] - current_standings['Goals Against']
            current_standings = current_standings[['Team', 'Points', 'Goals For', 'Goals Against', 'Goal Difference']]
            current_standings.reset_index(drop=True, inplace=True)
            current_standings = current_standings.astype(
                {'Points': 'int', 'Goals For': 'int', 'Goals Against': 'int', 'Goal Difference': 'int'})
            current_league_table = {
                team: {'Points': points, 'Goals For': gf, 'Goals Against': ga, 'Goal Difference': gd} for
                team, points, gf, ga, gd in current_standings.values}

            # Manual points adjustments — a deduction announced before
            # Sportmonks folds it into `points`. Skipped automatically once
            # the standings already show it, so it cannot double-count.
            current_league_table = await apply_points_adjustments(
                current_league_table, standings, league_id, current_season_id, teams, league)

            avg_table, all_tables = sim_multiple_seasons(season_score_preds, current_league_table, num_sims=10000)

            avg_table_with_probs_and_point_limits = get_avg_table_with_probs_and_point_limits(avg_table,
                                                                                              all_tables)

        stat_list = get_stat_list(league_id)

        # In[21]:

        models = load_all_models(stat_list, ProjectionService.MODEL_FILE_PATH)

        # In[22]:

        if next_fix.empty:
            return Response(status_code=204)

        todays_date = pd.to_datetime(next_fix['kickoff_datetime'].iloc[0]).date()

        # In[ ]:

        team_projections = get_team_round_predictions(next_fix, stat_list, fixtures_df, team_stats, teams, stats_types,
                                                      models, ratings=ratings,
                                                      league_weightings=[league_above_attack_weight,
                                                                         league_above_defense_weight,
                                                                         league_below_attack_weight,
                                                                         league_below_defense_weight],
                                                      season_id=[current_season_id, previous_season_id,
                                                                 previous_season_id_above, previous_season_id_below],
                                                      games=50,
                                                      comp_teams=comp_teams[comp_teams['competition_id'] == league_id])

        # In[ ]:

        ## NEW - Add historical stats to the model dataset and drop them from team projections afterwards

        new_rows = []

        for i in range(len(team_projections)):
            team_df = team_projections.iloc[[i]]
            new_row = {}
            new_row['id'] = team_df['fixture_id'].values[0]
            new_row['kickoff_datetime'] = team_df['kickoff_datetime'].values[0]
            new_row['comp_id'] = league_id
            new_row['Team'] = team_df['Team'].values[0]
            new_row['Opponent'] = team_df['Opponent'].values[0]
            new_row['Venue'] = team_df['Venue'].values[0]
            for stat in stat_list:
                new_row['Team ' + stat + ' History'] = team_df['Team ' + stat + ' History'].values[0]
                new_row['Opponent ' + stat + ' History Against'] = \
                    team_df['Opponent ' + stat + ' History Against'].values[0]
            new_rows.append(new_row)

        model_dataset_league = pd.concat([model_dataset_league, pd.DataFrame(new_rows)], ignore_index=True)
        model_dataset_all = pd.concat([model_dataset_all, pd.DataFrame(new_rows)], ignore_index=True)
        model_dataset_league.drop_duplicates(subset=['id', 'Team', 'Opponent', 'Venue'], keep='last', inplace=True)
        model_dataset_all.drop_duplicates(subset=['id', 'Team', 'Opponent', 'Venue'], keep='last', inplace=True)

        ProjectionService._write_df(model_dataset_league, f"{ProjectionService.DATA_FOLDER_PATH}/{league}_model_dataset_with_history")
        ProjectionService._write_df(model_dataset_all, f"{ProjectionService.DATA_FOLDER_PATH}/all_leagues_model_dataset_with_history")
        # Dual-write to DB (see projections() for rationale).
        try:
            from app.repository.projection_dataset_repo import insert_model_dataset_async
            await insert_model_dataset_async(model_dataset_league, league_id, league, teams, fixtures_df, comp_teams)
        except Exception as _db_err:
            logger.warning(f"[{league}] model_dataset DB dual-write failed: {_db_err}")

        # model_dataset_league.to_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\{league}_model_dataset_with_history.xlsx", index=False)
        # model_dataset_all.to_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\all_leagues_model_dataset_with_history.xlsx", index=False)

        team_projections.drop(
            columns=['Team ' + stat + ' History' for stat in stat_list] + ['Opponent ' + stat + ' History Against' for
                                                                           stat in stat_list], inplace=True)

        # In[ ]:

        avg_goals = (avg_home_goals + avg_away_goals) / 2

        league_team_stats = team_stats[
            team_stats['fixture_id'].isin(fixtures_df[fixtures_df['competition_id'] == league_id]['id'])]

        league_shots = league_team_stats[league_team_stats['stats_type_id'] == get_stat_id('Shots Total',
                                                                                           stats_types)].copy()  # NEW - all team shots for specific league
        league_shots['Date'] = league_shots['fixture_id'].map(
            fixtures_df.set_index('id')['kickoff_datetime'])  # NEW - map fixture dates
        league_shots['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(
            league_shots['Date'])).dt.days // 7  # NEW - calculate weeks since kickoff
        league_shots['Weight'] = 0.9 ** (
                league_shots['Weeks Since Kickoff'] - 5)  # NEW - apply weighting to more recent matches
        league_shots.loc[league_shots['Weeks Since Kickoff'] < 6, 'Weight'] = 1  # NEW - full weight for last 5 weeks
        league_shots['Weighted Shots'] = league_shots['Weight'] * league_shots[
            'value']  # NEW - calculate weighted shots
        avg_shots = league_shots['Weighted Shots'].sum() / league_shots[
            'Weight'].sum()  # UPDATED - new formula for average shots

        league_shots_on_target = league_team_stats[league_team_stats['stats_type_id'] == get_stat_id('Shots On Target',
                                                                                                     stats_types)].copy()  # NEW - all team shots on target for specific league
        league_shots_on_target['Date'] = league_shots_on_target['fixture_id'].map(
            fixtures_df.set_index('id')['kickoff_datetime'])  # NEW - map fixture dates
        league_shots_on_target['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(
            league_shots_on_target['Date'])).dt.days // 7  # NEW - calculate weeks since kickoff
        league_shots_on_target['Weight'] = 0.9 ** (
                league_shots_on_target['Weeks Since Kickoff'] - 5)  # NEW - apply weighting to more recent matches
        league_shots_on_target.loc[
            league_shots_on_target['Weeks Since Kickoff'] < 6, 'Weight'] = 1  # NEW - full weight for last 5 weeks
        league_shots_on_target['Weighted Shots On Target'] = league_shots_on_target['Weight'] * league_shots_on_target[
            'value']  # NEW - calculate weighted shots on target
        avg_shots_on_target = league_shots_on_target['Weighted Shots On Target'].sum() / league_shots_on_target[
            'Weight'].sum()  # UPDATED - new formula for average shots on target

        avg_shots_per_goal = avg_shots / avg_goals
        avg_shots_on_target_per_goal = avg_shots_on_target / avg_goals

        # In[ ]:

        # if 'team_projections' in globals():
        goals = []
        assists = []
        for i in range(len(team_projections)):
            team = team_projections['Team'].iloc[i]
            opp = team_projections['Opponent'].iloc[i]
            # try:
            #    team_pred = score_preds[score_preds['Home Team'] == team]['Home Goals'].values[0]
            # except:
            #    team_pred = score_preds[score_preds['Away Team'] == team]['Away Goals'].values[0]
            fixture = score_preds[score_preds['id'] == team_projections['fixture_id'].iloc[
                i]]  # NEW - Get the fixture from score_preds
            team_pred = fixture['Home Goals'].values[0] if fixture['Home Team'].values[0] == team else \
                fixture['Away Goals'].values[
                    0]  # UPDATED - new way to get team prediction that handles teams having multiple matches in a round
            opp_pred = fixture['Away Goals'].values[0] if fixture['Home Team'].values[0] == opp else \
                fixture['Home Goals'].values[
                    0]  # UPDATED - new way to get opponent prediction that handles teams having multiple matches in a round
            goals.append(team_pred)
            assists.append((team_pred * 0.82).round(2))
            projected_shots = team_projections['Shots Total'].iloc[i]
            projected_shots_on_target = team_projections['Shots On Target'].iloc[i]

            adjusted_shots, adjusted_shots_on_target = adjust_shots_projection(
                team_pred,
                projected_shots,
                projected_shots_on_target,
                avg_shots_per_goal,
                avg_shots_on_target_per_goal
            )
            team_projections.at[i, 'Shots Total'] = adjusted_shots
            team_projections.at[i, 'Shots On Target'] = adjusted_shots_on_target

        team_projections['Goals'] = goals
        team_projections['Assists'] = assists

        # PL only: project team-level Ball Recovery + CBI(FPL) per fixture.
        # No PoissonRegressor exists for these stats (Sportmonks contributes
        # zero team-level rows); use get_simple_team_stat_prediction's
        # closed-form opponent-adjusted weighted average.
        # distribute_team_predictions_to_players auto-projects per-player
        # values from any column on team_projections, so adding these here
        # gives us per-player Recoveries + CBI for the team-down CBIT calc.
        if fpl:
            _lw_def = [league_above_attack_weight, league_above_defense_weight,
                       league_below_attack_weight, league_below_defense_weight]
            _sid_def = [current_season_id, previous_season_id,
                        previous_season_id_above, previous_season_id_below]
            _cpl_def = comp_teams[comp_teams['competition_id'] == league_id]
            _rec_col = []
            _cbi_col = []
            _cbit_col = []
            for i in range(len(team_projections)):
                _row = team_projections.iloc[i]
                try:
                    rec_v, _, _ = get_simple_team_stat_prediction(
                        _row['Team'], _row['Opponent'], fixtures_df, 'Ball Recovery',
                        team_stats, teams, stats_types,
                        ratings=ratings, venue=_row['Venue'], comp_id=league_id,
                        league_weightings=_lw_def, season_id=_sid_def, games=50,
                        comp_teams=_cpl_def,
                    )
                except Exception:
                    rec_v = 0
                try:
                    cbi_v, _, _ = get_simple_team_stat_prediction(
                        _row['Team'], _row['Opponent'], fixtures_df,
                        'Clearances Blocks Interceptions (FPL)',
                        team_stats, teams, stats_types,
                        ratings=ratings, venue=_row['Venue'], comp_id=league_id,
                        league_weightings=_lw_def, season_id=_sid_def, games=50,
                        comp_teams=_cpl_def,
                    )
                except Exception:
                    cbi_v = 0
                # Combined CBIT — the quantity FPL scores the threshold
                # against. Projected as ONE stat rather than assembling a
                # modelled Tackles with a blended CBI lump: FPL and Sportmonks
                # agree on the total to 0.3%, so one definition costs nothing
                # and removes the three inconsistent notions the pipeline
                # carried. George, 2026-08-04.
                try:
                    cbit_v, _cbit_th, _cbit_oh = get_simple_team_stat_prediction(
                        _row['Team'], _row['Opponent'], fixtures_df,
                        'Clearances Blocks Interceptions Tackles (FPL)',
                        team_stats, teams, stats_types,
                        ratings=ratings, venue=_row['Venue'], comp_id=league_id,
                        league_weightings=_lw_def, season_id=_sid_def, games=50,
                        comp_teams=_cpl_def,
                    )
                except Exception:
                    cbit_v = 0
                    _cbit_th = _cbit_oh = None
                # Diagnostic for the ~5% under-projection measured 2026-08-05
                # (projected 51.88 CBIT/match against an actual 55.59; 13 of 15
                # clubs low, and NOT explained by opponent mix or missing
                # history). The blend is
                #     alpha * team_history + (1-alpha) * opponent_history
                # and both terms are returned but discarded, so log them for a
                # handful of fixtures to see which side is light before
                # guessing at the cause.
                if i < 6:
                    logger.info(
                        f"[{league}] CBIT blend probe: {_row['Team']} vs {_row['Opponent']} "
                        f"({_row['Venue']}) team_hist={_cbit_th} opp_hist={_cbit_oh} "
                        f"-> blended={cbit_v}"
                    )
                _rec_col.append(rec_v)
                _cbi_col.append(cbi_v)
                _cbit_col.append(cbit_v)
            team_projections['Ball Recovery'] = _rec_col
            team_projections['Clearances Blocks Interceptions (FPL)'] = _cbi_col
            team_projections['Clearances Blocks Interceptions Tackles (FPL)'] = _cbit_col
            # Tackles Won rides a team TACKLES total, and that total is
            # CBIT - CBI rather than the modelled 'Tackles' column. Measured
            # against 760 real PL team-matches: the model averages 14.37 against
            # an actual 16.69 (-13.9%), while CBIT - CBI gives 15.74 (-5.7%) —
            # the two blended quantities are uniformly ~7% light, so their
            # DIFFERENCE tracks tackles better than the dedicated model does.
            # George called it before it was measured, 2026-08-05.
            #
            # It also makes BPS reconcile with DefCon by construction: BPS scores
            # CBI + tackles = CBI + (CBIT - CBI) = CBIT, the exact quantity the
            # DefCon threshold uses. Projecting the two separately had them
            # disagreeing by 10-14% on promoted clubs (Coventry 58.05 vs 51.00).
            #
            # The SHARE is unchanged and still measured from history — his
            # tackles won over his team's actual tackles (_TEAM_DENOMINATOR_ALIAS
            # maps the denominator to 'Tackles'), so a player's personal success
            # rate is folded in automatically.
            _tkl_from_cbit = (team_projections['Clearances Blocks Interceptions Tackles (FPL)']
                              - team_projections['Clearances Blocks Interceptions (FPL)'])
            # A fixture where the CBI blend outruns the CBIT blend would give a
            # negative tackle count; fall back to the modelled column there.
            team_projections['Tackles Won'] = _tkl_from_cbit.where(
                _tkl_from_cbit > 0, team_projections['Tackles'])
            # Split the goal projection into penalty and non-penalty halves.
            #
            # Penalties are taken as a fixed PROPORTION of projected goals, not
            # a flat per-match constant, so an attacking team is projected more
            # of them (George's call, 2026-08-07). Note the measured
            # correlation between a team's goals and its penalties across PL
            # 25/26 was 0.029 — i.e. none — with Brentford winning 10 on 55
            # goals against Man City's 4 on 77. George's position is that the
            # mechanism is real and one season of 2-10 counts is too noisy to
            # disprove it; the risk is City runs high and Brentford low.
            #
            # Constants measured over PL 24/25 + 25/26 (175 penalties):
            # penalties are 6.76% of goals, converted at 83.4%.
            # A proportional split cannot leak or invent goals — the two halves
            # always sum back to the original projection.
            _pg = team_projections['Goals'] * PENALTY_GOAL_SHARE
            team_projections['Non-Penalty Goals'] = team_projections['Goals'] - _pg
            # Named 'Penalties Scored' rather than 'Penalty Goals' so it matches
            # the PLAYER stat of the same name (111). distribute() resolves a
            # share for every team column by looking up the identically-named
            # player stat, so a column with no player counterpart hands every
            # player a zero share. Attempts and misses derive from this
            # downstream (attempts = scored / conversion), avoiding a second
            # distributed column that would need its own share.
            team_projections['Penalties Scored'] = _pg

        saves = []
        for i in range(len(team_projections)):
            # opp = team_projections['Opponent'].iloc[i]
            # try:
            #    opp_pred = score_preds[score_preds['Home Team'] == opp]['Home Goals'].values[0]
            # except:
            #    opp_pred = score_preds[score_preds['Away Team'] == opp]['Away Goals'].values[0]
            # saves.append(team_projections[team_projections['Team'] == opp]['Shots On Target'].values[0] - opp_pred)
            fixture_id = team_projections['fixture_id'].iloc[i]  # NEW - Get fixture ID
            fixture_team_projections = team_projections[
                team_projections['fixture_id'] == fixture_id]  # NEW - Get both teams' projections for the fixture
            fixture_team_projections = fixture_team_projections.drop(
                i)  # NEW - Drop the current team to get the opponent projections
            saves.append(
                fixture_team_projections['Shots On Target'].values[0] - fixture_team_projections['Goals'].values[
                    0])  # UPDATED - New way to calculate saves based on opponent projections that handles teams having multiple matches in a round

        team_projections['Saves'] = saves
        team_projections['Saves'] = team_projections['Saves'].round(2)  # NEW - Round saves to 2 decimal places
        # PL projects Key Passes properly (get_stat_list). Everywhere else it
        # stays derived — measured 0.72-0.74 across the top 5, so 0.75 runs a
        # little high. George, 2026-08-02.
        if 'Key Passes' not in team_projections.columns:
            team_projections['Key Passes'] = (team_projections['Shots Total'] * 0.75).round(2)
        # Retain Ball Recovery + CBI(FPL) columns when present (added by the
        # PL-only block above). Other leagues skip these columns.
        _extra_def_cols = [c for c in ['Ball Recovery', 'Clearances Blocks Interceptions (FPL)', 'Clearances Blocks Interceptions Tackles (FPL)', 'Tackles Won', 'Non-Penalty Goals', 'Penalties Scored']
                           if c in team_projections.columns]
        team_projections = team_projections[
            ['fixture_id', 'kickoff_datetime', 'Team', 'Opponent', 'Venue', 'Goals', 'Assists',
             'Key Passes'] + [c for c in stat_list if c != 'Key Passes'] + ['Fouls Drawn', 'Saves'] + _extra_def_cols]
        team_projections.rename(columns={'Successful Passes': 'Accurate Passes'}, inplace=True)
        logger.debug(f"[{league}] team_projections columns ready")

        # ── Team-stat odds-blend ──
        # Reels each team's projected stats (corners/cards/shots/SoT/
        # fouls/tackles) toward bookmaker expected via the cascade
        # (Path 1 per-team ladder → Path 1.5 partial+match → Path 2
        # match-split via model ratio). Per-stat bookmaker priority
        # lists in TEAM_STAT_BOOKIE_PRIORITY. Falls back to model
        # unchanged for any (stat, fixture) with no usable book data.
        from app.services.odds_blend import (
            load_team_stat_odds, blend_team_stat,
            TEAM_STAT_BOOKIE_PRIORITY, STAT_COLUMN_TO_MARKET,
        )
        _fix_ids = team_projections['fixture_id'].astype(int).unique().tolist()
        _odds_per_market = {}
        _odds_conn = await get_source_connection()
        try:
            for _market, _books in TEAM_STAT_BOOKIE_PRIORITY.items():
                _odds_per_market[_market] = await load_team_stat_odds(
                    _odds_conn, _fix_ids, _market, _books,
                )
        finally:
            release_source_connection(_odds_conn)

        _fid_to_home_team = {}
        for _fid in _fix_ids:
            _row = next_fix[next_fix['id'] == _fid]
            if not _row.empty:
                _fid_to_home_team[_fid] = _row['home_team'].iloc[0]

        _seen_fixtures = set()
        for _i in range(len(team_projections)):
            fid = int(team_projections['fixture_id'].iloc[_i])
            if fid in _seen_fixtures:
                continue
            _seen_fixtures.add(fid)
            pair = team_projections[team_projections['fixture_id'] == fid]
            if len(pair) != 2:
                continue
            home_team_name = _fid_to_home_team.get(fid)
            if not home_team_name:
                continue
            home_mask = (team_projections['fixture_id'] == fid) & (team_projections['Team'] == home_team_name)
            away_mask = (team_projections['fixture_id'] == fid) & (team_projections['Team'] != home_team_name)

            for stat_col, market in STAT_COLUMN_TO_MARKET.items():
                if stat_col not in team_projections.columns:
                    continue
                try:
                    mh = float(team_projections.loc[home_mask, stat_col].iloc[0])
                    ma = float(team_projections.loc[away_mask, stat_col].iloc[0])
                except (IndexError, KeyError, ValueError):
                    continue
                fh, fa = blend_team_stat(
                    mh, ma,
                    _odds_per_market.get(market, {}).get(fid, {}),
                    market, odds_beta,
                )
                team_projections.loc[home_mask, stat_col] = round(fh, 2)
                team_projections.loc[away_mask, stat_col] = round(fa, 2)
        
        # print(team_projections['Assists', 'Key Passes'])
        # In[ ]:

        # team_projections_save = team_projections.copy()
        # team_projections_save.drop(['Assists', 'Fouls Drawn', 'Saves', 'Key Passes'], axis=1,
        #                            inplace=True)  # UPDATED - No longer dropping interceptions and accurate passes

        team_projections_save = team_projections.copy()
        
        team_projections_save.drop(
            ['Assists', 'Fouls Drawn', 'Saves', 'Key Passes'],
            axis=1,
            inplace=True,
            errors='ignore'  # <- ovo sprečava KeyError ako kolona ne postoji
        )

        team_projections_save = team_projections_save.round(2)

        team_projections_save.rename(columns={'Accurate Passes': 'Successful Passes'},
                                     inplace=True)  # NEW - Rename back for consistency with other datasets

        # In[ ]:

        ## NEW - Update projection accuracy dataset

        for fixture_id in team_projections_save['fixture_id'].unique():
            fixture_projections = team_projections_save[team_projections_save['fixture_id'] == fixture_id]
            # accuracy dataset has no columns for the PL-only stats
            for stat in accuracy_stat_list(stat_list):
                projection_accuracy_dataset_league.loc[
                    projection_accuracy_dataset_league['fixture_id'] == fixture_id, 'Home Projected ' + stat] = \
                    fixture_projections.loc[fixture_projections['Venue'] == 'H', stat].values[0]
                projection_accuracy_dataset_league.loc[
                    projection_accuracy_dataset_league['fixture_id'] == fixture_id, 'Away Projected ' + stat] = \
                    fixture_projections.loc[fixture_projections['Venue'] == 'A', stat].values[0]
                projection_accuracy_dataset_league.loc[
                    projection_accuracy_dataset_league['fixture_id'] == fixture_id, 'Total Projected ' + stat] = \
                    fixture_projections[stat].sum()

        projection_accuracy_dataset_league.drop_duplicates(subset=['fixture_id'], keep='last', inplace=True)
        projection_accuracy_dataset_league.reset_index(drop=True, inplace=True)
        # projection_accuracy_dataset_league.to_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\{league}_accuracy_dataset.xlsx", index=False)
        ProjectionService._write_df(projection_accuracy_dataset_league, f"{ProjectionService.DATA_FOLDER_PATH}/{league}_accuracy_dataset")
        # Dual-write to DB (see projections() for rationale).
        try:
            from app.repository.projection_dataset_repo import insert_accuracy_dataset_async
            await insert_accuracy_dataset_async(projection_accuracy_dataset_league, league_id, league, teams, fixtures_df, comp_teams)
        except Exception as _db_err:
            logger.warning(f"[{league}] accuracy_dataset DB dual-write failed: {_db_err}")

        projection_accuracy_dataset_all = pd.concat(
            [projection_accuracy_dataset_all, projection_accuracy_dataset_league], ignore_index=True)
        projection_accuracy_dataset_all.drop_duplicates(subset=['fixture_id'], keep='last', inplace=True)
        projection_accuracy_dataset_all.reset_index(drop=True, inplace=True)
        # projection_accuracy_dataset_all.to_excel(rf"{ProjectionService.DATA_FOLDER_PATH}\all_leagues_accuracy_dataset.xlsx", index=False)
        ProjectionService._write_df(projection_accuracy_dataset_all, f"{ProjectionService.DATA_FOLDER_PATH}/all_leagues_accuracy_dataset")

        #
        # # **Player Projections**
        #
        # Distributing the above dataframe's values to each player based on the % of teams total

        # In[ ]:

        # UPDATED: Removed xG parameter, added comps parameter and added season_id paramter
        # Pre-load confirmed XI + player-prop odds for the same fixture
        # batch (see the projections() site for the canonical comment).
        from app.services.odds_blend import (
            load_confirmed_lineups, load_player_odds,
            PLAYER_BLEND_BOOKS, PLAYER_BLEND_STAT_IDS,
        )
        _pl_fix_ids = next_fix['id'].astype(int).unique().tolist()
        _ll_conn = await get_source_connection()
        try:
            _confirmed_lineups = await load_confirmed_lineups(_ll_conn, _pl_fix_ids)
            _odds_for_fixture_players = await load_player_odds(
                _ll_conn, _pl_fix_ids, PLAYER_BLEND_STAT_IDS, PLAYER_BLEND_BOOKS,
            )
        finally:
            release_source_connection(_ll_conn)
        pl_projections = distribute_team_predictions_to_players(player_stats, team_stats, team_projections, stats_types,
                                                                fixtures_df, players, teams, comps, 0.97,
                                                                season_id=[current_season_id, previous_season_id,
                                                                           previous_season_id_above,
                                                                           previous_season_id_below],
                                                                competition_id=league_id, comp_teams=comp_teams,
                                                                confirmed_lineups=_confirmed_lineups,
                                                                odds_for_fixture_players=_odds_for_fixture_players,
                                                                odds_blend_weight=odds_beta)

        # Vectorized: player_lookup merge + Position + Start? in one pass.
        # Saves=0 always in player_props (no GK lookup needed here).
        _team_names = teams[['id', 'name']].rename(columns={'id': '_team_id', 'name': 'Team'})
        _player_lookup = players.merge(
            _team_names, left_on='current_team_id', right_on='_team_id', how='left'
        )[['display_name', 'Team', 'id', '_team_id', 'position']].rename(
            columns={'display_name': 'Player', 'id': '_player_id'}
        ).drop_duplicates(subset=['Player', 'Team'])

        pl_projections = pl_projections.merge(_player_lookup, on=['Player', 'Team'], how='left')

        _pos_map = {'goalkeeper': 'GK', 'defender': 'DEF', 'midfielder': 'MID', 'attacker': 'FWD'}
        # Final .fillna('Unknown') catches players whose Sportmonks row has
        # NULL position (3 Allsvenskan players hit this 2026-05-28). The
        # downstream player_prop_projections.position column is NOT NULL,
        # so leaving NaN here propagates through to the SQL insert and
        # kills the whole league's projection run.
        pl_projections['Position'] = pl_projections['position'].map(_pos_map).fillna(pl_projections['position']).fillna('Unknown')
        pl_projections.loc[pl_projections['Player'] == 'Caoimhin Kelleher', 'Position'] = 'GK'
        pl_projections['Saves'] = 0

        # Predicted starters (moved here from later — runs before column reorder strips _team_id/_player_id)
        _pred_starters = player_stats[player_stats['fixture_id'].isin(next_fix['id'])]
        _pred_starters = _pred_starters[_pred_starters['stats_type_id'] == 11]
        _starter_pairs = set(zip(
            _pred_starters['team_id'].astype('Int64'),
            _pred_starters['player_id'].astype('Int64')
        ))
        pl_projections['Start?'] = [
            'Yes' if (pd.notna(t) and pd.notna(p) and (int(t), int(p)) in _starter_pairs) else 'No'
            for t, p in zip(pl_projections['_team_id'], pl_projections['_player_id'])
        ]
        pl_projections.drop(columns=['_player_id', '_team_id', 'position'], inplace=True, errors='ignore')

        pl_projections = pl_projections[
            ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue',
             'Start?',
             'Assists', 'Key Passes', 'Accurate Passes', 'Goals',
             'Shots Total',
             'Shots On Target', 'Passes', 'Interceptions', 'Tackles', 'Total Crosses',
             'Yellowcards', 'Offsides', 'Fouls', 'Fouls Drawn', 'Saves']]

        pl_projections.rename(columns={'Yellowcards': 'Yellow Cards'}, inplace=True)

        # ## **Predicted Lineups**
        #
        # Which players are predicted to play?

        # In[ ]:

        logger.info(f"[{league}] Player projections: {len(pl_projections)} rows")
        pl_projections = pl_projections[
            ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue', 'Start?',
             'Shots Total',
             'Goals', 'Assists', 'Key Passes', 'Accurate Passes',
             'Shots On Target', 'Passes', 'Interceptions', 'Tackles', 'Total Crosses',
             'Yellow Cards', 'Offsides', 'Fouls', 'Fouls Drawn', 'Saves']]
        pl_projections = pl_projections.round(2)

        # In[ ]:

        # pl_projections.sort_values(by='Goals', ascending=False, inplace=True)
        pl_projections.reset_index(drop=True, inplace=True)
        pl_projections = pl_projections.round(2)
        # pl_projections.to_csv(rf"{save_file_path}\{league} Player.csv", index=False)

        pl_projections.rename(columns={'Fouls': 'Fouls Committed'}, inplace=True)


        perc_stats = ['Shots On Target', 'Fouls Committed', 'Fouls Drawn',
                      'Goals', 'Tackles', 'Shots Total', 'Offsides']
        lines = [1, 2, 3]


        player_stat_probs = get_poisson_probs(pl_projections, perc_stats, lines)
        # Note: 'Yellowcards' is renamed to 'Yellow Cards' upstream of this point.
        if 'Yellow Cards' in pl_projections.columns:
            yellow_probs = get_poisson_probs(pl_projections, ['Yellow Cards'], [1])
            player_stat_probs = pd.concat([player_stat_probs, yellow_probs], ignore_index=True)
        player_stat_probs = player_stat_probs.round(2)
        await insert_players_stats_async(player_stat_probs, teams=teams, competition_id=league_id, comp_teams=comp_teams)
