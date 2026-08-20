"""LeagueDataLoader — scoped per-league DB loader for projection runs.

Phase 2 skeleton (Direct DB Query Migration). Replaces the read paths of
DataCache for the 3 large tables (player_stats, team_stats, fixtures_df) by
querying only the rows needed for ONE league projection.

Scope (per `loader_scope_rules.md`):
  - team_ids = current + previous season squads of: target league + extras
    + league_above + league_below.
  - player_ids = players whose current_team_id ∈ team_ids.
  - team_fixture_ids = fixtures involving any team_id, last 2yr.
  - player_fixture_ids = fixtures any player_id appeared in, last 2yr —
    captures cross-club history (e.g. Marc Bernal's Barcelona stats while
    now at Palace) AND international stats (Saka for England).
  - fixture_ids = team_fixture_ids ∪ player_fixture_ids → drives
    fixtures_df so all merge keys resolve.

  Per-table scope:
  - fixtures_df: WHERE id IN fixture_ids (the union)
  - team_stats: WHERE fixture_id IN team_fixture_ids only — both teams'
    rows loaded (no team_id filter) so get_opp_stats sees opponents.
    Cross-club fixtures intentionally EXCLUDED — no projection path
    iterates team_stats for out-of-scope clubs.
  - player_stats: WHERE player_id IN player_ids AND fixture_id IN
    fixture_ids (union).

NOT YET WIRED IN. This file is a skeleton — Phase 3 adds shadow-mode hookup.

Output schema matches DataCache attribute-by-attribute (same column names,
same dedup keys, same fixtures_df bet365 LEFT JOIN, same team_ratings.Date
type) so projection services can swap source with no other changes.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from app.source_database import get_source_connection, release_source_connection

logger = logging.getLogger("data_loader")


# Match DataCache.fixtures_df bet365 column expectations exactly. Downstream
# code reads e.g. `bet365_home_odds_decimal` — these renames preserve those.
#
# Over 1.5 / 2.5 goals moved out of bet365_fixture_odds into
# bet365_totals_odds 2026-05-22 (commit 76ae9cf0 dropped the legacy
# columns). They're now pulled in via separate derived-table joins in
# _load_fixtures so the downstream column names stay identical.
_BET365_COLS_NEEDED = [
    "fixture_id",
    "home_win_odd", "draw_odd", "away_win_odd",
    "btts_yes_odd",
]
_BET365_RENAMES = {
    "home_win_odd": "bet365_home_odds_decimal",
    "draw_odd": "bet365_draw_odds_decimal",
    "away_win_odd": "bet365_away_odds_decimal",
    "btts_yes_odd": "bet365_btts_yes_odds_decimal",
}

# Fixture history depth. Matches the ~2-season rolling window used by the
# old SEASON_FILTER_FPS/FTS in fetch_all_data_service.py. Calendar-based
# (vs season-id based) is simpler and slightly more inclusive — promoted
# teams keep their lower-league history naturally.
_FIXTURE_LOOKBACK_YEARS = 2

# Chunk size for player_stats batched query. Picked so each query's
# player_id IN list × fixture_id IN list product stays within MySQL's
# default 8MB range_optimizer budget. Empirically: 500 players × 21k
# fixtures completes in ~1-2s; 10k players × 21k fixtures hangs.
_PLAYER_CHUNK_SIZE = 500


# Minimum distinct players for a fpl_player_snapshots date to be treated as a
# complete bootstrap. FPL carries ~560 players across 20 clubs; 400 is well
# below any real day and well above a partial write.
FPL_SNAPSHOT_MIN_PLAYERS = 400

# xG Sportmonks assigns to a penalty. MEASURED, not assumed: of 859
# (player, fixture) rows holding exactly one penalty and exactly one shot — so
# the xG is the penalty alone — 98.7% read 0.790. Confirms two things at once:
# their xG INCLUDES penalties, and this is the price.
#
# 0.79 rather than the 0.78 general football figure precisely because we
# subtract it from Sportmonks' own xG; at 0.78 a penalty-only appearance leaves
# 0.01 npxG behind, which is pure noise in a share denominator.
PENALTY_XG = 0.79

# Synthetic stats_type id holding Sportmonks' OWN xG for the Premier League,
# snapshotted before _overlay_fpl_stats replaces it with FPL/Opta values.
#
# Why it exists: the overlay is PL-only (FPL covers no other league) and FPL
# carries per-fixture xG only from 2025/26 — 2024/25 is 16,061 rows with the
# column entirely NULL. So a PL goal-average window spanning two seasons was
# built from Opta for the recent half and Sportmonks for the older half.
# Measured 2026-08-20: exactly 51% of loaded PL xG rows differed from the DB,
# matching the 380-of-750 season split, and the two providers sit 0.18 apart
# on identical fixtures.
#
# The league goal averages want ONE provider across the whole window, and
# Sportmonks is the one every other league uses. Player projections keep the
# Opta values — they are per-player and better — so only the league-level
# level/ratio reads this id. See stats_types_synthetic_ids memory note; 999001
# is xA, 999002 CBI, 999003 CBIT, 999004 Non-Penalty Goals, 999005 npxG --
# 999006 is the first free slot. These ids are NOT auto-increment; a collision
# would silently feed xG rows into whatever consumer owns that id.
SM_XG_STAT_ID = 999006


class LeagueDataLoader:
    """Loads scoped data for projecting ONE competition.

    Lifecycle: instantiate per projection run, call `load()`, read
    attributes, drop. Reference tables are loaded fresh each run for now;
    Phase 3+ may add a session-level cache for Run All Leagues bursts.
    """

    def __init__(
        self,
        league_id: int,
        *,
        # For Euro comps the scope spans multiple domestic top tiers. Caller
        # passes the list explicitly. None = single-league scope.
        extra_league_ids: Optional[Sequence[int]] = None,
        league_weightings_xlsx_path: Optional[str] = None,
        # Narrow the team_ids set directly (skips the
        # competition_season_teams + league_above/below resolution). Used
        # by euro-comp runs that already know the small set of teams in
        # their upcoming fixtures — collapses a ~248-team scope to a
        # ~2-team scope for finals, cutting loader time from ~7min to
        # <30s. None = derive scope from league_id + extras (default).
        restrict_team_ids: Optional[Sequence[int]] = None,
    ):
        self.league_id = int(league_id)
        self.extra_league_ids: List[int] = [int(x) for x in (extra_league_ids or [])]
        self.league_weightings_xlsx_path = league_weightings_xlsx_path
        self.restrict_team_ids: Optional[List[int]] = (
            sorted({int(x) for x in restrict_team_ids})
            if restrict_team_ids is not None else None
        )

        # Resolved scope (populated by _resolve_scope / _resolve_fixture_ids)
        self.team_ids: List[int] = []
        self.player_ids: List[int] = []
        self.team_fixture_ids: List[int] = []     # team-based set
        self.player_fixture_ids: List[int] = []   # player-based set (cross-club + intl)
        self.comp_fixture_ids: List[int] = []     # comp-based set (narrow scope only)
        self.fixture_ids: List[int] = []          # UNION — drives fixtures_df

        # Scoped tables (the 3 big ones)
        self.player_stats: Optional[pd.DataFrame] = None
        self.team_stats: Optional[pd.DataFrame] = None
        self.fixtures_df: Optional[pd.DataFrame] = None

        # Reference tables (loaded in full — small)
        self.standings: Optional[pd.DataFrame] = None
        self.seasons: Optional[pd.DataFrame] = None
        self.comps: Optional[pd.DataFrame] = None
        self.comp_teams: Optional[pd.DataFrame] = None
        self.teams: Optional[pd.DataFrame] = None
        self.b365_odds: Optional[pd.DataFrame] = None
        self.stats_types: Optional[pd.DataFrame] = None
        self.league_weightings: Optional[pd.DataFrame] = None
        self.projection_config: Optional[pd.DataFrame] = None
        self.promoted_team_ratings: Optional[pd.DataFrame] = None
        self.fpl_player_dials: Optional[pd.DataFrame] = None
        self.transfermarkt_team_mappings: Optional[pd.DataFrame] = None
        self.team_ratings: Optional[pd.DataFrame] = None
        self.fpl_player_mappings: Optional[pd.DataFrame] = None

        # Players scoped to teams in this run (current squad). Columns:
        # id, display_name, current_team_id, position. Replaces the legacy
        # pd.read_csv("players.csv") path which loaded ALL ~150k+ players
        # globally — projection code only ever uses team-scoped subsets so
        # this is functionally equivalent and avoids stale-CSV risk.
        self.players: Optional[pd.DataFrame] = None

        self._loaded = False

    # ── Public API ────────────────────────────────────────────────────────

    async def load(self) -> None:
        """Resolve scope → resolve fixture IDs → load tables.

        Reference tables and scope come first. Then fixture-ID resolution
        (two queries: team-based + player-based, UNION'd). Then the three
        scoped data loaders. Sequential on a single connection — queries
        are fast (milliseconds) on indexed tables, parallelism would just
        add pool churn."""
        conn = await get_source_connection()
        try:
            # Euro-comp scope can have 10k+ player_ids in the IN list,
            # exceeding MySQL's default 8MB range_optimizer_max_mem_size
            # and falling back to full table scan on fixture_player_stats
            # (15M rows). Lift the cap for this session — single connection,
            # released to pool when load() returns. NOT a global config
            # change; only this loader's queries see the bump.
            async with conn.cursor() as cur:
                await cur.execute(
                    "SET SESSION range_optimizer_max_mem_size = 0"
                )

            await self._load_reference_tables(conn)
            await self._resolve_scope(conn)
            await self._resolve_fixture_ids(conn)
            await self._load_fixtures(conn)
            await self._load_team_stats(conn)
            await self._load_player_stats(conn)
            await self._overlay_fpl_stats(conn)
            # Sportmonks fallback for CBIT + team-level recoveries. Runs for EVERY
            # league (the FPL overlay above is PL-only), so a promoted player's
            # Championship history carries defensive data instead of collapsing to
            # a ~0 share. Must come AFTER the overlay: FPL keeps precedence on PL.
            await self._derive_cbit_from_components(conn)
            await self._derive_non_penalty_goals(conn)
            await self._derive_non_penalty_xg(conn)
            self._load_local_files()
            self._loaded = True
            logger.info(
                "LeagueDataLoader loaded for comp_id=%s: "
                "%d teams, %d players, %d team_fixtures, %d player_fixtures, "
                "%d fixtures (union), %d team_stat rows, %d player_stat rows",
                self.league_id, len(self.team_ids), len(self.player_ids),
                len(self.team_fixture_ids), len(self.player_fixture_ids),
                len(self.fixture_ids),
                0 if self.team_stats is None else len(self.team_stats),
                0 if self.player_stats is None else len(self.player_stats),
            )
        finally:
            release_source_connection(conn)

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Scope resolution ──────────────────────────────────────────────────

    async def _resolve_scope(self, conn) -> None:
        """Compute team_ids and player_ids.

        Team scope = (target_league + extras + league_above + league_below)
        × current 2 seasons. Players = current_team_id IN team_ids.
        Fixture-ID resolution is a separate step (`_resolve_fixture_ids`).

        When `restrict_team_ids` is supplied (euro-comp single-fixture
        path), skip the comp-derived resolution entirely and use it
        directly — collapses ~248 → ~2 teams for finals.
        """
        if self.restrict_team_ids:
            self.team_ids = list(self.restrict_team_ids)
            logger.info(
                "LeagueDataLoader: restrict_team_ids supplied — skipping "
                "comp-derived scope, using %d teams directly",
                len(self.team_ids),
            )
            await self._resolve_players(conn)
            return

        comp_ids = self._all_scope_comp_ids()

        # Add league_above / league_below from competition_projection_config
        async with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(comp_ids))
            await cur.execute(
                f"""
                SELECT league_above_id, league_below_id
                FROM competition_projection_config
                WHERE competition_id IN ({placeholders})
                """,
                tuple(comp_ids),
            )
            rows = await cur.fetchall()
        for above, below in rows:
            if above is not None:
                comp_ids.add(int(above))
            if below is not None:
                comp_ids.add(int(below))

        # Resolve team_ids: top 2 seasons per competition in scope.
        # SELECT DISTINCT inside the window function so each season gets
        # ranked once (not once per team-row). Same fix as SEASON_FILTER_FPS
        # — the bug that silently capped to 1 season for months.
        async with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(comp_ids))
            await cur.execute(
                f"""
                SELECT DISTINCT cst.team_id
                FROM competition_season_teams cst
                JOIN (
                    SELECT competition_id, season_id FROM (
                        SELECT competition_id, season_id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY competition_id
                                   ORDER BY season_id DESC
                               ) AS rn
                        FROM (
                            SELECT DISTINCT competition_id, season_id
                            FROM competition_season_teams
                            WHERE competition_id IN ({placeholders})
                        ) cs
                    ) ranked
                    WHERE rn <= 2
                ) recent
                  ON recent.competition_id = cst.competition_id
                 AND recent.season_id = cst.season_id
                """,
                tuple(comp_ids),
            )
            self.team_ids = sorted({int(r[0]) for r in await cur.fetchall()})

        await self._resolve_players(conn)

    async def _resolve_players(self, conn) -> None:
        """Resolve self.players + self.player_ids from self.team_ids.

        Shared between the standard comp-derived scope path and the
        `restrict_team_ids` shortcut — both need current_team_id-based
        player resolution off the same team set.
        """
        if not self.team_ids:
            logger.warning(
                "LeagueDataLoader: scope resolution returned 0 teams for comp_id=%s",
                self.league_id,
            )
            self.player_ids = []
            self.players = pd.DataFrame(columns=['id', 'display_name', 'current_team_id', 'position'])
            return

        async with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(self.team_ids))
            await cur.execute(
                f"""
                SELECT id, display_name, current_team_id, position
                FROM players
                WHERE current_team_id IN ({placeholders})
                """,
                tuple(self.team_ids),
            )
            rows = await cur.fetchall()
        self.players = pd.DataFrame(rows, columns=['id', 'display_name', 'current_team_id', 'position'])
        self.players['display_name'] = self.players['display_name'].astype(str).str.strip()
        self.player_ids = sorted({int(x) for x in self.players['id'].tolist()})

    def _all_scope_comp_ids(self) -> set:
        ids = {self.league_id}
        ids.update(self.extra_league_ids)
        return ids

    # ── Fixture-ID resolution (two sources, UNION'd) ──────────────────────

    async def _resolve_fixture_ids(self, conn) -> None:
        """Resolve team_fixture_ids + player_fixture_ids, store union.

        Team-based: any in-scope team's fixtures in last 2yr.
        Player-based: any fixture an in-scope player appeared in (last 2yr) —
        captures cross-club history (e.g. Bernal's Barca games while now at
        Palace) AND international fixtures (e.g. Saka for England).
        Comp-based (narrow scope only): all fixtures from comps in scope —
        ensures league-wide averages (avg_shots/avg_goals/etc) have full
        coverage even when team_ids is collapsed to just the upcoming
        fixture's participants."""
        cutoff = datetime.utcnow() - timedelta(days=365 * _FIXTURE_LOOKBACK_YEARS)

        # Comp-based set — only needed when team_ids is artificially narrow
        # (otherwise team_fixture_ids already covers every team in those
        # comps, which IS every team in those comps' historical fixtures).
        self.comp_fixture_ids: List[int] = []
        if self.restrict_team_ids:
            comp_ids = list(self._all_scope_comp_ids())
            if comp_ids:
                comp_ph = ",".join(["%s"] * len(comp_ids))
                sql = f"""
                    SELECT id FROM fixtures
                    WHERE competition_id IN ({comp_ph})
                      AND kickoff_datetime >= %s
                """
                params = tuple(comp_ids) + (cutoff,)
                async with conn.cursor() as cur:
                    await cur.execute(sql, params)
                    self.comp_fixture_ids = sorted({int(r[0]) for r in await cur.fetchall()})
                logger.info(
                    "LeagueDataLoader: narrow scope — pulled %d comp fixtures "
                    "across %d comps for league-wide aggregations",
                    len(self.comp_fixture_ids), len(comp_ids),
                )

        # Team-based set
        if self.team_ids:
            team_ph = ",".join(["%s"] * len(self.team_ids))
            sql = f"""
                SELECT id FROM fixtures
                WHERE (home_team_id IN ({team_ph}) OR away_team_id IN ({team_ph}))
                  AND kickoff_datetime >= %s
            """
            params = tuple(self.team_ids) + tuple(self.team_ids) + (cutoff,)
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                self.team_fixture_ids = sorted({int(r[0]) for r in await cur.fetchall()})
        else:
            self.team_fixture_ids = []

        # Player-based set (cross-club + international)
        if self.player_ids:
            player_ph = ",".join(["%s"] * len(self.player_ids))
            sql = f"""
                SELECT DISTINCT fps.fixture_id
                FROM fixture_player_stats fps
                JOIN fixtures f ON f.id = fps.fixture_id
                WHERE fps.player_id IN ({player_ph})
                  AND f.kickoff_datetime >= %s
            """
            params = tuple(self.player_ids) + (cutoff,)
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                self.player_fixture_ids = sorted({int(r[0]) for r in await cur.fetchall()})
        else:
            self.player_fixture_ids = []

        self.fixture_ids = sorted(
            set(self.team_fixture_ids)
            | set(self.player_fixture_ids)
            | set(self.comp_fixture_ids)
        )

    # ── Scoped table loaders ──────────────────────────────────────────────

    async def _load_fixtures(self, conn) -> None:
        """All fixtures referenced by team_stats OR player_stats. UNION
        ensures downstream merges (player_stats.fixture_id ↔ fixtures.id)
        always resolve. bet365 LEFT JOIN preserves DataCache column names."""
        if not self.fixture_ids:
            self.fixtures_df = pd.DataFrame()
            return

        fix_ph = ",".join(["%s"] * len(self.fixture_ids))
        b365_select = ", ".join(
            f"b365.{col} AS {_BET365_RENAMES.get(col, col)}"
            for col in _BET365_COLS_NEEDED if col != "fixture_id"
        )
        # Goals totals (match-grain, side=over) live in bet365_totals_odds
        # as of 2026-05-22. team_id IS NULL marks match-level. The unique
        # index covers team_id but MySQL allows multiple NULLs, so we
        # collapse with MAX(price) per (fixture, line) — duplicate rows
        # typically share a price anyway.
        sql = f"""
            SELECT f.*, {b365_select},
                bt15.price AS over_1_5_odds_decimal,
                bt25.price AS over_2_5_odds_decimal
            FROM fixtures f
            LEFT JOIN bet365_fixture_odds b365 ON b365.fixture_id = f.id
            LEFT JOIN (
                SELECT fixture_id, MAX(price) AS price
                FROM bet365_totals_odds
                WHERE market = 'goals' AND team_id IS NULL
                  AND line = 1.5 AND side = 'over'
                GROUP BY fixture_id
            ) bt15 ON bt15.fixture_id = f.id
            LEFT JOIN (
                SELECT fixture_id, MAX(price) AS price
                FROM bet365_totals_odds
                WHERE market = 'goals' AND team_id IS NULL
                  AND line = 2.5 AND side = 'over'
                GROUP BY fixture_id
            ) bt25 ON bt25.fixture_id = f.id
            WHERE f.id IN ({fix_ph})
        """
        async with conn.cursor() as cur:
            await cur.execute(sql, tuple(self.fixture_ids))
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        df = pd.DataFrame(rows, columns=cols)

        if not df.empty:
            df.drop_duplicates(
                subset=["season_id", "home_team_id", "away_team_id", "kickoff_datetime"],
                inplace=True,
            )
            # Coerce DECIMAL columns (bet365 odds) — MySQL DECIMAL → Python
            # decimal.Decimal via aiomysql, but downstream code expects floats
            # (e.g. `1/odd`, `.round()`, `*` with floats). CSV mode dodges
            # this via pandas type inference.
            #
            # The over_1_5/2_5 columns come from the new bet365_totals_odds
            # derived-table joins and aren't in _BET365_RENAMES — coerce
            # them explicitly so they don't slip through as Decimal.
            _coerce_cols = list(_BET365_RENAMES.values()) + [
                'over_1_5_odds_decimal', 'over_2_5_odds_decimal'
            ]
            for col in _coerce_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        self.fixtures_df = df

    async def _load_team_stats(self, conn) -> None:
        """Team stats for the UNION fixture set, filtered to projection-relevant
        stat_types only.

        Two things changed 2026-04-30 vs the original Phase-2 design:

        1. Fixture scope = `self.fixture_ids` (UNION) NOT `self.team_fixture_ids`.
           The original comment claimed "no projection path reads team_stats
           from out-of-scope clubs" — wrong. `get_player_stats` does a per-row
           merge against `team_df` keyed on (fixture_id, team_id), where the
           team_id is whatever club the player was at for that fixture. For
           transferred players (Souza at Tottenham, history still at his old
           club) those rows fail to merge against in-league-scoped team_df and
           the share denominator collapses to 0 → NaN-guard fires, projection
           forced to 0.

        2. stats_type_id filter pulls only `TEAM_STAT_NAMES` (~13 of ~1,116
           stat types). Reduces ~70% of row volume — pays for the +cross-club
           rows from change #1 several times over.

        No `team_id` filter: get_opp_stats needs opposing-team rows.
        See loader_scope_rules.md.
        """
        from app.services.projection_stats import TEAM_STAT_NAMES, resolve_stat_ids

        if not self.fixture_ids:
            self.team_stats = pd.DataFrame()
            return

        team_stat_type_ids = resolve_stat_ids(TEAM_STAT_NAMES, self.stats_types)
        fix_ph = ",".join(["%s"] * len(self.fixture_ids))
        stat_ph = ",".join(["%s"] * len(team_stat_type_ids))
        sql = f"""
            SELECT * FROM fixture_team_stats
            WHERE fixture_id IN ({fix_ph})
              AND stats_type_id IN ({stat_ph})
        """
        async with conn.cursor() as cur:
            await cur.execute(sql, tuple(self.fixture_ids) + tuple(team_stat_type_ids))
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        if not df.empty:
            df.drop_duplicates(
                subset=["fixture_id", "team_id", "stats_type_id"],
                inplace=True,
            )
            # `value` is stored as VARCHAR in source DB; CSV mode inferred
            # it as float via pandas. Coerce here so downstream arithmetic
            # (home + away, > 2.5, etc.) doesn't string-concatenate.
            if "value" in df.columns:
                df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Reconcile every counting-stat share denominator to
        # max(team_row, player_sum). Each is a one-event-one-player count, so
        # the team total SHOULD equal the sum of player values — but the two
        # Sportmonks feeds disagree in a meaningful fraction of fixtures
        # (esp. international: team Assists missing/low ~33% of games, Key
        # Passes ~24%; sometimes a thin player import makes the player-sum
        # low instead). Both modes are undercounts, so max() picks the better
        # estimate: no-op where the feeds agree, player-sum where the team row
        # is short, team_row where the player import is thin. Mirrors the
        # intl/WC pipeline (wc_player_stat_service _COUNTING_DENOM_MAP). xG
        # (modeled), xA/Ball Recovery/CBI-FPL (overlay-injected, already
        # player-derived) and Fouls Drawn (opponent-based) are excluded.
        if not df.empty:
            name_to_id = dict(zip(self.stats_types["name"], self.stats_types["id"]))
            # (player_stat_name, team_stat_name) — same name except
            # Accurate Passes (player) → Successful Passes (team total).
            COUNTING_NAME_PAIRS = [
                ("Goals", "Goals"), ("Shots Total", "Shots Total"),
                ("Shots On Target", "Shots On Target"), ("Fouls", "Fouls"),
                ("Yellowcards", "Yellowcards"), ("Tackles", "Tackles"),
                ("Passes", "Passes"), ("Accurate Passes", "Successful Passes"),
                ("Total Crosses", "Total Crosses"), ("Interceptions", "Interceptions"),
                ("Offsides", "Offsides"), ("Assists", "Assists"),
                ("Key Passes", "Key Passes"),
            ]
            pid_to_tid: dict = {}
            for pname, tname in COUNTING_NAME_PAIRS:
                pid, tid = name_to_id.get(pname), name_to_id.get(tname)
                if pid is not None and tid is not None:
                    pid_to_tid[int(pid)] = int(tid)

            if pid_to_tid:
                player_ids = list(pid_to_tid.keys())
                psum_ph = ",".join(["%s"] * len(player_ids))
                # stats_imported=1 gate so thin imports don't feed a too-low
                # player-sum into the reconciliation (matches WC).
                sql_psum = f"""
                    SELECT fps.fixture_id, fps.team_id, fps.stats_type_id, SUM(fps.value) AS total
                    FROM fixture_player_stats fps
                    JOIN fixtures f ON f.id = fps.fixture_id
                    WHERE fps.fixture_id IN ({fix_ph})
                      AND fps.stats_type_id IN ({psum_ph})
                      AND f.stats_imported = 1
                    GROUP BY fps.fixture_id, fps.team_id, fps.stats_type_id
                """
                async with conn.cursor() as cur:
                    await cur.execute(
                        sql_psum, tuple(self.fixture_ids) + tuple(player_ids)
                    )
                    psum_rows = await cur.fetchall()

                if psum_rows:
                    # Map player_id → team_id so the player-sum lands on the
                    # team denominator's stats_type_id (Accurate→Successful).
                    psum_df = pd.DataFrame(
                        [
                            {
                                "fixture_id": int(r[0]),
                                "team_id": int(r[1]),
                                "stats_type_id": pid_to_tid[int(r[2])],
                                "psum": float(r[3]) if r[3] is not None else 0.0,
                            }
                            for r in psum_rows
                            if int(r[2]) in pid_to_tid
                        ]
                    )
                    # Successful Passes (team) AND Accurate Passes player-sum
                    # both map to the Successful Passes id — collapse dupes.
                    psum_df = (
                        psum_df.groupby(["fixture_id", "team_id", "stats_type_id"], as_index=False)["psum"].sum()
                    )
                    counting_tids = set(pid_to_tid.values())
                    mask = df["stats_type_id"].isin(counting_tids)
                    counting = df[mask]
                    rest = df[~mask]
                    merged = counting.merge(
                        psum_df, on=["fixture_id", "team_id", "stats_type_id"], how="outer"
                    )
                    # max() over the two candidate denominators, NaN-safe
                    # (missing team row → use psum; missing player data → keep team).
                    merged["value"] = merged[["value", "psum"]].max(axis=1)
                    merged.drop(columns=["psum"], inplace=True)
                    df = pd.concat([rest, merged], ignore_index=True)

        self.team_stats = df

    async def _load_player_stats(self, conn) -> None:
        """Player stats across UNION fixture set so cross-club + international
        history is captured. player_id filter ensures we only load rows for
        currently-in-scope players, not e.g. Real Madrid players from a
        Barcelona-vs-Real fixture pulled in via Bernal's history.

        Batched by player_id chunks to keep the query planner sane. Euro
        comp scope can have 10k+ player_ids × 20k+ fixture_ids — the dual
        IN clause blows past MySQL's range_optimizer_max_mem_size and
        falls back to full table scan on fixture_player_stats (15M rows).
        Splitting into player-id chunks of `_PLAYER_CHUNK_SIZE` keeps each
        query small enough for the optimizer to use indexes.

        2026-04-30: stats_type_id filter added to pull only PLAYER_STAT_NAMES
        (~23 of ~1,116 stat types). ~70% volume reduction without changing
        any caller behaviour — projection paths only read these stat names.
        """
        from app.services.projection_stats import PLAYER_STAT_NAMES, resolve_stat_ids

        if not self.player_ids or not self.fixture_ids:
            self.player_stats = pd.DataFrame()
            return

        player_stat_type_ids = resolve_stat_ids(PLAYER_STAT_NAMES, self.stats_types)
        chunks = []
        cols = None
        fix_ph = ",".join(["%s"] * len(self.fixture_ids))
        stat_ph = ",".join(["%s"] * len(player_stat_type_ids))
        fix_params = tuple(self.fixture_ids)
        stat_params = tuple(player_stat_type_ids)

        for i in range(0, len(self.player_ids), _PLAYER_CHUNK_SIZE):
            batch = self.player_ids[i : i + _PLAYER_CHUNK_SIZE]
            player_ph = ",".join(["%s"] * len(batch))
            sql = f"""
                SELECT * FROM fixture_player_stats
                WHERE player_id IN ({player_ph})
                  AND fixture_id IN ({fix_ph})
                  AND stats_type_id IN ({stat_ph})
            """
            params = tuple(batch) + fix_params + stat_params
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
                if cols is None:
                    cols = [d[0] for d in cur.description]
            if rows:
                chunks.append(pd.DataFrame(rows, columns=cols))

        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols or [])
        if not df.empty:
            df.drop_duplicates(
                subset=["fixture_id", "player_id", "stats_type_id"],
                inplace=True,
            )
            if "value" in df.columns:
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
        self.player_stats = df

    async def _overlay_fpl_stats(self, conn) -> None:
        """Premier League only: overlay FPL stats onto in-memory player_stats
        and team_stats DataFrames. PL only because the FPL API only covers PL.

        Why: FPL is Opta-sourced and considered authoritative for the stats
        it tracks. For stats Sportmonks ALSO has (xG, Tackles, Recoveries),
        we replace SM values with FPL where available — same definition,
        slightly more accurate. For stats Sportmonks LACKS (xA, combined CBI),
        we inject as new rows tagged with synthetic stats_type ids (see
        stats_types_synthetic_ids memory note).

        Stats applied here:
        - xG       (overlay onto SM 'Expected Goals (xG)')
        - xA       (inject as synthetic 'Expected Assists (xA)', id 999001)
        - Tackles  (overlay onto SM 'Tackles')
        - Recoveries (overlay onto SM 'Ball Recovery')
        - CBI      (inject as synthetic 'Clearances Blocks Interceptions
                    (FPL)', id 999002 — SM has 3 separate components but
                    no combined total; the components stay loaded for
                    other consumers, this row is just for the team-down
                    CBIT projection)

        Team-level totals are derived by summing player rows per
        (fixture_id, team_id). Mutates DataFrames in-memory only — DB
        is not touched. See projection_stats.py for which stat names are
        in TEAM_STAT_NAMES / PLAYER_STAT_NAMES (must include any stat we
        overlay, or the loader filter will drop the SM rows we'd be
        replacing — and we'd ALSO be unable to read our injected rows
        since the loader wouldn't know to load them).
        """
        # Premier League scope only.
        if self.league_id != 8:
            return
        if self.player_stats is None or self.player_stats.empty:
            return
        if not self.player_ids or not self.fixture_ids:
            return

        def _resolve(name: str, required: bool = True) -> int | None:
            m = self.stats_types[self.stats_types["name"] == name]
            if m.empty:
                level = "warning" if required else "info"
                getattr(logger, level)(
                    "[FPL overlay] '%s' stats_type missing — skipping its branch", name
                )
                return None
            return int(m["id"].iloc[0])

        xg_id = _resolve("Expected Goals (xG)")
        xa_id = _resolve("Expected Assists (xA)")
        tackles_id = _resolve("Tackles")
        recoveries_id = _resolve("Ball Recovery")
        cbi_id = _resolve("Clearances Blocks Interceptions (FPL)")
        cbit_id = _resolve("Clearances Blocks Interceptions Tackles (FPL)")

        if (xg_id is None and xa_id is None and tackles_id is None
                and recoveries_id is None and cbi_id is None and cbit_id is None):
            logger.warning("[FPL overlay] No FPL-overlayable stats_types resolved — skipping overlay entirely")
            return

        # Fetch FPL data. WHERE clause: drop rows where ALL FPL fields are
        # null — they have no overlay value. Per-field null filter happens
        # later, when each stat's branch picks its own rows.
        p_ph = ",".join(["%s"] * len(self.player_ids))
        f_ph = ",".join(["%s"] * len(self.fixture_ids))
        sql = f"""
            SELECT player_id, fixture_id, expected_goals, expected_assists,
                   tackles, recoveries, clearances_blocks_interceptions
            FROM fpl_player_stats
            WHERE player_id IN ({p_ph})
              AND fixture_id IN ({f_ph})
              AND (expected_goals IS NOT NULL
                   OR expected_assists IS NOT NULL
                   OR tackles IS NOT NULL
                   OR recoveries IS NOT NULL
                   OR clearances_blocks_interceptions IS NOT NULL)
        """
        async with conn.cursor() as cur:
            await cur.execute(sql, tuple(self.player_ids) + tuple(self.fixture_ids))
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]

        if not rows:
            logger.info("[FPL overlay] No FPL rows for in-scope players × fixtures.")
            return

        fpl = pd.DataFrame(rows, columns=cols)
        for col in ("expected_goals", "expected_assists", "tackles", "recoveries",
                    "clearances_blocks_interceptions"):
            fpl[col] = pd.to_numeric(fpl[col], errors="coerce")
        # Combined CBIT, the quantity FPL actually scores the defensive
        # contribution threshold against (DEF need 10; MID/FWD need 12 of
        # CBIT + recoveries). Injected as ONE stat so the standard share path
        # handles it — recency weighting, damping, cross-club and per-90 all
        # come free, and there is a single definition instead of the three the
        # pipeline used to carry. George, 2026-08-04.
        fpl["cbit"] = (
            fpl["clearances_blocks_interceptions"].fillna(0) + fpl["tackles"].fillna(0)
        ).where(
            fpl["clearances_blocks_interceptions"].notna() | fpl["tackles"].notna()
        )

        # Map player→team. Required to stamp injected rows with the correct
        # team_id and to aggregate per-team for team_stats overlay. Drop
        # rows where player_stats has no record for that (player, fixture):
        # those wouldn't pass the Minutes Played filter downstream anyway.
        team_lookup = (
            self.player_stats[["player_id", "fixture_id", "team_id", "season_id"]]
            .drop_duplicates(subset=["player_id", "fixture_id"])
        )
        fpl = fpl.merge(team_lookup, on=["player_id", "fixture_id"], how="left")
        fpl = fpl.dropna(subset=["team_id"])
        if fpl.empty:
            logger.info("[FPL overlay] FPL rows didn't team-stamp via player_stats.")
            return

        # Run each stat through the same overlay pattern. (id, value_col, label).
        # For "inject only" stats (xA, CBI) the overlay step is a no-op
        # because no SM rows exist with that stat_id — all rows go through
        # the append path. Same code, different distribution of counts.
        plan = [
            (xg_id, "expected_goals", "xG"),
            (xa_id, "expected_assists", "xA"),
            (tackles_id, "tackles", "Tackles"),
            (recoveries_id, "recoveries", "Recoveries"),
            (cbi_id, "clearances_blocks_interceptions", "CBI"),
            (cbit_id, "cbit", "CBIT"),
        ]

        # Snapshot Sportmonks' own team xG before the loop below overwrites
        # it, so the league goal averages can stay on a single provider.
        # Cheap: ~1.5k rows for a PL run, and only for the PL.
        if xg_id is not None and self.team_stats is not None and not self.team_stats.empty:
            sm_xg = self.team_stats[self.team_stats["stats_type_id"] == xg_id].copy()
            if not sm_xg.empty:
                sm_xg["stats_type_id"] = SM_XG_STAT_ID
                self.team_stats = pd.concat([self.team_stats, sm_xg], ignore_index=True)
                logger.info("[FPL overlay] preserved %d Sportmonks team xG rows as id %d",
                            len(sm_xg), SM_XG_STAT_ID)

        results = []
        for stat_id, col, label in plan:
            if stat_id is None:
                continue
            p_o, p_a = self._apply_player_stat_overlay(fpl, col, stat_id)
            t_o, t_a = self._apply_team_stat_overlay(fpl, col, stat_id)
            results.append((label, p_o, p_a, t_o, t_a))

        summary = ", ".join(
            f"{lbl}: p_ov={po} p_ap={pa} t_ov={to} t_ap={ta}"
            for lbl, po, pa, to, ta in results
        )
        logger.info("[FPL overlay] PL: %s", summary)

    def _apply_player_stat_overlay(self, fpl: pd.DataFrame, value_col: str, stat_id: int) -> tuple[int, int]:
        """Overlay one FPL field onto self.player_stats for one stat_type_id.

        For each (player, fixture) where FPL has a non-null value:
        - if a SM row exists with this stat_type_id → replace its value
        - else → append a new row tagged with stat_type_id

        Returns (overlaid_count, appended_count).
        """
        sub = fpl[fpl[value_col].notna()][
            ["player_id", "fixture_id", "team_id", "season_id", value_col]
        ].copy()
        if sub.empty:
            return 0, 0

        ps = self.player_stats
        val_map = sub.set_index(["player_id", "fixture_id"])[value_col]
        mask = ps["stats_type_id"] == stat_id

        n_overlaid = 0
        existing = ps[mask]
        if not existing.empty:
            idx = pd.MultiIndex.from_arrays(
                [existing["player_id"], existing["fixture_id"]],
                names=["player_id", "fixture_id"],
            )
            aligned = val_map.reindex(idx)
            overlay_mask = aligned.notna().values
            if overlay_mask.any():
                ps.loc[existing.index[overlay_mask], "value"] = aligned.values[overlay_mask]
                n_overlaid = int(overlay_mask.sum())

        sm_keys = set(zip(ps[mask]["player_id"], ps[mask]["fixture_id"]))
        fpl_only = sub[~sub.apply(lambda r: (r["player_id"], r["fixture_id"]) in sm_keys, axis=1)]
        n_appended = 0
        if not fpl_only.empty:
            new_rows = pd.DataFrame({
                "player_id": fpl_only["player_id"].astype("int64"),
                "fixture_id": fpl_only["fixture_id"].astype("int64"),
                "team_id": fpl_only["team_id"].astype("int64"),
                "season_id": fpl_only["season_id"].astype("int64"),
                "stats_type_id": stat_id,
                "value": fpl_only[value_col].astype(float),
            })
            self.player_stats = pd.concat([ps, new_rows], ignore_index=True)
            n_appended = len(new_rows)

        return n_overlaid, n_appended

    def _apply_team_stat_overlay(self, fpl: pd.DataFrame, value_col: str, stat_id: int) -> tuple[int, int]:
        """Aggregate FPL player rows to (fixture_id, team_id) and overlay
        team_stats for one stat_type_id. Same overlay/append pattern as
        the player-side helper.

        Returns (overlaid_count, appended_count).
        """
        if self.team_stats is None or self.team_stats.empty:
            return 0, 0

        team_agg = (
            fpl[fpl[value_col].notna()]
            .groupby(["fixture_id", "team_id"], as_index=False)[value_col]
            .sum()
        )
        if team_agg.empty:
            return 0, 0

        ts = self.team_stats
        val_map = team_agg.set_index(["fixture_id", "team_id"])[value_col]
        mask = ts["stats_type_id"] == stat_id

        n_overlaid = 0
        existing = ts[mask]
        if not existing.empty:
            idx = pd.MultiIndex.from_arrays(
                [existing["fixture_id"], existing["team_id"]],
                names=["fixture_id", "team_id"],
            )
            aligned = val_map.reindex(idx)
            overlay_mask = aligned.notna().values
            if overlay_mask.any():
                ts.loc[existing.index[overlay_mask], "value"] = aligned.values[overlay_mask]
                n_overlaid = int(overlay_mask.sum())

        sm_team_keys = set(zip(ts[mask]["fixture_id"], ts[mask]["team_id"]))
        fpl_only = team_agg[
            ~team_agg.apply(lambda r: (r["fixture_id"], r["team_id"]) in sm_team_keys, axis=1)
        ]
        n_appended = 0
        if not fpl_only.empty:
            new_rows = pd.DataFrame({
                "fixture_id": fpl_only["fixture_id"].astype("int64"),
                "team_id": fpl_only["team_id"].astype("int64"),
                "stats_type_id": stat_id,
                "value": fpl_only[value_col].astype(float),
            })
            self.team_stats = pd.concat([ts, new_rows], ignore_index=True)
            n_appended = len(new_rows)

        return n_overlaid, n_appended

    # ── Reference tables (small; bulk-loaded each run for now) ────────────

    async def _load_reference_tables(self, conn) -> None:
        """Load all small reference tables. Mirrors fetch_all_data_service.py
        query shapes so column names match DataCache exactly."""
        self.comps = await self._sql_to_df(conn, "SELECT * FROM competitions")
        self.seasons = await self._sql_to_df(conn, "SELECT * FROM seasons")
        self.comp_teams = await self._sql_to_df(
            conn, "SELECT * FROM competition_season_teams"
        )
        self.teams = await self._sql_to_df(conn, "SELECT * FROM teams")
        self.standings = await self._sql_to_df(conn, "SELECT * FROM standings")
        self.stats_types = await self._sql_to_df(conn, "SELECT * FROM stats_types")

        self.transfermarkt_team_mappings = await self._sql_to_df(
            conn,
            """
            SELECT c.name AS league_name, ttm.*
            FROM transfermarkt_team_mappings ttm
            JOIN competitions c ON c.id = ttm.competition_id
            WHERE ttm.to_name IS NOT NULL
            """,
        )
        self.promoted_team_ratings = await self._sql_to_df(
            conn,
            """
            SELECT c.name AS league_name, ptr.*
            FROM promoted_team_ratings ptr
            JOIN competitions c ON c.id = ptr.competition_id
            """,
        )
        # Admin FPL player dials (xmins-methodology §12 Phase 5): standing
        # per-player overrides, one row per player, NULL column = use model,
        # non-NULL REPLACES the model value at FPL assembly. All fractions 0-1.
        self.fpl_player_dials = await self._sql_to_df(
            conn,
            "SELECT player_id, p_play, p60, p90, goal_share, assist_share, defcon_share FROM fpl_player_dials",
        )
        self.projection_config = await self._sql_to_df(
            conn,
            """
            SELECT c.name AS league_name,
                   ca.name AS league_above_name,
                   cb.name AS league_below_name,
                   cpc.*
            FROM competition_projection_config cpc
            JOIN competitions c ON c.id = cpc.competition_id
            LEFT JOIN competitions ca ON ca.id = cpc.league_above_id
            LEFT JOIN competitions cb ON cb.id = cpc.league_below_id
            """,
        )

        # team_ratings — column rename + Date conversion identical to
        # DataCache so projection services see no diff.
        self.team_ratings = await self._sql_to_df(
            conn,
            """
            SELECT c.name AS League,
                   t.name AS Team,
                   tr.date AS Date,
                   tr.attack AS Attack,
                   tr.defense AS Defense,
                   tr.overall AS Overall,
                   tr.attack_xg AS Attack_xG,
                   tr.defense_xg AS Defense_xG,
                   tr.overall_xg AS Overall_xG,
                   tr.movement AS Movement,
                   tr.inverse AS Inverse,
                   tr.team_id,
                   tr.competition_id,
                   tr.id AS row_id
            FROM team_ratings tr
            JOIN competitions c ON c.id = tr.competition_id
            JOIN teams t ON t.id = tr.team_id
            """,
        )
        if not self.team_ratings.empty:
            self.team_ratings["Date"] = pd.to_datetime(self.team_ratings["Date"]).dt.date
            # MySQL DECIMAL → Python decimal.Decimal via aiomysql; CSV mode
            # gets float for free. Coerce so arithmetic in get_ratings
            # (Attack/Defense weighting, Movement subtraction) and the
            # euro-comp cached-ratings path both work.
            for col in ("Attack", "Defense", "Overall", "Attack_xG", "Defense_xG", "Overall_xG", "Movement"):
                if col in self.team_ratings.columns:
                    self.team_ratings[col] = pd.to_numeric(
                        self.team_ratings[col], errors="coerce"
                    )

        # b365_odds — DataCache keeps it as an empty frame; nothing critical
        # reads it directly. Same here for parity.
        self.b365_odds = pd.DataFrame()

        # FPL player mappings — Sportmonks player_id → FPL element data,
        # source of truth for FPL Position (1=GK, 2=DEF, 3=MID, 4=FWD via
        # fpl_element_type). Used by the FPL projection block in
        # projection_service.py / projection_all_teams_service.py.
        # Replaces the legacy `PL Fantasy Players.xlsx` lookup which
        # joined by Player NAME (fragile) — this joins by player_id.
        # GATED ON CURRENT FPL MEMBERSHIP (George, 2026-08-03: "for FPL we
        # should only be projecting players in FPL").
        #
        # fpl_player_mappings accumulates: a player who left keeps his old
        # fpl_team_id forever, so the raw table had 68 players at Bournemouth
        # against FPL's actual 25. That fed the FPL frame 59 players a side,
        # putting a club's projected minutes at 2,709 in a 90-minute match
        # (the 990 rule, methodology §10 step 6) and letting ~34 players per
        # club who are not in the game compete for bonus.
        #
        # last_verified_at is NOT a usable filter — the importer stamps all
        # 951 rows every run, so everything looks current. The latest
        # fpl_player_snapshots date IS the current bootstrap: 555 players,
        # Bournemouth exactly 25, matching FPL.
        #
        # The date is resolved to the most recent snapshot with a plausible
        # row count rather than MAX() outright, so a half-written or missed
        # snapshot day cannot silently gate every player out of FPL.
        self.fpl_player_mappings = await self._sql_to_df(
            conn,
            """
            SELECT m.player_id, m.fpl_id, m.fpl_code, m.fpl_element_type,
                   m.fpl_first_name, m.fpl_second_name, m.fpl_web_name,
                   ftm.team_id AS fpl_club_team_id,
                   -- Penalty order driving the cascade in fpl_penalties.
                   --
                   -- FPL's own `penalties_order` is the default, read from the
                   -- SAME snapshot row that gates membership so it can never
                   -- be staler than the squad it belongs to. Joined rather
                   -- than sub-selected for that reason; (player_id,
                   -- snapshot_date) is unique so it cannot fan out rows.
                   --
                   -- An admin override replaces it ALL-OR-NOTHING per club: if
                   -- the club has any fpl_penalty_orders row, FPL's list for
                   -- that club is discarded outright, including for players
                   -- the admin did not list. Mixing the two would leave FPL's
                   -- rank 1 beside an admin rank 1 with no way to order them.
                   CASE WHEN ovr_club.team_id IS NOT NULL
                        THEN ovr.penalty_rank
                        ELSE s.penalties_order
                   END AS penalties_order,
                   CASE WHEN ovr_club.team_id IS NOT NULL
                        THEN ovr.weight
                        ELSE NULL
                   END AS penalty_weight
            FROM fpl_player_mappings m
            LEFT JOIN fpl_team_mappings ftm ON ftm.fpl_id = m.fpl_team_id
            JOIN fpl_player_snapshots s
              ON s.player_id = m.player_id
             AND s.snapshot_date = (
                    SELECT snapshot_date FROM fpl_player_snapshots
                    GROUP BY snapshot_date
                    HAVING COUNT(DISTINCT player_id) >= %s
                    ORDER BY snapshot_date DESC
                    LIMIT 1
                )
            LEFT JOIN fpl_penalty_orders ovr ON ovr.player_id = m.player_id
            -- Which clubs have an override at all. Keyed on the SPORTMONKS
            -- team id (ftm.team_id), the same id the override rows store, so
            -- a club is matched by identity rather than by FPL's own id.
            LEFT JOIN (
                SELECT DISTINCT team_id FROM fpl_penalty_orders
            ) ovr_club ON ovr_club.team_id = ftm.team_id
            """,
            (FPL_SNAPSHOT_MIN_PLAYERS,),
        )
        logger.info(
            "[loader] fpl_player_mappings gated to current FPL squad: %d players",
            len(self.fpl_player_mappings),
        )

    # Sportmonks stat_type ids for the CBIT components. Resolved by name at
    # runtime rather than hardcoded, but listed here for reference.
    _CBIT_COMPONENTS = ("Clearances", "Blocked Shots", "Interceptions", "Tackles")

    async def _derive_cbit_from_components(self, conn) -> None:
        """Build CBIT + roll up Ball Recovery to team level for EVERY league.

        Why this exists: the combined CBIT stat (999003) and team-level Ball
        Recovery are both produced by the FPL overlay, which is Premier League
        only. A PL run loads a promoted player's Championship history, but that
        history carries no CBIT and no team-level recoveries — so his share
        collapsed to ~0 and no Coventry, Hull or Ipswich player could appear in
        the DefCon rankings at all. George spotted their absence.

        Sportmonks has the parts everywhere. Championship coverage checked
        2026-08-04: derived team CBIT mean 57.5 (PL 54.6), derived team
        recoveries mean 47.5 (PL 46.2) — close enough that a promoted player's
        history transfers meaningfully. FPL and Sportmonks also agree on PL
        CBIT to 0.3%, so the two sources are interchangeable.

        Runs AFTER _overlay_fpl_stats so FPL keeps precedence on PL fixtures:
        only (player, fixture) pairs with no CBIT row get one derived here.

        Sportmonks stores no zero rows, so a component with no row means the
        player genuinely did none — absent is summed as 0, not as missing.
        """
        if self.player_stats is None or self.player_stats.empty:
            return
        if self.team_stats is None:
            return

        def _sid(name):
            m = self.stats_types[self.stats_types["name"] == name]
            return int(m["id"].iloc[0]) if not m.empty else None

        cbit_id = _sid("Clearances Blocks Interceptions Tackles (FPL)")
        self._cbit_stat_id = cbit_id
        rec_id = _sid("Ball Recovery")
        comp_ids = [i for i in (_sid(n) for n in self._CBIT_COMPONENTS) if i is not None]
        if cbit_id is None or not comp_ids:
            logger.warning("[CBIT derive] missing stats_types — skipped")
            return

        ps = self.player_stats
        ps["value"] = pd.to_numeric(ps["value"], errors="coerce")

        # ---- player-level CBIT where absent -------------------------------
        have = set(
            map(tuple, ps.loc[ps["stats_type_id"] == cbit_id, ["player_id", "fixture_id"]].values)
        )
        comp = ps[ps["stats_type_id"].isin(comp_ids)]
        derived = (
            comp.groupby(["player_id", "fixture_id", "team_id"], as_index=False)["value"]
            .sum()
        )
        if not derived.empty and have:
            derived = derived[
                ~derived.apply(lambda r: (r["player_id"], r["fixture_id"]) in have, axis=1)
            ]
        n_player = 0
        if not derived.empty:
            new_rows = pd.DataFrame({
                "player_id": derived["player_id"].astype("int64"),
                "fixture_id": derived["fixture_id"].astype("int64"),
                "team_id": derived["team_id"].astype("int64"),
                "stats_type_id": cbit_id,
                "value": derived["value"].astype(float),
            })
            self.player_stats = pd.concat([ps, new_rows], ignore_index=True)
            ps = self.player_stats
            n_player = len(new_rows)

        # ---- team-level totals, read from the DB ------------------------
        # NOT summed from self.player_stats. That frame is scoped to
        # self.player_ids, so for a cross-club fixture we hold ONE player's
        # rows — summing those as a "team total" made Abdulkadir Omur's own 7
        # CBIT stand in for his team's real 92 across 23 players, giving him a
        # ~100% share and a 99.89% DefCon that shipped to the site.
        #
        # The player-scope restriction is a VOLUME decision (the loader exists
        # to stop pulling the whole DB per run) and cross-club history is
        # deliberately loaded — see the module docstring. So the fix is not to
        # load more players, it is to ask the DB for the totals directly: one
        # pre-aggregated query, measured at 1.32s for 3,905 fixtures returning
        # 35,518 rows — 0.09% of a ~25 min run. George, 2026-08-04.
        n_team = 0
        want = [i for i in (cbit_id, rec_id) if i is not None]
        fixture_ids = sorted({int(x) for x in ps["fixture_id"].dropna().unique()})
        if want and fixture_ids:
            comp_sum = await self._team_totals_from_db(conn, fixture_ids, comp_ids, rec_id)
            for stat_id in want:
                sub = comp_sum.get(stat_id)
                if sub is None or sub.empty:
                    continue
                ts = self.team_stats
                # REPLACE any existing team row, do not merely fill the gaps.
                # The FPL overlay writes team CBIT / recoveries by summing the
                # FPL rows, which cover FPL-MAPPED PLAYERS ONLY — so its team
                # total silently omits anyone unmapped. Measured 2026-08-05
                # over 760 PL team-matches: FPL sums 54.62 CBIT per match
                # against 55.59 from every player's Sportmonks components, so
                # the history the projection learns from ran 1.7% light and
                # every team CBIT projection inherited that.
                #
                # This is the same partial-squad trap that gave Abdulkadir Omur
                # a 99.89% DefCon; it was fixed for the player side and left on
                # the team side. _team_totals_from_db aggregates over ALL
                # players in SQL, so it is strictly the better number wherever
                # the two disagree. George, 2026-08-05.
                keys = set(map(tuple, sub[["fixture_id", "team_id"]].astype("int64").values))
                if not ts.empty:
                    _same_stat = ts["stats_type_id"] == stat_id
                    _pairs = list(zip(ts["fixture_id"], ts["team_id"]))
                    _drop = _same_stat & pd.Series(
                        [(int(f), int(t)) in keys if pd.notna(f) and pd.notna(t) else False
                         for f, t in _pairs],
                        index=ts.index,
                    )
                    if _drop.any():
                        ts = ts[~_drop]
                self.team_stats = pd.concat([ts, pd.DataFrame({
                    "fixture_id": sub["fixture_id"].astype("int64"),
                    "team_id": sub["team_id"].astype("int64"),
                    "stats_type_id": stat_id,
                    "value": sub["value"].astype(float),
                })], ignore_index=True)
                n_team += len(sub)

        logger.info(
            "[CBIT derive] player rows added %d, team rows added %d (all leagues)",
            n_player, n_team,
        )

    async def _derive_non_penalty_goals(self, conn) -> None:
        """Build Non-Penalty Goals (999004) = Goals - Penalties Scored.

        FPL goal shares come from `Goals`, which INCLUDES penalties, so a
        designated taker's share is inflated by spot-kicks and keeps paying him
        for them even if he loses the duty. Splitting the projection into
        penalty and non-penalty halves only works if the share feeding the
        non-penalty half excludes them too — otherwise takers are counted twice.

        Both inputs are already imported and reconcile: across the PL 25/26 we
        hold 92 penalties taken against the 92 actually awarded (George
        confirmed), 77 scored and 15 missed.

        Player rows: emitted for every player with a Goals row. A player with
        goals and no penalty row keeps his full total — Sportmonks stores no
        zero rows, so absent means none taken, not missing.

        Team rows: summed from the DB over ALL players, not from
        self.player_stats, for the same reason _team_totals_from_db exists —
        that frame is scoped to the run's players, so summing it would understate
        the team and inflate every share. That mistake put Abdulkadir Omur on a
        99.89% DefCon.
        """
        if self.player_stats is None or self.player_stats.empty:
            return
        if self.team_stats is None:
            return

        def _sid(name):
            m = self.stats_types[self.stats_types["name"] == name]
            return int(m["id"].iloc[0]) if not m.empty else None

        npg_id = _sid("Non-Penalty Goals")
        goals_id = _sid("Goals")
        pens_id = _sid("Penalties Scored")
        if npg_id is None or goals_id is None:
            logger.warning("[NPG derive] missing stats_types — skipped")
            return

        ps = self.player_stats
        ps["value"] = pd.to_numeric(ps["value"], errors="coerce")

        goals = ps[ps["stats_type_id"] == goals_id][
            ["player_id", "fixture_id", "team_id", "value"]
        ].rename(columns={"value": "goals"})
        if goals.empty:
            return
        if pens_id is not None:
            pens = ps[ps["stats_type_id"] == pens_id][
                ["player_id", "fixture_id", "value"]
            ].rename(columns={"value": "pens"})
            merged = goals.merge(pens, on=["player_id", "fixture_id"], how="left")
        else:
            merged = goals.assign(pens=0.0)
        merged["pens"] = merged["pens"].fillna(0.0)
        merged["npg"] = (merged["goals"] - merged["pens"]).clip(lower=0)

        have = set(
            map(tuple, ps.loc[ps["stats_type_id"] == npg_id, ["player_id", "fixture_id"]].values)
        )
        if have:
            merged = merged[
                ~merged.apply(lambda r: (r["player_id"], r["fixture_id"]) in have, axis=1)
            ]

        n_player = 0
        if not merged.empty:
            self.player_stats = pd.concat([ps, pd.DataFrame({
                "player_id": merged["player_id"].astype("int64"),
                "fixture_id": merged["fixture_id"].astype("int64"),
                "team_id": merged["team_id"].astype("int64"),
                "stats_type_id": npg_id,
                "value": merged["npg"].astype(float),
            })], ignore_index=True)
            ps = self.player_stats
            n_player = len(merged)

        # ---- team-level totals, from the DB over ALL players ----
        n_team = 0
        fixture_ids = sorted({int(x) for x in ps["fixture_id"].dropna().unique()})
        if fixture_ids:
            totals = await self._npg_team_totals_from_db(conn, fixture_ids, goals_id, pens_id)
            # Team-level rows for BOTH synthetic quantities. Penalties Scored
            # needs a team total of its own because that is the DENOMINATOR of
            # the player's penalty share — team_stats holds no such row
            # otherwise, and the share would collapse to zero for everyone.
            for stat_id, col in ((npg_id, "value"), (pens_id, "pens")):
                if stat_id is None or totals is None or totals.empty or col not in totals.columns:
                    continue
                ts = self.team_stats
                keys = set(map(tuple, totals[["fixture_id", "team_id"]].astype("int64").values))
                if not ts.empty:
                    _same = ts["stats_type_id"] == stat_id
                    _drop = _same & pd.Series(
                        [(int(f), int(t)) in keys if pd.notna(f) and pd.notna(t) else False
                         for f, t in zip(ts["fixture_id"], ts["team_id"])],
                        index=ts.index,
                    )
                    if _drop.any():
                        ts = ts[~_drop]
                self.team_stats = pd.concat([ts, pd.DataFrame({
                    "fixture_id": totals["fixture_id"].astype("int64"),
                    "team_id": totals["team_id"].astype("int64"),
                    "stats_type_id": stat_id,
                    "value": totals[col].astype(float),
                })], ignore_index=True)
                n_team += len(totals)

        logger.info(
            "[NPG derive] player rows added %d, team rows added %d", n_player, n_team
        )

    async def _derive_non_penalty_xg(self, conn) -> None:
        """Build Non-Penalty Expected Goals (999005) = xG - 0.79 x pens taken.

        Why this exists
        ---------------
        `Goals` blends 50/50 with xG at assembly; `Non-Penalty Goals` had no
        expected-goals partner, so its share came from raw NPG alone. A player
        with in-window xG but no in-window goals therefore projected a positive
        Goals share and a ZERO non-penalty share, and his goals disappeared
        from the projection: 77 players, 1,276 rows, 3.1% of all projected PL
        goals on run 3758. Milner is the purest case — his only goal in the
        sample WAS a penalty, so his NPG history is honestly zero.

        Sportmonks ships no npxG (stat 7943 exists but holds zero rows), so it
        is derived. Their xG DOES include penalties and prices one at exactly
        0.79 — measured, not assumed: of 859 (player, fixture) rows with one
        penalty and exactly one shot, so the xG is the penalty alone, 98.7%
        read 0.790. 0.79 rather than the 0.78 general figure precisely because
        we subtract from Sportmonks' own number; at 0.78 a penalty-only
        appearance leaves 0.01 behind, pure noise in a share denominator.

        Penalties TAKEN, not scored: a missed penalty still carries its xG.

        Team rows come from the DB over ALL players, not self.player_stats,
        for the same reason _npg_team_totals_from_db does — that frame is
        scoped to the run's players, so summing it would understate the team
        and inflate every share.
        """
        if self.player_stats is None or self.player_stats.empty:
            return
        if self.team_stats is None:
            return

        def _sid(name):
            m = self.stats_types[self.stats_types["name"] == name]
            return int(m["id"].iloc[0]) if not m.empty else None

        npxg_id = _sid("Non-Penalty Expected Goals")
        xg_id = _sid("Expected Goals (xG)")
        scored_id = _sid("Penalties Scored")
        missed_id = _sid("Penalties Missed")
        if npxg_id is None or xg_id is None:
            logger.warning("[npxG derive] missing stats_types — skipped")
            return

        ps = self.player_stats
        ps["value"] = pd.to_numeric(ps["value"], errors="coerce")

        xg = ps[ps["stats_type_id"] == xg_id][
            ["player_id", "fixture_id", "team_id", "value"]
        ].rename(columns={"value": "xg"})
        if xg.empty:
            return

        pen_ids = [i for i in (scored_id, missed_id) if i is not None]
        if pen_ids:
            pens = (ps[ps["stats_type_id"].isin(pen_ids)]
                    .groupby(["player_id", "fixture_id"], as_index=False)["value"].sum()
                    .rename(columns={"value": "taken"}))
            merged = xg.merge(pens, on=["player_id", "fixture_id"], how="left")
        else:
            merged = xg.assign(taken=0.0)
        merged["taken"] = merged["taken"].fillna(0.0)
        merged["npxg"] = (
            merged["xg"] - PENALTY_XG * merged["taken"]
        ).clip(lower=0)

        have = set(map(tuple, ps.loc[
            ps["stats_type_id"] == npxg_id, ["player_id", "fixture_id"]
        ].values))
        if have:
            merged = merged[~merged.apply(
                lambda r: (r["player_id"], r["fixture_id"]) in have, axis=1
            )]

        n_player = 0
        if not merged.empty:
            self.player_stats = pd.concat([ps, pd.DataFrame({
                "player_id": merged["player_id"].astype("int64"),
                "fixture_id": merged["fixture_id"].astype("int64"),
                "team_id": merged["team_id"].astype("int64"),
                "stats_type_id": npxg_id,
                "value": merged["npxg"].astype(float),
            })], ignore_index=True)
            ps = self.player_stats
            n_player = len(merged)

        # ---- team totals, from the DB over ALL players ----
        n_team = 0
        fixture_ids = sorted({int(x) for x in ps["fixture_id"].dropna().unique()})
        if fixture_ids:
            totals = await self._npxg_team_totals_from_db(
                conn, fixture_ids, xg_id, pen_ids
            )
            if totals is not None and not totals.empty:
                ts = self.team_stats
                keys = set(map(tuple, totals[["fixture_id", "team_id"]].astype("int64").values))
                if not ts.empty:
                    _drop = (ts["stats_type_id"] == npxg_id) & pd.Series(
                        [(int(f), int(t)) in keys if pd.notna(f) and pd.notna(t) else False
                         for f, t in zip(ts["fixture_id"], ts["team_id"])],
                        index=ts.index,
                    )
                    if _drop.any():
                        ts = ts[~_drop]
                self.team_stats = pd.concat([ts, pd.DataFrame({
                    "fixture_id": totals["fixture_id"].astype("int64"),
                    "team_id": totals["team_id"].astype("int64"),
                    "stats_type_id": npxg_id,
                    "value": totals["value"].astype(float),
                })], ignore_index=True)
                n_team = len(totals)

        logger.info(
            "[npxG derive] player rows added %d, team rows added %d", n_player, n_team
        )

    async def _npxg_team_totals_from_db(self, conn, fixture_ids, xg_id, pen_ids):
        """Team Non-Penalty xG per (fixture, team) = TEAM xG row - 0.79 x pens.

        The base comes from fixture_TEAM_stats, not from summing players — see
        _npg_team_totals_from_db for why that distinction matters. It barely
        moves xG (team row 1120.4 vs player sum 1124.4 over PL 25/26, -0.36%),
        but the two totals are built identically so the pair cannot drift.

        Penalties still come from the player table: every penalty is credited
        to a player, so there is no own-goal analogue to miss.
        """
        import pandas as pd
        team_rows, pen_rows = [], []
        for i in range(0, len(fixture_ids), 1500):
            chunk = fixture_ids[i:i + 1500]
            fph = ",".join(["%s"] * len(chunk))
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""SELECT fixture_id, team_id, SUM(value)
                          FROM fixture_team_stats
                         WHERE stats_type_id = %s AND fixture_id IN ({fph})
                         GROUP BY fixture_id, team_id""",
                    (xg_id,) + tuple(chunk),
                )
                team_rows.extend(await cur.fetchall())
                if pen_ids:
                    sph = ",".join(["%s"] * len(pen_ids))
                    await cur.execute(
                        f"""SELECT fixture_id, team_id, SUM(value)
                              FROM fixture_player_stats
                             WHERE stats_type_id IN ({sph}) AND fixture_id IN ({fph})
                             GROUP BY fixture_id, team_id""",
                        tuple(pen_ids) + tuple(chunk),
                    )
                    pen_rows.extend(await cur.fetchall())
        if not team_rows:
            return None
        x = pd.DataFrame(team_rows, columns=["fixture_id", "team_id", "value"])
        x["value"] = pd.to_numeric(x["value"], errors="coerce").fillna(0.0)
        if not pen_rows:
            return x
        p = pd.DataFrame(pen_rows, columns=["fixture_id", "team_id", "taken"])
        p["taken"] = pd.to_numeric(p["taken"], errors="coerce").fillna(0.0)
        out = x.merge(p, on=["fixture_id", "team_id"], how="left")
        out["taken"] = out["taken"].fillna(0.0)
        out["value"] = (out["value"] - PENALTY_XG * out["taken"]).clip(lower=0)
        return out[["fixture_id", "team_id", "value"]]

    async def _npg_team_totals_from_db(self, conn, fixture_ids, goals_id, pens_id):
        """Team Non-Penalty Goals per (fixture, team) = TEAM Goals row - pens.

        The goals base MUST come from fixture_TEAM_stats, not from summing
        players. OWN GOALS count for the team but belong to no player of it, so
        the player sum is short: PL 25/26 reads 1045 team goals against 1011
        player goals, a 3.25% gap.

        That matters because the ordinary `Goals` share divides by the TEAM row.
        Players' goal shares therefore sum to ~96.75% and own goals are
        allocated to nobody, which is correct. Building the NPG denominator
        from the player sum instead made NPG shares sum to ~100%, and since
        both are multiplied by a projected team total that DOES include own
        goals, every player's non-penalty goals came out ~3.4% high. George
        caught it by asking whether the NPG share really is the goal share with
        penalties removed from both ends — it now is.

        Penalties still come from the player table: every penalty is credited
        to a player, so there is no own-goal analogue on that side.

        A team-fixture with no team Goals row (nobody scored — Sportmonks
        stores no zero) simply produces no row, exactly as the player-sum
        version did, and the share code drops those fixtures from the window.
        """
        import pandas as pd
        team_rows, pen_rows = [], []
        for i in range(0, len(fixture_ids), 1500):
            chunk = fixture_ids[i:i + 1500]
            fph = ",".join(["%s"] * len(chunk))
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""SELECT fixture_id, team_id, SUM(value)
                          FROM fixture_team_stats
                         WHERE stats_type_id = %s AND fixture_id IN ({fph})
                         GROUP BY fixture_id, team_id""",
                    (goals_id,) + tuple(chunk),
                )
                team_rows.extend(await cur.fetchall())
                if pens_id is not None:
                    await cur.execute(
                        f"""SELECT fixture_id, team_id, SUM(value)
                              FROM fixture_player_stats
                             WHERE stats_type_id = %s AND fixture_id IN ({fph})
                             GROUP BY fixture_id, team_id""",
                        (pens_id,) + tuple(chunk),
                    )
                    pen_rows.extend(await cur.fetchall())
        if not team_rows:
            return None
        g = pd.DataFrame(team_rows, columns=["fixture_id", "team_id", "value"])
        g["value"] = pd.to_numeric(g["value"], errors="coerce").fillna(0.0)
        if pens_id is None:
            return g
        if pen_rows:
            p = pd.DataFrame(pen_rows, columns=["fixture_id", "team_id", "pens"])
            p["pens"] = pd.to_numeric(p["pens"], errors="coerce").fillna(0.0)
            out = g.merge(p, on=["fixture_id", "team_id"], how="left")
        else:
            out = g.assign(pens=0.0)
        out["pens"] = out["pens"].fillna(0.0)
        out["value"] = (out["value"] - out["pens"]).clip(lower=0)
        # `pens` retained: the caller writes team-level Penalties Scored from it.
        return out[["fixture_id", "team_id", "value", "pens"]]

    async def _team_totals_from_db(self, conn, fixture_ids, comp_ids, rec_id):
        """{stat_id: DataFrame[fixture_id, team_id, value]} — true team totals.

        CBIT is summed from its four Sportmonks components; Ball Recovery is
        taken as-is. Aggregated in SQL over ALL players in each fixture, which
        is the whole point: self.player_stats holds only the run's players.

        Chunked at 1,500 fixtures. Measured 1.32s for 3,905 fixtures / 35,518
        rows against a ~25 min run.
        """
        import pandas as pd
        want = list(comp_ids) + ([rec_id] if rec_id is not None else [])
        rows = []
        for i in range(0, len(fixture_ids), 1500):
            chunk = fixture_ids[i:i + 1500]
            fph = ",".join(["%s"] * len(chunk))
            sph = ",".join(["%s"] * len(want))
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""SELECT fixture_id, team_id, stats_type_id, SUM(value)
                          FROM fixture_player_stats
                         WHERE stats_type_id IN ({sph}) AND fixture_id IN ({fph})
                         GROUP BY fixture_id, team_id, stats_type_id""",
                    tuple(want) + tuple(chunk),
                )
                rows.extend(await cur.fetchall())
        if not rows:
            return {}
        df = pd.DataFrame(rows, columns=["fixture_id", "team_id", "stats_type_id", "value"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
        out = {}
        cbit = (
            df[df["stats_type_id"].isin(comp_ids)]
            .groupby(["fixture_id", "team_id"], as_index=False)["value"].sum()
        )
        out[self._cbit_stat_id] = cbit
        if rec_id is not None:
            out[rec_id] = (
                df[df["stats_type_id"] == rec_id][["fixture_id", "team_id", "value"]]
                .reset_index(drop=True)
            )
        return out

    def _load_local_files(self) -> None:
        """League Weightings.xlsx is the only non-DB reference. Loaded if
        path supplied; otherwise empty frame (CSV-mode parity)."""
        if self.league_weightings_xlsx_path and os.path.exists(self.league_weightings_xlsx_path):
            self.league_weightings = pd.read_excel(self.league_weightings_xlsx_path)
        else:
            self.league_weightings = pd.DataFrame()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    async def _sql_to_df(conn, sql: str, params: tuple = ()) -> pd.DataFrame:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)


# ── Shadow-mode capture (Phase 3) ────────────────────────────────────────

SHADOW_OUTPUT_DIR = Path("/tmp/loader_shadow")


async def capture_shadow_snapshot(
    league_name: str,
    league_id: int,
    *,
    extra_league_ids: Optional[Sequence[int]] = None,
    league_weightings_xlsx_path: Optional[str] = None,
) -> Optional[Path]:
    """Run LeagueDataLoader for one league and dump its DataFrames to parquet.

    Phase 4's diff tool will compare these against CSV-mode equivalents.
    Failures swallowed and logged — this must NEVER break the surrounding
    projection (it's purely observational while we validate parity).

    Returns the output dir path on success, None on failure.
    """
    try:
        SHADOW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        league_slug = league_name.replace(" ", "-").replace(".", "").lower()
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = SHADOW_OUTPUT_DIR / f"{league_slug}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        loader = LeagueDataLoader(
            league_id,
            extra_league_ids=extra_league_ids,
            league_weightings_xlsx_path=league_weightings_xlsx_path,
        )
        await loader.load()

        # Dump scope diagnostics + the actual ID lists so Phase 4's diff
        # tool can apply the exact same filter to CSV-mode DataFrames.
        scope_meta = pd.DataFrame([{
            "league_name": league_name,
            "league_id": league_id,
            "n_team_ids": len(loader.team_ids),
            "n_player_ids": len(loader.player_ids),
            "n_team_fixture_ids": len(loader.team_fixture_ids),
            "n_player_fixture_ids": len(loader.player_fixture_ids),
            "n_fixture_ids_union": len(loader.fixture_ids),
            "captured_at": ts,
        }])
        scope_meta.to_parquet(out_dir / "_scope.parquet", index=False)

        pd.DataFrame({"team_id": loader.team_ids}).to_parquet(
            out_dir / "_team_ids.parquet", index=False)
        pd.DataFrame({"player_id": loader.player_ids}).to_parquet(
            out_dir / "_player_ids.parquet", index=False)
        pd.DataFrame({"fixture_id": loader.team_fixture_ids}).to_parquet(
            out_dir / "_team_fixture_ids.parquet", index=False)
        pd.DataFrame({"fixture_id": loader.player_fixture_ids}).to_parquet(
            out_dir / "_player_fixture_ids.parquet", index=False)
        pd.DataFrame({"fixture_id": loader.fixture_ids}).to_parquet(
            out_dir / "_fixture_ids.parquet", index=False)

        # The 3 scoped tables — these are what Phase 4 diffs against
        # equivalent CSV slices to prove parity.
        for attr in ("fixtures_df", "team_stats", "player_stats"):
            df = getattr(loader, attr)
            if df is not None and not df.empty:
                df.to_parquet(out_dir / f"{attr}.parquet", index=False)

        logger.info(
            "[%s] shadow snapshot captured at %s "
            "(teams=%d players=%d fixtures=%d team_stats=%d player_stats=%d)",
            league_name, out_dir,
            len(loader.team_ids), len(loader.player_ids),
            len(loader.fixture_ids),
            0 if loader.team_stats is None else len(loader.team_stats),
            0 if loader.player_stats is None else len(loader.player_stats),
        )
        return out_dir
    except Exception as e:
        logger.warning(
            "[%s] shadow snapshot FAILED (non-fatal): %s",
            league_name, e, exc_info=True,
        )
        return None
