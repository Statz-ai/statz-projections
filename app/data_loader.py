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

        if xg_id is None and xa_id is None and tackles_id is None and recoveries_id is None and cbi_id is None:
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
        ]

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
            "SELECT player_id, p_play, p60, p90, goal_share, assist_share, defcon_pct FROM fpl_player_dials",
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
        self.fpl_player_mappings = await self._sql_to_df(
            conn,
            """
            SELECT m.player_id, m.fpl_id, m.fpl_code, m.fpl_element_type,
                   m.fpl_first_name, m.fpl_second_name, m.fpl_web_name,
                   ftm.team_id AS fpl_club_team_id
            FROM fpl_player_mappings m
            LEFT JOIN fpl_team_mappings ftm ON ftm.fpl_id = m.fpl_team_id
            """,
        )

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
