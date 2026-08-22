import asyncio
import logging
import time
from scipy.stats import poisson
import warnings
from dataclasses import dataclass

from app.repository.fixtures_repo import insert_fixtures_async
from app.repository.team_repo import insert_teams_async
from app.repository.player_stat_repo import insert_players_stats_async
from app.repository.player_repo import insert_player_async, get_players_from_league
from app.data_loader import LeagueDataLoader

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from .statz_functions import *
from pathlib import Path
import os

logger = logging.getLogger("projection")


class EuroCompProjectionService:
    CURRENT_DIR = Path(__file__).resolve().parent
    APP_DIR = CURRENT_DIR.parent

    DATA_FOLDER_PATH = APP_DIR / "data"
    MODEL_FILE_PATH = APP_DIR / "model-builds"
    SAVE_FILE_PATH = APP_DIR / "projection-outputs"
    DAYS = 5

    EURO_COMPS = ['Champions League', 'Europa League', 'Conference League', 'Europa Conference League']

    # ── Multi-league scopes ───────────────────────────────────────────────
    #
    # This service exists for competitions whose teams come from SEVERAL
    # domestic leagues, so ratings built inside each league have to be put on
    # one scale before they can meet. That is true of the European comps and
    # equally true of the domestic cups, which pair clubs from four tiers.
    #
    # Registered as scopes rather than copied into a second service: a fourth
    # parallel implementation of the fixture pipeline is exactly what was
    # deleted on 2026-08-20 (see docs/projection-services-design-scope.md).
    #
    #   leagues              — domestic leagues supplying teams. Each MUST have
    #                          team_ratings rows and a competition_projection_config
    #                          row, or the scale step has nothing to anchor on.
    #   goal_avg_league_ids  — which leagues set the Poisson baseline. Euro comps
    #                          use the top 5 only: including smaller leagues
    #                          pulled PSG/Arsenal projections toward Nordic
    #                          scoring rates (2026-05-27).
    #   coefficient          — how ratings are made comparable across leagues.
    #                          'uefa' reads competitions.uefa_coefficient_index.
    #                          'flat' applies 1.0, i.e. no cross-league scaling
    #                          at all — only valid while same_league_only is on.
    #   same_league_only     — project only ties where both clubs are in the SAME
    #                          domestic league. Those need no cross-league bridge:
    #                          the two ratings are already mutually consistent, so
    #                          the tie is arithmetically a league game.
    #   days                 — how far ahead to look. Cup rounds are weeks apart,
    #                          so the 5-day domestic window would never see them.
    @dataclass(frozen=True)
    class MultiLeagueScope:
        comps: tuple
        leagues: tuple
        goal_avg_league_ids: tuple
        coefficient: str = 'uefa'
        same_league_only: bool = False
        days: int = 5
        # --- domestic-cup tier model (agreed with George 2026-08-21) ---
        # tier_ladder: relative difficulty per competition_id, chained from the
        #   promotion/relegation steps George maintains — 1.25 (L1->L2),
        #   1.35 (Ch->L1), 1.60 (PL->Ch). Only ratios matter; the pin is
        #   arbitrary.
        # tier_compression: exponent applied to the ratio, keyed by how many
        #   divisions apart. 1 division is untouched (1.00) so the steps stay
        #   exactly as tuned; wider gaps are pulled in, because a single match
        #   is 11 v 11 and the favourite rests players — the gap can only be so
        #   big on the day. Set D: 1.00 / 0.85 / 0.75.
        # guardrail_max: ceiling on the odds blend for this scope.
        tier_ladder: tuple = ()
        tier_compression: tuple = ()
        guardrail_max: float = None
        # stat_ladder: the same idea as tier_ladder but for TEAM STATS, and
        # deliberately gentler — chained from the DAMPED promotion weights
        # (1.24 / 1.14 / 1.10, what get_team_weighted_average actually applies)
        # rather than the raw config values behind tier_ladder.
        #
        # Shots should separate less than goals: a division gap buys both more
        # shots AND better conversion, and goals collect both while shots only
        # collect the volume half. Two-division example — goals ratio 1.924,
        # stats ratio 1.342.
        #
        # There is no market safety net here. Every cup tie is priced for
        # GOALS, so the guardrail corrects us there, but across all five books
        # not one Carabao tie carries a corners, cards or shots market
        # (measured 2026-08-22). Whatever this produces is what publishes.
        stat_ladder: tuple = ()

    # NOTE the cup scope is deliberately incomplete: with coefficient='flat'
    # and same_league_only=True it projects 6 of the 23 upcoming Carabao ties
    # (measured 2026-08-21). The other 17 are cross-tier and need a real
    # coefficient — that is the next step, and it is isolated to one field.
    SCOPES = {
        'euro': MultiLeagueScope(
            comps=('Champions League', 'Europa League', 'Conference League',
                   'Europa Conference League'),
            leagues=('Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1',
                     'Eredivisie', 'Liga Portugal', 'Scottish Premiership',
                     'Austrian Bundesliga', 'Belgian Pro League', 'Eliteserien',
                     'Super League', 'Super Lig', 'Superliga', 'Allsvenskan'),
            goal_avg_league_ids=(8, 564, 384, 82, 301),
            coefficient='uefa',
        ),
        'domestic_cup': MultiLeagueScope(
            comps=('Carabao Cup', 'FA Cup'),
            leagues=('Premier League', 'Championship', 'League One', 'League Two'),
            goal_avg_league_ids=(8, 9, 12, 14),
            coefficient='flat',
            same_league_only=False,
            days=21,
            tier_ladder=((8, 2.700), (9, 1.6875), (12, 1.250), (14, 1.000)),
            tier_compression=((1, 1.00), (2, 0.85), (3, 0.75)),
            guardrail_max=0.75,
            stat_ladder=((8, 1.555), (9, 1.254), (12, 1.100), (14, 1.000)),
        ),
    }

    @staticmethod
    def scope_for(league: str):
        for scope in EuroCompProjectionService.SCOPES.values():
            if league in scope.comps:
                return scope
        return None

    # The former TOP_5_LEAGUE_IDS and LEAGUE_COUNTRY_DICT constants moved into
    # SCOPES above (2026-08-21) when domestic cups joined this service — two
    # scopes need two sets of values, and leaving the constants behind would
    # have been a second source of truth. Their reasoning is preserved in the
    # SCOPES comments; LEAGUE_COUNTRY_DICT's country values were never read.

    @staticmethod
    def _read_df(path_no_ext: str) -> pd.DataFrame:
        parquet_path = f"{path_no_ext}.parquet"
        excel_path = f"{path_no_ext}.xlsx"
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        elif os.path.exists(excel_path):
            return pd.read_excel(excel_path)
        raise FileNotFoundError(f"No data file found at {parquet_path} or {excel_path}")

    @staticmethod
    def is_euro_comp(league: str) -> bool:
        return league in EuroCompProjectionService.EURO_COMPS

    @staticmethod
    def handles(league: str) -> bool:
        """True for any competition this service projects — euro comps and
        the domestic cups. Routing should ask this, not is_euro_comp."""
        return EuroCompProjectionService.scope_for(league) is not None

    @staticmethod
    async def _resolve_upcoming_fixture_teams(comp_id: int, date_from, date_to):
        """Return distinct home+away team_ids for upcoming euro-comp
        fixtures in the projection window.

        Returns None if zero upcoming fixtures — callers should fall
        back to the full comp-derived scope so the loader has something
        sensible to load (the projection then skips cleanly via the
        empty-next_fix guard downstream).
        """
        from app.source_database import get_source_connection, release_source_connection
        conn = await get_source_connection()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT DISTINCT home_team_id FROM fixtures
                     WHERE competition_id = %s
                       AND kickoff_datetime >= %s AND kickoff_datetime <= %s
                       AND home_team_id IS NOT NULL
                    UNION
                    SELECT DISTINCT away_team_id FROM fixtures
                     WHERE competition_id = %s
                       AND kickoff_datetime >= %s AND kickoff_datetime <= %s
                       AND away_team_id IS NOT NULL
                    """,
                    (comp_id, date_from, date_to, comp_id, date_from, date_to),
                )
                rows = await cur.fetchall()
        finally:
            release_source_connection(conn)
        ids = sorted({int(r[0]) for r in rows})
        return ids if ids else None

    async def projections(self, league_request):
        league = league_request.league or 'Champions League'
        _start_time = time.time()
        # Pin this run's history cutoff before any history is computed. Also
        # stops a cutoff pinned by a PREVIOUS run in the same process from
        # being reused — every entry point sets its own.
        _pinned = set_run_cutoff()
        scope = EuroCompProjectionService.scope_for(league)
        if scope is None:
            raise ValueError(f"{league!r} is not a registered multi-league scope")
        logger.info(f'[{league}] START multi-league projections (history cutoff pinned {_pinned})')

        data_folder_path = EuroCompProjectionService.DATA_FOLDER_PATH
        model_file_path = EuroCompProjectionService.MODEL_FILE_PATH
        save_file_path = EuroCompProjectionService.SAVE_FILE_PATH

        date_from = pd.to_datetime('today')
        date_to = date_from + pd.DateOffset(days=scope.days)
        odds_weight = 0.5

        # Euro comp scope spans the comp itself + 8 domestic top-tiers
        # (scope.leagues). Resolve IDs up-front via direct DB queries,
        # then pass them to LeagueDataLoader so the team scope covers all
        # relevant clubs.
        from app.services.projection_service import ProjectionService
        comp_id_for_load = await ProjectionService._resolve_league_id_db(league)
        domestic_ids = []
        for dom_league in scope.leagues:
            domestic_ids.append(await ProjectionService._resolve_league_id_db(dom_league))
        league_weightings_path = os.path.join(data_folder_path, "League Weightings.xlsx")

        # Loader scope narrowing (2026-05-27): we only need history for
        # the teams playing in upcoming fixtures, not every team across
        # all 8 domestic top tiers. For a final, that's 2 teams instead
        # of 248. Loader time goes from ~7min → <30s. team_ratings is
        # still loaded in full (it's a reference table, not scoped) so
        # cross-league rating computation downstream still works.
        #
        # If 0 upcoming fixtures, restrict_team_ids stays None and the
        # loader falls back to its full comp-derived scope — safer than
        # loading nothing, and the projection skips cleanly downstream
        # via the `len(next_fix) == 0` guard.
        restrict_team_ids = await self._resolve_upcoming_fixture_teams(
            comp_id_for_load, date_from, date_to
        )

        _loader = LeagueDataLoader(
            comp_id_for_load,
            extra_league_ids=domestic_ids,
            league_weightings_xlsx_path=league_weightings_path,
            restrict_team_ids=restrict_team_ids,
        )
        await _loader.load()
        source = _loader
        logger.info(
            f"[{league}] Data source: LeagueDataLoader "
            f"({'narrow scope, ' + str(len(restrict_team_ids)) + ' teams' if restrict_team_ids else 'full scope, +8 domestic comps'})"
        )
        ProjectionService._current_source = source
        # Loader is per-call so mutation safety isn't a concern. _maybe_copy
        # kept as a no-op shim so call sites don't churn.
        def _maybe_copy(df):
            return df

        player_stats = _maybe_copy(source.player_stats)
        team_stats = _maybe_copy(source.team_stats)
        standings = _maybe_copy(source.standings)
        seasons = source.seasons
        comps = source.comps
        comp_teams = source.comp_teams
        teams = source.teams
        fixtures_df = _maybe_copy(source.fixtures_df)
        stats_types = source.stats_types

        # Players from LeagueDataLoader (DB-direct, scoped to comp + 8 domestic
        # comps). display_name already stripped upstream.
        players = source.players
        logger.info(f"[{league}] Loaded {len(players)} players from DB-loader")

        # Ratings Dataset — DB-sourced (cache or per-league loader).
        all_team_ratings = _maybe_copy(source.team_ratings)

        # League Weightings (for domestic rating calculations) — loader
        # populates this from competition_projection_config; xlsx fallback
        # only fires if the DB table is unexpectedly empty.
        _lw_xlsx = os.path.join(data_folder_path, "League Weightings.xlsx")
        league_weightings_df = source.league_weightings if (source.league_weightings is not None and not source.league_weightings.empty) else (pd.read_excel(_lw_xlsx) if os.path.exists(_lw_xlsx) else pd.DataFrame())

        # UEFA Coefficients — DB-sourced from competitions.uefa_coefficient_index
        # (backfilled 2026-04-22). Built into the same shape the legacy xlsx
        # had (League / Coefficient Index columns) so downstream lookups work
        # unchanged. xlsx is no longer read.
        # Cast to float: aiomysql reads MySQL DECIMAL → decimal.Decimal,
        # which doesn't divide with float (downstream `team_projections /
        # (diff + 1)` raises TypeError). xlsx path got float for free.
        uefa_coef = comps[['name', 'uefa_coefficient_index']].rename(
            columns={'name': 'League', 'uefa_coefficient_index': 'Coefficient Index'}
        ).dropna(subset=['Coefficient Index']).reset_index(drop=True)
        uefa_coef['Coefficient Index'] = pd.to_numeric(uefa_coef['Coefficient Index'], errors='coerce')

        comp_id = get_league_id(league, comps)
        league_ids = [get_league_id(l, comps) for l in scope.leagues]

        # Shadow capture deliberately SKIPPED for euro comps. Scope spans
        # the comp + 8 domestic top tiers → 10k+ players, 8M+ player_stat
        # rows in the loader DataFrame. Stacked on top of the already-loaded
        # 4GB DataCache it OOM-kills the gunicorn worker mid-run. Domestic
        # leagues still capture (smaller scope, no OOM risk). Euro-comp
        # parity is covered by the standalone test_on_mode.py CL smoke
        # test from Phase 5d.

        fixtures = fixtures_df[fixtures_df['competition_id'] == comp_id]
        current_season_id = get_season_id(comp_id, seasons, False)
        if current_season_id is None:
            # Typical for euro comps right after the final — Sportmonks
            # hasn't created the next season yet. Skip cleanly.
            raise RuntimeError(
                f"no current season in seasons table for competition_id={comp_id} — skipping"
            )
        stat_list = get_stat_list(comp_id)

        logger.info(f'[{league}] Building cross-league ratings...')

        # ── Build cross-league ratings using UEFA coefficients ──

        def rescale_to_range(series, new_min=0.5, new_max=2.0):
            old_min = series.min()
            old_max = series.max()
            return new_min + (series - old_min) * (new_max - new_min) / (old_max - old_min)

        ratings_df = pd.DataFrame()

        # CACHED-RATINGS PATH (2026-05-27): inner-league ratings are read
        # from the team_ratings DB table instead of recomputed per league.
        # The domestic projection cron writes fresh, post-MV, post-dial,
        # rescaled-to-mean-100 rows nightly — recomputing here was ~25s
        # per league × 15 leagues = ~6 min of wasted work every euro
        # comp run. Now we just pick the latest row per (competition_id,
        # team_id), apply the UEFA coefficient on top, and concat.
        #
        # Things that USED to happen in this loop and now don't, because
        # they're already baked into team_ratings:
        #   - get_ratings() weighted compute
        #   - promoted-team blend (handled by domestic projection)
        #   - market-value adjustment
        #   - team dials apply
        #   - per-league rescale-to-mean-100
        latest_ratings_by_id = {}
        if all_team_ratings is not None and not all_team_ratings.empty:
            # Pick latest row per (competition_id, team_id). Frame includes
            # all leagues so we filter as we iterate.
            sorted_tr = all_team_ratings.sort_values('Date', ascending=False)
            latest_ratings_by_id = sorted_tr.drop_duplicates(
                subset=['competition_id', 'team_id'], keep='first'
            )

        # Each club belongs to exactly ONE league here: the scope league it
        # holds its most recent rating for.
        #
        # Without this a promoted or relegated club appears twice. Cardiff City
        # carried a Championship rating (Attack 91.6, Overall -30.8) alongside
        # the League One rating it won promotion with (Attack 136.3, Overall
        # +46.6). make_round_goal_prediction looks a club up by NAME and takes
        # .values[0], and this frame is sorted by Overall, so Cardiff were
        # projected on their League One attack — 1.91 expected goals against a
        # market saying 1.5, in a Championship-vs-Championship cup tie. George
        # spotted it in the output: "seems like an incorrect projection".
        #
        # Barely bites the euro scope, whose 15 leagues are all top tiers in
        # different countries, so a club rarely has two. Across four tiers of
        # one pyramid it is routine — promotion and relegation guarantee it.
        _scope_ids_by_name = {get_league_id(l, comps): l for l in scope.leagues}
        _current_comp_by_team = {}
        if isinstance(latest_ratings_by_id, pd.DataFrame) and not latest_ratings_by_id.empty:
            _in_scope = latest_ratings_by_id[
                latest_ratings_by_id['competition_id'].isin(_scope_ids_by_name.keys())
            ].sort_values('Date', ascending=False).drop_duplicates(subset=['team_id'], keep='first')
            _current_comp_by_team = dict(zip(_in_scope['team_id'], _in_scope['competition_id']))

        for league_name in scope.leagues:
            league_id = get_league_id(league_name, comps)

            if isinstance(latest_ratings_by_id, pd.DataFrame):
                league_rows = latest_ratings_by_id[latest_ratings_by_id['competition_id'] == league_id]
                # Drop clubs whose newest scope rating belongs to another
                # league — they are carried here only by an older season.
                if not league_rows.empty and _current_comp_by_team:
                    _belongs = league_rows['team_id'].map(_current_comp_by_team) == league_id
                    _stale = int((~_belongs).sum())
                    if _stale:
                        logger.info(
                            f"[{league}] {league_name}: dropped {_stale} club(s) rated more "
                            f"recently in another scope league"
                        )
                    league_rows = league_rows[_belongs]
            else:
                league_rows = pd.DataFrame()
            if league_rows.empty:
                logger.warning(f"[{league}] {league_name}: no team_ratings rows in DB — skipping (run the domestic projection first to seed it)")
                continue

            ratings = league_rows[['Team', 'Attack', 'Defense', 'Overall',
                                   'Attack_xG', 'Defense_xG', 'Overall_xG']].copy()
            # Defensive — strip whitespace on team names so cross-league
            # joins downstream match cleanly (transfermarkt mapping uses
            # exact strings).
            ratings['Team'] = ratings['Team'].astype(str).str.strip()
            logger.info(f"[{league}] {league_name}: loaded {len(ratings)} teams from team_ratings cache")

            # Apply UEFA coefficient — same scaling applies to both the
            # indexed and the xG/game columns so euro-comp rankings stay
            # cross-league comparable.
            # Try DB first (competitions.uefa_coefficient_index, added
            # 2026-04-22 migration), fall back to League Coefficients.xlsx
            # for any league not yet backfilled in DB.
            if scope.coefficient == 'flat':
                # No cross-league scaling. Only sound because same_league_only
                # is on: within one league the ratings are already mutually
                # consistent, so a tie between two clubs from it is
                # arithmetically a league game. The moment cross-tier ties are
                # projected this MUST become a real coefficient — an EFL tier
                # gap is not 1.0 and pretending otherwise would hand League Two
                # sides Premier League strength.
                coef = 1.0
            else:
                comp_row = comps[comps['id'] == league_id]
                db_coef = comp_row['uefa_coefficient_index'].iloc[0] if (
                    not comp_row.empty and 'uefa_coefficient_index' in comps.columns
                    and pd.notna(comp_row['uefa_coefficient_index'].iloc[0])
                ) else None
                if db_coef is not None:
                    coef = float(db_coef)
                else:
                    xlsx_match = uefa_coef[uefa_coef['League'] == league_name]['Coefficient Index']
                    if xlsx_match.empty:
                        logger.warning(f"[{league}] No UEFA coefficient for {league_name} in DB or xlsx — defaulting to 1.0")
                        coef = 1.0
                    else:
                        coef = xlsx_match.values[0]
            ratings['League'] = league_name
            ratings['coef'] = coef
            ratings['Attack'] *= coef
            ratings['Defense'] /= coef
            ratings['Attack_xG'] *= coef
            ratings['Defense_xG'] /= coef
            ratings['Overall_xG'] = ratings['Attack_xG'] - ratings['Defense_xG']
            ratings_df = pd.concat([ratings_df, ratings], ignore_index=True)

        ratings_df['Overall'] = ratings_df['Attack'] - ratings_df['Defense']
        ratings_df.sort_values(by='Overall', ascending=False, inplace=True)
        ratings = ratings_df.copy()

        logger.info(f'[{league}] Ratings built for {len(ratings)} teams across {len(scope.leagues)} leagues')

        # Save UEFA-coefficient-adjusted ratings to the team_ratings DB table
        # under the euro comp's competition_id. This replaces the previous
        # no-op (euro_comp_service never wrote ratings → Champions League
        # and Europa League hadn't been updated since Mar 20).
        from app.repository.team_ratings_repo import insert_team_ratings_async
        await insert_team_ratings_async(
            ratings[['Team', 'Attack', 'Defense', 'Overall', 'Attack_xG', 'Defense_xG', 'Overall_xG']].copy(),
            league, comp_id, teams,
            comp_teams=comp_teams,
            # Ratings are written under the euro comp's id but cover teams
            # from all 8 domestic top tiers (Barcelona/Bayern/PSG aren't
            # in EL's competition_season_teams pool, so a comp_id-scoped
            # lookup misses them — ~11 fallback warnings/run pre-fix).
            lookup_competition_ids=league_ids + [comp_id],
        )

        # ── Fixture projections ──

        fixtures['kickoff_datetime'] = pd.to_datetime(fixtures['kickoff_datetime'])
        next_fix = fixtures[(fixtures['kickoff_datetime'] >= date_from) & (fixtures['kickoff_datetime'] <= date_to)]

        # Qualifying rounds are never projected — Statz only covers the actual
        # competition (decision 2026-07-03). The statz-side trigger already
        # filters these out of scheduled runs (ProjectionTriggerEvaluator);
        # this guard covers full-comp/manual runs with no fixture_ids, which
        # would otherwise pick up e.g. the August CL qualifying play-offs once
        # both teams are rated. Column-presence check keeps this robust on DBs
        # that pre-date the is_qualifying migration.
        if 'is_qualifying' in next_fix.columns:
            _pre_qual = len(next_fix)
            next_fix = next_fix[next_fix['is_qualifying'] != 1]
            if len(next_fix) < _pre_qual:
                logger.info(f'[{league}] Skipped {_pre_qual - len(next_fix)} qualifying-round fixture(s)')
        if hasattr(league_request, 'fixture_ids') and league_request.fixture_ids:
            next_fix = next_fix[next_fix['id'].isin(league_request.fixture_ids)]
            logger.info(f'[{league}] Filtered to {len(next_fix)} fixtures by IDs')
        # Carry `neutral_venue` through — read at projection time by
        # make_round_goal_prediction + get_team_round_predictions to
        # disable home-advantage bias for finals at neutral grounds.
        # Defaults to False if the source DF doesn't have the column
        # yet (legacy fixtures pre-migration 2026-05-27).
        _has_neutral = 'neutral_venue' in next_fix.columns
        _cols = ['id', 'kickoff_datetime', 'name', 'home_team_id', 'away_team_id',
                 'bet365_home_odds_decimal', 'bet365_draw_odds_decimal', 'bet365_away_odds_decimal']
        if _has_neutral:
            _cols.append('neutral_venue')
        next_fix = next_fix[_cols]
        if not _has_neutral:
            next_fix['neutral_venue'] = False

        # Qualifying-round guard: a fixture can carry a NULL team_id (TBD
        # placeholder for a prior-round winner — e.g. "TBD v BATE" in the
        # ECL first qualifying round) or, in principle, a team missing from
        # the teams table. get_team() IndexErrors on either BEFORE the
        # placeholder/ratings guards below get a chance to skip the fixture,
        # killing the whole comp run (2026-07-03: the first 2026/27 CL + ECL
        # qualifying fixtures entered the window and crashed every nightly
        # run). Skip-and-warn, same idiom as the "not in ratings" guard.
        known_team_ids = set(teams['id'].values)
        _pre_known = len(next_fix)
        next_fix = next_fix[
            next_fix['home_team_id'].isin(known_team_ids)
            & next_fix['away_team_id'].isin(known_team_ids)
        ]
        if len(next_fix) < _pre_known:
            logger.warning(
                f'[{league}] Skipped {_pre_known - len(next_fix)} fixture(s) with NULL/unknown '
                f'team ids (TBD qualifying slots or teams absent from teams table)'
            )

        next_fix['home_team'] = next_fix['home_team_id'].apply(lambda x: get_team(x, teams))
        next_fix['away_team'] = next_fix['away_team_id'].apply(lambda x: get_team(x, teams))
        next_fix = next_fix.drop(columns=['home_team_id', 'away_team_id'])
        next_fix = drop_placeholder_fixtures(next_fix, league)
        next_fix.sort_values(by=['kickoff_datetime', 'home_team'], inplace=True)
        next_fix.reset_index(drop=True, inplace=True)

        # Domestic cups, step 1: only ties where both clubs are in the SAME
        # domestic league. Those need no cross-league bridge — the two ratings
        # were built inside one league and are already mutually consistent, so
        # the tie is arithmetically a league game. Cross-tier ties are dropped
        # rather than projected wrong; they need a real tier coefficient, which
        # is the next step and is isolated to scope.coefficient.
        #
        # Placed after home_team/away_team are resolved from ids — the ratings
        # frame is keyed on team NAME.
        if scope.tier_ladder:
            # Stamp each club's division so the baseline and the tier ratio can
            # both be taken per fixture below.
            #
            # `ratings` holds one row per club — its CURRENT league — so the
            # frame is safe to map from. Do NOT rebuild this with a plain
            # dict(zip(...)) on an unsorted frame: `ratings` is sorted by
            # Overall, and a club with rows in two divisions would then resolve
            # last-wins by RATING rather than by league.
            _team_league = dict(zip(
                ratings['Team'].astype(str).str.strip(), ratings['League']
            ))
            _pre = len(next_fix)
            _hl = next_fix['home_team'].astype(str).str.strip().map(_team_league)
            _al = next_fix['away_team'].astype(str).str.strip().map(_team_league)
            _keep = _hl.notna() & _al.notna()
            next_fix = next_fix[_keep].copy()
            next_fix['_home_league'] = _hl[_keep]
            next_fix['_away_league'] = _al[_keep]
            if _pre - len(next_fix):
                # Unrated clubs are dropped, never guessed at. This is what
                # keeps the FA Cup honest — its qualifying rounds are non-league
                # sides we hold nothing for (1 of 273 rated, 2026-08-21).
                logger.info(f'[{league}] dropped {_pre - len(next_fix)} tie(s) with an unrated club')
            # Pair-major ordering so the per-pair score_preds concat below stays
            # aligned with next_fix, which downstream code indexes positionally.
            next_fix = next_fix.sort_values(
                by=['_home_league', '_away_league', 'kickoff_datetime', 'home_team']
            ).reset_index(drop=True)

        # Drop fixtures where teams don't have ratings
        drop_indices = []
        for i in range(len(next_fix)):
            home_team = next_fix['home_team'][i]
            away_team = next_fix['away_team'][i]
            if home_team not in ratings['Team'].values or away_team not in ratings['Team'].values:
                logger.warning(f'[{league}] Skipping fixture: {home_team} vs {away_team} — team not in ratings')
                drop_indices.append(i)
        next_fix = next_fix.drop(drop_indices).reset_index(drop=True)

        if len(next_fix) == 0:
            logger.info(f"[{league}] No fixtures to project"); logger.info(f"[{league}] DONE euro comp projections (nothing to do)")
            return

        logger.info(f'[{league}] Projecting {len(next_fix)} fixtures...')

        # Goal averages from the top-5 leagues only (PL, La Liga, Serie A,
        # Bundesliga, Ligue 1). All scope.leagues entries get
        # ratings, but smaller leagues' goal rates don't reflect realistic
        # euro-comp scoring — using them in the Poisson baseline pulled
        # PSG/Arsenal/etc.'s projections toward Nordic / Austrian averages.
        # NaN-filter keeps the math safe for between-season leagues whose
        # team_stats might return None.
        _goal_avg_pool = [lid for lid in scope.goal_avg_league_ids if lid in league_ids]
        avg_home_goals_list = [get_home_goal_avg(lid, team_stats, fixtures_df, stats_types, standings, seasons) for lid in _goal_avg_pool]
        avg_away_goals_list = [get_away_goal_avg(lid, team_stats, fixtures_df, stats_types, standings, seasons) for lid in _goal_avg_pool]
        avg_home_goals_list = [v for v in avg_home_goals_list if v is not None and not np.isnan(v)]
        avg_away_goals_list = [v for v in avg_away_goals_list if v is not None and not np.isnan(v)]
        avg_home_goals = np.mean(avg_home_goals_list) if avg_home_goals_list else 1.5
        avg_away_goals = np.mean(avg_away_goals_list) if avg_away_goals_list else 1.2
        logger.info(f"[{league}] Goal averages: avg_home={avg_home_goals:.3f} avg_away={avg_away_goals:.3f} (from {len(avg_home_goals_list)} top-5 leagues)")

        if scope.tier_ladder and '_home_league' in next_fix.columns and not next_fix.empty:
            # Per-fixture baseline and tier ratio, grouped by the pair of
            # divisions involved so both are constant within a group.
            #
            # BASELINE — the merge George settled on: average the two clubs'
            # own league averages. It matters that this is per-pair: measured
            # over 3 years the Premier League runs 2.99 goals a game against
            # 2.53-2.60 for the three EFL divisions, so pooling all four would
            # push a League Two tie up by roughly 0.4 goals.
            #
            # TIER RATIO — R = ladder[home] / ladder[away], then raised to an
            # exponent set by how many divisions apart the clubs are. Folded
            # into the baseline rather than applied to the ratings, so no club
            # is ever converted into another division's terms: converting
            # produced a 242 defence for Plymouth and made the answer depend on
            # which direction you converted in.
            _ladder = dict(scope.tier_ladder)
            _compress = dict(scope.tier_compression)
            _rank = {cid: i for i, cid in enumerate(
                sorted(_ladder, key=lambda c: -_ladder[c]))}
            _parts = []
            for (_hl, _al), _grp in next_fix.groupby(['_home_league', '_away_league'], sort=True):
                _hid, _aid = get_league_id(_hl, comps), get_league_id(_al, comps)
                _bh = get_home_goal_avg(_hid, team_stats, fixtures_df, stats_types, standings, seasons)
                _ba = get_away_goal_avg(_hid, team_stats, fixtures_df, stats_types, standings, seasons)
                _bh2 = get_home_goal_avg(_aid, team_stats, fixtures_df, stats_types, standings, seasons)
                _ba2 = get_away_goal_avg(_aid, team_stats, fixtures_df, stats_types, standings, seasons)
                _vals = [v for v in (_bh, _bh2) if v is not None and not np.isnan(v)]
                _home_base = float(np.mean(_vals)) if _vals else avg_home_goals
                _vals = [v for v in (_ba, _ba2) if v is not None and not np.isnan(v)]
                _away_base = float(np.mean(_vals)) if _vals else avg_away_goals

                _R = 1.0
                if _hid in _ladder and _aid in _ladder:
                    _steps = abs(_rank[_hid] - _rank[_aid])
                    _R = (_ladder[_hid] / _ladder[_aid]) ** _compress.get(_steps, 1.0)
                logger.info(
                    f"[{league}] {_hl} v {_al}: {len(_grp)} tie(s), "
                    f"baseline {_home_base:.3f}/{_away_base:.3f}, tier ratio {_R:.3f}"
                )
                _parts.append(make_round_goal_prediction(
                    _grp, ratings, _home_base * _R, _away_base / _R))
            score_preds = pd.concat(_parts, ignore_index=True)
        else:
            score_preds = make_round_goal_prediction(next_fix, ratings, avg_home_goals, avg_away_goals)

        boost = 1.1
        score_preds['Home Odds %'] = ((1 / next_fix['bet365_home_odds_decimal']) * 100)
        score_preds['Draw Odds %'] = ((1 / next_fix['bet365_draw_odds_decimal']) * 100)
        score_preds['Away Odds %'] = ((1 / next_fix['bet365_away_odds_decimal']) * 100)

        # Pre-load bet365 goals over/under for the fixtures we're about
        # to project. The blend cascade (paths 1-3) consumes per-team
        # and match-total ladders directly; path 4 (legacy 1X2-only) is
        # the fall-through when those markets aren't priced.
        from app.services.odds_blend import (
            load_goals_odds_for_fixtures,
            compute_final_goals_and_probs,
        )
        from app.source_database import get_source_connection, release_source_connection
        _conn = await get_source_connection()
        try:
            goals_odds_map = await load_goals_odds_for_fixtures(
                _conn, next_fix['id'].tolist(),
            )
        finally:
            release_source_connection(_conn)

        home_win = []
        draw = []
        away_win = []
        home_clean = []
        away_clean = []
        over_1 = []
        over_2 = []
        btts = []

        for i in range(len(score_preds)):
            bookie_margin = 1 + (score_preds.loc[i, 'Home Odds %'] + score_preds.loc[i, 'Draw Odds %'] + score_preds.loc[i, 'Away Odds %'] - 100) / 100
            score_preds.loc[i, 'Home Odds %'] = (score_preds.loc[i, 'Home Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Draw Odds %'] = (score_preds.loc[i, 'Draw Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Away Odds %'] = (score_preds.loc[i, 'Away Odds %'] / bookie_margin).round(2)

            home_goals = score_preds['Home Goals'][i]
            away_goals = score_preds['Away Goals'][i]

            # Bookie 1X2 as fractions (margin-stripped above), or None if
            # the row had no 1X2 priced at all.
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
                    odds_weight,
                    boost,
                    # Domestic cups take the per-side distance-weighted
                    # guardrail rather than a flat weight, because our cup
                    # errors are concentrated rather than uniform: 9 of 17
                    # cross-tier ties landed within 0.25 goals of the book
                    # while Chelsea v Luton was out by 1.2. A flat weight
                    # would drag the nine we had right in order to help the
                    # one we didn't. Euro comps keep the flat weight.
                    guardrail=bool(scope.guardrail_max),
                    guardrail_max=scope.guardrail_max,
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

        score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'], inplace=True)
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

        logger.info(f'[{league}] Fixture projections complete')

        # Save fixture projections
        await insert_fixtures_async(score_preds, teams=teams, competition_id=comp_id, comp_teams=comp_teams)

        # ── Team projections ──

        logger.info(f'[{league}] Building team projections...')

        # Load pre-trained models (no retraining)
        models = load_all_models(stat_list, str(model_file_path))

        team_projections = get_team_round_predictions(
            next_fix, stat_list, fixtures_df, team_stats, teams, stats_types, models,
            ratings=ratings, comp_id=league_ids, games=50,
            # comp_teams must cover all 8 domestic leagues (league_ids), not
            # just the euro comp itself — downstream get_team_id scopes via
            # .isin(league_ids) and would miss every PSG/Bayern/Atletico/etc
            # if comp_teams was filtered to comp_id alone (~880 fallback
            # warnings per nightly run pre-fix).
            comp_teams=comp_teams[comp_teams['competition_id'].isin(league_ids + [comp_id])],
        )

        team_projections.drop(
            columns=['Team ' + stat + ' History' for stat in stat_list] +
                    ['Opponent ' + stat + ' History Against' for stat in stat_list],
            inplace=True
        )

        # Adjust team stats for the opposition's division.
        _stat_ladder = {cid: v for cid, v in scope.stat_ladder}
        _stat_comp = dict(scope.tier_compression)
        _stat_rank = {cid: i for i, cid in enumerate(
            sorted(_stat_ladder, key=lambda c: -_stat_ladder[c]))} if _stat_ladder else {}
        _league_id_cache = {}
        for i in range(len(team_projections)):
            team = team_projections['Team'].iloc[i]
            team_league = ratings.loc[ratings['Team'] == team, 'League'].values
            if len(team_league) == 0:
                continue
            team_league = team_league[0]
            # Read the coefficient the ratings were actually built with rather
            # than re-deriving it from uefa_coef: EFL tiers have no UEFA index,
            # so that lookup returns an empty frame and .values[0] raises.
            # Identical for euro comps — ratings['coef'] was set from the same
            # source in the loop above. Under a flat scope both sides are 1.0,
            # so diff is 0 and the adjustment below is a no-op.
            league_rating = ratings.loc[ratings['Team'] == team, 'coef'].values[0]

            opponent = team_projections['Opponent'].iloc[i]
            opp_league = ratings.loc[ratings['Team'] == opponent, 'League'].values
            if len(opp_league) == 0:
                continue
            opp_league = opp_league[0]
            opp_league_rating = ratings.loc[ratings['Team'] == opponent, 'coef'].values[0]
            if _stat_ladder:
                for _nm in (team_league, opp_league):
                    if _nm not in _league_id_cache:
                        _league_id_cache[_nm] = get_league_id(_nm, comps)
                _t_id, _o_id = _league_id_cache[team_league], _league_id_cache[opp_league]

            if _stat_ladder:
                # Domestic cups: multiply by a tier RATIO, exactly as the
                # scoreline does, rather than the euro form below.
                #
                # The euro form divides by (opponent_coef - team_coef + 1),
                # which is only well-behaved while the two coefficients sit
                # within 1.0 of each other. Our ladder spans 1.00 to 1.555, so
                # it would survive — but the GOALS ladder spans 1.00 to 2.700,
                # and a Premier League side against League Two would give a
                # divisor of -0.70 and negative shot projections. A ratio is
                # well-behaved at any spread, so use it here and don't leave
                # that landmine for whoever widens the ladder next.
                _n = abs(_stat_rank.get(_t_id, 0) - _stat_rank.get(_o_id, 0))
                _sr = (_stat_ladder[_t_id] / _stat_ladder[_o_id]) ** _stat_comp.get(_n, 1.0) \
                    if (_t_id in _stat_ladder and _o_id in _stat_ladder) else 1.0
                for _col in ('Shots Total', 'Shots On Target', 'Corners',
                             'Passes', 'Successful Passes', 'Total Crosses'):
                    team_projections.at[team_projections.index[i], _col] = round(
                        team_projections[_col].iloc[i] * _sr, 2)
                # Fouls, cards, tackles, interceptions and offsides are left
                # alone deliberately. They should move INVERSELY with the tier
                # gap — the weaker side does the chasing — but there is no
                # measured basis for the size, and no market on any cup tie to
                # catch a wrong sign. Getting the direction wrong on tackles
                # and cards would surface directly in player props.
                continue
            diff = (opp_league_rating - league_rating)
            team_projections.at[team_projections.index[i], 'Shots Total'] = (team_projections['Shots Total'].iloc[i] / (diff + 1)).round(2)
            team_projections.at[team_projections.index[i], 'Shots On Target'] = (team_projections['Shots On Target'].iloc[i] / (diff + 1)).round(2)
            team_projections.at[team_projections.index[i], 'Corners'] = (team_projections['Corners'].iloc[i] / (diff + 1)).round(2)
            team_projections.at[team_projections.index[i], 'Passes'] = (team_projections['Passes'].iloc[i] / (diff + 1)).round(2)
            team_projections.at[team_projections.index[i], 'Successful Passes'] = (team_projections['Successful Passes'].iloc[i] / (diff + 1)).round(2)
            team_projections.at[team_projections.index[i], 'Total Crosses'] = (team_projections['Total Crosses'].iloc[i] / (diff + 1)).round(2)

        # Bake goals into shots projections
        avg_goals = (avg_home_goals + avg_away_goals) / 2

        league_team_stats = team_stats[team_stats['fixture_id'].isin(fixtures_df[fixtures_df['competition_id'] == comp_id]['id'])]

        league_shots = league_team_stats[league_team_stats['stats_type_id'] == get_stat_id('Shots Total', stats_types)].copy()
        league_shots['Date'] = league_shots['fixture_id'].map(fixtures_df.set_index('id')['kickoff_datetime'])
        league_shots['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(league_shots['Date'])).dt.days // 7
        league_shots['Weight'] = 0.9 ** (league_shots['Weeks Since Kickoff'] - 5)
        league_shots.loc[league_shots['Weeks Since Kickoff'] < 6, 'Weight'] = 1
        league_shots['Weighted Shots'] = league_shots['Weight'] * league_shots['value']
        avg_shots = league_shots['Weighted Shots'].sum() / league_shots['Weight'].sum()

        league_shots_on_target = league_team_stats[league_team_stats['stats_type_id'] == get_stat_id('Shots On Target', stats_types)].copy()
        league_shots_on_target['Date'] = league_shots_on_target['fixture_id'].map(fixtures_df.set_index('id')['kickoff_datetime'])
        league_shots_on_target['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(league_shots_on_target['Date'])).dt.days // 7
        league_shots_on_target['Weight'] = 0.9 ** (league_shots_on_target['Weeks Since Kickoff'] - 5)
        league_shots_on_target.loc[league_shots_on_target['Weeks Since Kickoff'] < 6, 'Weight'] = 1
        league_shots_on_target['Weighted Shots On Target'] = league_shots_on_target['Weight'] * league_shots_on_target['value']
        avg_shots_on_target = league_shots_on_target['Weighted Shots On Target'].sum() / league_shots_on_target['Weight'].sum()

        avg_shots_per_goal = avg_shots / avg_goals
        avg_shots_on_target_per_goal = avg_shots_on_target / avg_goals

        goals = []
        assists = []
        for i in range(len(team_projections)):
            team = team_projections['Team'].iloc[i]
            try:
                team_pred = score_preds[score_preds['Home Team'] == team]['Home Goals'].values[0]
            except:
                team_pred = score_preds[score_preds['Away Team'] == team]['Away Goals'].values[0]
            goals.append(team_pred)
            assists.append((team_pred * 0.82).round(2))
            projected_shots = team_projections['Shots Total'].iloc[i]
            projected_shots_on_target = team_projections['Shots On Target'].iloc[i]

            adjusted_shots, adjusted_shots_on_target = adjust_shots_projection(
                team_pred, projected_shots, projected_shots_on_target,
                avg_shots_per_goal, avg_shots_on_target_per_goal
            )
            team_projections.at[team_projections.index[i], 'Shots Total'] = adjusted_shots
            team_projections.at[team_projections.index[i], 'Shots On Target'] = adjusted_shots_on_target

        team_projections['Goals'] = goals
        team_projections['Assists'] = assists

        saves = []
        for i in range(len(team_projections)):
            opp = team_projections['Opponent'].iloc[i]
            try:
                opp_pred = score_preds[score_preds['Home Team'] == opp]['Home Goals'].values[0]
            except:
                opp_pred = score_preds[score_preds['Away Team'] == opp]['Away Goals'].values[0]
            saves.append(team_projections[team_projections['Team'] == opp]['Shots On Target'].values[0] - opp_pred)

        team_projections['Saves'] = saves
        team_projections['Key Passes'] = (team_projections['Shots Total'] * 0.75).round(2)
        team_projections = team_projections[
            ['fixture_id', 'kickoff_datetime', 'Team', 'Opponent', 'Venue', 'Goals', 'Assists', 'Key Passes'] +
            stat_list + ['Fouls Drawn', 'Saves']
        ]
        team_projections.rename(columns={'Successful Passes': 'Accurate Passes'}, inplace=True)

        # ── Team-stat odds-blend ──
        # Reels each team's projected stat toward bookie expected via
        # the cascade (Path 1 per-team ladder → Path 1.5 partial+match
        # → Path 2 match-split via model ratio → fall through). Per
        # stat, books tried in TEAM_STAT_BOOKIE_PRIORITY order; first
        # to return usable λs wins. Six stats covered in v1: corners,
        # cards, shots, SoT, fouls, tackles.
        from app.services.odds_blend import (
            load_team_stat_odds, blend_team_stat,
            TEAM_STAT_BOOKIE_PRIORITY, STAT_COLUMN_TO_MARKET,
        )
        _fix_ids = team_projections['fixture_id'].astype(int).unique().tolist()
        _odds_per_market = {}
        _conn = await get_source_connection()
        try:
            for _market, _books in TEAM_STAT_BOOKIE_PRIORITY.items():
                _odds_per_market[_market] = await load_team_stat_odds(
                    _conn, _fix_ids, _market, _books,
                )
        finally:
            release_source_connection(_conn)

        # Pre-build fixture → home_team_name map so we can identify
        # which team_projections row is home regardless of Venue tag
        # (neutral-venue finals tag both rows 'N', so we can't rely on
        # Venue='H'/'A' alone).
        _team_id_to_name = dict(zip(teams['id'], teams['name'])) if teams is not None else {}
        _fid_to_home_team = {}
        for _fid in _fix_ids:
            _row = next_fix[next_fix['id'] == _fid]
            if _row.empty:
                continue
            _fid_to_home_team[_fid] = _row['home_team'].iloc[0]

        # Apply per-fixture for each stat. One pass per fixture updates
        # both home+away rows for every stat column.
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
                    market, odds_weight,
                )
                team_projections.loc[home_mask, stat_col] = round(fh, 2)
                team_projections.loc[away_mask, stat_col] = round(fa, 2)

        # Save team projections
        team_projections_save = team_projections.copy()
        team_projections_save.drop(['Assists', 'Fouls Drawn', 'Saves', 'Key Passes'], axis=1, inplace=True)
        team_projections_save = team_projections_save.round(2)
        await insert_teams_async(team_projections_save, teams=teams, competition_id=comp_id, comp_teams=comp_teams)

        logger.info(f'[{league}] Team projections complete')

        # ── Player projections ──

        logger.info(f'[{league}] Building player projections...')

        # Pre-load confirmed XI + player-prop odds (Goals/Shots/SoT v1).
        # See the canonical comment in projection_service.py's projections()
        # site. Blend α = odds_weight (0.5 for euro comps).
        from app.services.odds_blend import (
            load_confirmed_lineups, load_player_odds,
            PLAYER_BLEND_BOOKS, PLAYER_BLEND_STAT_IDS,
        )
        _pl_fix_ids = next_fix['id'].astype(int).unique().tolist()
        _conn = await get_source_connection()
        try:
            _confirmed_lineups = await load_confirmed_lineups(_conn, _pl_fix_ids)
            _odds_for_fixture_players = await load_player_odds(
                _conn, _pl_fix_ids, PLAYER_BLEND_STAT_IDS, PLAYER_BLEND_BOOKS,
            )
        finally:
            release_source_connection(_conn)

        pl_projections = distribute_team_predictions_to_players(
            player_stats, team_stats, team_projections, stats_types, fixtures_df, players, teams, comps, 0.97,
            competition_id=comp_id, comp_teams=comp_teams,
            confirmed_lineups=_confirmed_lineups,
            odds_for_fixture_players=_odds_for_fixture_players,
            odds_blend_weight=odds_weight,
        )

        player_pos = []
        player_saves = []
        for player, team in pl_projections[['Player', 'Team']].values:
            pos = get_player_position(player, team, players, teams, comp_id, comp_teams)
            if pos == 'GK':
                player_saves.append(team_projections[team_projections['Team'] == team]['Saves'].values[0])
            else:
                player_saves.append(0)
            player_pos.append(pos)
        pl_projections['Position'] = player_pos
        pl_projections['Saves'] = player_saves

        pl_projections = pl_projections[
            ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue',
             'Goals', 'Assists', 'Shots Total', 'Shots On Target', 'Key Passes', 'Passes', 'Accurate Passes',
             'Interceptions', 'Tackles', 'Total Crosses', 'Yellowcards', 'Offsides', 'Fouls', 'Fouls Drawn', 'Saves']
        ]
        pl_projections.rename(columns={'Yellowcards': 'Yellow Cards'}, inplace=True)

        # Predict starters
        pred_starters = player_stats[player_stats['fixture_id'].isin(next_fix['id'])].copy()
        pred_starters = pred_starters[pred_starters['stats_type_id'] == 11]

        start = []
        for i in range(len(pl_projections)):
            team = pl_projections['Team'].iloc[i]
            player_name = pl_projections['Player'].iloc[i]
            try:
                player_id = get_player_id(player_name, players, team, teams, comp_id, comp_teams)
            except:
                start.append('No')
                continue
            team_starters = pred_starters[pred_starters['team_id'] == get_team_id(team, teams, comp_id, comp_teams)]
            if player_id in team_starters['player_id'].values:
                start.append('Yes')
            else:
                start.append('No')
        pl_projections['Start?'] = start

        pl_projections = pl_projections[
            ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue',
             'Start?', 'Goals', 'Assists', 'Shots Total', 'Shots On Target', 'Key Passes', 'Passes',
             'Accurate Passes', 'Interceptions', 'Tackles', 'Total Crosses', 'Yellow Cards', 'Offsides',
             'Fouls', 'Fouls Drawn', 'Saves']
        ]
        pl_projections = pl_projections.round(2)
        pl_projections.sort_values(by='Goals', ascending=False, inplace=True)
        pl_projections.reset_index(drop=True, inplace=True)

        # Save player projections
        await insert_player_async(pl_projections, teams=teams, competition_id=comp_id, comp_teams=comp_teams)

        logger.info(f'[{league}] Player projections complete')

        # ── Player stat props ──

        logger.info(f'[{league}] Building player stat props...')

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

        # Save player stat props
        await insert_players_stats_async(player_stat_probs, teams=teams, competition_id=comp_id, comp_teams=comp_teams)

        _elapsed = round(time.time() - _start_time, 1)
        logger.info(f'[{league}] DONE euro comp projections in {_elapsed}s')
