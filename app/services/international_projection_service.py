"""
World Cup 2026 fixture projections.

Self-contained — ratings AND projections are both refreshed every run,
mirroring the domestic projection pattern.

Score prediction is the **same code path as domestic** (calls the shared
get_result_probs + find_inputs_for_probs helpers in statz_functions).
The only differences are the ratings source (international_ratings rather
than the domestic get_ratings pipeline) and the per-fixture lambda
construction (single AVG_GOALS + host bonus, no league-specific weightings).

Pipeline:
1. Compute fresh Statz international ratings (TEB v5 FIFA opp + MV v8
   nudge + symmetric caps) for today's date, committing to team_ratings.
2. Cross-Poisson per fixture from those ratings: λ_h = (h.atk/100) ×
   (a.def/100) × AVG_GOALS, λ_a similarly. 1.10× host bonus for
   USA/Canada/Mexico at home.
3. Model 1X2 from get_result_probs(home_goals, away_goals, boost).
4. If bet365 1X2 odds exist: de-vig (divide by overround) + linear
   blend at ODDS_BETA → adjusted 1X2 → find_inputs_for_probs to re-solve
   λs that produce the adjusted 1X2.
5. Downstream markets (CS, O1.5, O2.5, BTTS) from the final Poisson grid.
6. Write to fixture_projections (idempotent — DELETE comp=732 + INSERT).

Skipped: fixtures with unknown team rating (knockout bracket placeholders
like "1st Group A vs 3rd Group C/E/F/H/I").

After the fixture markets: a Monte Carlo tournament simulation, then
per-team stat projections (InternationalTeamStatService → team_projections), then
per-player stat projections (WcPlayerStatService → player_projections,
scoped to nations with a confirmed tournament_squads entry).
"""
import logging
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
from scipy.stats import poisson

from app.services.international_ratings import compute_international_ratings
from app.services.statz_functions import get_result_probs, find_inputs_for_probs, set_run_cutoff
from app.services.tournament_configs import WC_2026
from app.services.tournament_simulation_service import TournamentSimulator
from app.services.international_team_stat_service import InternationalTeamStatService
from app.services.wc_player_stat_service import WcPlayerStatService
from app.services.wc_fantasy_points_service import WcFantasyPointsService
from app.source_database import get_source_connection, release_source_connection

logger = logging.getLogger("international_projection")

# team_ratings storage bucket for international football. All 211 national
# teams' ratings live under this single competition_id regardless of which
# intl comp is being projected — friendlies, qualifiers, WC all read from
# the same pool. Distinct from `scope.competition_id` which is the comp
# whose FIXTURES are being projected.
INTL_RATINGS_BUCKET_COMP_ID = 732
AVG_GOALS = 1.3

# Per-fixture neutral-venue path constants (friendlies and any future
# intl comp that opts in via scope.use_per_fixture_neutral_venue).
#
# DECAY_BASE / DECAY_CLAMP_WEEKS — mirror the get_home_goal_avg shape
#   (`base^(weeks - (clamp-1))`, clamped to 1 for weeks < clamp), tuned
#   for intl windows that only recur Mar/Jun/Sep/Oct/Nov:
#     - domestic 0.9^(w-5) → ~10% weight at 6mo; too aggressive for intl
#     - intl    0.98^(w-26) → ~80% at 6mo, ~60% at 1y, ~22% at 2y
#   The 27-week clamp covers ~5 most-recent FIFA windows at full weight.
#
# LOOKBACK_MONTHS — same 36mo window we used to validate the constants.
# GOALS_BLEND / XG_BLEND — match domestic's 30/70 Goals/xG mix exactly.
# Stat IDs — Goals=52, Expected Goals (xG)=5304 in stats_types.
INTL_GOAL_AVG_DECAY_BASE = 0.98
INTL_GOAL_AVG_DECAY_CLAMP_WEEKS = 27
INTL_GOAL_AVG_LOOKBACK_MONTHS = 36
INTL_GOAL_AVG_GOALS_WEIGHT = 0.3
INTL_GOAL_AVG_XG_WEIGHT = 0.7
GOALS_STAT_ID = 52
XG_STAT_ID = 5304

# odds_beta = bet365 goal-line blend weight (0 = pure model, 1 = pure bookie).
# Bumped to 0.5 alongside the goals odds-blend cascade rewrite 2026-05-29 —
# matches euro_comp_projection_service. boost=1.0 = no draw inflation
# (international games don't need the draw nudge that league football
# gets per-comp via projection_config).
ODDS_BETA = 0.5
BOOST = 1.0


@dataclass(frozen=True)
class IntlProjectionScope:
    """Per-comp configuration for the international projection pipeline.

    competition_id    — the comp whose fixtures are being projected
    competition_name  — human-readable name (matches competitions.name)
    hosts             — host nation names (only WC has these); empty for
                        non-tournament intl comps
    host_bonus/penalty — λ multipliers when a host plays at home / opp
                        plays in a host country
    bracket_config    — TournamentConfig for tournaments with a knockout
                        bracket (WC, Euros, Copa, AFCON). None for
                        friendlies / qualifiers / Nations League group
                        stage etc. — those skip Step 3.
    has_squad_source  — True when wc_squads/wc_players (or future Euros
                        equivalent) carries the named squad. False routes
                        the player-stat service through the recent-caps
                        provider in intl_squad_provider.py.
    fantasy_rules     — 'fifa_wc_2026' enables Step 6 (WC fantasy points).
                        None for non-WC comps; future Euros fantasy keys
                        would be added here.
    fanteam_rules     — 'fanteam_wc_2026' enables Step 6b (FanTeam WC
                        fantasy points). Sibling of fantasy_rules; FanTeam
                        and FIFA can both be on for the same comp.
    """
    competition_id: int
    competition_name: str
    hosts: frozenset = field(default_factory=frozenset)
    host_bonus: float = 1.0
    host_penalty: float = 1.0
    bracket_config: object = None
    has_squad_source: bool = False
    fantasy_rules: object = None
    fanteam_rules: object = None
    # When True, replace the symmetric AVG_GOALS=1.3 baseline with comp-
    # specific weighted avg_home/avg_away goals AND classify each fixture
    # per-venue (home_at_home / away_at_home / true_neutral / no_venue).
    # WC keeps host_bonus/host_penalty path; friendlies opt in.
    use_per_fixture_neutral_venue: bool = False
    # When True, skip any fixture that has no 1X2 result odds across our
    # 5 fixture-odds books (bet365 / Coral / Ladbrokes / Midnite /
    # Boylesports). For friendlies — bookmakers price ~3-5 days before
    # kickoff, so this gates projections behind "is this a real, live,
    # bookmaker-confirmed fixture". WC doesn't opt in (every WC fixture
    # gets priced; the gate would just add latency).
    require_result_odds: bool = False
    # When set, skip a fixture where BOTH teams' overall Statz rating
    # (attack − defense) is below this floor. Friendlies set it to 0.0 so
    # minnow-vs-minnow exhibitions (two below-average sides) aren't
    # projected — they clog the pages, draw little interest, and are the
    # hardest to predict (thin, volatile data). One below-average side vs a
    # decent one still projects (interest in the bigger team). Teams with no
    # Statz rating are already dropped by the carry-forward gate. None = off
    # (WC / quals / tournaments project every fixture regardless of strength).
    skip_both_below_rating: float = None
    # Blend weight toward bookmaker lines in the goals + team-stat odds
    # blends: final = (1 − β)·model + β·market. WC keeps 0.5 (trust model
    # and market equally). Friendlies bump to 0.7 — the model is noisier on
    # thin-data nations, so hug the book more and stop surfacing model error
    # as inflated "edges" (cards/corners/etc.). Applies to BOTH the 1X2/goals
    # blend and every team-stat blend for the scope.
    odds_beta: float = 0.5


# Registry of every comp the international projection pipeline knows about.
# Adding a comp = adding an entry here + flipping competitions.is_projected
# = 1 in the DB. Renamed comps trip the name lookup; keys must match
# competitions.name exactly.
INTL_SCOPES = {
    'World Cup': IntlProjectionScope(
        competition_id=732,
        competition_name='World Cup',
        hosts=frozenset({'United States', 'Mexico', 'Canada'}),
        host_bonus=1.10,
        host_penalty=0.90,
        bracket_config=WC_2026,
        has_squad_source=True,
        fantasy_rules='fifa_wc_2026',
        fanteam_rules='fanteam_wc_2026',
    ),
    'Friendly International': IntlProjectionScope(
        competition_id=1082,
        competition_name='Friendly International',
        hosts=frozenset(),       # no host nation
        host_bonus=1.0,          # → no λ adjustment via the hosts-list path
        host_penalty=1.0,
        bracket_config=None,     # no bracket → Step 3 skipped
        has_squad_source=False,  # no tournament_squads entry → RecentCapsSquadProvider
        fantasy_rules=None,      # no FIFA fantasy → Step 6 skipped
        skip_both_below_rating=0.0,  # skip minnow-vs-minnow (both overall < 0)
        odds_beta=0.7,           # hug the book harder — model noisier on friendlies
        # Friendlies are played across UEFA-vs-CONMEBOL tour matches,
        # one-off games at the away team's invite, true neutrals (Dubai
        # showcase fixtures), etc. — per-fixture venue classification is
        # the only way to get the home/away baseline right.
        use_per_fixture_neutral_venue=True,
        # Friendlies feed includes a lot of obscure exhibition / micro-
        # confederation fixtures that bookmakers don't price. Gate on
        # 1X2 odds presence so we only project games with real
        # bookmaker interest. Nightly full-comp re-runs pick up new
        # odds as books start pricing closer to kickoff.
        require_result_odds=True,
    ),
}


async def _compute_intl_comp_goal_avgs(conn, competition_id: int):
    """Weighted home/away goal averages for an international competition.

    Mirrors `get_home_goal_avg` / `get_away_goal_avg` (statz_functions.py)
    but with two intl-specific adjustments:

    1. **Slower decay**: 0.98^(weeks-26) clamped to 1 for weeks<27 (vs the
       domestic 0.9^(weeks-5) clamped<6). Intl windows recur 5x/year, so
       the slower fall-off keeps ~5 windows weighted equally.

    2. **xG-completeness filter**: only counts fixtures that have BOTH
       Goals AND xG for BOTH teams. Without this, the 30/70 Goals+xG
       blend mixes a wide-but-noisy Goals sample with a narrow xG sample
       (intl Sportmonks xG covers ~60% of fixtures; the missing 40% are
       weighted toward smaller-confederation games which skews the
       home-vs-away spread).

    Returns (avg_home, avg_away) or (None, None) if there's no usable
    history.
    """
    import numpy as np
    import pandas as pd

    async with conn.cursor() as cur:
        await cur.execute(
            f"""
            SELECT id, home_team_id, away_team_id, kickoff_datetime
            FROM fixtures
            WHERE competition_id = %s
              AND kickoff_datetime < NOW()
              AND kickoff_datetime > DATE_SUB(NOW(), INTERVAL {INTL_GOAL_AVG_LOOKBACK_MONTHS} MONTH)
            """,
            (competition_id,),
        )
        fx_rows = await cur.fetchall()
        if not fx_rows:
            return None, None
        fx = pd.DataFrame(fx_rows, columns=[d[0] for d in cur.description])
        fx['weeks'] = ((pd.Timestamp.now() - pd.to_datetime(fx['kickoff_datetime'])).dt.days // 7).astype(int)

        fid_ph = ",".join(["%s"] * len(fx))
        await cur.execute(
            f"""
            SELECT fixture_id, team_id, stats_type_id,
                   CAST(value AS DECIMAL(10,4)) AS value
            FROM fixture_team_stats
            WHERE fixture_id IN ({fid_ph})
              AND stats_type_id IN (%s, %s)
            """,
            tuple(int(x) for x in fx['id']) + (GOALS_STAT_ID, XG_STAT_ID),
        )
        ts_rows = await cur.fetchall()
    if not ts_rows:
        return None, None
    ts = pd.DataFrame(ts_rows, columns=['fixture_id', 'team_id', 'stats_type_id', 'value'])
    ts['value'] = ts['value'].astype(float)

    # Keep only fixtures where every cell of the 4-row block is present:
    # home_team Goals, away_team Goals, home_team xG, away_team xG.
    h_map = dict(zip(fx['id'], fx['home_team_id']))
    a_map = dict(zip(fx['id'], fx['away_team_id']))
    have = ts.groupby('fixture_id').apply(lambda g: set(zip(g['team_id'], g['stats_type_id']))).to_dict()
    complete = set()
    for fid in fx['id']:
        needed = {(h_map[fid], GOALS_STAT_ID), (a_map[fid], GOALS_STAT_ID),
                  (h_map[fid], XG_STAT_ID),   (a_map[fid], XG_STAT_ID)}
        if fid in have and needed.issubset(have[fid]):
            complete.add(fid)
    if not complete:
        return None, None

    df = fx[fx['id'].isin(complete)].merge(
        ts[ts['fixture_id'].isin(complete)],
        left_on='id', right_on='fixture_id', how='inner',
    )

    base = INTL_GOAL_AVG_DECAY_BASE
    clamp = INTL_GOAL_AVG_DECAY_CLAMP_WEEKS
    df['weight'] = np.where(df['weeks'] < clamp, 1.0, base ** (df['weeks'] - (clamp - 1)))

    def _weighted_avg(side: str, stat_id: int):
        sub = df[(df['stats_type_id'] == stat_id) & (df['team_id'] == df[f'{side}_team_id'])]
        if sub.empty or sub['weight'].sum() == 0:
            return None
        return float((sub['value'] * sub['weight']).sum() / sub['weight'].sum())

    h_goals = _weighted_avg('home', GOALS_STAT_ID)
    a_goals = _weighted_avg('away', GOALS_STAT_ID)
    h_xg = _weighted_avg('home', XG_STAT_ID)
    a_xg = _weighted_avg('away', XG_STAT_ID)
    if h_goals is None or a_goals is None or h_xg is None or a_xg is None:
        return None, None

    avg_home = h_goals * INTL_GOAL_AVG_GOALS_WEIGHT + h_xg * INTL_GOAL_AVG_XG_WEIGHT
    avg_away = a_goals * INTL_GOAL_AVG_GOALS_WEIGHT + a_xg * INTL_GOAL_AVG_XG_WEIGHT
    return float(avg_home), float(avg_away)


def _classify_fixture_venue(home_team_country_id, away_team_country_id,
                             venue_id, venue_country_id) -> str:
    """Bucket an intl fixture into 4 venue cases — drives the per-fixture
    λ baseline selection.

    - 'no_venue'      → fixture.venue_id is NULL. Caller skips projection.
    - 'home_at_home'  → venue is in the listed home team's country (most
                        common; treat as a true home game).
    - 'away_at_home'  → inversion: venue is in the listed AWAY team's
                        country. Sportmonks gets the home/away flip
                        wrong for some friendlies (e.g. Argentina v
                        Iceland @ Reykjavik). Caller swaps the
                        avg_home / avg_away baselines.
    - 'true_neutral'  → venue is in a third country. Tour matches at
                        showcase venues (Dubai, Miami, etc.). Caller
                        uses the midpoint baseline.
    - 'unknown'       → venue exists but its country couldn't be
                        derived (coastal/island edge cases the polygon
                        geocoder can't match). Caller skips the fixture
                        rather than guessing — fail-closed policy.
    """
    if venue_id is None:
        return 'no_venue'
    if venue_country_id is None:
        return 'unknown'
    if venue_country_id == home_team_country_id:
        return 'home_at_home'
    if venue_country_id == away_team_country_id:
        return 'away_at_home'
    return 'true_neutral'


class InternationalProjectionService:
    """Routes any international fixture (national-team comp) through the
    same pipeline. WC is one comp among several — friendlies, qualifiers,
    Euros etc. are added by registering scopes in INTL_SCOPES.

    Stateless — instance method only for parity with EuroCompProjectionService.
    """

    @staticmethod
    def is_international_comp(league: str) -> bool:
        return league in INTL_SCOPES

    async def _fetch_upcoming_team_ids(
        self,
        competition_id: int,
        fixture_ids_filter,
    ) -> list:
        """Return the distinct team_ids playing in upcoming fixtures of
        competition_id (optionally narrowed by fixture_ids_filter).
        Used to scope RecentCapsSquadProvider for non-WC intl comps."""
        fid_sql = ""
        fid_params: tuple = ()
        if fixture_ids_filter:
            ph = ",".join(["%s"] * len(fixture_ids_filter))
            fid_sql = f" AND id IN ({ph})"
            fid_params = tuple(fixture_ids_filter)
        conn = await get_source_connection()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT DISTINCT team_id FROM (
                        SELECT home_team_id AS team_id FROM fixtures
                        WHERE competition_id = %s
                          AND kickoff_datetime > NOW()
                          AND state_id = 1
                          {fid_sql}
                        UNION
                        SELECT away_team_id AS team_id FROM fixtures
                        WHERE competition_id = %s
                          AND kickoff_datetime > NOW()
                          AND state_id = 1
                          {fid_sql}
                    ) t
                    WHERE team_id IS NOT NULL
                    """,
                    (competition_id,) + fid_params + (competition_id,) + fid_params,
                )
                rows = await cur.fetchall()
            return sorted({int(r[0]) for r in rows})
        finally:
            release_source_connection(conn)

    async def projections(self, league_request=None, commit: bool = True) -> dict:
        """Compute + (optionally) write international fixture projections.

        Self-contained: refreshes Statz international ratings inline
        (committing to team_ratings) before computing fixture projections.
        Score prediction uses the same shared helpers as domestic
        (get_result_probs + find_inputs_for_probs).

        Per-comp behaviour is driven by the IntlProjectionScope looked up
        from INTL_SCOPES by request.league. Hosts/host_bonus, bracket
        config, squad source, fantasy rules all flow from the scope.

        Returns a stats dict:
          {n_total, n_projected, n_blended, n_skipped_unknown_team,
           n_ratings, ratings_snapshot_date, committed, competition_id, ...}
        """
        league_name = league_request.league if league_request else 'World Cup'
        if league_name not in INTL_SCOPES:
            raise ValueError(
                f"InternationalProjectionService called for unknown league {league_name!r}. "
                f"Known: {sorted(INTL_SCOPES.keys())}"
            )
        scope = INTL_SCOPES[league_name]
        # Pin this run's history cutoff before any history is computed, and so
        # a cutoff pinned by a previous run in the same process is never reused.
        set_run_cutoff()

        # Per-fixture mode (set via LeagueRequest.fixture_ids) skips
        # the slow fixture-independent / bracket-wide steps:
        #   - Step 1 (rating refresh): uses last nightly snapshot from
        #     team_ratings instead of recomputing
        #   - Step 3 (tournament simulator): bracket-wide, not per-fixture
        # Steps 2/4/5/6 each filter SQL by the supplied fixture_ids.
        fixture_ids_filter = None
        if league_request is not None and getattr(league_request, 'fixture_ids', None):
            fixture_ids_filter = [int(x) for x in league_request.fixture_ids]

        # Lean tournament-only re-sim: run the bracket-wide steps (1 ratings,
        # 2 1X2, 3 sim, 7 top-scorer) but SKIP the expensive per-fixture steps
        # (4 team-stat, 5 player-stat, 6/6b fantasy). Only meaningful for a
        # bracket-wide run — a fixture_ids filter would itself skip 1/3/7, so
        # lean is ignored when one is present.
        lean = bool(getattr(league_request, 'lean', False)) if league_request is not None else False
        if lean and fixture_ids_filter:
            logger.warning(
                f"{scope.competition_name}: lean=True ignored — a fixture_ids filter "
                f"was supplied, which already skips the bracket-wide steps."
            )
            lean = False

        # Ratings-only: recompute team_ratings (Step 1) then return. Forces a
        # bracket-wide recompute (no fixture filter → Step 1 takes the recompute
        # branch, not the cache branch).
        ratings_only = bool(getattr(league_request, 'ratings_only', False)) if league_request is not None else False
        if ratings_only:
            fixture_ids_filter = None

        logger.info(
            f"International projection start — comp={scope.competition_name} "
            f"(id={scope.competition_id}), commit={commit}, odds_beta={scope.odds_beta}, "
            f"boost={BOOST}, fixture_ids={fixture_ids_filter}, lean={lean}, ratings_only={ratings_only}"
        )

        # Step 1: refresh Statz ratings inline (skipped in per-fixture mode).
        ratings: dict
        statz_rated_team_names: set = set()
        fifa_carry_forward_names: set = set()
        if fixture_ids_filter:
            # Read latest cached team_ratings rows from team_ratings table
            # without recomputing. Fast — single SQL query.
            _r_conn = await get_source_connection()
            try:
                async with _r_conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT t.name, tr.attack, tr.defense
                        FROM team_ratings tr
                        JOIN teams t ON t.id = tr.team_id
                        WHERE tr.competition_id = %s
                          AND tr.date = (
                              SELECT MAX(date) FROM team_ratings
                              WHERE competition_id = %s AND team_id = tr.team_id
                          )
                        """,
                        (INTL_RATINGS_BUCKET_COMP_ID, INTL_RATINGS_BUCKET_COMP_ID),
                    )
                    rows = await cur.fetchall()
                ratings = {r[0]: (float(r[1]), float(r[2])) for r in rows}
            finally:
                release_source_connection(_r_conn)
            logger.info(f"Per-fixture mode: loaded {len(ratings)} ratings from cache (skipped recompute)")
        else:
            ratings_date = date.today()
            statz_df, _ = await compute_international_ratings(ratings_date, commit=commit)
            # `ratings` dict — only the freshly-computed Statz ratings
            # (~180 teams). compute_international_ratings ALSO writes
            # ~31 FIFA carry-forwards to team_ratings for micro-nations
            # without enough recent intl data (BVI, Vanuatu, etc.), but
            # those flat defaults (typically atk=40 def=40) carry no
            # real signal — every carry-forward team would project
            # identically. Per policy, fixtures involving carry-forward
            # teams get skipped (n_skipped_fifa_carry_forward counter).
            # We layer carry-forwards into a SEPARATE set just for the
            # diagnostic distinction "FIFA-only" vs "truly unrated".
            ratings = {
                row['team_name']: (float(row['attack']), float(row['defense']))
                for _, row in statz_df.iterrows()
            } if not statz_df.empty else {}
            statz_rated_team_names: set = set(ratings.keys())
            fifa_carry_forward_names: set = set()
            if commit:
                _r_conn = await get_source_connection()
                try:
                    async with _r_conn.cursor() as cur:
                        await cur.execute(
                            """
                            SELECT t.name FROM team_ratings tr
                            JOIN teams t ON t.id = tr.team_id
                            WHERE tr.competition_id = %s AND tr.date = %s
                            """,
                            (INTL_RATINGS_BUCKET_COMP_ID, ratings_date),
                        )
                        for (name,) in await cur.fetchall():
                            if name and name not in statz_rated_team_names:
                                fifa_carry_forward_names.add(name)
                finally:
                    release_source_connection(_r_conn)
            logger.info(
                f"Refreshed {len(statz_rated_team_names)} Statz ratings + "
                f"{len(fifa_carry_forward_names)} FIFA carry-forwards "
                f"(carry-forward fixtures will skip) for {ratings_date}"
            )

        # Ratings-only run: Step 1 has recomputed + committed team_ratings —
        # stop here, skip the sim / per-fixture stats / fantasy / top-scorer.
        if ratings_only:
            logger.info(
                f"{scope.competition_name}: ratings_only run complete — "
                f"recomputed {len(ratings)} ratings, skipped projection steps."
            )
            return {
                'ratings_only': True,
                'n_ratings': len(ratings),
                'committed': commit,
                'competition_id': scope.competition_id,
            }

        conn = await get_source_connection()
        try:
            async with conn.cursor() as cur:
                # Upcoming WC fixtures + bet365 1X2 odds (LEFT JOIN, null-safe).
                # When fixture_ids_filter is set, scope the SELECT to that
                # list — typically exactly one fixture for the per-fixture
                # re-projection trigger.
                fid_filter_sql = ""
                fid_filter_params: tuple = ()
                if fixture_ids_filter:
                    placeholders = ",".join(["%s"] * len(fixture_ids_filter))
                    fid_filter_sql = f" AND f.id IN ({placeholders})"
                    fid_filter_params = tuple(fixture_ids_filter)
                # venue.country_id (post-backfill, ~99% populated) drives
                # the per-fixture neutral-venue classifier. teams.country_id
                # lets us spot Sportmonks home/away inversions where the
                # listed home team is playing in the listed away team's
                # country (Argentina v Iceland @ Reykjavik etc.).
                await cur.execute(
                    f"""
                    SELECT f.id, f.home_team_id, f.away_team_id, f.kickoff_datetime,
                           th.name AS h, ta.name AS a,
                           bo.home_win_odd, bo.draw_odd, bo.away_win_odd,
                           f.venue_id, v.country_id AS venue_country_id,
                           th.country_id AS home_country_id,
                           ta.country_id AS away_country_id
                    FROM fixtures f
                    JOIN teams th ON th.id = f.home_team_id
                    JOIN teams ta ON ta.id = f.away_team_id
                    LEFT JOIN venues v ON v.id = f.venue_id
                    LEFT JOIN bet365_fixture_odds bo ON bo.fixture_id = f.id
                    WHERE f.competition_id = %s
                      AND f.kickoff_datetime > NOW()
                      AND f.state_id = 1
                      {fid_filter_sql}
                    ORDER BY f.kickoff_datetime
                    """,
                    (scope.competition_id,) + fid_filter_params,
                )
                fixtures = await cur.fetchall()
                logger.info(
                    f"Loaded {len(fixtures)} upcoming {scope.competition_name} fixtures"
                )

            # Pre-load bet365 goals over/under for these fixtures. The
            # blend cascade (paths 1-3) uses per-team and match-total
            # ladders directly; path 4 (legacy 1X2) is the fallback.
            from app.services.odds_blend import (
                load_goals_odds_for_fixtures,
                compute_final_goals_and_probs,
            )
            wc_fixture_ids = [row[0] for row in fixtures]
            goals_odds_map = await load_goals_odds_for_fixtures(conn, wc_fixture_ids)

            # When scope.require_result_odds, build the set of fixtures
            # that have complete 1X2 result odds across any of our 5
            # fixture-odds books. Used as the FIRST gate in the per-
            # fixture loop — friendlies feed includes obscure exhibition
            # / micro-confederation games bookies don't price, so we
            # skip them rather than emit a model-only projection. Stake
            # has no 1X2 for football so it's not included.
            fixtures_with_1x2: set = set()
            if scope.require_result_odds and wc_fixture_ids:
                async with conn.cursor() as cur:
                    fid_ph = ",".join(["%s"] * len(wc_fixture_ids))
                    fid_tuple = tuple(int(x) for x in wc_fixture_ids)
                    await cur.execute(
                        f"""
                        SELECT fixture_id FROM bet365_fixture_odds
                          WHERE fixture_id IN ({fid_ph}) AND home_win_odd IS NOT NULL AND draw_odd IS NOT NULL AND away_win_odd IS NOT NULL
                        UNION SELECT fixture_id FROM coral_fixture_odds
                          WHERE fixture_id IN ({fid_ph}) AND home_win_odd IS NOT NULL AND draw_odd IS NOT NULL AND away_win_odd IS NOT NULL
                        UNION SELECT fixture_id FROM ladbrokes_fixture_odds
                          WHERE fixture_id IN ({fid_ph}) AND home_win_odd IS NOT NULL AND draw_odd IS NOT NULL AND away_win_odd IS NOT NULL
                        UNION SELECT fixture_id FROM midnite_fixture_odds
                          WHERE fixture_id IN ({fid_ph}) AND home_win_odd IS NOT NULL AND draw_odd IS NOT NULL AND away_win_odd IS NOT NULL
                        UNION SELECT fixture_id FROM boylesports_fixture_odds
                          WHERE fixture_id IN ({fid_ph}) AND home_win_odd IS NOT NULL AND draw_odd IS NOT NULL AND away_win_odd IS NOT NULL
                        """,
                        fid_tuple * 5,
                    )
                    fixtures_with_1x2 = {int(r[0]) for r in await cur.fetchall()}
                logger.info(
                    f"{scope.competition_name}: {len(fixtures_with_1x2)}/{len(wc_fixture_ids)} "
                    f"fixtures have 1X2 result odds across 5 books — others will skip"
                )

            # Per-fixture neutral-venue path needs comp-specific home/away
            # goal averages (cached once per run). Falls back to symmetric
            # AVG_GOALS if the helper returns None (e.g. brand-new comp
            # with no history yet).
            avg_home_goals = AVG_GOALS
            avg_away_goals = AVG_GOALS
            if scope.use_per_fixture_neutral_venue:
                avg_h, avg_a = await _compute_intl_comp_goal_avgs(conn, scope.competition_id)
                if avg_h is not None and avg_a is not None:
                    avg_home_goals = avg_h
                    avg_away_goals = avg_a
                    logger.info(
                        f"{scope.competition_name} comp goal avgs (xG-complete, "
                        f"0.98^(w-26) decay): home={avg_home_goals:.4f} "
                        f"away={avg_away_goals:.4f} adv={avg_home_goals - avg_away_goals:+.4f}"
                    )
                else:
                    logger.warning(
                        f"{scope.competition_name}: comp goal avg helper returned None — "
                        f"falling back to symmetric AVG_GOALS={AVG_GOALS}"
                    )
            neutral_baseline = (avg_home_goals + avg_away_goals) / 2

            n_blended = 0
            n_skipped_no_odds = 0
            n_skipped_unknown_team = 0
            n_skipped_fifa_carry_forward = 0
            n_skipped_both_minnow = 0
            n_skipped_no_venue = 0
            n_home_at_home = 0
            n_away_at_home = 0
            n_true_neutral = 0
            n_unknown_venue = 0
            inserts = []
            for (fid, h_tid, a_tid, ko, home, away, oh, od, oa,
                 venue_id, venue_country_id, home_country_id, away_country_id) in fixtures:
                # Gate 1: bookmaker-confirmed (only when scope opts in).
                # First gate so the strongest skip reason wins — a fixture
                # with no odds AND no rating counts as no_odds, not
                # unknown_team.
                if scope.require_result_odds and fid not in fixtures_with_1x2:
                    n_skipped_no_odds += 1
                    continue
                if home not in ratings or away not in ratings:
                    # `ratings` dict (full-comp mode) holds Statz-rated
                    # teams only — FIFA carry-forward teams are tracked
                    # separately. Bucket the skip accordingly:
                    #   - FIFA-only team in the fixture → carry-forward
                    #     skip (flat 40/40 defaults are noise, not signal)
                    #   - Neither rated nor carry-forward → truly unknown
                    if home in fifa_carry_forward_names or away in fifa_carry_forward_names:
                        n_skipped_fifa_carry_forward += 1
                    else:
                        n_skipped_unknown_team += 1
                    continue

                # λ from cross-Poisson rating product. Baseline selection:
                #
                # - scope.use_per_fixture_neutral_venue OFF (WC etc.):
                #   symmetric AVG_GOALS for both sides + host-list
                #   bonus/penalty (preserves existing WC behaviour byte
                #   for byte).
                #
                # - scope.use_per_fixture_neutral_venue ON (friendlies):
                #   classify the fixture by venue.country_id vs each
                #   team's country, then pick comp-specific avg_home /
                #   avg_away baselines accordingly. Inversions swap
                #   the baselines so the real-home team gets the boost.
                h_atk, h_def = ratings[home]
                a_atk, a_def = ratings[away]

                # Gate 3: minnow-vs-minnow. Skip when BOTH teams' overall
                # Statz rating (atk − def) is below the scope floor. Placed
                # before the venue classify so these fixtures bucket as
                # "minnow" rather than "no_venue". Both-sides test → a
                # below-average team vs a decent one still projects.
                if scope.skip_both_below_rating is not None:
                    if ((h_atk - h_def) < scope.skip_both_below_rating
                            and (a_atk - a_def) < scope.skip_both_below_rating):
                        n_skipped_both_minnow += 1
                        continue

                if scope.use_per_fixture_neutral_venue:
                    case = _classify_fixture_venue(
                        home_country_id, away_country_id,
                        venue_id, venue_country_id,
                    )
                    if case == 'no_venue':
                        n_skipped_no_venue += 1
                        continue
                    if case == 'unknown':
                        # Venue exists but its country_id is NULL and the
                        # geocoder couldn't match (coastal/island edge
                        # cases — Bahrain Gulf coast, French Polynesia,
                        # etc.). Skip rather than guess — better to drop
                        # a fixture than project it with possibly-wrong
                        # home-advantage assumptions.
                        n_unknown_venue += 1
                        continue
                    if case == 'home_at_home':
                        h_baseline, a_baseline = avg_home_goals, avg_away_goals
                        n_home_at_home += 1
                    elif case == 'away_at_home':
                        # Inversion — swap the baselines so the actual
                        # home team (the listed away side) gets the
                        # avg_home_goals boost.
                        h_baseline, a_baseline = avg_away_goals, avg_home_goals
                        n_away_at_home += 1
                    else:  # 'true_neutral'
                        h_baseline = a_baseline = neutral_baseline
                        n_true_neutral += 1
                    home_goals = (h_atk / 100) * (a_def / 100) * h_baseline
                    away_goals = (a_atk / 100) * (h_def / 100) * a_baseline
                else:
                    home_goals = (h_atk / 100) * (a_def / 100) * AVG_GOALS
                    away_goals = (a_atk / 100) * (h_def / 100) * AVG_GOALS
                    if home in scope.hosts:
                        home_goals *= scope.host_bonus
                        away_goals *= scope.host_penalty
                    elif away in scope.hosts:
                        away_goals *= scope.host_bonus
                        home_goals *= scope.host_penalty

                # Score prediction via the shared odds-blend cascade.
                # Paths 1-3 consume bet365 goals over/under directly;
                # path 4 (legacy 1X2 blend + reverse-solve) is the
                # fallback; path 5 (no odds at all) leaves model output
                # unchanged.
                bookie_1x2_pct = None
                if oh and od and oa:
                    home_odds_pct = (1.0 / float(oh)) * 100
                    draw_odds_pct = (1.0 / float(od)) * 100
                    away_odds_pct = (1.0 / float(oa)) * 100
                    bookie_margin = 1 + (home_odds_pct + draw_odds_pct + away_odds_pct - 100) / 100
                    bookie_1x2_pct = (
                        home_odds_pct / bookie_margin / 100.0,
                        draw_odds_pct / bookie_margin / 100.0,
                        away_odds_pct / bookie_margin / 100.0,
                    )
                    n_blended += 1

                new_home_goals, new_away_goals, adjusted_home_win_prob, adjusted_draw_prob, adjusted_away_win_prob = (
                    compute_final_goals_and_probs(
                        fid,
                        float(home_goals), float(away_goals),
                        bookie_1x2_pct,
                        goals_odds_map.get(fid, {}),
                        scope.odds_beta,
                        BOOST,
                    )
                )

                home_clean_sheet = poisson.pmf(0, new_away_goals)
                away_clean_sheet = poisson.pmf(0, new_home_goals)
                x = np.arange(0, 9)
                y = np.arange(0, 9)
                X, Y = np.meshgrid(x, y)
                Z = poisson.pmf(X, new_home_goals) * poisson.pmf(Y, new_away_goals)
                over_1_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1]) * 100
                over_2_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1] - Z[2, 0] - Z[0, 2] - Z[1, 1]) * 100
                both_teams_score_prob = (1 - Z[0, :].sum() - Z[:, 0].sum() + Z[0, 0]) * 100
                # === End shared score-prediction block ===

                inserts.append((
                    fid, h_tid, a_tid,
                    str(round(new_home_goals, 2)), str(round(new_away_goals, 2)),
                    round(adjusted_home_win_prob, 2),
                    round(adjusted_away_win_prob, 2),
                    round(adjusted_draw_prob, 2),
                    round(home_clean_sheet * 100, 2),
                    round(away_clean_sheet * 100, 2),
                    round(over_1_goals, 2),
                    round(over_2_goals, 2),
                    round(both_teams_score_prob, 2),
                    ko,
                ))

            if scope.use_per_fixture_neutral_venue:
                logger.info(
                    f"{scope.competition_name} projection ready: total={len(fixtures)} "
                    f"projected={len(inserts)} blended={n_blended} | skips: "
                    f"no_odds={n_skipped_no_odds} "
                    f"unknown_team={n_skipped_unknown_team} "
                    f"fifa_carry_forward={n_skipped_fifa_carry_forward} "
                    f"both_minnow={n_skipped_both_minnow} "
                    f"no_venue={n_skipped_no_venue} "
                    f"unknown_venue={n_unknown_venue} | "
                    f"venue breakdown: home_at_home={n_home_at_home} "
                    f"away_at_home={n_away_at_home} true_neutral={n_true_neutral}"
                )
            else:
                logger.info(
                    f"{scope.competition_name} projection ready: total={len(fixtures)} "
                    f"projected={len(inserts)} blended={n_blended} "
                    f"skipped_unknown_team={n_skipped_unknown_team}"
                )

            if commit and inserts:
                async with conn.cursor() as cur:
                    # Per-fixture mode scopes the DELETE to the requested
                    # fixture_ids so we don't wipe the rest of the WC
                    # projection rows. Full-comp mode keeps the original
                    # "delete-all-WC-then-reinsert" idempotent pattern.
                    if fixture_ids_filter:
                        del_ph = ",".join(["%s"] * len(fixture_ids_filter))
                        await cur.execute(
                            f"DELETE FROM fixture_projections WHERE fixture_id IN ({del_ph})",
                            tuple(fixture_ids_filter),
                        )
                    else:
                        await cur.execute(
                            """
                            DELETE fp FROM fixture_projections fp
                            JOIN fixtures f ON f.id = fp.fixture_id
                            WHERE f.competition_id = %s
                            """,
                            (scope.competition_id,),
                        )
                    deleted = cur.rowcount
                    now = datetime.now()
                    rows = [(
                        fid, h_tid, a_tid, hg, ag,
                        hw, aw, dw, hcs, acs, o15, o25, btts,
                        ko, now, now
                    ) for fid, h_tid, a_tid, hg, ag, hw, aw, dw, hcs, acs, o15, o25, btts, ko in inserts]
                    for i in range(0, len(rows), 100):
                        await cur.executemany(
                            """INSERT INTO fixture_projections
                               (fixture_id, home_team_id, away_team_id,
                                home_goals, away_goals,
                                home_win_percent, away_win_percent, draw_percent,
                                home_clean_sheet_percent, away_clean_sheet_percent,
                                over_15_goals_percent, over_25_goals_percent,
                                both_teams_shore_percent,
                                kickoff_datetime, created_at, updated_at)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                            rows[i:i+100],
                        )
                await conn.commit()
                logger.info(f"WROTE: deleted={deleted}, inserted={len(rows)}")
        finally:
            release_source_connection(conn)

        # Step 3: Monte Carlo tournament simulator (bracket-wide).
        # Skipped in per-fixture mode — the sim reads the entire bracket
        # fixture projection set and walks the bracket; a single-fixture
        # change doesn't shift bracket outcomes enough to be worth a full
        # 10k-sim rerun. The next nightly full pass will refresh it.
        # Also skipped for non-bracket comps (friendlies, quals, Nations
        # League group stage) — their scope.bracket_config is None.
        sim_result = None
        if commit and not fixture_ids_filter and scope.bracket_config is not None:
            sim_result = await TournamentSimulator().run(scope.bracket_config, num_sims=10_000)
            logger.info(f"{scope.competition_name} tournament simulation: {sim_result}")
        elif scope.bracket_config is None:
            logger.info(
                f"Step 3: skipped (no bracket config for {scope.competition_name})"
            )

        # Step 3b (lean only): re-sync team_projections.goals from the freshly
        # rewritten fixture_projections. In a full run, Step 4 (team-stat)
        # overwrites team_projections.goals with fixture_projections.home/away_
        # goals — so the two tables agree. A lean re-runs Step 2 (rewrites
        # fixture_projections off the updated ratings) but SKIPS Step 4, which
        # left team_projections.goals stale → the Fixtures-tab and Teams-tab
        # scorelines diverged (e.g. Argentina 1.65 vs 1.68). This lightweight
        # join-update restores the invariant (team goals == fixture-side goal)
        # without re-running the expensive shots/corners suite. Runs before
        # Step 7, which reads team_projections.goals. (Shots/corners stay
        # anchored to the last full pass until the next full run — accepted.)
        if commit and lean:
            _sync_conn = await get_source_connection()
            try:
                async with _sync_conn.cursor() as cur:
                    await cur.execute(
                        """
                        UPDATE team_projections tp
                        JOIN fixtures f ON f.id = tp.fixture_id
                        JOIN fixture_projections fp ON fp.fixture_id = tp.fixture_id
                        SET tp.goals = CASE WHEN tp.team_id = f.home_team_id
                                            THEN fp.home_goals ELSE fp.away_goals END,
                            tp.updated_at = NOW()
                        WHERE f.competition_id = %s
                          AND f.kickoff_datetime > NOW()
                        """,
                        (scope.competition_id,),
                    )
                    _synced = cur.rowcount
                await _sync_conn.commit()
                logger.info(
                    f"{scope.competition_name}: lean goals-sync — "
                    f"team_projections.goals re-aligned to fixture_projections ({_synced} rows)"
                )
            except Exception as e:
                logger.exception(f"{scope.competition_name} lean goals-sync failed: {e}")
            finally:
                release_source_connection(_sync_conn)

        # Step 4: Per-team stat projections. (Skipped in lean — the post-match
        # re-sim reuses the last full pass's per-fixture lines.)
        team_stat_result = None
        if commit and not lean:
            try:
                team_stat_result = await InternationalTeamStatService(scope=scope).project(
                    commit=commit, fixture_ids=fixture_ids_filter,
                )
                logger.info(f"{scope.competition_name} team-stat projection: {team_stat_result}")
            except Exception as e:
                logger.exception(f"{scope.competition_name} team-stat projection failed: {e}")
                team_stat_result = {'error': str(e)}

        # Step 5: Per-player stat projections.
        #
        # Squad-source strategy depends on the scope:
        #   - WC scope (has_squad_source=True) → WcSquadProvider, reads
        #     wc_squads (the FIFA fantasy roster).
        #   - Non-WC scopes (friendlies / quals / Nations League — no
        #     formal roster) → RecentCapsSquadProvider, derives from
        #     fixture_player_stats minutes played in the last 24 months
        #     for each nation in the upcoming fixture batch.
        player_stat_result = None
        if commit and not lean:
            try:
                from app.services.intl_squad_provider import (
                    WcSquadProvider, RecentCapsSquadProvider,
                )
                if scope.has_squad_source:
                    squad_provider = WcSquadProvider()
                else:
                    # Pre-fetch the team_ids playing in the upcoming
                    # fixture batch for this scope. RecentCapsSquadProvider
                    # uses them to know which nations to source squads for.
                    team_ids_in_scope = await self._fetch_upcoming_team_ids(
                        scope.competition_id, fixture_ids_filter,
                    )
                    squad_provider = RecentCapsSquadProvider(team_ids=team_ids_in_scope)
                    logger.info(
                        f"{scope.competition_name} player-stat squad source: "
                        f"recent-caps for {len(team_ids_in_scope)} nations"
                    )
                player_stat_result = await WcPlayerStatService(
                    scope=scope, squad_provider=squad_provider,
                ).project(
                    commit=commit, fixture_ids=fixture_ids_filter,
                )
                logger.info(f"{scope.competition_name} player-stat projection: {player_stat_result}")
            except Exception as e:
                logger.exception(f"{scope.competition_name} player-stat projection failed: {e}")
                player_stat_result = {'error': str(e)}

        # Step 6: Per-(fixture, player) Fantasy point projections.
        # FIFA WC fantasy scoring is the only ruleset wired today. Other
        # comps (Euros / Copa / AFCON) will need their own scoring-rule
        # configs before opting in via scope.fantasy_rules.
        fantasy_points_result = None
        if commit and not lean and scope.fantasy_rules == 'fifa_wc_2026':
            try:
                fantasy_points_result = await WcFantasyPointsService(scope=scope).project(
                    commit=commit, fixture_ids=fixture_ids_filter,
                )
                logger.info(f"{scope.competition_name} fantasy points projection: {fantasy_points_result}")
            except Exception as e:
                logger.exception(f"{scope.competition_name} fantasy points projection failed: {e}")
                fantasy_points_result = {'error': str(e)}

        # Step 6b: Per-(fixture, player) FanTeam fantasy point projections.
        # Sibling of Step 6 with the FanTeam PL ruleset applied to WC stat
        # bags. Reads position from `fanteam_wc_player_mappings` so the
        # scoring matches FanTeam's own classification of each player.
        fanteam_points_result = None
        if commit and not lean and scope.fanteam_rules == 'fanteam_wc_2026':
            try:
                from app.services.fanteam_wc_projection_service import FanTeamWcProjectionService
                fanteam_points_result = await FanTeamWcProjectionService(scope=scope).project(
                    commit=commit, fixture_ids=fixture_ids_filter,
                )
                logger.info(f"{scope.competition_name} FanTeam projection: {fanteam_points_result}")
            except Exception as e:
                logger.exception(f"{scope.competition_name} FanTeam projection failed: {e}")
                fanteam_points_result = {'error': str(e)}

        # Step 7: Per-(competition, player) tournament-total goals + assists.
        # Aggregates the group-stage player projections + scales them by the
        # team's expected_goals_for from tournament_projections. Gated on
        # bracket_config — same condition as the tournament simulator since
        # both need the Monte Carlo bracket output.
        # Skipped in per-fixture mode (tournament totals are bracket-wide,
        # not per-fixture).
        tournament_player_result = None
        if commit and not fixture_ids_filter and scope.bracket_config is not None:
            try:
                from app.services.wc_tournament_player_projection_service import (
                    WcTournamentPlayerProjectionService,
                )
                tournament_player_result = await WcTournamentPlayerProjectionService(
                    scope=scope
                ).project(commit=commit)
                logger.info(
                    f"{scope.competition_name} tournament player projection: {tournament_player_result}"
                )
            except Exception as e:
                logger.exception(
                    f"{scope.competition_name} tournament player projection failed: {e}"
                )
                tournament_player_result = {'error': str(e)}

        result = {
            'n_total': len(fixtures),
            'n_projected': len(inserts),
            'n_blended': n_blended,
            'n_skipped_unknown_team': n_skipped_unknown_team,
            'n_ratings': len(ratings),
            'ratings_snapshot_date': 'cached' if fixture_ids_filter else str(ratings_date),
            'fixture_ids': fixture_ids_filter,
            'competition_id': scope.competition_id,
            'competition_name': scope.competition_name,
            'committed': commit,
            'tournament_simulation': sim_result,
            'team_stat_projection': team_stat_result,
            'player_stat_projection': player_stat_result,
            'fantasy_points_projection': fantasy_points_result,
            'fanteam_points_projection': fanteam_points_result,
            'tournament_player_projection': tournament_player_result,
        }
        if scope.use_per_fixture_neutral_venue:
            result.update({
                'avg_home_goals': avg_home_goals,
                'avg_away_goals': avg_away_goals,
                'n_skipped_no_venue': n_skipped_no_venue,
                'n_home_at_home': n_home_at_home,
                'n_away_at_home': n_away_at_home,
                'n_true_neutral': n_true_neutral,
                'n_unknown_venue': n_unknown_venue,
            })
        if scope.require_result_odds:
            result['n_skipped_no_odds'] = n_skipped_no_odds
        result['n_skipped_fifa_carry_forward'] = n_skipped_fifa_carry_forward
        if scope.skip_both_below_rating is not None:
            result['n_skipped_both_minnow'] = n_skipped_both_minnow
        return result
