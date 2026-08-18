"""One team rating, used everywhere.

Replaces two separate things that disagreed with each other:

  * the `mv_beta` nudge inside _prepare_league, which tilted the form
    ratings that drive FIXTURE predictions, and
  * team_strength.build_strength_ratings, which built a different rating
    that drove the SEASON SIMULATION only.

That split let the model say Liverpool finish 3rd and Brentford 10th while
making Brentford favourites when they played each other (George,
2026-07-31). Everything here produces a single rating consumed by both.

The blend has three components, all expressed as OVERALL strength in index
points above the league average, so they are directly comparable:

    overall = (attack_index - defence_index) / 2

  form   the existing get_ratings output — the only component that knows
         about style, so it also sets the attack/defence SHAPE
  MV     squad value as a z-score, soft-capped, scaled by the form frame's
         spread. Known weakness: it therefore inherits any form
         over-dispersion instead of being an independent opinion. Left
         as-is deliberately (George, 2026-08-03) — see the memory note on
         giving MV an absolute anchor.
  odds   outright prices -> expected points -> goal difference. Fully
         absolute; borrows nothing from form.

Blending is LEVEL-AND-SHAPE: the market components only speak to overall
strength, so they set the LEVEL and form keeps the SHAPE. The blended
overall minus the form overall is a single delta, added to attack and
subtracted from defence. Blending attack directly would flatten each
team's own character (Arsenal defence-led, City attack-led).
"""
import logging
import math
import os
import statistics

import numpy as np
import pandas as pd

from app.services import team_strength as ts

logger = logging.getLogger("projection")

# Retired 2026-08-04. This was the rollout switch while the legacy mv_beta
# nudge still existed alongside; that path is now deleted, so there is
# nothing to fall back to and the flag has no meaningful "off" state.
# Kept as a constant so any straggling reference still reads True rather
# than silently disabling the only rating the pipeline has.
UNIFIED_ENABLED = True

# --- weight schedule -------------------------------------------------------
# p = fraction of the season played.
#
#   odds  0.30 flat     (George, 2026-08-04. The book keeps repricing all
#                         season so it never becomes uninformative, and with
#                         the pre-season figure equal to the floor there is
#                         nothing to fade — its share stays constant.)
#   MV    0.30 -> 0.00   (gone by three-quarter distance — squad value is a
#                         PRE-season prior, and once there is real evidence
#                         it is just a proxy for what form now measures)
#   form  the remainder, 0.40 -> 0.70
#
# George's rule: market value heaviest pre-season and fading, form lightest
# pre-season and growing. One schedule for every league; only MV's starting
# weight is per-competition (see blend_weights).
#
# Pre-season split settled at 40/30/30 after comparing it against 30/30/40
# and 45/25/30 on a full Premier League simulation (2026-08-04). All three
# gave near-identical tables — Arsenal 57/56/57% for the title, Hull 94/96/96%
# for relegation — because form and the market already agree about most of
# the league. The differences show only where they disagree: Liverpool 64.7
# -> 62.9 -> 62.2 and Tottenham 56.1 -> 54.4 -> 53.5 as form gains weight.
# Reweighting is NOT the lever for making projections diverge from the book.
W_ODDS_PRE = float(os.getenv("UNIFIED_W_ODDS_PRE", "0.30"))
W_ODDS_MIN = float(os.getenv("UNIFIED_W_ODDS_MIN", "0.30"))
W_MV_PRE = float(os.getenv("UNIFIED_W_MV_PRE", "0.30"))
MV_FADE_BY = float(os.getenv("UNIFIED_MV_FADE_BY", "0.75"))

# Squad value is heavy-tailed — one oligarch club sits several SDs clear and
# would otherwise drag the whole component. tanh compresses the extremes
# while leaving the middle of the table essentially linear.
MV_SOFT_CAP = float(os.getenv("UNIFIED_MV_SOFT_CAP", "2.5"))

# Below this many usable outright markets the full odds path is not
# trusted. A lone title market is one point at position 1: it says a great
# deal about the favourite and almost nothing about 14th. Leagues under
# this threshold fall through to the PARTIAL path below rather than losing
# the market entirely.
MIN_ODDS_MARKETS = int(os.getenv("UNIFIED_MIN_ODDS_MARKETS", "3"))

# --- partial odds, for thin books only -------------------------------------
# Nine of the leagues we project carry only a title market. Dropping it
# outright throws away real information about the favourites; using all of
# it invents information about everyone else. So use the prices that are
# genuinely opinions and ignore the rest.
#
# A price is an opinion if its de-vigged probability clears this floor.
# Expressed as probability, not odds, because 101/1 means different things
# in different books — in a league with three co-favourites the tail is far
# flatter. Measured across Serie A, Liga Portugal, Belgium and the Super
# Lig (2026-08-04): at 0.5% no unpriced team moves by more than 0.1 points
# and the real disagreements are caught (Istanbul Basaksehir 5th on form at
# 67/1 -> corrected 6 points; AC Milan mid-table on form at 6/1 -> +5.5).
# At 0.2% it breaks — Moreirense gains 6.4 points off a 101/1 price.
PARTIAL_ODDS_MIN_PROB = float(os.getenv("UNIFIED_PARTIAL_MIN_PROB", "0.005"))

# Sigma cannot be self-calibrated from a single market — one bar per team
# fits any spread perfectly — so it is fixed. 12 sits just above what the
# deep books calibrate to (Premier League 10.9, La Liga 9.8, Bundesliga
# 9.0) and is the gentler choice where it matters: teams near the floor are
# the most sigma-sensitive, the short prices barely move with it.
PARTIAL_ODDS_SIGMA = float(os.getenv("UNIFIED_PARTIAL_SIGMA", "12.0"))

# This path applies ONLY to thin books. Leagues clearing MIN_ODDS_MARKETS
# keep the full path unchanged — the point of this is to rescue leagues
# with poor coverage, not to alter the ones that are already well served.

# Attack/defence floor as a fraction of the league mean, so no blend can
# drive a goal rate to zero or negative.
RATING_FLOOR_FRACTION = float(os.getenv("UNIFIED_RATING_FLOOR", "0.2"))

# Where each outright market's points bar sits, and which end of the table
# it is anchored from. The bar VALUE is not hardcoded — it is the median
# points that finishing position actually took across the seasons we hold,
# so it self-maintains as standings accumulate.
MARKET_POS = {
    75:    (1, 't'),        # Win League
    50464: (2, 't'),        # Top 2
    50457: (3, 't'),        # Top 3
    50130: (4, 't'),        # Top 4
    50141: (5, 't'),        # Top 5
    50131: (6, 't'),        # Top 6
    50455: (7, 't'),        # Top 7
    60015: (8, 't'),        # Top 8
    10141: ('half', 't'),   # Top Half
    10142: ('half', 'b'),   # Bottom Half
    344:   ('survive', 'b'),  # Relegation
    10106: ('bottom', 'b'),   # Finish Bottom
}

# How many selections each market's de-vigged probabilities must sum to.
BAND = {75: 1, 50464: 2, 50457: 3, 50130: 4, 50141: 5, 50131: 6,
        50455: 7, 60015: 8, 10141: 'half', 10142: 'half',
        344: 'releg', 10106: 1}

# Relegation places per competition, for the 'survive' bar and the
# relegation market's band. Defaults to 3 with a log when unknown.
RELEGATION_PLACES = {
    8: 3, 9: 3, 12: 4, 14: 2, 72: 2, 82: 2, 181: 2, 208: 1, 271: 2,
    301: 2, 384: 3, 444: 2, 462: 2, 501: 1, 564: 3, 573: 2, 591: 1,
    600: 3, 648: 4, 779: 0, 944: 3,
}


def relegation_places(competition_id):
    cid = int(competition_id)
    if cid in RELEGATION_PLACES:
        return RELEGATION_PLACES[cid]
    logger.info("  no relegation count for competition %s — assuming 3", cid)
    return 3


def blend_weights(matches_played, games_in_season, mv_pre=None):
    """(form, MV, odds) weights at this point in the season.

    `mv_pre` is squad value's PRE-SEASON weight and is set per competition
    (competition_projection_config.mv_beta), because money predicts the
    table far better in some leagues than others. Measured 2026-08-03
    against each league's last completed season: correlation runs from 0.95
    in Liga Portugal down to 0.57 in League One, and in the lower EFL squad
    value is close to noise while carrying the widest spread of any
    component — which is how York City's manual prior of 1.68 attack came
    out at 1.45. Defaults to W_MV_PRE when not supplied.
    """
    if not games_in_season or games_in_season <= 0:
        p = 0.0
    else:
        p = min(1.0, max(0.0, float(matches_played) / float(games_in_season)))
    w_odds = max(W_ODDS_MIN, W_ODDS_PRE - (W_ODDS_PRE - W_ODDS_MIN) * (p / MV_FADE_BY)) \
        if MV_FADE_BY > 0 else W_ODDS_MIN
    w_odds = max(W_ODDS_MIN, min(W_ODDS_PRE, w_odds))
    mv_pre = W_MV_PRE if mv_pre is None else float(mv_pre)
    w_mv = mv_pre * max(0.0, 1.0 - (p / MV_FADE_BY)) if MV_FADE_BY > 0 else 0.0
    w_form = max(0.0, 1.0 - w_odds - w_mv)
    return w_form, w_mv, w_odds


async def games_per_team(conn, competition_id, season_id, n_teams):
    """How many games each team actually plays this season. Counted, not assumed.

    Callers derive this as `(n_teams - 1) * 2`, which is exactly right for a
    balanced double round-robin — everyone plays everyone home and away. That
    covers most of what we project, which is why it survived unquestioned. It
    is wrong for any other structure:

        MLS                   30 teams, conference-based   34 games, not 58
        Scottish Premiership  12 teams, triple + split     33 games, not 22

    Both are wrong twice over: the strength divisor (MLS compressed 41%,
    Scottish inflated 50%) and the season fraction that drives the MV fade,
    the odds weight and the fixture-strength crossfade (MLS at 19 of 34 is
    56% through but reads as 33%).

    Returns None when the fixture list looks incomplete — a season part-way
    through loading would otherwise report a short schedule, which reads as
    "further through the season than we are" and fades the market component
    early. Falling back to the caller's assumption is the safer failure.
    """
    if not competition_id or not season_id or not n_teams:
        return None
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT team_id, COUNT(*) FROM (
                    SELECT home_team_id AS team_id FROM fixtures
                    WHERE competition_id = %s AND season_id = %s
                    UNION ALL
                    SELECT away_team_id FROM fixtures
                    WHERE competition_id = %s AND season_id = %s
                ) x
                GROUP BY team_id
                """,
                (int(competition_id), int(season_id)) * 2,
            )
            rows = await cur.fetchall()
    except Exception as err:
        logger.warning("  games-per-team query failed (%s) — using caller's value", err)
        return None

    if len(rows) < 2:
        return None
    counts = [int(c) for _, c in rows]
    # Mode, for the same reason matches_played uses it: a team with a fixture
    # yet to be scheduled should not set the league's schedule length.
    games = statistics.mode(counts)

    # Sanity gate: at least a single round-robin's worth. Below that the
    # fixture list is being loaded, not short.
    if games < n_teams - 1:
        return None
    return int(games)


# --- odds -> expected points ----------------------------------------------

def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def power_devig(odds, band):
    """Strip the bookmaker's margin with a POWER de-vig.

    Solve p_i = (1/o_i)^k for the k that makes the probabilities sum to the
    number of selections the market pays out on. Proportional de-vig
    (dividing by the book percentage) leaves longshots systematically
    over-priced, which produced contradictory markets — a team shorter for
    the title yet also likelier to be relegated.
    """
    lo, hi = 0.5, 5.0
    for _ in range(60):
        k = (lo + hi) / 2
        if sum((1.0 / o) ** k for o in odds) > band:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    return [(1.0 / o) ** k for o in odds]


def _scan(lo, hi, step, err_fn):
    """Smallest err_fn on a grid, returned as (x, err)."""
    best, best_err = lo, 1e18
    x = lo
    while x <= hi:
        e = err_fn(x)
        if e < best_err:
            best_err, best = e, x
        x += step
    return best, best_err


def fit_points(dots, sigma):
    """The points total best explaining one team's markets, and its residual.

    Each market is read as P(this team's points exceed this bar), modelled
    as a normal with the given spread. Least squares over every bar the
    team is priced in.

    Coarse-then-fine rather than one fine sweep of the whole range: the
    error surface is a sum of squared differences of monotone CDFs, so it
    is smooth and single-troughed in points, and a 1-point grid followed by
    a 0.05 grid over the winning interval finds the same minimum for about
    a fifteenth of the work. That matters — this runs once per team per
    candidate sigma, so the naive version cost ~21s on the Premier League
    alone, on every projection run.
    """
    def err(mu):
        return sum((_phi((mu - bar) / sigma) - p) ** 2 for bar, p in dots)

    coarse, _ = _scan(5.0, 125.0, 1.0, err)
    return _scan(max(5.0, coarse - 1.0), min(125.0, coarse + 1.0), 0.05, err)


def calibrate_sigma(all_dots, lo=4.0, hi=18.0, step=0.1):
    """The spread that best explains the whole book at once.

    A team's several markets are only mutually consistent at one spread:
    too tight and no single points total satisfies win + top-6 + relegation
    together, too wide and the fit goes slack. Sweep and take the minimum
    total residual, so nothing is hand-picked per league.

    Only teams with 2+ markets inform it — a single bar fits perfectly at
    any sigma and would just add noise.
    """
    usable = [d for d in all_dots if len(d) >= 2]
    if not usable:
        return None

    def total_err(s):
        return sum(fit_points(d, s)[1] for d in usable)

    # Coarse-then-fine, same reasoning as fit_points — the residual curve
    # measured across several leagues is a smooth single trough.
    coarse, _ = _scan(lo, hi, 0.5, total_err)
    best_s, _ = _scan(max(lo, coarse - 0.5), min(hi, coarse + 0.5), step, total_err)
    return round(best_s, 1)


async def league_constants(conn, competition_id, releg, n_teams):
    """Points bars, league mean points, and goals-per-point — all measured.

    Returns (bars, mean_points, gd_per_point, n_seasons) or None when there
    is no usable standings history.

    gd_per_point is the slope of GOAL DIFFERENCE REGRESSED ON POINTS. An
    earlier version regressed points on goal difference and inverted the
    slope, which overstates goal difference by 6-13% depending on league —
    a least-squares fit minimises error in its dependent variable only and
    cannot be read backwards.
    """
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT st.season_id, st.points, st.goals_for - st.goals_against
            FROM standings st
            JOIN seasons s ON s.id = st.season_id
            WHERE st.played >= 25 AND s.competition_id = %s
            """,
            (int(competition_id),),
        )
        rows = await cur.fetchall()
    if not rows:
        return None

    by_season = {}
    for season_id, points, gd in rows:
        by_season.setdefault(int(season_id), []).append((float(points), float(gd)))
    seasons = {k: v for k, v in by_season.items() if len(v) >= 8}
    if not seasons:
        return None

    all_points = [p for v in seasons.values() for p, _ in v]
    mean_points = sum(all_points) / len(all_points)

    pairs = [(p, g) for v in seasons.values() for p, g in v]
    mean_p = sum(p for p, _ in pairs) / len(pairs)
    mean_g = sum(g for _, g in pairs) / len(pairs)
    sxx = sum((p - mean_p) ** 2 for p, _ in pairs)
    gd_per_point = (sum((p - mean_p) * (g - mean_g) for p, g in pairs) / sxx) if sxx else 1.0

    def bar(position):
        vals = [sorted([p for p, _ in v], reverse=True)[position - 1]
                for v in seasons.values() if len(v) >= position]
        return statistics.median(vals) if vals else None

    bars = {}
    for market_id, (pos, _) in MARKET_POS.items():
        if pos == 'half':
            k = n_teams // 2
        elif pos == 'survive':
            k = n_teams - releg
        elif pos == 'bottom':
            k = n_teams - 1
        else:
            k = pos
        bars[market_id] = bar(k) if 1 <= k <= n_teams else None
    return bars, mean_points, gd_per_point, len(seasons)


async def implied_points(conn, competition_id, season_id, bars, team_ids,
                         n_teams, releg, now=None):
    """Outright prices -> one expected points total per team.

    Returns (points_by_team_id, n_markets, sigma, pivot). `pivot` is the
    mean of the fitted totals; subtracting it is what makes the league's
    goal difference sum to zero, which is an arithmetic identity — every
    goal scored is conceded by somebody.
    """
    odds_df = await ts.load_outright_odds(conn, competition_id, season_id)
    if odds_df is None or odds_df.empty:
        return {}, 0, None, None
    now = now or pd.Timestamp.utcnow().tz_localize(None)

    dots = {int(t): [] for t in team_ids}
    used = []
    for market_id, (pos, anchor) in MARKET_POS.items():
        if bars.get(market_id) is None:
            continue
        rows = ts._pick_provider_rows(odds_df, market_id, list(dots), now)
        if rows is None or rows.empty:
            continue
        # One price across the whole book means the market was never
        # actually priced — a placeholder, not a 24-way coin flip.
        if rows['odd_decimal'].nunique() <= 1 and len(rows) > 3:
            continue
        band_key = BAND[market_id]
        band = (n_teams // 2 if band_key == 'half'
                else releg if band_key == 'releg' else band_key)
        if not band:
            continue
        probs = power_devig(rows['odd_decimal'].astype(float).tolist(), band)
        used.append(f"{market_id}@{bars[market_id]:.0f}({rows['provider'].iloc[0]})")
        for team_id, p in zip(rows['team_id'].astype(int), probs):
            if team_id not in dots:
                continue
            # Bottom-anchored markets carry a real but binary tail for a
            # few named clubs (pending points deductions). Excluded here
            # and modelled as a points penalty in the sim instead.
            if anchor == 'b' and team_id in ts.NO_TAIL_TEAM_IDS:
                continue
            p_above = p if anchor == 't' else 1.0 - p
            dots[team_id].append((bars[market_id], min(max(p_above, 1e-6), 1 - 1e-6)))

    sigma = calibrate_sigma([d for d in dots.values() if d])
    if sigma is None:
        return {}, len(used), None, None
    points = {t: (fit_points(d, sigma)[0] if d else None) for t, d in dots.items()}
    priced = [v for v in points.values() if v is not None]
    if not priced:
        return {}, len(used), sigma, None
    pivot = sum(priced) / len(priced)
    logger.info("  odds: %d markets [%s], sigma %.1f, %d/%d teams priced, fitted mean %.1f",
                len(used), ", ".join(used), sigma, len(priced), len(dots), pivot)
    return points, len(used), sigma, pivot


async def implied_points_partial(conn, competition_id, season_id, bars, team_ids,
                                 n_teams, releg, now=None):
    """Expected points from a THIN book, using only the informative prices.

    Same machinery as implied_points, with three differences:

      * a price is used only if its de-vigged probability clears
        PARTIAL_ODDS_MIN_PROB — beyond that the book is filling shelves,
        not expressing a view, and the fit cannot separate 500/1 from
        750/1;
      * sigma is fixed rather than calibrated, because a single bar per
        team fits any sigma perfectly;
      * no pivot is returned. The caller anchors the result on where form
        and squad value already place these teams, because a thin book
        knows the ORDER of the teams it prices but not where that group
        sits relative to the rest of the league.

    Returns {team_id: points} for the qualifying teams only.
    """
    odds_df = await ts.load_outright_odds(conn, competition_id, season_id)
    if odds_df is None or odds_df.empty:
        return {}
    now = now or pd.Timestamp.utcnow().tz_localize(None)

    dots = {int(t): [] for t in team_ids}
    for market_id, (pos, anchor) in MARKET_POS.items():
        if bars.get(market_id) is None:
            continue
        rows = ts._pick_provider_rows(odds_df, market_id, list(dots), now)
        if rows is None or rows.empty:
            continue
        if rows['odd_decimal'].nunique() <= 1 and len(rows) > 3:
            continue
        band_key = BAND[market_id]
        band = (n_teams // 2 if band_key == 'half'
                else releg if band_key == 'releg' else band_key)
        if not band:
            continue
        probs = power_devig(rows['odd_decimal'].astype(float).tolist(), band)
        for team_id, p in zip(rows['team_id'].astype(int), probs):
            if team_id not in dots:
                continue
            if anchor == 'b' and team_id in ts.NO_TAIL_TEAM_IDS:
                continue
            # Saturated at either end tells us only "far from this bar",
            # never how far.
            if p < PARTIAL_ODDS_MIN_PROB or p > 1.0 - PARTIAL_ODDS_MIN_PROB:
                continue
            p_above = p if anchor == 't' else 1.0 - p
            dots[team_id].append((bars[market_id], min(max(p_above, 1e-6), 1 - 1e-6)))

    return {t: fit_points(d, PARTIAL_ODDS_SIGMA)[0] for t, d in dots.items() if d}


# --- the blend -------------------------------------------------------------

async def apply_unified_ratings(conn, ratings, *, competition_id, season_id,
                                team_ids_by_name, mv_index, matches_played,
                                games_in_season, goals_per_game,
                                mv_weight_pre=None, now=None):
    """Blend form, squad value and outright odds into one rating.

    `ratings` carries Attack/Defense in RAW goal units (what _prepare_league
    holds at this point); the blend happens on the mean=100 index and is
    written back in raw units, so the caller's later rescale is unaffected.

    Returns (ratings, audit) where audit is a per-team DataFrame of the
    components in comparable units, for logging and the dry run. `ratings`
    is modified in place and also returned.
    """
    n_teams = len(ratings)
    if n_teams == 0:
        return ratings, pd.DataFrame()

    # Count the schedule rather than trusting the caller's (n_teams-1)*2.
    _counted = await games_per_team(conn, competition_id, season_id, n_teams)
    if _counted and _counted != games_in_season:
        logger.info("  schedule: %d games per team (counted), not %s (assumed) "
                    "— season is %.0f%% gone, not %.0f%%",
                    _counted, games_in_season,
                    (matches_played or 0) / _counted * 100,
                    (matches_played or 0) / games_in_season * 100 if games_in_season else 0)
        games_in_season = _counted

    attack_mean = float(pd.to_numeric(ratings['Attack'], errors='coerce').mean())
    defence_mean = float(pd.to_numeric(ratings['Defense'], errors='coerce').mean())
    if not np.isfinite(attack_mean) or not np.isfinite(defence_mean) \
            or attack_mean == 0 or defence_mean == 0:
        logger.warning("  unified blend skipped — attack/defence mean is not usable")
        return ratings, pd.DataFrame()

    a_idx = pd.to_numeric(ratings['Attack'], errors='coerce') / attack_mean * 100.0
    d_idx = pd.to_numeric(ratings['Defense'], errors='coerce') / defence_mean * 100.0
    form_overall = (a_idx - d_idx) / 2.0
    spread = float((a_idx.std(ddof=0) + d_idx.std(ddof=0)) / 2.0)
    team_index = {t: i for i, t in enumerate(ratings['Team'])}

    releg = relegation_places(competition_id)
    consts = await league_constants(conn, competition_id, releg, n_teams)
    odds_overall = {}
    odds_is_partial = False
    if consts is None:
        logger.info("  no standings history for competition %s — odds component unavailable",
                    competition_id)
    else:
        bars, mean_points, gd_per_point, n_seasons = consts
        ids = [int(v) for v in team_ids_by_name.values() if v is not None]
        points, n_markets, _sigma, pivot = await implied_points(
            conn, competition_id, season_id, bars, ids, n_teams, releg, now=now)
        if n_markets < MIN_ODDS_MARKETS or pivot is None:
            # Thin book: keep only the prices that are genuinely opinions.
            partial = await implied_points_partial(
                conn, competition_id, season_id, bars, ids, n_teams, releg, now=now)
            if not partial:
                logger.info("  odds component DROPPED — %d usable market(s), "
                            "none clearing the %.1f%% floor",
                            n_markets, PARTIAL_ODDS_MIN_PROB * 100)
            else:
                n_games = max(1, (n_teams - 1) * 2)
                for name, team_id in team_ids_by_name.items():
                    if team_id is None:
                        continue
                    pts = partial.get(int(team_id))
                    if pts is None:
                        continue
                    gd_per_match = (pts - mean_points) * gd_per_point / n_games
                    odds_overall[name] = (gd_per_match / 2.0) / goals_per_game * 100.0
                odds_is_partial = True
                logger.info("  odds component PARTIAL — %d market(s), %d of %d teams "
                            "clear the %.1f%% floor",
                            n_markets, len(odds_overall), n_teams,
                            PARTIAL_ODDS_MIN_PROB * 100)
        else:
            logger.info("  league constants from %d seasons: mean pts %.1f, GD per point %.3f",
                        n_seasons, mean_points, gd_per_point)
            n_games = max(1, (n_teams - 1) * 2)
            for name, team_id in team_ids_by_name.items():
                if team_id is None:
                    continue
                pts = points.get(int(team_id))
                if pts is None:
                    continue
                # Pivot on the FITTED mean so the league's goal difference
                # sums to zero by construction. Only gaps reach the rating,
                # so sliding the whole column changes nothing.
                gd_per_match = (pts - pivot) * gd_per_point / n_games
                # Half the goal difference to each end, then express as
                # index points above average — the same units as form.
                odds_overall[name] = (gd_per_match / 2.0) / goals_per_game * 100.0

    # Squad value: a z-score, soft-capped, sized by the form frame's spread.
    mv_overall = {}
    if mv_index:
        vals = {t: mv_index.get(t) for t in ratings['Team']}
        clean = {t: float(v) for t, v in vals.items()
                 if v is not None and np.isfinite(pd.to_numeric(v, errors='coerce'))}
        if len(clean) >= 3:
            mu = sum(clean.values()) / len(clean)
            sd = math.sqrt(sum((v - mu) ** 2 for v in clean.values()) / len(clean))
            if sd > 0:
                for t, v in clean.items():
                    z = (v - mu) / sd
                    z = MV_SOFT_CAP * math.tanh(z / MV_SOFT_CAP)
                    mv_overall[t] = z * spread

    w_form, w_mv, w_odds = blend_weights(matches_played, games_in_season,
                                         mv_pre=mv_weight_pre)

    if odds_is_partial and odds_overall:
        # Anchor the thin-book component on where form and squad value
        # already put these same teams.
        #
        # The fit is pivoted on the league's HISTORICAL mean points, which
        # assumes the market can tell you where the mean is. A lone title
        # market cannot: with the bar around 88 points and sigma 12, any
        # non-zero title chance implies an above-average total, so every
        # priced team floats upward and the rest of the league is pushed
        # down to compensate. Liga Portugal prices 12 of 18 teams and
        # drifted +4.9 points a team that way, lifting a 101/1 shot by 12.8
        # while taxing Benfica 5.8 to pay for it (2026-08-04).
        #
        # A thin book knows the ORDER of the teams it prices, not where that
        # group belongs. Form and squad value know the latter. After this,
        # teams the market never mentioned do not move at all.
        priced = list(odds_overall)
        odds_mu = sum(odds_overall.values()) / len(priced)
        base_mu = 0.0
        for t in priced:
            i = team_index.get(t)
            f = float(form_overall.iloc[i]) if i is not None else 0.0
            m = mv_overall.get(t)
            wf, wm = w_form, (w_mv if m is not None else 0.0)
            base_mu += (wf * f + wm * (m or 0.0)) / (wf + wm)
        base_mu /= len(priced)
        shift = base_mu - odds_mu
        odds_overall = {k: v + shift for k, v in odds_overall.items()}
        logger.info("  partial odds anchored: shifted %+.1f onto the form+MV level "
                    "of the %d priced team(s)", shift, len(priced))
    logger.info("  unified weights: form %.2f / MV %.2f / odds %.2f (%s of %s matches played)",
                w_form, w_mv, w_odds, matches_played, games_in_season)

    audit = []
    new_attack, new_defence = [], []
    for i, team in enumerate(ratings['Team']):
        f = float(form_overall.iloc[i]) if np.isfinite(form_overall.iloc[i]) else 0.0
        m = mv_overall.get(team)
        o = odds_overall.get(team)

        # A missing component hands its weight to FORM, never to the others.
        #
        # Proportional renormalising looks natural but misbehaves: 9 of the
        # 21 leagues we project carry only a title market, so the odds
        # component drops out, and rescaling would promote squad value from
        # 0.30 to 0.50 — putting half the rating on the least trustworthy
        # input precisely where we have least corroboration. Squad value is
        # also over-dispersed in several of those leagues (Serie A spread
        # 32.1 against form's 20.5), so it would arrive amplified as well.
        # Form is measured from matches actually played, so it is the right
        # place for the slack to land. (George, 2026-08-03.)
        wf, wm, wo = w_form, w_mv, w_odds
        if o is None:
            wf, wo = wf + wo, 0.0
        if m is None:
            wf, wm = wf + wm, 0.0
        total_w = wf + wm + wo
        blended = ((wf * f + wm * (m or 0.0) + wo * (o or 0.0)) / total_w) if total_w > 0 else f
        delta = blended - f
        new_attack.append((float(a_idx.iloc[i]) + delta) * attack_mean / 100.0)
        new_defence.append((float(d_idx.iloc[i]) - delta) * defence_mean / 100.0)
        audit.append({
            'Team': team, 'form': f, 'mv': m, 'odds': o,
            'blended': blended, 'delta': delta,
            'w_form': wf, 'w_mv': wm, 'w_odds': wo,
        })

    ratings['Attack'] = new_attack
    ratings['Defense'] = new_defence
    for col, mean_val in (('Attack', attack_mean), ('Defense', defence_mean)):
        ratings[col] = ratings[col].clip(lower=RATING_FLOOR_FRACTION * mean_val)

    audit_df = pd.DataFrame(audit)
    n_mv = int(audit_df['mv'].notna().sum()) if not audit_df.empty else 0
    n_odds = int(audit_df['odds'].notna().sum()) if not audit_df.empty else 0
    logger.info("  unified blend applied to %d teams (MV for %d, odds for %d)",
                n_teams, n_mv, n_odds)
    return ratings, audit_df
