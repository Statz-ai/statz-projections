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

# Global kill switch. Per-competition control is the caller's business.
UNIFIED_ENABLED = os.getenv("UNIFIED_RATINGS", "1") not in ("0", "false", "False")

# --- weight schedule -------------------------------------------------------
# p = fraction of the season played.
#
#   odds  0.40 -> 0.30   (never below 0.30: the book keeps repricing all
#                         season, so it never becomes uninformative)
#   MV    0.30 -> 0.00   (gone by three-quarter distance — squad value is a
#                         PRE-season prior, and once there is real evidence
#                         it is just a proxy for what form now measures)
#   form  the remainder, 0.30 -> 0.70
#
# George's rule: market value heaviest pre-season and fading, form lightest
# pre-season and growing. One schedule for every league — no per-league
# tuning, which is what mv_beta had turned into.
W_ODDS_PRE = float(os.getenv("UNIFIED_W_ODDS_PRE", "0.40"))
W_ODDS_MIN = float(os.getenv("UNIFIED_W_ODDS_MIN", "0.30"))
W_MV_PRE = float(os.getenv("UNIFIED_W_MV_PRE", "0.30"))
MV_FADE_BY = float(os.getenv("UNIFIED_MV_FADE_BY", "0.75"))

# Squad value is heavy-tailed — one oligarch club sits several SDs clear and
# would otherwise drag the whole component. tanh compresses the extremes
# while leaving the middle of the table essentially linear.
MV_SOFT_CAP = float(os.getenv("UNIFIED_MV_SOFT_CAP", "2.5"))

# Below this many usable outright markets the odds component is dropped
# rather than trusted. A lone title market is one point at position 1: it
# says a great deal about the favourite and almost nothing about 14th.
MIN_ODDS_MARKETS = int(os.getenv("UNIFIED_MIN_ODDS_MARKETS", "3"))

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


def blend_weights(matches_played, games_in_season):
    """(form, MV, odds) weights at this point in the season."""
    if not games_in_season or games_in_season <= 0:
        p = 0.0
    else:
        p = min(1.0, max(0.0, float(matches_played) / float(games_in_season)))
    w_odds = max(W_ODDS_MIN, W_ODDS_PRE - (W_ODDS_PRE - W_ODDS_MIN) * (p / MV_FADE_BY)) \
        if MV_FADE_BY > 0 else W_ODDS_MIN
    w_odds = max(W_ODDS_MIN, min(W_ODDS_PRE, w_odds))
    w_mv = W_MV_PRE * max(0.0, 1.0 - (p / MV_FADE_BY)) if MV_FADE_BY > 0 else 0.0
    w_form = max(0.0, 1.0 - w_odds - w_mv)
    return w_form, w_mv, w_odds


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


def fit_points(dots, sigma):
    """The points total best explaining one team's markets, and its residual.

    Each market is read as P(this team's points exceed this bar), modelled
    as a normal with the given spread. Least squares over every bar the
    team is priced in.
    """
    best, best_err, mu = None, 1e18, 5.0
    while mu <= 125.0:
        err = sum((_phi((mu - bar) / sigma) - p) ** 2 for bar, p in dots)
        if err < best_err:
            best_err, best = err, mu
        mu += 0.05
    return best, best_err


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
    best_s, best_err, s = None, 1e18, lo
    while s <= hi:
        total = sum(fit_points(d, s)[1] for d in usable)
        if total < best_err:
            best_err, best_s = total, s
        s += step
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


# --- the blend -------------------------------------------------------------

async def apply_unified_ratings(conn, ratings, *, competition_id, season_id,
                                team_ids_by_name, mv_index, matches_played,
                                games_in_season, goals_per_game, now=None):
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

    releg = relegation_places(competition_id)
    consts = await league_constants(conn, competition_id, releg, n_teams)
    odds_overall = {}
    if consts is None:
        logger.info("  no standings history for competition %s — odds component unavailable",
                    competition_id)
    else:
        bars, mean_points, gd_per_point, n_seasons = consts
        ids = [int(v) for v in team_ids_by_name.values() if v is not None]
        points, n_markets, _sigma, pivot = await implied_points(
            conn, competition_id, season_id, bars, ids, n_teams, releg, now=now)
        if n_markets < MIN_ODDS_MARKETS or pivot is None:
            logger.info("  odds component DROPPED — %d usable market(s), need %d",
                        n_markets, MIN_ODDS_MARKETS)
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

    w_form, w_mv, w_odds = blend_weights(matches_played, games_in_season)
    logger.info("  unified weights: form %.2f / MV %.2f / odds %.2f (%s of %s matches played)",
                w_form, w_mv, w_odds, matches_played, games_in_season)

    audit = []
    new_attack, new_defence = [], []
    for i, team in enumerate(ratings['Team']):
        f = float(form_overall.iloc[i]) if np.isfinite(form_overall.iloc[i]) else 0.0
        parts = [(w_form, f)]
        m = mv_overall.get(team)
        if m is not None:
            parts.append((w_mv, m))
        o = odds_overall.get(team)
        if o is not None:
            parts.append((w_odds, o))
        # A team missing a component keeps the others in proportion rather
        # than being dragged toward the mean by a zeroed input.
        total_w = sum(w for w, _ in parts)
        blended = (sum(w * v for w, v in parts) / total_w) if total_w > 0 else f
        delta = blended - f
        new_attack.append((float(a_idx.iloc[i]) + delta) * attack_mean / 100.0)
        new_defence.append((float(d_idx.iloc[i]) - delta) * defence_mean / 100.0)
        audit.append({
            'Team': team, 'form': f, 'mv': m, 'odds': o,
            'blended': blended, 'delta': delta,
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
