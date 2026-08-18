"""Team attack/defence strengths inverted from bookmaker fixture odds.

Bookmaker knowledge reaches a rating through outright markets, which decay:
forward strength is (implied final points - banked) / games remaining, so a
fixed error on the implied finish is ~3% at matchweek 5 and ~17% by matchweek
30. Books reprice outrights weekly — it is a precision problem, not staleness.

Match odds have neither problem. Every team is priced every week, all season.
So invert them:

    ln lambda_home = c_home + ln(attack_home) + ln(defence_away)
    ln lambda_away = c_away + ln(attack_away) + ln(defence_home)

Linear in logs, therefore least squares. Identified by fixing
mean(ln attack) = mean(ln defence) = 0, leaving c_home / c_away to carry the
level and the home advantage.

Validated on a full Championship season (555 fixtures, 24 teams): recovered
attack correlated +0.83 with actual goals scored and defence +0.81 with goals
conceded, from prices alone, without seeing a result. Home advantage came out
c_home 0.301 against c_away 0.041.

Two callers will want this, for different reasons:

  - DEEP-OUTRIGHT leagues hand over from outrights around mid-season, because
    the outright signal is decaying.
  - THIN-OUTRIGHT leagues switch as soon as the fit identifies, because their
    outright signal was never good: live runs show competitions on the partial
    path with ONE market pricing 5 of 18 teams. An uneven signal across a
    third of the field is worse than none.

See statz/docs/fixture-derived-strengths-spec.md.
"""
import logging
import math

logger = logging.getLogger("fixture_strengths")

# Recency half-life in days. THE ONE PARAMETER NOT SET BY GEORGE — flagged for
# review.
#
# Calibrating it out-of-sample on one league-season put the optimum near 7
# days, but moved likelihood by 0.011 per fixture, which is nothing, and a
# 7-day half-life leaves a fixture 9 rounds back on 0.2% weight — which would
# collapse the window to ~2 rounds and destroy the connectivity the fit needs
# to compare teams at all.
#
# 21 days keeps genuine recency (a month-old price is worth a third of
# today's) while a 10-round-old fixture still carries ~10%, enough to hold the
# who-played-whom graph together.
HALF_LIFE_DAYS = 21.0

# Rolling window, in rounds. Below ~5 the graph is too sparse to compare
# teams; 2 rounds is the bare algebraic minimum (48 unknowns in a 24-team
# league, 24 equations per round) and badly conditioned. 10 survives a blank
# midweek and still reflects a January signing within weeks.
WINDOW_ROUNDS = 10

# Below this many usable fixtures the fit is not trustworthy and the caller
# should keep whatever market source it was using.
MIN_FIXTURES_PER_TEAM = 4

_FIT_ITERATIONS = 200


def power_devig_1x2(home_odds, draw_odds, away_odds):
    """Margin-stripped (p_h, p_d, p_a) by the power method.

    p_i = (1/o_i)^k, k solved so they sum to 1.

    Deliberately NOT the proportional divide-by-overround used elsewhere in
    the pipeline. Bookmakers load more margin onto longshots, so removing it
    uniformly leaves longshots overstated. Measured across 263 live bet365
    prices, the power method moves +2.3pp onto favourites and -1.4pp off
    longshots. New code, so it starts correct rather than inheriting the
    older method.
    """
    if not (home_odds and draw_odds and away_odds):
        return None
    if min(home_odds, draw_odds, away_odds) <= 1.0:
        return None
    raw = [1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds]
    if sum(raw) <= 1.0:
        return tuple(raw)                      # no margin to strip
    lo, hi = 1.0, 6.0
    for _ in range(80):
        k = (lo + hi) / 2.0
        if sum(r ** k for r in raw) > 1.0:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2.0
    return tuple(r ** k for r in raw)


def fit_strengths(observations, weights, n_teams, iterations=_FIT_ITERATIONS):
    """Weighted least squares for attack/defence in log space.

    observations: list of (home_idx, away_idx, ln_lambda_home, ln_lambda_away)
    weights:      one non-negative weight per observation

    Returns (attack, defence, c_home, c_away) with attack/defence as
    MULTIPLICATIVE factors averaging 1 — the units the caller wants.

    Alternating least squares rather than a matrix solve: it is a convex
    quadratic so coordinate descent converges, it needs no numpy linear
    algebra, and it degrades gracefully when a team is under-represented
    (its estimate simply stays near the mean).
    """
    alpha = [0.0] * n_teams
    delta = [0.0] * n_teams
    c_home = c_away = 0.0
    total_w = sum(weights)
    if total_w <= 0:
        return None

    for _ in range(iterations):
        c_home = sum(w * (lh - alpha[h] - delta[a])
                     for (h, a, lh, _la), w in zip(observations, weights)) / total_w
        c_away = sum(w * (la - alpha[a] - delta[h])
                     for (h, a, _lh, la), w in zip(observations, weights)) / total_w

        num = [0.0] * n_teams
        den = [0.0] * n_teams
        for (h, a, lh, la), w in zip(observations, weights):
            num[h] += w * (lh - c_home - delta[a]); den[h] += w
            num[a] += w * (la - c_away - delta[h]); den[a] += w
        alpha = [num[i] / den[i] if den[i] else 0.0 for i in range(n_teams)]
        mean_a = sum(alpha) / n_teams
        alpha = [x - mean_a for x in alpha]

        num = [0.0] * n_teams
        den = [0.0] * n_teams
        for (h, a, lh, la), w in zip(observations, weights):
            num[a] += w * (lh - c_home - alpha[h]); den[a] += w
            num[h] += w * (la - c_away - alpha[a]); den[h] += w
        delta = [num[i] / den[i] if den[i] else 0.0 for i in range(n_teams)]
        mean_d = sum(delta) / n_teams
        delta = [x - mean_d for x in delta]

    return ([math.exp(x) for x in alpha],
            [math.exp(x) for x in delta],
            c_home, c_away)


def to_overall_index(attack, defence):
    """Multiplicative factors -> the 'overall' index the unified blend uses.

    The blend's odds component emits
        overall = (gd_per_match / 2) / goals_per_game * 100

    Expected goals for a team against an average opponent is
    L * attack, and against it L * defence, where L is the league's goals per
    team per game. So gd_per_match = L * (attack - defence), and since the
    blend divides by goals_per_game which is that same L, the L cancels:

        overall = 50 * (attack - defence)

    Which is why this needs no league-average argument.
    """
    return 50.0 * (attack - defence)


async def load_1x2_for_fixtures(conn, fixture_ids, books=None):
    """{fixture_id: (p_home, p_draw, p_away)} — margin stripped, first book
    in priority order that priced the fixture.

    Cascades for the same reason the goals ladders do: no single book prices
    every fixture, and a fixture we cannot price is a fixture that drops out
    of the fit entirely. bet365 leads (house rule) and the rest fill gaps.
    """
    from app.services.odds_blend import GOALS_BOOKIE_PRIORITY
    books = books or GOALS_BOOKIE_PRIORITY
    if not fixture_ids:
        return {}

    ph = ",".join(["%s"] * len(fixture_ids))
    out = {}
    for book in books:
        missing = [f for f in fixture_ids if f not in out]
        if not missing:
            break
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT fixture_id, home_win_odd, draw_odd, away_win_odd
                    FROM {book}_fixture_odds
                    WHERE fixture_id IN ({ph})
                      AND home_win_odd > 1 AND draw_odd > 1 AND away_win_odd > 1
                    """,
                    tuple(fixture_ids),
                )
                rows = await cur.fetchall()
        except Exception as err:
            logger.debug("1X2 cascade: %s unavailable (%s)", book, err)
            continue
        for fid, h, d, a in rows:
            if int(fid) in out:
                continue
            try:
                out[int(fid)] = power_devig_1x2(float(h), float(d), float(a))
            except Exception:
                continue
    return out


async def derive_fixture_strengths(conn, competition_id, season_id, n_teams,
                                   now=None):
    """Invert recent fixture odds into per-team attack/defence strength.

    Returns {team_id: overall_index} on the same units as the unified blend's
    odds component, or None when the market data cannot identify a fit.

    The split comes from the OVER/UNDER goals ladders, not from the 1X2. A
    1X2 price fixes the supremacy between two sides but barely constrains how
    many goals the game holds, so inverting it alone leaves the total sitting
    wherever the solver started — every team comes out with attack and
    defence as near mirror images and the "split" carries no information.
    The goals ladder is what says whether 1-0 or 3-2; only with it does
    attack separate from defence.

    So every fixture here is priced through derive_bookie_lambdas, which
    cascades the five books and prefers, in order: both per-team ladders,
    one per-team ladder plus the match total, then the match total split by
    the 1X2-implied supremacy. Its last path splits a total by the MODEL
    ratio — that one is deliberately unreachable here, because we always
    hand it a 1X2, and a strength fitted from our own model would be
    circular. Fixtures with no goals ladder are dropped rather than guessed.

    Never raises: a failure leaves the caller on its existing market source.
    """
    from datetime import datetime

    if not competition_id or not season_id or not n_teams:
        return None

    now = now or datetime.utcnow()
    window_fixtures = max(1, int(WINDOW_ROUNDS * n_teams / 2))

    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT f.id, f.kickoff_datetime, f.home_team_id, f.away_team_id
                FROM fixtures f
                WHERE f.competition_id = %s AND f.season_id = %s
                  AND f.state_id = 5
                ORDER BY f.kickoff_datetime DESC
                LIMIT %s
                """,
                (int(competition_id), int(season_id), window_fixtures),
            )
            rows = await cur.fetchall()
    except Exception as err:
        logger.warning("fixture strengths: query failed (%s) — skipping", err)
        return None

    if not rows:
        return None

    from app.services.odds_blend import derive_bookie_lambdas, load_goals_odds_for_fixtures

    fixture_ids = [int(r[0]) for r in rows]
    try:
        goals_odds_map = await load_goals_odds_for_fixtures(conn, fixture_ids)
        probs_map = await load_1x2_for_fixtures(conn, fixture_ids)
    except Exception as err:
        logger.warning("fixture strengths: odds load failed (%s) — skipping", err)
        return None

    team_ids = sorted({int(r[2]) for r in rows} | {int(r[3]) for r in rows})
    idx = {t: i for i, t in enumerate(team_ids)}

    observations, weights = [], []
    appearances = {t: 0 for t in team_ids}
    no_ladder = 0
    for fid, ko, home_id, away_id in rows:
        fid = int(fid)
        ladders = goals_odds_map.get(fid, {})
        if not ladders:
            no_ladder += 1
            continue
        # Model lambdas are placeholders only: with a 1X2 in hand the model
        # path is never reached, and without one we would rather drop the
        # fixture than fit our own output back onto itself.
        probs = probs_map.get(fid)
        if probs is None:
            no_ladder += 1
            continue
        lambdas = derive_bookie_lambdas(fid, 1.0, 1.0, probs, ladders)
        if lambdas is None:
            no_ladder += 1
            continue
        lh, la = lambdas
        if lh <= 0 or la <= 0:
            continue

        days_ago = max(0.0, (now - ko).total_seconds() / 86400.0) if hasattr(ko, 'year') else 0.0
        observations.append((idx[int(home_id)], idx[int(away_id)],
                             math.log(lh), math.log(la)))
        weights.append(0.5 ** (days_ago / HALF_LIFE_DAYS))
        appearances[int(home_id)] += 1
        appearances[int(away_id)] += 1

    thin = [t for t, n in appearances.items() if n < MIN_FIXTURES_PER_TEAM]
    if len(observations) < n_teams or thin:
        logger.info(
            "  fixture strengths not identifiable yet — %d/%d fixtures priced "
            "(%d without a usable goals ladder), %d team(s) under %d appearances",
            len(observations), len(rows), no_ladder, len(thin), MIN_FIXTURES_PER_TEAM)
        return None

    fitted = fit_strengths(observations, weights, len(team_ids))
    if fitted is None:
        return None
    attack, defence, c_home, c_away = fitted

    out = {t: to_overall_index(attack[idx[t]], defence[idx[t]]) for t in team_ids}
    logger.info(
        "  fixture strengths fitted from %d/%d fixtures (%d teams, half-life %.0fd, "
        "home %.3f away %.3f)",
        len(observations), len(rows), len(team_ids), HALF_LIFE_DAYS, c_home, c_away)
    return out
