"""Bookmaker odds blending for goal projections.

Reels the model's expected goals (λ_h, λ_a) toward bookie expected goals
when bookie totals markets are priced. Replaces the older blend-in-
probability-space + 2D reverse-solve approach with a cleaner cascade
that exploits the goals over/under markets directly.

Cascade per fixture, per bookmaker (bet365 first, others as fallback):

  Path 1  per-team goals O/U ladders → fit each side's λ independently
  Path 1.5 single-team ladder + match total → derive missing side by subtraction
  Path 2  match total + 1X2 → fit match λ, reverse-solve share to match 1X2
  Path 3  match total only → fit match λ, split via model's per-team ratio
  Path 4  1X2 only → caller's existing reverse-solve (not handled here)
  Path 5  nothing → return None; caller leaves model unchanged

The caller blends our returned (λ_h_bookie, λ_a_bookie) with their model
output at the service-specific blend weight:

    λ_h_final = (1-w) * λ_h_model + w * λ_h_bookie

If we return None, the caller falls back to its existing 1X2-only path.
"""

import math
from typing import Optional, Tuple

import logging
logger = logging.getLogger("odds_blend")

# Per-book priority for the goals-blend cascade. bet365 always first
# (it covers ~50K upcoming fixtures vs 2-3K for each fallback), but the
# fallback books DO carry meaningful coverage for fixtures bet365 is
# dark on — primarily international friendlies and the lower divisions
# of La Liga / Serie B / Eredivisie etc.. derive_bookie_lambdas
# iterates this list and falls through book-by-book per fixture.
GOALS_BOOKIE_PRIORITY = ['bet365', 'coral', 'ladbrokes', 'midnite', 'boylesports']

# Per-stat bookmaker priority for team-stat blending. bet365 always
# tried first per user rule (2026-05-29); rest ordered by per-team
# coverage observed on UCL final. Maps the bookie market name (column
# `market` in *_totals_odds) to the priority list.
TEAM_STAT_BOOKIE_PRIORITY = {
    'corners': ['bet365', 'midnite', 'boylesports', 'coral', 'ladbrokes'],
    'cards':   ['bet365', 'midnite', 'boylesports', 'coral'],
    'shots':   ['midnite', 'boylesports', 'coral'],     # no bet365 coverage
    'sot':     ['midnite', 'boylesports', 'coral'],     # no bet365 coverage
    'fouls':   ['midnite'],                              # midnite only
    'tackles': ['boylesports'],                          # boyle only
}

# Mapping from the team_projections column (statz internal stat name)
# to the bookie market key in TEAM_STAT_BOOKIE_PRIORITY.
STAT_COLUMN_TO_MARKET = {
    'Corners':         'corners',
    'Yellowcards':     'cards',
    'Shots Total':     'shots',
    'Shots On Target': 'sot',
    'Fouls':           'fouls',
    'Tackles':         'tackles',
}


# Per-stat bookmaker priority for PLAYER-PROP blending. bet365 always
# leads regardless of aggregate coverage — cascade falls through per
# (player, fixture) row when bet365 has no quote. Rest ordered by
# observed coverage. Keys are stats_types.id (52=Goals, 42=Shots Total,
# 86=Shots On Target). v1 is these three; extend in v1.5 with Tackles
# (78), Fouls (56), Fouls Drawn (96), Yellow Cards (84), Assists (79).
# Blended player-prop markets, all at the flat PLAYER_ODDS_BLEND_WEIGHT (0.3).
# Passes deliberately excluded (thin market). NOTE: these are still over-only
# with no de-margin, so the λ's run hot (worst on short-priced fouls/tackles)
# — accepted at the low 0.3 weight; tighten via the de-margin (backlog).
PLAYER_STAT_BOOKIE_PRIORITY = {
    52: ['bet365', 'coral', 'ladbrokes', 'midnite', 'boylesports'],  # Goals
    42: ['bet365', 'midnite', 'coral', 'ladbrokes', 'boylesports'],  # Shots Total
    86: ['bet365', 'midnite', 'coral', 'ladbrokes', 'boylesports'],  # Shots On Target
    79: ['bet365', 'coral', 'ladbrokes', 'midnite', 'boylesports'],  # Assists (bet365 = "Or Their Substitute" variant)
    56: ['bet365', 'midnite', 'coral', 'ladbrokes', 'boylesports'],  # Fouls (committed)
    96: ['bet365', 'coral', 'ladbrokes', 'midnite', 'boylesports'],  # Fouls Drawn (To Be Fouled)
    78: ['bet365', 'midnite', 'coral', 'ladbrokes', 'boylesports'],  # Tackles
    84: ['bet365', 'coral', 'ladbrokes', 'midnite', 'boylesports'],  # Yellow Cards (booked)
}

# Derived constants — callers import these directly rather than
# recomputing the union per call site.
PLAYER_BLEND_STAT_IDS = list(PLAYER_STAT_BOOKIE_PRIORITY.keys())
PLAYER_BLEND_BOOKS = sorted({b for lst in PLAYER_STAT_BOOKIE_PRIORITY.values() for b in lst})

# DataFrame column / row-dict key → stats_type_id for the v1 blend.
# Used by callers (statz_functions.distribute_team_predictions_to_players
# and wc_player_stat_service._build_player_rows) to translate the stat
# name they're iterating over to the integer key in
# PLAYER_STAT_BOOKIE_PRIORITY. v1.5+ stats added here propagate to both
# call sites automatically.
PLAYER_BLEND_STAT_NAMES = {
    'Goals':           52,
    'Shots Total':     42,
    'Shots On Target': 86,
    'Assists':         79,
    'Fouls':           56,
    'Fouls Drawn':     96,
    'Tackles':         78,
    'Yellow Cards':    84,
}


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    # Log-space to avoid overflow on factorial
    logp = -lam + k * math.log(lam)
    for i in range(2, k + 1):
        logp -= math.log(i)
    return math.exp(logp)


def _poisson_p_geq(k: int, lam: float) -> float:
    s = 0.0
    for i in range(k):
        s += _poisson_pmf(i, lam)
    return 1.0 - s


def _margin_stripped_over_prob(over_price: Optional[float], under_price: Optional[float]) -> Optional[float]:
    """Return margin-stripped P(over) if BOTH sides priced; raw 1/over if
    only over is priced; 1 - raw 1/under if only under is priced; None
    if neither. Single-side fallback inflates λ slightly (~5%) but is
    still much better than nothing for fixtures where bet365 prices only
    one side of a line.
    """
    if over_price and under_price:
        po = 1.0 / over_price
        pu = 1.0 / under_price
        return po / (po + pu)
    if over_price:
        return 1.0 / over_price
    if under_price:
        return 1.0 - 1.0 / under_price
    return None


def fit_lambda_from_ladder(
    ladder: list,
    prob_filter: Tuple[float, float] = (0.10, 0.90),
) -> Optional[float]:
    """Fit a single Poisson λ to a list of over/under lines via
    squared-error MLE in probability space.

    ladder = [(line: float, over_price: float|None, under_price: float|None), ...]

    For each usable line, compute margin-stripped P(X > floor(line)) and
    minimise the sum of squared errors over all lines:

        λ* = argmin Σ_i (P_poisson(X ≥ k_i, λ) - p_i_bookie)^2

    This automatically down-weights extreme lines: their margin-stripped
    probabilities are tiny and barely change with λ, so they contribute
    little to the fit. Near-50/50 lines, by contrast, are highly
    λ-sensitive and dominate. No special tier logic needed.

    Stub-line rejection is band-relative via prob_filter, not a fixed
    price cutoff: the (lo, hi) bounds on the derived over-probability
    implicitly reject prices outside [1/hi, 1/lo]. Default (0.10, 0.90)
    means prices ≤ ~1.11 and ≥ 10 are dropped — that's where bookmaker
    no-take stubs cluster on deep team-stat ladders. Player-prop callers
    pass (0.02, 0.90) → keeps prices up to ~50 (a legitimate "3+ goals
    for a striker" rung) while still rejecting ≤ ~1.11 near-cert stubs.

    prob_filter: (lo, hi) bounds on the derived over-probability. Lines
    outside the band are dropped. Default (0.10, 0.90) for team-stat
    ladders (corners 8.5/9.5/10.5/.. , cards 3.5/4.5/.. — deep enough
    that the band has plenty of rungs left). Player-prop callers pass
    (0.02, 0.90) because typical player ladders are shallow (1+/2+/3+),
    so even a 0.04-probability rung carries meaningful signal.

    Returns None if no usable line.
    """
    if not ladder:
        return None

    lo, hi = prob_filter

    # Build the (k, p_over) pairs to fit. Drop lines where the
    # margin-stripped over-probability is outside [lo, hi] —
    # outside that band three things go wrong simultaneously:
    #   1. The Poisson tail is insensitive to λ → line carries no
    #      real info anyway (squared error stays near 0 for any λ).
    #   2. Bookmaker margins skew asymmetrically at the extremes
    #      (long-shots carry wider margin than the lay side), so the
    #      "margin-stripped" probability isn't actually stripped clean.
    #   3. Most extreme-price lines are bookmaker stubs ("won't take
    #      this bet" placeholder prices) rather than real markets.
    # Filtering on the derived probability subsumes both "over @ 20.0"
    # (P_over ~0.05) and "over @ 1.05" (P_over ~0.95) without
    # hardcoding magic price thresholds. Works the same for over-only
    # ladders (single-side raw 1/price still goes through the filter).
    rows = []
    max_line_kept = 0.0
    for line, over_price, under_price in ladder:
        p_over = _margin_stripped_over_prob(over_price, under_price)
        if p_over is None:
            continue
        if p_over < lo or p_over > hi:
            continue
        k = int(math.floor(line)) + 1
        rows.append((k, p_over))
        if line > max_line_kept:
            max_line_kept = line

    if not rows:
        return None

    # Data-driven λ search cap: the deepest kept line + 5 buffer.
    # True λ is essentially always below the deepest priced line —
    # otherwise the bookmaker would offer further lines. +5 gives the
    # grid headroom without slowing the fit. Self-scales to any stat:
    # goals (line up to ~6.5 → cap ~12), corners (~16 → ~21), tackles
    # (~22 → ~27), passes (~30 → ~35).
    lambda_upper_hundredths = int((max_line_kept + 5.0) * 100)

    best_lam = None
    best_err = float('inf')
    lam_hundredths = 5
    while lam_hundredths <= lambda_upper_hundredths:
        lam = lam_hundredths / 100.0
        err = 0.0
        for k, p_bookie in rows:
            err += (_poisson_p_geq(k, lam) - p_bookie) ** 2
        if err < best_err:
            best_err = err
            best_lam = lam
        lam_hundredths += 1

    return best_lam


def reverse_solve_share(lambda_total: float, p_h: float, p_d: float, p_a: float) -> float:
    """Find share_home ∈ [0.05, 0.95] such that Poisson joint
    distribution of (lambda_total * share, lambda_total * (1-share))
    best matches the target 1X2 probabilities.

    1D bounded grid search (step 0.01) + local refinement. Result fits
    the 1X2 spread while honouring the goals-total constraint exactly.
    Used by Path 2 only.
    """
    best_s = 0.5
    best_err = float('inf')

    # Coarse search
    si = 5
    while si <= 95:
        s = si / 100.0
        lh = lambda_total * s
        la = lambda_total * (1.0 - s)
        ph, pd, pa = _hda_from_lambdas(lh, la)
        err = (ph - p_h) ** 2 + (pd - p_d) ** 2 + (pa - p_a) ** 2
        if err < best_err:
            best_err = err
            best_s = s
        si += 1

    # Refine around best_s with step 0.005
    for delta_hundredths in range(-10, 11):
        s = best_s + delta_hundredths * 0.005
        if s < 0.05 or s > 0.95:
            continue
        lh = lambda_total * s
        la = lambda_total * (1.0 - s)
        ph, pd, pa = _hda_from_lambdas(lh, la)
        err = (ph - p_h) ** 2 + (pd - p_d) ** 2 + (pa - p_a) ** 2
        if err < best_err:
            best_err = err
            best_s = s

    return best_s


def _hda_from_lambdas(lh: float, la: float) -> Tuple[float, float, float]:
    """Joint Poisson grid 0..9 → (P_home, P_draw, P_away)."""
    ph = pd_ = pa = 0.0
    for x in range(10):
        px = _poisson_pmf(x, lh)
        for y in range(10):
            py = _poisson_pmf(y, la)
            p = px * py
            if x > y:
                ph += p
            elif x == y:
                pd_ += p
            else:
                pa += p
    return ph, pd_, pa


def derive_bookie_lambdas(
    fixture_id: int,
    lambda_h_model: float,
    lambda_a_model: float,
    bookie_1x2: Optional[Tuple[float, float, float]],
    goals_odds: dict,
) -> Optional[Tuple[float, float]]:
    """Try paths 1 → 3 to derive bookie-implied (λ_h, λ_a) for the
    fixture. Returns None if no path succeeded; caller should then
    fall through to its existing 1X2-only logic.

    bookie_1x2: margin-stripped (p_home, p_draw, p_away) or None.
    goals_odds: dict keyed by bookmaker name → dict with:
        'match':  list[(line, over, under)]    # team_id IS NULL rows
        'home':   list[(line, over, under)]    # team_id = home_team_id
        'away':   list[(line, over, under)]    # team_id = away_team_id

    Order of fallback through bookmakers controlled by GOALS_BOOKIE_PRIORITY.
    """
    for bookie in GOALS_BOOKIE_PRIORITY:
        result = _try_paths_for_bookie(
            goals_odds.get(bookie, {}),
            lambda_h_model, lambda_a_model,
            bookie_1x2,
        )
        if result is not None:
            return result
    return None


def _try_paths_for_bookie(
    bookie_data: dict,
    lambda_h_model: float,
    lambda_a_model: float,
    bookie_1x2: Optional[Tuple[float, float, float]],
) -> Optional[Tuple[float, float]]:
    home_ladder = bookie_data.get('home', [])
    away_ladder = bookie_data.get('away', [])
    match_ladder = bookie_data.get('match', [])

    lambda_h_bookie = fit_lambda_from_ladder(home_ladder)
    lambda_a_bookie = fit_lambda_from_ladder(away_ladder)
    lambda_total_bookie = fit_lambda_from_ladder(match_ladder)

    # PATH 1 — both per-team ladders
    if lambda_h_bookie is not None and lambda_a_bookie is not None:
        return lambda_h_bookie, lambda_a_bookie

    # PATH 1.5 — single per-team + match total → derive missing side
    if lambda_h_bookie is not None and lambda_total_bookie is not None:
        return lambda_h_bookie, max(0.05, lambda_total_bookie - lambda_h_bookie)
    if lambda_a_bookie is not None and lambda_total_bookie is not None:
        return max(0.05, lambda_total_bookie - lambda_a_bookie), lambda_a_bookie

    # PATH 2 — match total + 1X2 → reverse-solve share
    if lambda_total_bookie is not None and bookie_1x2 is not None:
        p_h, p_d, p_a = bookie_1x2
        share = reverse_solve_share(lambda_total_bookie, p_h, p_d, p_a)
        return lambda_total_bookie * share, lambda_total_bookie * (1.0 - share)

    # PATH 3 — match total only → split via model ratio
    if lambda_total_bookie is not None:
        denom = lambda_h_model + lambda_a_model
        share = (lambda_h_model / denom) if denom > 0 else 0.5
        return lambda_total_bookie * share, lambda_total_bookie * (1.0 - share)

    # No goals-ladder data → caller falls back to its own 1X2-only path
    return None


def blend_lambdas(
    lambda_h_model: float,
    lambda_a_model: float,
    lambda_h_bookie: float,
    lambda_a_bookie: float,
    w: float,
) -> Tuple[float, float]:
    """Linear blend in goal space.
        λ_final = (1-w) * λ_model + w * λ_bookie
    """
    lh = (1.0 - w) * lambda_h_model + w * lambda_h_bookie
    la = (1.0 - w) * lambda_a_model + w * lambda_a_bookie
    return lh, la


async def load_goals_odds_for_fixtures(conn, fixture_ids: list) -> dict:
    """Pre-load goals over/under odds across all 5 books for a batch.

    Returns dict keyed by fixture_id:
        {fid: {book: {'match': [...], 'home': [...], 'away': [...]}}}

    where `book` is one of bet365 / coral / ladbrokes / midnite /
    boylesports — the cascade in derive_bookie_lambdas iterates them in
    GOALS_BOOKIE_PRIORITY order and falls through per fixture when the
    leading book has no usable ladder.

    Each list element is (line: float, over_price: float|None,
    under_price: float|None). Rows deduped via MAX(price) per (fixture,
    team_id, line, side).

    Delegates to load_team_stat_odds with market='goals' — same SQL
    shape, same per-book schema, same dedupe. Keeps the goals path on
    one implementation path with team-stat / corners / cards / etc..

    Coverage as of 2026-06-01: bet365 ~52K upcoming fixtures, others
    ~2-3K each. The fallback meaningfully fires on bet365-dark fixtures
    (international friendlies, lower-division mid-week games — ~12
    such fixtures in any given 14-day window).
    """
    return await load_team_stat_odds(
        conn, fixture_ids, market='goals', books=GOALS_BOOKIE_PRIORITY,
    )


async def load_confirmed_lineups(conn, fixture_ids: list) -> dict:
    """Return {(fixture_id, team_id): set(player_id)} for confirmed=1
    rows in fixture_player_lineup. Caller passes this to
    distribute_team_predictions_to_players so the player iteration
    drops bench players for any (fixture, team) with a confirmed XI.

    Empty dict if no confirmed lineups exist for the supplied fixtures
    — the distribute function then falls back to full squad (current
    behaviour for league-wide nightly runs).
    """
    if not fixture_ids:
        return {}
    ph = ",".join(["%s"] * len(fixture_ids))
    result = {}
    async with conn.cursor() as cur:
        await cur.execute(
            f"""
            SELECT fixture_id, team_id, player_id
            FROM fixture_player_lineup
            WHERE fixture_id IN ({ph}) AND confirmed = 1
            """,
            tuple(fixture_ids),
        )
        rows = await cur.fetchall()
    for fid, tid, pid in rows:
        result.setdefault((int(fid), int(tid)), set()).add(int(pid))
    if result:
        logger.info(
            "Loaded confirmed lineups for %d (fixture, team) pairs",
            len(result),
        )
    return result


async def load_team_stat_odds(conn, fixture_ids: list, market: str, books: list) -> dict:
    """Generalised totals-odds loader for any market across multiple
    books. Same shape as load_goals_odds_for_fixtures but parameterised:

        market ∈ {'goals', 'corners', 'cards', 'shots', 'sot', 'fouls', 'tackles', ...}
        books  ⊆ {'bet365', 'ladbrokes', 'coral', 'midnite', 'boylesports'}

    Returns nested dict keyed by fixture_id:
        {fid: {book: {'match': [...], 'home': [...], 'away': [...]}}}

    Each list element is (line, over_price, under_price). Books not
    carrying the market for a fixture get an empty dict; downstream
    cascade will fall through them.

    Single SELECT per book — all schemas identical (fixture_id, team_id,
    market, line, side, price). Rows deduped via MAX(price) per
    (fixture, team, line, side) to handle the multi-fetch repeats.
    """
    if not fixture_ids or not books:
        return {}

    # Fixture → (home_team_id, away_team_id) map for tagging per-team rows.
    fix_ph = ",".join(["%s"] * len(fixture_ids))
    async with conn.cursor() as cur:
        await cur.execute(
            f"SELECT id, home_team_id, away_team_id FROM fixtures WHERE id IN ({fix_ph})",
            tuple(fixture_ids),
        )
        fixture_teams = {row[0]: (row[1], row[2]) for row in await cur.fetchall()}

    result = {}
    for book in books:
        table = f"{book}_totals_odds"
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT fixture_id, team_id, line, side, MAX(price) AS price
                FROM {table}
                WHERE market = %s AND fixture_id IN ({fix_ph})
                GROUP BY fixture_id, team_id, line, side
                """,
                (market,) + tuple(fixture_ids),
            )
            rows = await cur.fetchall()

        buckets = {}  # buckets[(fid, role)][line] = {'over': p, 'under': p}
        for fid, team_id, line, side, price in rows:
            teams = fixture_teams.get(fid)
            if not teams:
                continue
            home_tid, away_tid = teams
            if team_id is None:
                role = 'match'
            elif team_id == home_tid:
                role = 'home'
            elif team_id == away_tid:
                role = 'away'
            else:
                continue
            buckets.setdefault((fid, role), {}).setdefault(float(line), {})[side] = float(price)

        for (fid, role), by_line in buckets.items():
            ladder = [(line, sides.get('over'), sides.get('under'))
                      for line, sides in sorted(by_line.items())]
            result.setdefault(fid, {}).setdefault(book, {})[role] = ladder

    n_with_data = sum(1 for fid in fixture_ids if result.get(fid))
    logger.info(
        "Loaded %s odds for %d/%d fixtures across %s",
        market, n_with_data, len(fixture_ids), ",".join(books),
    )
    return result


def derive_team_stat_lambdas(
    odds_for_fixture: dict,
    model_home: float,
    model_away: float,
    books_priority: list,
) -> Optional[Tuple[float, float]]:
    """Cascade for team-stat (corners, cards, shots etc.) lambdas.

    Per priority book: try per-team ladders first (full or partial),
    then match-total split via model ratio. First book that returns a
    usable result wins.

    No 1X2 analog for team stats — cascade is shorter than goals:
      Path 1   per-team ladders (both teams)     → fit each side
      Path 1.5 one per-team + match total        → derive missing side
      Path 2   match total only                  → split via model ratio
      Path 3   nothing for this book             → fall through to next

    Returns (lambda_home_bookie, lambda_away_bookie) or None.
    """
    for book in books_priority:
        book_data = odds_for_fixture.get(book, {})
        if not book_data:
            continue

        home_ladder = book_data.get('home', [])
        away_ladder = book_data.get('away', [])
        match_ladder = book_data.get('match', [])

        lam_h = fit_lambda_from_ladder(home_ladder)
        lam_a = fit_lambda_from_ladder(away_ladder)
        lam_t = fit_lambda_from_ladder(match_ladder)

        # Path 1 — both per-team ladders
        if lam_h is not None and lam_a is not None:
            return lam_h, lam_a

        # Path 1.5 — one per-team + match total
        if lam_h is not None and lam_t is not None:
            return lam_h, max(0.01, lam_t - lam_h)
        if lam_a is not None and lam_t is not None:
            return max(0.01, lam_t - lam_a), lam_a

        # Path 2 — match total only, split by model ratio
        if lam_t is not None:
            denom = model_home + model_away
            if denom > 0:
                share = model_home / denom
                return lam_t * share, lam_t * (1.0 - share)

        # This book has nothing usable — fall through.

    return None


def blend_team_stat(
    model_home: float,
    model_away: float,
    odds_for_fixture: dict,
    market: str,
    blend_weight: float,
) -> Tuple[float, float]:
    """Blend a single team-stat (e.g. corners) per fixture in goal space.

    Returns (final_home, final_away). Falls back to model unchanged
    if no book in the priority list has usable data.
    """
    books = TEAM_STAT_BOOKIE_PRIORITY.get(market, ['bet365'])
    bookie_lambdas = derive_team_stat_lambdas(
        odds_for_fixture, model_home, model_away, books,
    )
    if bookie_lambdas is None:
        return model_home, model_away

    lh_b, la_b = bookie_lambdas
    fh = (1.0 - blend_weight) * model_home + blend_weight * lh_b
    fa = (1.0 - blend_weight) * model_away + blend_weight * la_b
    return fh, fa


# ─────────────────────────────────────────────────────────────────────
# Player-prop blend
# ─────────────────────────────────────────────────────────────────────
#
# Mirrors the team-stat blend in shape but operates per (fixture, player,
# stats_type). Schema-level differences from team stats:
#
#   - Player odds are stored per book in `{book}_player_odds` tables
#     with columns (fixture_id, player_id, stats_type_id, stat_min,
#     price). All 5 books use the same column names.
#   - All player ladders are OVER-ONLY (no paired under). v1 uses raw
#     1/price as the implied probability — no margin stripping.
#   - Ladders are typically shallow (1+/2+/3+) so the probability filter
#     widens to (0.02, 0.90) — see fit_lambda_from_ladder.
#   - bet365 always leads the priority cascade per the global rule; the
#     cascade falls through per (fixture, player) row when bet365 has
#     no quote for that specific player.

# Player-prop ladders are over-only; this is the wider filter band
# passed to fit_lambda_from_ladder. Player ladders are shallow (1+/2+/3+)
# so the [0.10, 0.90] team-stat band drops most rungs; [0.02, 0.90]
# keeps "3+ goals for a striker" (~0.04) and "5+ tackles for a CDM"
# (~0.03) while still rejecting bookmaker stubs.
PLAYER_PROB_FILTER = (0.02, 0.90)


async def load_player_odds(
    conn,
    fixture_ids: list,
    stats_type_ids: list,
    books: list,
) -> dict:
    """Pre-load per-book player-prop odds for a batch of fixtures.

    Returns a nested dict:
        {fixture_id: {player_id: {stats_type_id: {book: [(line, over_price, None), ...sorted asc]}}}}

    Each ladder element matches the (line, over_price, under_price)
    shape expected by fit_lambda_from_ladder — under_price is always
    None because player props are over-only. `line = stat_min - 0.5` so
    fit_lambda_from_ladder's `k = floor(line) + 1 = stat_min`, i.e. an
    "X+" market maps to P(value ≥ X).

    Rows deduped via MAX(price) per (fixture, player, stats_type,
    stat_min). bet365 has a unique constraint on this tuple as of
    2026-05-25 + ladbrokes/coral use updateOrCreate → no-op for them.
    Midnite + BoyleSports use raw insert() with no unique key, so the
    MAX() is the canonical dedupe for those two.

    Rows with NULL player_id (midnite/boyle ~4-9% unmatched scrape
    rows) and NULL stat_min are filtered out — they carry no usable
    signal until the ingest-side player-id backfill ships.
    """
    if not fixture_ids or not stats_type_ids or not books:
        return {}

    fix_ph = ",".join(["%s"] * len(fixture_ids))
    st_ph = ",".join(["%s"] * len(stats_type_ids))
    params = tuple(fixture_ids) + tuple(stats_type_ids)

    result = {}
    for book in books:
        table = f"{book}_player_odds"
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT fixture_id, player_id, stats_type_id, stat_min,
                           MAX(price) AS price
                    FROM {table}
                    WHERE fixture_id IN ({fix_ph})
                      AND stats_type_id IN ({st_ph})
                      AND player_id IS NOT NULL
                      AND stat_min IS NOT NULL
                      AND stat_min > 0
                    GROUP BY fixture_id, player_id, stats_type_id, stat_min
                    """,
                    params,
                )
                rows = await cur.fetchall()
        except Exception as e:
            # Only swallow the "table doesn't exist" path (MySQL errno
            # 1146) — anything else (lost connection, syntax break,
            # schema drift) must surface so it gets investigated rather
            # than silently demoting the cascade for that book.
            if getattr(e, 'args', None) and e.args[0] == 1146:
                logger.warning("load_player_odds: table %s missing — skipping book", table)
                continue
            raise

        # Group rows into [(line, over, None)] ladders per
        # (fid, pid, stat_type, book). stat_min → line = stat_min - 0.5.
        buckets = {}  # buckets[(fid, pid, st)][stat_min] = price
        for fid, pid, st, stat_min, price in rows:
            buckets.setdefault((int(fid), int(pid), int(st)), {})[int(stat_min)] = float(price)

        for (fid, pid, st), by_min in buckets.items():
            ladder = [(float(sm) - 0.5, price, None)
                      for sm, price in sorted(by_min.items())]
            (result.setdefault(fid, {})
                   .setdefault(pid, {})
                   .setdefault(st, {})[book]) = ladder

    n_fix_with_data = len(result)
    logger.info(
        "Loaded player odds for %d/%d fixtures across %s (stats_type_ids=%s)",
        n_fix_with_data, len(fixture_ids),
        ",".join(books), stats_type_ids,
    )
    return result


def derive_player_lambdas(
    ladders_by_book: dict,
    books_priority: list,
) -> Optional[float]:
    """First-book-wins cascade for a single (player, fixture, stats_type).

    ladders_by_book: {book: [(line, over_price, None), ...]} — the
        innermost dict from load_player_odds, keyed by book.
    books_priority: ordered list of book names to try.

    Returns the first usable λ produced by fit_lambda_from_ladder on
    any book's ladder, or None if no book has data.

    Single-stat (not per-team like the team-stat equivalent) because a
    player prop is one λ, not a pair.
    """
    for book in books_priority:
        ladder = ladders_by_book.get(book)
        if not ladder:
            continue
        lam = fit_lambda_from_ladder(ladder, prob_filter=PLAYER_PROB_FILTER)
        if lam is not None:
            return lam
    return None


# Master switch for the player-odds blend (WC + domestic both route through
# blend_player_stat). Re-enabled 2026-06-10 for the ORIGINAL v1 stats only
# (Goals/Shots/SoT — see PLAYER_BLEND_STAT_NAMES). The 2026-06-10 extension to
# assists/fouls/tackles/etc was reverted because blending toward un-de-margined
# over-only props inflated the aggregate ~40% (Germany R1 goals ×1.43).
# Re-add those stats only after de-margining the over-only ladders (backlog).
PLAYER_ODDS_BLEND_ENABLED = True

# Fixed player-odds blend weight — 0.3 for ALL competitions (deliberately NOT
# the per-comp odds_beta). The over-only ladders carry uncorrected vig, so we
# keep the market's pull modest and uniform until the de-margin lands; then
# revisit whether to scale by odds_beta. blend_player_stat uses THIS, ignoring
# whatever weight the caller passes. See backlog "Player-odds blend".
PLAYER_ODDS_BLEND_WEIGHT = 0.3


def blend_player_stat(
    model_lambda: float,
    ladders_by_book: dict,
    stats_type_id: int,
    blend_weight: float,
) -> float:
    """Blend a single (player, fixture, stats_type) λ in lambda space.

    ladders_by_book: {book: ladder} — pre-loaded by load_player_odds,
        already scoped to this (fixture, player, stats_type). Pass {}
        when no book has data; this returns model_lambda unchanged.

    stats_type_id: drives the per-stat bookmaker priority lookup
        via PLAYER_STAT_BOOKIE_PRIORITY.

    blend_weight: IGNORED — the blend uses the fixed module-level
        PLAYER_ODDS_BLEND_WEIGHT (0.3 for all comps) regardless of what the
        caller passes, until the over-only de-margin lands.
    """
    if not PLAYER_ODDS_BLEND_ENABLED:
        return model_lambda

    if not ladders_by_book:
        return model_lambda

    books = PLAYER_STAT_BOOKIE_PRIORITY.get(stats_type_id)
    if not books:
        return model_lambda

    lam_bookie = derive_player_lambdas(ladders_by_book, books)
    if lam_bookie is None:
        return model_lambda

    w = PLAYER_ODDS_BLEND_WEIGHT
    return (1.0 - w) * model_lambda + w * lam_bookie


def compute_final_goals_and_probs(
    fixture_id: int,
    lambda_h_model: float,
    lambda_a_model: float,
    bookie_1x2_pct: Optional[Tuple[float, float, float]],
    goals_odds: dict,
    odds_weight: float,
    boost: float,
    rho: float = 0.0,
) -> Tuple[float, float, float, float, float]:
    """Single entry point for the goal-blend logic across all 3 services.

    Returns: (final_home_goals, final_away_goals,
              final_home_win_pct, final_draw_pct, final_away_win_pct)

    Path priority:
      1-3 (this module): if goals O/U markets present, derive bookie λs
                         and blend in goal space.
      4   (legacy):      if only 1X2 present, fall back to the original
                         probability-space blend + 2D reverse-solve.
      5   (no odds):     model unchanged.

    bookie_1x2_pct: margin-stripped (p_h, p_d, p_a) as fractions, or None.
    goals_odds: nested dict keyed by bookmaker → {match, home, away}.
    odds_weight: per-service blend weight (0.3 domestic/WC, 0.5 euro comp).
    boost: draw-bias multiplier (1.1 across all services today).
    """
    # Try the new cascade first.
    bookie_lambdas = derive_bookie_lambdas(
        fixture_id, lambda_h_model, lambda_a_model,
        bookie_1x2_pct, goals_odds,
    )

    if bookie_lambdas is not None:
        # Paths 1 / 1.5 / 2 / 3 — blend in goal space.
        lh_b, la_b = bookie_lambdas
        new_h, new_a = blend_lambdas(
            lambda_h_model, lambda_a_model, lh_b, la_b, odds_weight,
        )
        # Compute final H/D/A from the blended λs so they're internally
        # consistent with the output goals.
        ph, pd_, pa = _hda_from_lambdas(new_h, new_a)
        # Apply the same draw shaping get_result_probs uses, so every path
        # emits consistent percentages. With Dixon-Coles configured the grid
        # has already been corrected cell by cell, so the flat boost must NOT
        # be layered on top — recompute from the corrected grid instead.
        if rho:
            from app.services.statz_functions import get_result_probs
            ph_pct, pd_pct, pa_pct = get_result_probs(new_h, new_a, 1.0, rho)
            return new_h, new_a, ph_pct, pd_pct, pa_pct
        pd_boosted = pd_ * boost
        remaining = 1.0 - pd_boosted
        if (ph + pa) > 0:
            ph_norm = (ph / (ph + pa)) * remaining
            pa_norm = (pa / (ph + pa)) * remaining
        else:
            ph_norm = pa_norm = remaining / 2.0
        return new_h, new_a, ph_norm * 100, pd_boosted * 100, pa_norm * 100

    # PATH 4 — legacy 1X2-only blend + 2D reverse-solve. Same maths as
    # before the cascade was added; preserved for fixtures without
    # goals O/U markets.
    if bookie_1x2_pct is None:
        # PATH 5 — no odds at all. Use model unchanged; emit model H/D/A.
        from app.services.statz_functions import get_result_probs
        h, d, a = get_result_probs(lambda_h_model, lambda_a_model, boost, rho)
        return lambda_h_model, lambda_a_model, h, d, a

    from app.services.statz_functions import get_result_probs, find_inputs_for_probs
    p_h, p_d, p_a = bookie_1x2_pct
    bookie_h_pct, bookie_d_pct, bookie_a_pct = p_h * 100, p_d * 100, p_a * 100

    model_h_pct, model_d_pct, model_a_pct = get_result_probs(
        lambda_h_model, lambda_a_model, boost, rho,
    )

    adj_h = model_h_pct + (bookie_h_pct - model_h_pct) * odds_weight
    adj_d = model_d_pct + (bookie_d_pct - model_d_pct) * odds_weight
    adj_a = model_a_pct + (bookie_a_pct - model_a_pct) * odds_weight

    new_h, new_a = find_inputs_for_probs(
        lambda_h_model, lambda_a_model,
        adj_h, adj_d, adj_a, boost, rho,
    )
    return float(new_h), float(new_a), adj_h, adj_d, adj_a
