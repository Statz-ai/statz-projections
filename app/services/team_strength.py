"""Team Strength ratings — market-anchored season-simulation ratings.

Why this exists (George, 2026-07-30): `get_ratings()` is a FORM marker, not a
team-strength marker. Its 30-game window with 0.96^(weeks-3) decay means a
team's last 6 games outweigh their previous 15 combined, and the only
non-form input (the market-value nudge) is worth about +/-0.10 net xG/game —
roughly 5% of the signal. Pre-season that produces indefensible outputs:
Liverpool's base rating is NEGATIVE (-0.03 net xG/g -> projected 9th) while
the market has them 3rd favourite; Hull project 19 points; and last season
Sunderland sat rank 20 from mid-September to late October before finishing 7th.

This module produces a SEPARATE strength rating for the season simulation by
blending three views on a common z-scale:

    A. base    — the existing get_ratings() output (our own evidence)
    B. market  — implied finishing position from bookmaker outright odds
    C. squad   — Transfermarkt squad market value

Pre-season leans on the market (which prices transfers, managers, and
off-pitch risk we will never model), then glides back to our own evidence as
real matches accumulate. "More in line with market, with a touch of Statz
sauce."

SCOPE: the season simulation ONLY. Match projections, team_projections,
player props and FPL keep using the form-tilted `ratings` — form-tilting is
CORRECT for next weekend. The Man City case makes the distinction concrete:
their short relegation price reflects pending financial charges and a
possible points deduction, which belongs in a final-league-position
projection and emphatically does not belong in "will City beat Brentford".
If this is ever extended to match projections, the non-footballing component
must be stripped out first.
"""

import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger("projection")

# Master switch. Per-competition rollout is controlled by
# competition_projection_config.use_strength_ratings; this env var is the
# global kill switch on top of it.
STRENGTH_ENABLED = os.getenv("TEAM_STRENGTH", "1") not in ("0", "false", "False")

# Blend weights (base, market, squad_value). Judgment-picked per George,
# exposed as tunables rather than fitted — see [[no-simulation-tuning]].
W_PRE_SEASON = (
    float(os.getenv("STRENGTH_W_BASE_PRE", "0.60")),
    float(os.getenv("STRENGTH_W_MARKET_PRE", "0.30")),
    float(os.getenv("STRENGTH_W_MV_PRE", "0.10")),
)
W_SETTLED = (
    float(os.getenv("STRENGTH_W_BASE_SETTLED", "0.80")),
    float(os.getenv("STRENGTH_W_MARKET_SETTLED", "0.10")),
    float(os.getenv("STRENGTH_W_MV_SETTLED", "0.10")),
)
# Promoted sides pre-season: a tier-scaled Championship extrapolation is the
# LEAST informative source about top-flight ability, yet today it is the only
# one. The market already prices the promoted-team base rate (Hull 1.25 to be
# relegated, not a 19-point catastrophe), so no separate empirical floor is
# needed on top of this.
W_PRE_SEASON_PROMOTED = (
    float(os.getenv("STRENGTH_W_BASE_PROMOTED", "0.20")),
    float(os.getenv("STRENGTH_W_MARKET_PROMOTED", "0.50")),
    float(os.getenv("STRENGTH_W_MV_PROMOTED", "0.30")),
)
# Matches played before the weights reach their settled values. A hard switch
# on GW1 morning would jump every rating overnight; this mirrors the decay
# idiom already used twice in projection_service (mv_beta * 0.95^matches,
# promoted priors 0.85^matches).
GLIDE_MATCHES = int(os.getenv("STRENGTH_GLIDE_MATCHES", "10"))

# Outright odds older than this are ignored (and logged).
ODDS_MAX_AGE_DAYS = int(os.getenv("STRENGTH_ODDS_MAX_AGE_DAYS", "7"))

# Which statistic summarises the market's implied finishing distribution.
# "mean" (default) uses the whole distribution; "median" is available and is
# robust to catastrophe tails, but we handle the one live case explicitly
# instead — see NO_TAIL_TEAM_IDS below.
MARKET_STAT = os.getenv("STRENGTH_MARKET_STAT", "mean").lower()

# Teams whose bottom-anchored markets (relegation / finish bottom / bottom
# half) are EXCLUDED from the strength derivation, with the remaining
# top-anchored markets renormalised into a proper distribution.
#
# Live case: Manchester City (team_id 9). Their book carries a real tail from
# the pending financial charges — 13 for relegation, 26 to finish bottom while
# 3.75 to win the league. That price is CORRECT, not a data error: the market
# has the possible points deduction priced in (George, 2026-07-30). But it is
# a binary event — "they will either get deducted 80 points or they won't" —
# and the outrights model should project the branch where they aren't. Smeared
# into a continuous strength rating it would drag the second favourites toward
# mid-table on football they will actually play.
#
# Deliberately an explicit, named, logged exception rather than a league-wide
# statistical device: it is one known situation, and it should be REMOVED the
# moment the case resolves. If a deduction is confirmed, model it as an
# explicit points penalty in the simulation, not as reduced strength.
NO_TAIL_TEAM_IDS = {
    int(t) for t in os.getenv("STRENGTH_NO_TAIL_TEAM_IDS", "9").split(",") if t.strip()
}

# Provider preference for outright odds. bet365 carries all 20 PL teams
# across 9 markets refreshed daily; others are patchier.
ODDS_PROVIDER_PRIORITY = ["bet365", "ladbrokes", "coral", "boylesports"]

# Default number of relegation places when the competition doesn't say.
DEFAULT_RELEGATION_PLACES = int(os.getenv("STRENGTH_RELEGATION_PLACES", "3"))

# Attack/Defense floor as a fraction of the league mean, so a heavy blend can
# never produce a zero or negative goal rate. Scale-agnostic (the pipeline
# feeds indexed mean=100 values; tests feed raw xG).
RATING_FLOOR_FRACTION = float(os.getenv("STRENGTH_RATING_FLOOR", "0.2"))

# Outright market_id -> finishing-position band, as (from, to) given the
# league size and the first relegation position.
#
# SOURCE OF TRUTH: app/Services/Projections/LeagueMarketRegistry.php in the
# statz repo. Mirrored here because the projection server can't call PHP.
# Keep the two in sync when markets are added.
MARKET_BANDS = {
    75:    lambda size, releg_from: (1, 1),                      # Win League
    50464: lambda size, releg_from: (1, 2),                      # Top 2
    50457: lambda size, releg_from: (1, 3),                      # Top 3
    50130: lambda size, releg_from: (1, 4),                      # Top 4
    50141: lambda size, releg_from: (1, 5),                      # Top 5
    50131: lambda size, releg_from: (1, 6),                      # Top 6
    50455: lambda size, releg_from: (1, 7),                      # Top 7
    10141: lambda size, releg_from: (1, size // 2),              # Top Half
    50502: lambda size, releg_from: (5, size),                   # Not Top 4
    10142: lambda size, releg_from: (size // 2 + 1, size),       # Bottom Half
    344:   lambda size, releg_from: (releg_from, size),          # Relegation
    10106: lambda size, releg_from: (size, size),                # Finish Bottom
}


async def load_outright_odds(conn, competition_id: int, season_id: int) -> pd.DataFrame:
    """Outright odds for one competition/season.

    Returns [team_id, market_id, provider, odd_decimal, updated_at]; empty
    frame when nothing is priced (the caller then falls back to base-only).
    """
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT team_id, market_id, provider, odd_decimal, updated_at
            FROM team_outright_odds
            WHERE competition_id = %s AND season_id = %s
              AND odd_decimal IS NOT NULL AND odd_decimal > 1.0
            """,
            (int(competition_id), int(season_id)),
        )
        rows = await cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["team_id", "market_id", "provider", "odd_decimal", "updated_at"])
    df = pd.DataFrame(rows, columns=["team_id", "market_id", "provider", "odd_decimal", "updated_at"])
    df["odd_decimal"] = pd.to_numeric(df["odd_decimal"], errors="coerce")
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    return df.dropna(subset=["odd_decimal"])


def _pick_provider_rows(odds_df, market_id, team_ids, now):
    """Rows for one market from the best available provider.

    Best = first provider in ODDS_PROVIDER_PRIORITY with fresh prices for
    every team; failing that, the provider covering the most teams.
    Returns None when no provider has usable prices.
    """
    fresh_cutoff = now - timedelta(days=ODDS_MAX_AGE_DAYS)
    market_rows = odds_df[odds_df["market_id"] == market_id]
    if market_rows.empty:
        return None

    stale = market_rows[market_rows["updated_at"] < fresh_cutoff]
    market_rows = market_rows[market_rows["updated_at"] >= fresh_cutoff]
    if market_rows.empty:
        logger.info(
            "  market %s skipped — all %d prices older than %d days (latest %s)",
            market_id, len(stale), ODDS_MAX_AGE_DAYS,
            stale["updated_at"].max() if not stale.empty else "?",
        )
        return None

    best, best_n = None, 0
    ordered = ODDS_PROVIDER_PRIORITY + [
        p for p in market_rows["provider"].unique() if p not in ODDS_PROVIDER_PRIORITY
    ]
    for provider in ordered:
        rows = market_rows[market_rows["provider"] == provider]
        rows = rows[rows["team_id"].isin(team_ids)].drop_duplicates("team_id")
        if len(rows) == len(team_ids):
            return rows
        if len(rows) > best_n:
            best, best_n = rows, len(rows)
    return best


def market_position_cdf(odds_df, team_ids, size, relegation_from, now=None):
    """Implied cumulative finishing-position distribution per team.

    Every outright market is a contiguous band of finishing positions, so a
    de-vigged band probability is a point on the team's position CDF:
      band starting at 1  ("top K")    -> P(pos <= to)     = p
      band ending at size ("bottom K") -> P(pos <= from-1) = 1 - p

    Returns {team_id: [(k, P(pos <= k)), ...]} sorted by k, monotone
    non-decreasing and clipped to [0, 1].
    """
    now = pd.Timestamp.utcnow().tz_localize(None) if now is None else pd.Timestamp(now)
    points = {int(t): [] for t in team_ids}
    used = []
    skipped_tail = set()

    for market_id, band_fn in MARKET_BANDS.items():
        rows = _pick_provider_rows(odds_df, market_id, team_ids, now)
        if rows is None or rows.empty:
            continue
        band_from, band_to = band_fn(size, relegation_from)
        band_size = band_to - band_from + 1
        if band_size <= 0 or band_size >= size:
            continue

        # De-vig: raw implied probabilities carry the book's overround, so
        # normalise them to sum to the number of teams the band holds.
        raw = 1.0 / rows["odd_decimal"].to_numpy(dtype=float)
        total = raw.sum()
        if total <= 0:
            continue
        devigged = raw * (band_size / total)

        if band_from == 1:
            k = band_to
            values = devigged
            bottom_anchored = False
        elif band_to == size:
            k = band_from - 1
            values = 1.0 - devigged
            bottom_anchored = True
        else:
            continue  # mid-table band — no single CDF point

        if k <= 0 or k >= size:
            continue
        for team_id, value in zip(rows["team_id"].to_numpy(), values):
            tid = int(team_id)
            if tid not in points:
                continue
            # Excluded teams contribute only top-anchored markets; their
            # bottom tail is a pending points deduction, not football.
            if bottom_anchored and tid in NO_TAIL_TEAM_IDS:
                skipped_tail.add(tid)
                continue
            points[tid].append((int(k), float(np.clip(value, 0.0, 1.0))))
        used.append((market_id, rows["provider"].iloc[0], len(rows)))

    if used:
        logger.info(
            "  market CDF built from %d markets: %s",
            len(used), ", ".join(f"{m}({p},{n})" for m, p, n in used),
        )

    if skipped_tail:
        logger.info(
            "  tail markets EXCLUDED for team_id(s) %s — pending points-deduction risk is a "
            "binary event, not football; projecting the no-deduction branch (see NO_TAIL_TEAM_IDS)",
            sorted(skipped_tail),
        )

    # Enforce a monotone non-decreasing CDF. This corrects genuinely broken
    # prices without special-casing.
    #
    # DO NOT add a "contradictory prices" filter on top of this. A team priced
    # short to win AND short to go down is usually telling you something real
    # (the Man City charges being the live example), and a tidy-up filter would
    # delete exactly the information the market component exists to supply.
    # Where such a tail must be ignored, exclude it explicitly and visibly via
    # NO_TAIL_TEAM_IDS rather than silently by heuristic.
    out = {}
    for team_id, pairs in points.items():
        if not pairs:
            continue
        merged = {}
        for k, v in sorted(pairs):
            merged.setdefault(k, []).append(v)
        ks = sorted(merged)
        vs = np.maximum.accumulate([float(np.mean(merged[k])) for k in ks])
        vs = np.clip(vs, 0.0, 1.0)
        # Tail-excluded teams: the surviving top markets stop short of 1.0
        # (City's deepest is P(top half)=0.878). Renormalise so they form a
        # proper distribution over the range those markets describe — i.e.
        # condition on the no-catastrophe branch — instead of leaving the
        # missing mass to be smeared across the positions below.
        if team_id in NO_TAIL_TEAM_IDS and len(vs) and vs[-1] > 0:
            vs = np.clip(vs / vs[-1], 0.0, 1.0)
        out[team_id] = list(zip(ks, vs))
    return out


def implied_expected_position(cdf_points, size):
    """E[finishing position] from CDF points, via E[X] = sum_k P(X > k).

    Linearly interpolates between the known points, anchored at P(pos<=0)=0
    and P(pos<=size)=1.

    NOTE: sensitive to tail mass — see implied_median_position, which is the
    default for strength precisely because of that.
    """
    ks = [0] + [k for k, _ in cdf_points] + [size]
    vs = [0.0] + [v for _, v in cdf_points] + [1.0]
    grid = np.arange(0, size)
    cdf = np.interp(grid, ks, vs)
    return float(np.sum(1.0 - cdf))


def implied_median_position(cdf_points, size):
    """Median implied finishing position — the k where the CDF crosses 0.5.

    This is the DEFAULT strength statistic, and the reason is Man City
    (George, 2026-07-30). Their book carries a real tail from the pending
    financial charges: P(pos<=6)=0.853 but P(pos<=17)=0.932, i.e. ~7% on a
    bottom-three finish that is a possible points deduction, not a football
    forecast. That single binary event drags E[position] from ~2.7 to ~4.7
    and would rate the second favourites below mid-table on football.

    "They will either get deducted 80 points or they won't — project as if
    they won't." The median does exactly that: it is set by where the bulk of
    the distribution sits and is unmoved by tail mass, so catastrophe risk
    (deductions, administration, points penalties anywhere in the league)
    drops out with no team-specific special-casing. If a deduction is ever
    confirmed it belongs in the simulation as an explicit points penalty, not
    smeared into a strength rating.
    """
    ks = [0] + [k for k, _ in cdf_points] + [size]
    vs = [0.0] + [v for _, v in cdf_points] + [1.0]
    grid = np.linspace(0, size, size * 20 + 1)
    cdf = np.interp(grid, ks, vs)
    crossing = np.searchsorted(cdf, 0.5)
    if crossing >= len(grid):
        return float(size)
    return float(max(1.0, grid[crossing]))


def implied_position(cdf_points, size):
    """Market strength statistic — median by default, mean when
    STRENGTH_MARKET_STAT=mean."""
    if MARKET_STAT == "mean":
        return implied_expected_position(cdf_points, size)
    return implied_median_position(cdf_points, size)


def _zscore(series):
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / sd


def blend_weights(matches_played, is_promoted):
    """(base, market, squad) weights for a team, glided by matches played."""
    pre = W_PRE_SEASON_PROMOTED if is_promoted else W_PRE_SEASON
    if GLIDE_MATCHES <= 0:
        return W_SETTLED
    g = min(1.0, max(0.0, float(matches_played) / GLIDE_MATCHES))
    return tuple(p + (s - p) * g for p, s in zip(pre, W_SETTLED))


def build_strength_ratings(ratings, odds_df, mv_index, team_ids_by_name,
                           promoted_teams, matches_played, size,
                           relegation_from=None, now=None, base_ratings=None):
    """Blend base / market / squad value into season-simulation ratings.

    ratings          DataFrame [Team, Attack, Defense] on the scale the
                     simulation consumes. In projection_service that is the
                     mean=100 INDEXED frame (see the rescale after the MV
                     block); make_round_goal_prediction defaults a missing
                     team to 100, so the output must stay on that scale.
    base_ratings     optional [Team, Attack, Defense] used for the base
                     component's z-score. Pass the PRE-market-value ratings
                     so squad value isn't counted twice (once inside the base
                     via mv_beta, once as its own 10% component). z-scores are
                     scale-invariant, so this frame can be in raw xG units
                     while the output stays indexed. Falls back to `ratings`.
    odds_df          load_outright_odds output (may be empty).
    mv_index         {team_name: MV Index} — the index already computed in
                     projection_service (may be empty).
    team_ids_by_name {team_name: team_id} for joining odds.
    promoted_teams   set of team names with no top-flight history this cycle.
    matches_played   matches played so far this season.
    size             number of teams in the competition.

    Returns a copy of `ratings` with Attack/Defense replaced by the blended
    values, plus the per-team component columns for persistence/audit.
    """
    df = ratings.copy()
    df["Attack"] = pd.to_numeric(df["Attack"], errors="coerce")
    df["Defense"] = pd.to_numeric(df["Defense"], errors="coerce")
    df = df.dropna(subset=["Attack", "Defense"]).reset_index(drop=True)
    if df.empty:
        return df

    relegation_from = relegation_from or (size - DEFAULT_RELEGATION_PLACES + 1)
    df["team_id"] = df["Team"].map(lambda t: team_ids_by_name.get(t))
    df["base_overall"] = df["Attack"] - df["Defense"]

    # Base z from the pre-MV frame when supplied — same teams, different
    # scale, so z-score there and carry the standardised value across.
    if base_ratings is not None and not base_ratings.empty:
        _b = base_ratings.copy()
        _b["_ov"] = (pd.to_numeric(_b["Attack"], errors="coerce")
                     - pd.to_numeric(_b["Defense"], errors="coerce"))
        _b = _b.dropna(subset=["_ov"])
        _bz = dict(zip(_b["Team"], _zscore(_b["_ov"])))
        df["base_z"] = df["Team"].map(_bz)
        missing = df["base_z"].isna()
        if missing.any():
            logger.info("  %d team(s) absent from pre-MV ratings — base z from indexed frame",
                        int(missing.sum()))
            df.loc[missing, "base_z"] = _zscore(df["base_overall"])[missing]
    else:
        df["base_z"] = _zscore(df["base_overall"])

    # --- market component ---------------------------------------------
    df["market_position"] = np.nan
    if odds_df is not None and not odds_df.empty:
        known_ids = [int(t) for t in df["team_id"].dropna().unique()]
        cdfs = market_position_cdf(odds_df, known_ids, size, relegation_from, now=now)
        df["market_position"] = df["team_id"].map(
            lambda t: implied_position(cdfs[int(t)], size)
            if pd.notna(t) and int(t) in cdfs and cdfs[int(t)] else np.nan
        )
    # Lower implied position = stronger, so negate before z-scoring.
    df["market_z"] = _zscore(-df["market_position"]) if df["market_position"].notna().any() else np.nan
    df.loc[df["market_position"].isna(), "market_z"] = np.nan

    # --- squad-value component ----------------------------------------
    df["mv_raw"] = df["Team"].map(lambda t: mv_index.get(t) if mv_index else None)
    df["mv_z"] = _zscore(df["mv_raw"]) if df["mv_raw"].notna().any() else np.nan
    df.loc[df["mv_raw"].isna(), "mv_z"] = np.nan

    # --- blend ---------------------------------------------------------
    blended_z, w_base_col, w_market_col, w_mv_col = [], [], [], []
    for row in df.itertuples(index=False):
        is_promoted = row.Team in promoted_teams
        w_base, w_market, w_mv = blend_weights(matches_played, is_promoted)

        # A team missing a component keeps the others in proportion rather
        # than being dragged toward the mean by a zeroed input.
        parts = [(w_base, row.base_z)]
        if pd.notna(row.market_z):
            parts.append((w_market, row.market_z))
        else:
            w_market = 0.0
        if pd.notna(row.mv_z):
            parts.append((w_mv, row.mv_z))
        else:
            w_mv = 0.0
        total_w = sum(w for w, _ in parts)
        blended_z.append(sum(w * z for w, z in parts) / total_w if total_w > 0 else row.base_z)
        w_base_col.append(w_base / total_w if total_w > 0 else 1.0)
        w_market_col.append(w_market / total_w if total_w > 0 else 0.0)
        w_mv_col.append(w_mv / total_w if total_w > 0 else 0.0)

    df["blended_z"] = blended_z
    df["w_base"], df["w_market"], df["w_mv"] = w_base_col, w_market_col, w_mv_col

    # Map back onto the base rating's own scale so the league's overall
    # spread of goal expectations is preserved.
    mean_overall = df["base_overall"].mean()
    sd_overall = df["base_overall"].std(ddof=0)
    df["blended_overall"] = mean_overall + df["blended_z"] * (sd_overall if np.isfinite(sd_overall) else 0.0)

    # Market and squad value speak to OVERALL strength, not style — so split
    # the delta 50/50 across attack and defence, preserving each team's own
    # attack/defence asymmetry (Arsenal defence-led, City attack-led).
    delta = df["blended_overall"] - df["base_overall"]
    df["Attack"] = df["Attack"] + delta / 2.0
    df["Defense"] = df["Defense"] - delta / 2.0
    # Never let the blend drive a rate to zero or negative. The floor is a
    # fraction of the column mean so it holds on either scale (indexed
    # mean=100 in the pipeline, raw xG in tests).
    for col in ("Attack", "Defense"):
        floor = RATING_FLOOR_FRACTION * float(ratings[col].astype(float).mean())
        df[col] = df[col].clip(lower=floor)

    df["is_promoted"] = df["Team"].isin(promoted_teams)
    df["matches_played"] = matches_played

    n_market = int(df["market_z"].notna().sum())
    logger.info(
        "  strength blend: %d teams, market for %d, weights base/market/mv = %.2f/%.2f/%.2f (matches=%s)",
        len(df), n_market, df["w_base"].mean(), df["w_market"].mean(), df["w_mv"].mean(), matches_played,
    )
    return df


def promoted_team_names(fixtures_df, teams_df, competition_id, previous_season_id, comp_team_names):
    """Teams with no fixtures in this competition last season — i.e. promoted
    (or otherwise new to the tier), which is exactly the population whose base
    rating is a cross-tier extrapolation."""
    if previous_season_id is None or fixtures_df is None or fixtures_df.empty:
        return set()
    prev = fixtures_df[
        (fixtures_df["competition_id"] == competition_id)
        & (fixtures_df["season_id"] == previous_season_id)
    ]
    if prev.empty:
        return set()
    seen_ids = set(prev["home_team_id"].tolist()) | set(prev["away_team_id"].tolist())
    id_by_name = teams_df.set_index("name")["id"].to_dict()
    return {
        name for name in comp_team_names
        if id_by_name.get(name) is not None and int(id_by_name[name]) not in seen_ids
    }
