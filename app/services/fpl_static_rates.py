"""Per-90 rates for BPS stats we do not project.

Spec: statz `docs/fpl-bps-rebuild-spec.md`. Successful dribbles (+1 BPS) and
big chances missed (-3 BPS) are player traits rather than a share of a team
total, so they get a simple historical per-90 rate instead of going through
the team-projection → share pipeline:

    lambda = per90_rate[player] x xMins / 90

George, 2026-08-03: last TWO seasons of history; players under a 450-minute
floor fall back to the average for their FPL position.

Deliberately NOT the projection pipeline. Both stats DO have team-level
Sportmonks coverage (dribbles 100%, big chances missed 94%), so they could be
projected properly like Big Chances Created — that was offered and declined
for now on cost grounds. The consequence to be aware of: these two are the
only terms in the BPS line that ignore the opponent.
"""
import logging

import pandas as pd

logger = logging.getLogger("projection")

MINUTES_STAT_ID = 119  # stats_types 'Minutes Played'

# Minutes below which a player's own rate is too noisy to trust and we use his
# position's average instead. A straight cutoff, not shrinkage — George's call.
MIN_MINUTES = 450.0

# Rates are per-90, so a player with no position and no history gets 0 rather
# than a guess. Only used when the positional average is also unavailable
# (e.g. a competition where the stat isn't collected at all).
FALLBACK_RATE = 0.0


def compute_per90_rates(player_stats, stat_id, positions_by_player, fixture_ids=None):
    """{player_id: per-90 rate} for one stat.

    player_stats: the pipeline's long-format per-player rows
        (player_id, fixture_id, stats_type_id, value).
    stat_id: stats_types id of the stat being rated.
    positions_by_player: {player_id: 'GK'|'DEF'|'MID'|'FWD'} — drives the
        below-floor fallback.
    fixture_ids: optional set of fixture_ids to restrict history to — used
        to scope to the last two seasons. fixture_player_stats has no
        season_id column (season lives on fixtures), so the caller resolves
        the seasons to fixtures and passes those.

    Sportmonks stores no zero rows, so a player who never dribbled has no rows
    at all — his numerator is legitimately 0 while his minutes still count.
    That is why minutes are summed independently of the stat rows.
    """
    ps = player_stats
    if fixture_ids is not None:
        ps = ps[ps["fixture_id"].isin(fixture_ids)]
    if ps.empty:
        return {}

    mins = (
        ps[ps["stats_type_id"] == MINUTES_STAT_ID]
        .drop_duplicates(subset=["player_id", "fixture_id"])
        .assign(m=lambda d: pd.to_numeric(d["value"], errors="coerce").fillna(0.0))
        .groupby("player_id")["m"].sum()
    )
    vals = (
        ps[ps["stats_type_id"] == stat_id]
        .drop_duplicates(subset=["player_id", "fixture_id"])
        .assign(v=lambda d: pd.to_numeric(d["value"], errors="coerce").fillna(0.0))
        .groupby("player_id")["v"].sum()
    )
    if mins.empty:
        return {}

    df = pd.DataFrame({"minutes": mins}).join(pd.DataFrame({"total": vals}), how="left")
    df["total"] = df["total"].fillna(0.0)
    df["per90"] = df["total"] / df["minutes"].replace(0, pd.NA) * 90.0
    df["position"] = df.index.map(lambda p: positions_by_player.get(p))

    qualified = df[(df["minutes"] >= MIN_MINUTES) & df["per90"].notna()]
    # Positional averages come from qualified players only — including the
    # thin-history players would let the noise we are trying to avoid leak
    # into the very number meant to replace it.
    pos_avg = qualified.groupby("position")["per90"].mean().to_dict()
    overall = float(qualified["per90"].mean()) if len(qualified) else FALLBACK_RATE

    rates = {}
    for pid, row in df.iterrows():
        if row["minutes"] >= MIN_MINUTES and pd.notna(row["per90"]):
            rates[pid] = float(row["per90"])
        else:
            rates[pid] = float(pos_avg.get(row["position"], overall))

    logger.info(
        "[static-rates] stat_id=%s players=%d own_rate=%d below_floor=%d pos_avg=%s",
        stat_id, len(rates), len(qualified), len(rates) - len(qualified),
        {k: round(v, 3) for k, v in pos_avg.items() if k},
    )
    return rates


def stamp_rate_columns(frame, rates_by_column):
    """Write lambda = per90 x xmin_bands / 90 onto the FPL-local frame.

    Called AFTER stamp_xmin_columns. These columns are already at expected
    minutes, so they must NOT also be in XMIN_SCALED_STAT_COLS — that would
    apply the minutes term twice.
    """
    if "xmin_bands" not in frame.columns:
        logger.warning("[static-rates] no xmin_bands column — skipping")
        return frame
    for col, rates in rates_by_column.items():
        per90 = pd.to_numeric(
            frame["player_id"].map(lambda p: rates.get(p, FALLBACK_RATE)),
            errors="coerce",
        ).fillna(FALLBACK_RATE)
        # Same per-90 companion convention as apply_per90_scaling, so the bonus
        # simulator reads every stat's rate the same way.
        frame[col + " per90"] = per90
        frame[col] = per90 * frame["xmin_bands"] / 90.0
    return frame
