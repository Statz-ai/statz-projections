"""Penalty-taker shares from FPL's own designated order.

The problem
-----------
A player's projected `Penalties Scored` used to come from his own penalty
history, like any other share. That is backward-looking in exactly the wrong
way: a new signing who has just been handed the duty projects zero penalties,
and last season's taker keeps being paid for them after losing the job. Penalty
duty is a *designation*, not a rate — it changes on a single team-sheet decision
and no amount of history sees it coming.

The model (George, 2026-08-07)
------------------------------
FPL publishes the designation itself as `penalties_order` (1, 2, 3...) per
player. The rule is a strict cascade:

    rank 1 takes the penalty if he is on the pitch;
    rank 2 takes it if he is on and rank 1 is not; and so on.

so, with m_i the probability taker i is on the pitch,

    P(1) = m1
    P(2) = (1 - m1) * m2
    P(3) = (1 - m1)(1 - m2) * m3

The leftover probability — nobody listed was on the pitch — goes to the club's
UNLISTED players, weighted by their own minutes, because in reality somebody
still steps up. It is deliberately NOT renormalised back over the listed takers:
doing that hands rank 1 the case where he was absent, which pushed him to 80.7%
against a measured 71.2%. Leaving the residual where it belongs keeps him at the
74.7% the validation below is based on.

Why m_i is xMins/90 and NOT p_play
----------------------------------
This is the difference between the model working and not. `p_play` is "did he
appear at all" — about 0.9 for a nailed starter — which puts rank 1 near 86%.
Penalties arrive roughly uniformly through a match, so what matters is the
probability he is on the pitch *at the moment one is awarded*, which is his
expected share of the 90 minutes.

Validated on PL 2024/25 + 2025/26, 30 club-seasons with 3+ penalties:

    top taker's ACTUAL share of his club's penalties   71.2%
    this rule's prediction from his minutes            74.7%

3.5 points apart on ~5 penalties per club-season, comfortably inside sampling
noise. No fitted weights, and deliberately no deference constant to close that
gap — a fudge factor would quietly rot as FPL's order data improves.

Consequences worth knowing
--------------------------
- A player with NO `penalties_order` no longer projects penalties from his own
  history: FPL's designation overrides history outright. He can still pick up a
  slice of the residual, which is the only route to a penalty for an unlisted
  player and matches the small rank-3/4 tail in the data (5.7% / 1.1%).
- The club's penalties always reconcile: listed shares plus residual sum to the
  team projection exactly.
- Independence across takers is an approximation — substitutions correlate who
  is on the pitch — but the validation above absorbs it.
"""

import logging

import numpy as np
import pandas as pd

from app.services.xminutes import PER90_SUFFIX

logger = logging.getLogger("projection")

PENS_COL = "Penalties Scored"
ORDER_COL = "penalties_order"


def cascade_shares(minutes_fracs):
    """P(each listed taker takes the penalty), in designated order.

    `minutes_fracs` is xMins/90 per taker, already in order. Returns
    (shares, residual) where shares has the same length as the input and
    `residual` is the probability that none of them was on the pitch.

    These are true probabilities and are NOT renormalised — see the module
    docstring. shares + residual == 1 by construction. An all-zero input
    returns a residual of 1.0, which the caller must read as "no opinion",
    not as "nobody takes penalties".
    """
    remaining = 1.0
    raw = []
    for m in minutes_fracs:
        m = min(max(float(m), 0.0), 1.0)
        raw.append(remaining * m)
        remaining *= (1.0 - m)
    return raw, remaining


def apply_penalty_order_shares(frame, order_by_pid, team_predictions):
    """Overwrite `Penalties Scored` from FPL's designated order.

    Mutates and returns `frame`. Must run AFTER per-90 scaling: the cascade
    already carries the minutes term (m_i = xMins/90), so the result is the
    player's expected penalties in THIS fixture and must not be scaled by
    minutes a second time. The per-90 companion is back-derived to preserve
    the identity the bonus simulator relies on:

        column == per90 * xmin_bands / 90
    """
    if PENS_COL not in frame.columns or not order_by_pid:
        return frame
    for col in ("fixture_id", "Team", "player_id", "xmin_bands"):
        if col not in frame.columns:
            logger.warning("[FPL pens] '%s' absent from frame — order shares skipped", col)
            return frame
    if PENS_COL not in getattr(team_predictions, "columns", []):
        logger.warning("[FPL pens] team '%s' absent — order shares skipped", PENS_COL)
        return frame

    tp = (
        team_predictions[["fixture_id", "Team", PENS_COL]]
        .rename(columns={PENS_COL: "_team_pens"})
        .drop_duplicates(subset=["fixture_id", "Team"])
    )
    merged = frame.merge(tp, on=["fixture_id", "Team"], how="left")
    team_pens = pd.to_numeric(merged["_team_pens"], errors="coerce")

    order = merged["player_id"].map(
        lambda p: order_by_pid.get(int(p)) if pd.notna(p) else None
    )
    order = pd.to_numeric(order, errors="coerce")
    mins = pd.to_numeric(merged["xmin_bands"], errors="coerce").fillna(0.0)

    new_pens = pd.Series(np.nan, index=merged.index, dtype=float)
    n_clubs = 0
    n_takers = 0

    for _, idx in merged.groupby(["fixture_id", "Team"], sort=False).groups.items():
        rows = merged.loc[idx]
        listed = rows[order.loc[idx].notna()]
        if listed.empty:
            # No designated taker for this club — leave every row's historical
            # share untouched rather than zeroing the whole club's penalties.
            continue
        tp_val = team_pens.loc[idx].dropna()
        if tp_val.empty:
            continue
        tp_val = float(tp_val.iloc[0])

        listed = listed.assign(_ord=order.loc[listed.index]).sort_values("_ord")
        fracs = (mins.loc[listed.index] / 90.0).tolist()
        shares, residual = cascade_shares(fracs)
        if sum(shares) <= 0:
            # Every listed taker at zero minutes. No opinion — history stands.
            continue

        # Start the whole club at zero so it reconciles to the team total, then
        # pay the listed takers their cascade probability.
        new_pens.loc[idx] = 0.0
        new_pens.loc[listed.index] = [s * tp_val for s in shares]

        # Residual: none of the listed takers was on the pitch, so an unlisted
        # one took it. Spread by minutes over the club's unlisted players. If
        # the club has none with minutes (every player designated), there is
        # nobody to receive it and it falls back to the listed takers in
        # proportion — the only case where renormalising is the right answer.
        if residual > 1e-9:
            unlisted = rows.index.difference(listed.index)
            # Goalkeepers are excluded as residual recipients. They play the
            # full 90, so minutes-weighting would hand the GK the single
            # largest slice of the residual — the one player who essentially
            # never takes one. An explicitly DESIGNATED GK still gets his
            # cascade share above; this only governs the leftover.
            if len(unlisted) and "FPL Position" in rows.columns:
                unlisted = unlisted[
                    rows.loc[unlisted, "FPL Position"].ne("GK").to_numpy()
                ]
            u_mins = mins.loc[unlisted] if len(unlisted) else pd.Series(dtype=float)
            u_tot = float(u_mins.sum())
            if u_tot > 0:
                new_pens.loc[unlisted] = (
                    (u_mins / u_tot) * residual * tp_val
                ).to_numpy()
            else:
                s_tot = sum(shares)
                if s_tot > 0:
                    new_pens.loc[listed.index] = [
                        (s / s_tot) * tp_val for s in shares
                    ]

        n_clubs += 1
        n_takers += len(listed)

    applied = new_pens.notna()
    if not applied.any():
        logger.warning("[FPL pens] no club could be resolved — order shares skipped")
        return frame

    out = pd.to_numeric(frame[PENS_COL], errors="coerce")
    out.loc[applied.to_numpy()] = new_pens[applied].to_numpy()
    frame[PENS_COL] = out.to_numpy()

    # Per-90 companion, back-derived so col == per90 * xmin_bands / 90 still
    # holds. A player at zero expected minutes has no meaningful rate; leave
    # it at 0 rather than dividing by zero.
    p90_col = PENS_COL + PER90_SUFFIX
    if p90_col in frame.columns:
        denom = mins.to_numpy() / 90.0
        rate = np.divide(
            frame[PENS_COL].to_numpy(dtype=float), denom,
            out=np.zeros(len(frame), dtype=float), where=denom > 0,
        )
        p90 = pd.to_numeric(frame[p90_col], errors="coerce").to_numpy(dtype=float)
        p90[applied.to_numpy()] = rate[applied.to_numpy()]
        frame[p90_col] = p90

    logger.info(
        "[FPL pens] order shares applied: %d club-fixtures, %d designated "
        "taker rows, %d unlisted rows (residual only)",
        n_clubs, n_takers, int(applied.sum()) - n_takers,
    )
    return frame
