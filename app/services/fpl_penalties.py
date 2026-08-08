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


def cascade_shares(takers):
    """P(each listed taker takes the penalty).

    `takers` is a list of (rank, weight, minutes_frac):
        rank    the tier. Players SHARING a rank share the duty.
        weight  split within the tier (any positive scale; 60/40 or 6/4).
                Ignored when a tier holds one player.
        minutes_frac  xMins/90 — probability he is on the pitch.

    Returns (shares, residual), shares aligned to the input order, residual
    the probability nobody listed was on. shares + residual == 1 exactly.
    Not renormalised — see the module docstring.

    Tiers cascade: a tier only takes the penalty when NO player in any
    earlier tier is on the pitch. Within a tier the duty is split by weight
    across whoever is actually on, which is why a 60/40 pair does not come
    out at exactly 60/40 — when one is absent the other takes all of it.

    An all-zero-minutes input returns residual 1.0, which the caller must
    read as "no opinion", not "nobody takes penalties".
    """
    idx_by_rank = {}
    for i, (rank, _w, _m) in enumerate(takers):
        idx_by_rank.setdefault(rank, []).append(i)

    shares = [0.0] * len(takers)
    reached = 1.0   # P(no one in any earlier tier was on the pitch)

    for rank in sorted(idx_by_rank):
        members = idx_by_rank[rank]
        ms = [min(max(float(takers[i][2]), 0.0), 1.0) for i in members]
        ws = [max(float(takers[i][1] or 0.0), 0.0) for i in members]
        # A tier whose weights are all zero (or unset) splits evenly — an
        # admin who set a rank but no percentages means "these share it".
        if sum(ws) <= 0:
            ws = [1.0] * len(members)

        # Enumerate which members of THIS tier are on the pitch. Tiers hold
        # 1-3 players in practice, so 2^k is trivial and exact — no need to
        # approximate the "both on" / "one on" cases separately.
        for mask in range(1, 1 << len(members)):
            present = [j for j in range(len(members)) if mask & (1 << j)]
            p = 1.0
            for j in range(len(members)):
                p *= ms[j] if (mask & (1 << j)) else (1.0 - ms[j])
            if p <= 0:
                continue
            wsum = sum(ws[j] for j in present)
            if wsum <= 0:
                continue
            for j in present:
                shares[members[j]] += reached * p * (ws[j] / wsum)

        # Move to the next tier only in the worlds where nobody here was on.
        none_on = 1.0
        for m in ms:
            none_on *= (1.0 - m)
        reached *= none_on

    return shares, reached


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

    # order_by_pid values are either a bare rank (FPL's penalties_order) or a
    # (rank, weight) pair once an admin has split a tier. Both accepted so the
    # loader can keep handing over FPL's raw column untouched.
    def _rank_of(p):
        v = order_by_pid.get(int(p)) if pd.notna(p) else None
        return (v[0] if isinstance(v, (tuple, list)) else v)

    def _weight_of(p):
        v = order_by_pid.get(int(p)) if pd.notna(p) else None
        return (v[1] if isinstance(v, (tuple, list)) and len(v) > 1 else None)

    order = pd.to_numeric(merged["player_id"].map(_rank_of), errors="coerce")
    weight = pd.to_numeric(merged["player_id"].map(_weight_of), errors="coerce")
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
        takers = [
            (
                float(order.loc[i]),
                float(weight.loc[i]) if pd.notna(weight.loc[i]) else 0.0,
                float(mins.loc[i]) / 90.0,
            )
            for i in listed.index
        ]
        shares, residual = cascade_shares(takers)
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

    # Reconciliation. The cascade assigns `share x team_pens` with shares plus
    # residual summing to 1, so the allocated total MUST equal the team total
    # over the clubs it touched. Emitted every run because the alternative is
    # measuring it by hand afterwards — which is how a whole evening went into
    # "penalties are 25% light" that turned out to be the wrong frame entirely
    # (the bundle snapshot, which deliberately has no cascade). If this line
    # ever reads anything but ~1.000, the allocation really is broken.
    _alloc = float(new_pens[applied].sum())
    _team_total = 0.0
    for _, idx in merged.groupby(["fixture_id", "Team"], sort=False).groups.items():
        if not applied.loc[idx].any():
            continue
        _tp = team_pens.loc[idx].dropna()
        if not _tp.empty:
            _team_total += float(_tp.iloc[0])
    _ratio = (_alloc / _team_total) if _team_total > 0 else float("nan")
    logger.info(
        "[FPL pens] order shares applied: %d club-fixtures, %d designated "
        "taker rows, %d unlisted rows (residual only) | allocated %.3f vs "
        "team %.3f = %.4f (expect 1.0000)",
        n_clubs, n_takers, int(applied.sum()) - n_takers,
        _alloc, _team_total, _ratio,
    )
    if _team_total > 0 and abs(_ratio - 1.0) > 0.005:
        logger.warning(
            "[FPL pens] allocation does NOT reconcile (%.4f) — player penalties "
            "do not sum to the team projection", _ratio,
        )
    return frame
