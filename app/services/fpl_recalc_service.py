"""Instant per-player FPL recalc (George, 2026-07-30).

A dial edit invalidates nothing expensive — team projections and shares are
untouched — so points reassemble from the last run's persisted scoring
snapshot (fpl_assembly_bundles) in seconds:

    1. load bundles for every fixture the player appears in (full casts,
       so bonus ranks against real context)
    2. patch the dialed player's rows:
       - bands: ratio-rescale count columns by new_xmin_bands / old_xmin_bands
         (columns are linear in xmin_bands — exact), restamp band columns
       - goal/assist share dials: column := team stat projection × dial ×
         new_xmin_bands / 90 (replacement semantics, §12 Phase 5)
       - defcon dial: CBIT Hit Rate / def_con_pct := dial
    3. re-score with the run's own functions (get_fpl_points +
       bonus_points_score + get_bonus_points, same constants)
    4. UPDATE fpl_projections for the dialed player only; write patched
       bundle rows back so consecutive edits compose.

Full runs remain the truth-refresher (odds blend, new shares, teammates'
bonus ripple); this fast-forwards the edited player between runs.
"""

import logging

import pandas as pd

from app.repository.fpl_recalc_repo import (
    load_bundles_for_player, load_player_dial_and_bands,
    update_player_fpl_points,
)
from app.services.fpl_scoring_constants import (
    FPL_POINTS_GK, FPL_POINTS_DEF, FPL_POINTS_MID, FPL_POINTS_FWD,
    FPL_BONUS_GK, FPL_BONUS_DEF, FPL_BONUS_MID, FPL_BONUS_FWD,
)
from app.services.statz_functions import (
    get_fpl_points, bonus_points_score, get_bonus_points,
)
from app.services.xminutes import bands_to_xmins, XMIN_SCALED_STAT_COLS

logger = logging.getLogger("fpl_recalc")


def _f(v, default=None):
    if v is None:
        return default
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(v) else v


async def recalc_fpl_player(player_id: int) -> dict:
    player_id = int(player_id)
    dial, model_bands = await load_player_dial_and_bands(player_id)
    if dial is None:
        return {"ok": False, "error": "no dial row for player"}

    frame, score_preds, team_stats = await load_bundles_for_player(player_id)
    if frame is None or frame.empty:
        return {"ok": False, "error": "no assembly bundle — run a full PL projection first"}

    d_p_play, d_p60, d_p90, d_goal, d_assist, d_defcon = (_f(x) for x in dial)
    m_p_play, m_p60, m_p90 = ((_f(x) for x in model_bands) if model_bands
                              else (None, None, None))

    mask = frame['player_id'] == player_id
    if not mask.any():
        return {"ok": False, "error": "player missing from bundle frame"}

    # --- bands: dial overlays model, monotone chain, band-derived xMins ---
    p_play = d_p_play if d_p_play is not None else (m_p_play if m_p_play is not None else float(frame.loc[mask, 'xmin_p_play'].iloc[0]))
    p60 = d_p60 if d_p60 is not None else (m_p60 if m_p60 is not None else float(frame.loc[mask, 'xmin_p60'].iloc[0]))
    p90 = d_p90 if d_p90 is not None else (m_p90 if m_p90 is not None else float(frame.loc[mask, 'xmin_p90'].iloc[0]))
    p60 = min(p60, p_play)
    p90 = min(p90, p60)
    new_bands = round(bands_to_xmins(p_play, p60, p90), 1)

    old_bands = frame.loc[mask, 'xmin_bands'].astype(float).replace(0, pd.NA)
    ratio = (new_bands / old_bands).fillna(1.0)
    for col in XMIN_SCALED_STAT_COLS:
        if col in frame.columns:
            frame.loc[mask, col] = frame.loc[mask, col].astype(float) * ratio

    frame.loc[mask, 'xmin_p_play'] = round(p_play, 4)
    frame.loc[mask, 'xmin_p60'] = round(p60, 4)
    frame.loc[mask, 'xmin_p90'] = round(p90, 4)
    frame.loc[mask, 'xmin_bands'] = new_bands

    # --- share dials: replacement λ = team stat × dial × xmin_bands/90 ---
    for stat_col, team_key, dial_val in (('Goals', 'Goals', d_goal), ('Assists', 'Assists', d_assist)):
        if dial_val is None:
            continue
        for idx in frame.index[mask]:
            fid = int(frame.at[idx, 'fixture_id'])
            team = str(frame.at[idx, 'Team'])
            team_proj = _f((team_stats.get(fid, {}).get(team, {}) or {}).get(team_key), 0.0) or 0.0
            frame.at[idx, stat_col] = team_proj * dial_val * new_bands / 90.0

    # --- defcon dial: replaces the hit rate ---
    if d_defcon is not None:
        frame.loc[mask, 'CBIT Hit Rate'] = d_defcon
        frame.loc[mask, 'def_con_pct'] = round(d_defcon * 100, 2)

    # --- re-score with the run's own functions ---
    frame = frame.reset_index(drop=True)
    pts = get_fpl_points(frame, score_preds, FPL_POINTS_GK, FPL_POINTS_DEF, FPL_POINTS_MID, FPL_POINTS_FWD)
    bps = bonus_points_score(frame, score_preds, FPL_BONUS_GK, FPL_BONUS_DEF, FPL_BONUS_MID, FPL_BONUS_FWD)
    bonus = get_bonus_points(bps, score_preds, expo_factor=0.1)

    fpl_df = pts.merge(bonus, on=['Player', 'Team', 'Opponent'], how='left', suffixes=('', '_Bonus'))
    fpl_df['Bonus Points'] = fpl_df['Bonus Points'].fillna(0)
    fpl_df['FPL Points'] = fpl_df['PTS'] + fpl_df['Bonus Points']

    mine = fpl_df[fpl_df['player_id'] == player_id]
    def_con_val = float(frame.loc[mask, 'def_con_pct'].iloc[0]) if 'def_con_pct' in frame.columns else None
    updates = [
        (round(float(r['FPL Points']), 2), round(float(r['Bonus Points']), 2),
         def_con_val, new_bands, player_id, int(r['fixture_id']))
        for _, r in mine.iterrows()
    ]
    n = await update_player_fpl_points(updates)
    # NOTE: no bundle write-back — bundles hold the PURE MODEL frame and every
    # recalc layers the current dial state on top from scratch, so Reset-to-
    # model is instant and consecutive edits can't compound (2026-07-30).

    total = round(float(mine['FPL Points'].sum()), 2)
    logger.info(f"[fpl_recalc] player {player_id}: {n} fixtures updated, "
                f"bands {p_play:.2f}/{p60:.2f}/{p90:.2f} → xMins {new_bands}, total {total}")
    return {"ok": True, "player_id": player_id, "fixtures_updated": n,
            "xmins": new_bands, "points_total": total}
