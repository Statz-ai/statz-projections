"""Instant FPL recalc for edited players (George, 2026-07-30).

Dial edits invalidate nothing expensive — team projections and shares are
untouched — so points reassemble from the last run's PURE-MODEL scoring
snapshot (fpl_assembly_bundles) in seconds:

    1. requested player_ids define the FIXTURE closure (bonus is
       fixture-scoped, so an edit's true blast radius is the fixtures the
       player appears in); full casts load for exact bonus ranking
    2. EVERY dial present in the loaded frame is applied from scratch on
       the model snapshot — not just the requested players — so updating
       full casts can never regress another dialed player, Reset-to-model
       is instant/exact, and consecutive edits can't compound
    3. re-score with the run's own functions + verbatim constants
    4. bulk UPDATE fpl_projections for all players in the re-scored
       fixtures (trues the teammate bonus ripple within the closure too)

Full runs stay the model refresher (new data, odds blend); this
fast-forwards assembly between them. Duration is measured and returned —
the admin panel displays the real number.
"""

import logging
import time

import pandas as pd

from app.repository.fpl_recalc_repo import (
    load_bundles_for_players, load_all_dials_and_bands,
    update_player_fpl_points,
)
from app.services.fpl_scoring_constants import (
    FPL_POINTS_GK, FPL_POINTS_DEF, FPL_POINTS_MID, FPL_POINTS_FWD,
    FPL_BONUS_GK, FPL_BONUS_DEF, FPL_BONUS_MID, FPL_BONUS_FWD,
)
from app.services.statz_functions import (
    get_fpl_points,
)
from app.services.xminutes import (
    bands_to_xmins, XMIN_SCALED_STAT_COLS, DC_RATE_COL,
    defcon_threshold, defcon_band_hit_rate,
)

logger = logging.getLogger("fpl_recalc")


def _f(v, default=None):
    if v is None:
        return default
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(v) else v


def _apply_dials_to_player(frame, mask, dial, model_band, team_stats):
    """Apply one player's dial columns onto his pure-model rows in frame.
    Returns the player's new band-derived xMins."""
    d_p_play, d_p60, d_p90, d_goal, d_assist, d_defcon = (_f(x) for x in dial)
    m_p_play, m_p60, m_p90 = ((_f(x) for x in model_band) if model_band
                              else (None, None, None))

    p_play = d_p_play if d_p_play is not None else (m_p_play if m_p_play is not None else float(frame.loc[mask, 'xmin_p_play'].iloc[0]))
    p60 = d_p60 if d_p60 is not None else (m_p60 if m_p60 is not None else float(frame.loc[mask, 'xmin_p60'].iloc[0]))
    p90 = d_p90 if d_p90 is not None else (m_p90 if m_p90 is not None else float(frame.loc[mask, 'xmin_p90'].iloc[0]))
    p60 = min(p60, p_play)
    p90 = min(p90, p60)
    new_bands = round(bands_to_xmins(p_play, p60, p90), 1)

    # Count columns are linear in xmin_bands → exact ratio rescale from model.
    old_bands = frame.loc[mask, 'xmin_bands'].astype(float).replace(0, pd.NA)
    ratio = (new_bands / old_bands).fillna(1.0)
    for col in XMIN_SCALED_STAT_COLS:
        if col in frame.columns:
            frame.loc[mask, col] = frame.loc[mask, col].astype(float) * ratio

    frame.loc[mask, 'xmin_p_play'] = round(p_play, 4)
    frame.loc[mask, 'xmin_p60'] = round(p60, 4)
    frame.loc[mask, 'xmin_p90'] = round(p90, 4)
    frame.loc[mask, 'xmin_bands'] = new_bands

    # Share dials: replacement λ = team stat projection × dial × xMins/90.
    for stat_col, dial_val in (('Goals', d_goal), ('Assists', d_assist)):
        if dial_val is None:
            continue
        for idx in frame.index[mask]:
            fid = int(frame.at[idx, 'fixture_id'])
            team = str(frame.at[idx, 'Team'])
            team_proj = _f((team_stats.get(fid, {}).get(team, {}) or {}).get(stat_col), 0.0) or 0.0
            frame.at[idx, stat_col] = team_proj * dial_val * new_bands / 90.0

    # Defensive contribution. defcon_share is a share of the team's DC total
    # (CBIT for DEF, CBIT + recoveries for MID/FWD), so it becomes a RATE and
    # then bands like any other — it is not a flat percentage override.
    # Re-banded even with NO dc dial, because a MINUTES dial changes the band
    # weights and the threshold is not linear in minutes; skipping that left
    # def_con_pct stale on every band edit.
    if DC_RATE_COL in frame.columns:
        for idx in frame.index[mask]:
            pos = frame.at[idx, 'FPL Position'] if 'FPL Position' in frame.columns else None
            threshold = defcon_threshold(pos)
            if threshold is None:
                continue
            if d_defcon is not None:
                fid = int(frame.at[idx, 'fixture_id'])
                team = str(frame.at[idx, 'Team'])
                _ts = (team_stats.get(fid, {}).get(team, {}) or {})
                team_total = _f(_ts.get('Clearances Blocks Interceptions Tackles (FPL)'), 0.0) or 0.0
                if pos != 'DEF':
                    team_total += _f(_ts.get('Ball Recovery'), 0.0) or 0.0
                rate90 = team_total * d_defcon
                frame.at[idx, DC_RATE_COL] = rate90
            else:
                rate90 = _f(frame.at[idx, DC_RATE_COL], 0.0) or 0.0
            hit = defcon_band_hit_rate(rate90, p_play, p60, p90, threshold)
            frame.at[idx, 'CBIT Hit Rate'] = hit
            frame.at[idx, 'def_con_pct'] = round(hit * 100, 2)

    return new_bands


async def recalc_fpl_players(player_ids) -> dict:
    t0 = time.monotonic()
    player_ids = [int(p) for p in player_ids]
    if not player_ids:
        return {"ok": False, "error": "player_ids required"}

    frame, score_preds, team_stats = await load_bundles_for_players(player_ids)
    if frame is None or frame.empty:
        return {"ok": False, "error": "no assembly bundle — run a full PL projection first",
                "status": {pid: "no_bundle" for pid in player_ids}}
    # Per-player status (panel warns on no_bundle): a requested player can
    # lack bundle rows when he was FPL-mapped after the last full run — the
    # membership gate (2026-07-30) bundles every mapped player each run, so
    # this self-heals at the next run.
    _present = set(frame['player_id'].dropna().astype(int))
    status = {pid: ("updated" if pid in _present else "no_bundle") for pid in player_ids}

    dials, model_bands = await load_all_dials_and_bands()

    # Apply EVERY dial whose player appears in the closure (see docstring).
    applied = 0
    for pid, dial in dials.items():
        mask = frame['player_id'] == pid
        if mask.any():
            _apply_dials_to_player(frame, mask, dial, model_bands.get(pid), team_stats)
            applied += 1

    frame = frame.reset_index(drop=True)

    # Penalty-taker cascade, re-derived here rather than trusted from the
    # bundle. It depends on xmin_bands, which the dials above may just have
    # moved, and on the admin's penalty order, which may have changed since the
    # run. Runs AFTER the dial loop for the first reason and after
    # reset_index because the cascade groups by position.
    #
    # Bonus does NOT refresh here (see the note below) — and bonus is where a
    # penalty reassignment bites hardest, since a penalty is 12 BPS flat
    # against 12/18/24 by position. Points do move: duty carries ~4-5 points
    # per penalty goal with it. So the panel shows the points effect instantly
    # and the bonus effect at the next full run, exactly like every other dial.
    try:
        from app.repository.fpl_recalc_repo import load_penalty_orders
        from app.services.fpl_penalties import apply_penalty_order_shares
        _pen_orders = await load_penalty_orders()
        if _pen_orders and team_stats:
            _tp_rows = [
                {'fixture_id': int(fid), 'Team': tname, **{
                    k: v for k, v in stats.items() if not isinstance(v, (list, dict))
                }}
                for fid, teams in team_stats.items()
                for tname, stats in (teams or {}).items()
            ]
            if _tp_rows:
                frame = apply_penalty_order_shares(
                    frame, _pen_orders, pd.DataFrame(_tp_rows)
                )
    except Exception as _pen_err:
        # Non-fatal: the bundle's stored penalty split stands, which is the
        # last full run's answer rather than a broken one.
        logger.warning(f"[fpl_recalc] penalty order shares skipped: {_pen_err}")

    pts = get_fpl_points(frame, score_preds, FPL_POINTS_GK, FPL_POINTS_DEF, FPL_POINTS_MID, FPL_POINTS_FWD)
    # Bonus is NOT recomputed here (George, 2026-08-04). It is a RANK, so
    # changing one player forces re-simulating every fixture he appears in —
    # ~1.7s per fixture at 5,000 samples, which took the panel's Update button
    # past its 60s timeout (24.3s for just 20 fixtures).
    #
    # Instead the stored bonus is carried through unchanged: points update
    # instantly and bonus refreshes on the next full run. The cost is a window
    # of up to one run where a dialled player's bonus — and his team-mates',
    # since it is competitive — reflect the pre-dial minutes.
    #
    # Carrying it is not optional: FPL Points = PTS + Bonus, so computing zero
    # would silently strip bonus from every dialled player.
    from app.repository.fpl_recalc_repo import load_existing_bonus
    _fix_ids = sorted({int(f) for f in pts['fixture_id'].dropna().unique()})
    bonus = await load_existing_bonus(_fix_ids)
    logger.info(f"[fpl_recalc] bonus carried through unchanged for {len(bonus)} rows "
                f"across {len(_fix_ids)} fixtures (not recomputed)")

    fpl_df = pts.merge(bonus, on=['fixture_id', 'player_id'], how='left')
    fpl_df['Bonus Points'] = fpl_df['Bonus Points'].fillna(0)
    fpl_df['FPL Points'] = fpl_df['PTS'] + fpl_df['Bonus Points']
    fpl_df = fpl_df.drop_duplicates(['player_id', 'fixture_id'])

    # Row context (def_con_pct / xmin_bands) back onto the scored rows.
    ctx = frame[['fixture_id', 'player_id', 'def_con_pct', 'xmin_bands']].drop_duplicates(['fixture_id', 'player_id'])
    fpl_df = fpl_df.merge(ctx, on=['fixture_id', 'player_id'], how='left')

    updates = [
        (round(float(r['FPL Points']), 2), round(float(r['Bonus Points']), 2),
         _f(r['def_con_pct']), _f(r['xmin_bands']),
         int(r['player_id']), int(r['fixture_id']))
        for _, r in fpl_df.iterrows()
        if pd.notna(r['player_id']) and pd.notna(r['fixture_id'])
    ]
    n = await update_player_fpl_points(updates)

    # Totals over rows that EXIST in fpl_projections only: bundles span the
    # full 200-fixture horizon but the FPL insert scopes to the fantasy
    # gameweek window, so summing every bundle fixture overstated totals
    # (2026-07-30 verification catch). The UPDATE is a no-op for the extras.
    from app.repository.fpl_recalc_repo import load_existing_fpl_pairs
    existing = await load_existing_fpl_pairs(player_ids)
    edited = fpl_df[fpl_df.apply(
        lambda r: pd.notna(r['player_id']) and pd.notna(r['fixture_id'])
        and (int(r['player_id']), int(r['fixture_id'])) in existing, axis=1)]
    totals = {int(pid): round(float(g['FPL Points'].sum()), 2)
              for pid, g in edited.groupby('player_id')}
    duration = round(time.monotonic() - t0, 1)
    n_fixtures = int(frame['fixture_id'].nunique())
    logger.info(f"[fpl_recalc] {len(player_ids)} requested / {applied} dials applied / "
                f"{n_fixtures} fixtures re-scored / {n} rows updated in {duration}s")
    return {"ok": True, "requested": player_ids, "dials_applied": applied,
            "fixtures_rescored": n_fixtures, "rows_updated": n,
            "points_totals": totals, "status": status,
            "duration_seconds": duration}


async def recalc_fpl_player(player_id: int) -> dict:
    """Back-compat single-player wrapper."""
    return await recalc_fpl_players([int(player_id)])
