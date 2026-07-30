"""Writer for fpl_player_per90_projections (xmins-methodology.md §11 Task 1).

Per-90 view of the FPL share model: share90 = share_legacy × 90 ÷ m_bar,
captured additively during distribute_team_predictions_to_players. Written
ONLY by the FPL branch (PL full runs, behind FPL_PER90_WRITE); read by
nothing yet — FPL point projections adopt it in a later, flagged step.

Grain: (competition_id, player_id, stat_name). per_match_series is display
material for the future admin panel, never an input to any estimate.
"""

import json
import logging

from app.repository.db_utils import execute_chunked

logger = logging.getLogger("fpl_per90")


async def insert_per90_shares_async(rows: list, competition_id: int):
    """rows: collector entries from distribute — dicts with player_id,
    stat_name, share_legacy, share90, m_bar, n_games, series."""
    if not rows:
        return 0

    values = []
    for r in rows:
        if r.get('share90') is None or r.get('m_bar') is None:
            # Empty sample — nothing meaningful to store (share_legacy is 0
            # and assembly multiplies to 0 regardless).
            continue
        values.append((
            competition_id,
            r['player_id'],
            r['stat_name'],
            round(float(r['share90']), 5),
            round(float(r['share_legacy']), 5),
            float(r['m_bar']),
            int(r['n_games']),
            json.dumps(r.get('series') or []),
        ))

    if not values:
        return 0

    sql = """
    INSERT INTO fpl_player_per90_projections (
        competition_id, player_id, stat_name, share90, share_legacy,
        m_bar, n_games, per_match_series, created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE
        share90 = new.share90,
        share_legacy = new.share_legacy,
        m_bar = new.m_bar,
        n_games = new.n_games,
        per_match_series = new.per_match_series,
        updated_at = NOW()
    """
    affected = await execute_chunked(sql, values, label="[fpl_player_per90]")
    logger.info(f"[fpl_player_per90] upserted {len(values)} (player, stat) share90 rows")
    return affected


async def insert_player_bands_async(profiles: dict, competition_id: int):
    """Persist standing per-player minutes bands (xmins-methodology §12
    Phase 0). profiles: {player_id: get_expected_minutes dict} — the
    PRE-confirmed-XI standing values (the profiles dict is built before any
    starter_override is applied to frame rows, so it is exactly that).
    The admin panel reads this table; bands are never recomputed in PHP."""
    values = []
    for pid, prof in profiles.items():
        if prof is None:
            continue
        p_play = prof.get('p_play')
        p60 = prof.get('p60')
        p90 = prof.get('p90')
        if p_play is None or p60 is None or p90 is None:
            continue
        values.append((competition_id, int(pid), float(p_play), float(p60), float(p90)))

    if not values:
        return 0

    sql = """
    INSERT INTO fpl_player_bands (
        competition_id, player_id, p_play, p60, p90, created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE
        p_play = new.p_play,
        p60 = new.p60,
        p90 = new.p90,
        updated_at = NOW()
    """
    affected = await execute_chunked(sql, values, label="[fpl_player_bands]")
    logger.info(f"[fpl_player_bands] upserted {len(values)} standing band rows")
    return affected
