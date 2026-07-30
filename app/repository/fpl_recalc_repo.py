"""Assembly-bundle persistence + loaders for the instant FPL recalc path.

Each PL run snapshots its final FPL scoring frame per (player, fixture) into
fpl_assembly_bundles, plus one player_id=0 context row per fixture (score
prediction + team stat projections). Recalc patches the dialed player inside
that snapshot and re-scores with the run's own functions — exact, seconds.
"""

import json
import logging

import pandas as pd

from app.database import get_connection
from app.repository.db_utils import execute_chunked
import app.database as _db

logger = logging.getLogger("fpl_recalc")

# Frame columns the scoring functions consume (get_fpl_points +
# bonus_points_score). Serialized per row; missing columns default 0.
BUNDLE_COLS = [
    'fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'FPL Position',
    'Team', 'Opponent', 'Venue', 'Gameweek',
    'Goals', 'Assists', 'Yellow Cards', 'Saves', 'Key Passes',
    'Interceptions', 'Shots Total', 'Shots On Target', 'Passes',
    'Accurate Passes', 'Fouls', 'Fouls Drawn', 'Offsides', 'Total Crosses',
    'Clearances Average', 'Blocked Shots Average', 'Ball Recovery Average',
    'Tackles Won Average', 'CBIT Average', 'CBIT Hit Rate',
    'Full Match Hit Rate', 'def_con_pct',
    'xmin_p_play', 'xmin_p60', 'xmin_p90', 'xmin_bands', 'xmin_start_len',
]

SCORE_PRED_COLS = ['id', 'Home Team', 'Away Team', 'Home Goals', 'Away Goals',
                   'Home Clean Sheet %', 'Away Clean Sheet %']


async def save_assembly_bundles(frame, score_preds, team_predictions):
    """Replace all bundles with this run's snapshot (PL-only feature, so a
    full swap is correct: DELETE + insert)."""
    rows = []
    present = [c for c in BUNDLE_COLS if c in frame.columns]
    for rec in frame[present].to_dict('records'):
        fid = rec.get('fixture_id')
        pid = rec.get('player_id')
        if fid is None or pid is None or pd.isna(fid) or pd.isna(pid):
            continue
        rec['kickoff_datetime'] = str(rec.get('kickoff_datetime'))
        rows.append((int(fid), int(pid), json.dumps(rec, default=str)))

    # Context rows: score prediction + team stat projections per fixture.
    team_cols = [c for c in team_predictions.columns if c not in ('fixture_id', 'Team')]
    tp_by_fix = {}
    for rec in team_predictions.to_dict('records'):
        fid = rec.get('fixture_id')
        if fid is None or pd.isna(fid):
            continue
        tp_by_fix.setdefault(int(fid), {})[str(rec.get('Team'))] = {
            c: (None if pd.isna(rec.get(c)) else rec.get(c))
            for c in team_cols if not isinstance(rec.get(c), (list, dict))
        }
    sp_present = [c for c in SCORE_PRED_COLS if c in score_preds.columns]
    for rec in score_preds[sp_present].to_dict('records'):
        fid = rec.get('id')
        if fid is None or pd.isna(fid):
            continue
        ctx = {'score_pred': rec, 'team_stats': tp_by_fix.get(int(fid), {})}
        rows.append((int(fid), 0, json.dumps(ctx, default=str)))

    if not rows:
        return 0

    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM fpl_assembly_bundles")
        await conn.commit()
    finally:
        if _db.pool:
            _db.pool.release(conn)

    sql = """
    INSERT INTO fpl_assembly_bundles (fixture_id, player_id, payload, created_at, updated_at)
    VALUES (%s, %s, %s, NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE payload = new.payload, updated_at = NOW()
    """
    affected = await execute_chunked(sql, rows, label="[fpl_assembly_bundles]")
    logger.info(f"[fpl_assembly_bundles] snapshotted {len(rows)} rows")
    return affected


async def load_bundles_for_player(player_id: int):
    """All bundle rows (players + context) for every fixture the player has a
    bundle row in. Returns (frame_df, score_preds_df, team_stats_by_fixture)."""
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT fixture_id FROM fpl_assembly_bundles WHERE player_id = %s",
                (player_id,),
            )
            fixture_ids = [r[0] for r in await cur.fetchall()]
            if not fixture_ids:
                return None, None, None
            ph = ",".join(["%s"] * len(fixture_ids))
            await cur.execute(
                f"SELECT fixture_id, player_id, payload FROM fpl_assembly_bundles WHERE fixture_id IN ({ph})",
                tuple(fixture_ids),
            )
            rows = await cur.fetchall()
    finally:
        if _db.pool:
            _db.pool.release(conn)

    frame_rows, score_rows, team_stats = [], [], {}
    for fid, pid, payload in rows:
        data = json.loads(payload)
        if pid == 0:
            score_rows.append(data.get('score_pred', {}))
            team_stats[int(fid)] = data.get('team_stats', {})
        else:
            frame_rows.append(data)

    frame = pd.DataFrame(frame_rows)
    for col in BUNDLE_COLS:
        if col not in frame.columns:
            frame[col] = 0
    score_preds = pd.DataFrame(score_rows)
    return frame, score_preds, team_stats


async def load_player_dial_and_bands(player_id: int):
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT p_play, p60, p90, goal_share, assist_share, defcon_pct FROM fpl_player_dials WHERE player_id = %s",
                (player_id,),
            )
            dial = await cur.fetchone()
            await cur.execute(
                "SELECT p_play, p60, p90 FROM fpl_player_bands WHERE player_id = %s",
                (player_id,),
            )
            bands = await cur.fetchone()
    finally:
        if _db.pool:
            _db.pool.release(conn)
    return dial, bands


async def update_player_fpl_points(updates):
    """updates: list of (fpl_points, bonus, def_con_pct, expected_minutes,
    player_id, fixture_id)."""
    if not updates:
        return 0
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.executemany(
                """UPDATE fpl_projections
                   SET fpl_points = %s, bonus = %s, def_con_pct = %s,
                       expected_minutes = %s, updated_at = NOW()
                   WHERE player_id = %s AND fixture_id = %s""",
                updates,
            )
        await conn.commit()
    finally:
        if _db.pool:
            _db.pool.release(conn)
    return len(updates)


async def update_player_bundles(frame, player_id: int):
    """Write the dialed player's patched rows back so consecutive recalcs
    compose from current state, not the last full run's."""
    sub = frame[frame['player_id'] == player_id]
    rows = [
        (int(r['fixture_id']), int(r['player_id']), json.dumps(r, default=str))
        for r in sub.to_dict('records')
    ]
    if not rows:
        return 0
    sql = """
    INSERT INTO fpl_assembly_bundles (fixture_id, player_id, payload, created_at, updated_at)
    VALUES (%s, %s, %s, NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE payload = new.payload, updated_at = NOW()
    """
    return await execute_chunked(sql, rows, label="[fpl_assembly_bundles:patch]")
