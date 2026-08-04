"""Assembly-bundle persistence + loaders for the instant FPL recalc path.

Each PL run snapshots its final FPL scoring frame per (player, fixture) into
fpl_assembly_bundles, plus one player_id=0 context row per fixture (score
prediction + team stat projections). Recalc patches the dialed player inside
that snapshot and re-scores with the run's own functions — exact, seconds.
"""

import json
import logging
import math

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
    'Big Chances Created', 'Successful Dribbles', 'Big Chances Missed',
    'Interceptions', 'Shots Total', 'Shots On Target', 'Passes',
    'Accurate Passes', 'Fouls', 'Fouls Drawn', 'Offsides', 'Total Crosses',
    'Clearances Average', 'Blocked Shots Average', 'Ball Recovery Average',
    'Tackles Won Average', 'CBIT Average', 'CBIT Hit Rate',
    'Full Match Hit Rate', 'def_con_pct',
    # The assembled defensive-contribution rate per 90 (CBIT for DEF, CBIT +
    # recoveries for MID/FWD, or the defcon_share dial where set). Banked as
    # ONE column rather than its components so recalc can re-band the DefCon
    # threshold when a minutes dial moves, without re-deriving the
    # position-dependent denominator. Without it recalc left def_con_pct stale
    # on every band edit. George, 2026-08-04.
    'dc_rate90',
    'xmin_p_play', 'xmin_p60', 'xmin_p90', 'xmin_bands', 'xmin_start_len',
] + [
    # Per-90 companions (xminutes.PER90_SUFFIX). The bonus simulator samples a
    # minutes band per player, so it needs the rate BEFORE the minutes term.
    # Without these the recalc path would silently simulate off lambda-at-
    # expected-minutes and understate everyone, worst for subs.
    c + ' per90' for c in (
        'Goals', 'Assists', 'Key Passes', 'Big Chances Created',
        'Big Chances Missed', 'Shots Total', 'Shots On Target', 'Passes',
        'Accurate Passes', 'Fouls', 'Fouls Drawn', 'Offsides', 'Yellow Cards',
        'Saves', 'Interceptions', 'Total Crosses', 'Successful Dribbles',
        'Clearances Average', 'Blocked Shots Average', 'Ball Recovery Average',
        'Tackles Won Average',
    )
]

SCORE_PRED_COLS = ['id', 'Home Team', 'Away Team', 'Home Goals', 'Away Goals',
                   'Home Clean Sheet %', 'Away Clean Sheet %']


async def save_assembly_bundles(frame, score_preds, team_predictions):
    """Replace all bundles with this run's snapshot (PL-only feature, so a
    full swap is correct: DELETE + insert)."""
    def _clean(d):
        # json.dumps emits literal NaN for float('nan') (allow_nan default),
        # which MySQL's JSON type rejects ("Invalid JSON text ... position
        # 129") — both 2026-07-30 seeding runs failed on it. NaN -> null.
        return {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in d.items()
        }

    rows = []
    present = [c for c in BUNDLE_COLS if c in frame.columns]
    for rec in frame[present].to_dict('records'):
        fid = rec.get('fixture_id')
        pid = rec.get('player_id')
        if fid is None or pid is None or pd.isna(fid) or pd.isna(pid):
            continue
        rec['kickoff_datetime'] = str(rec.get('kickoff_datetime'))
        rows.append((int(fid), int(pid), json.dumps(_clean(rec), default=str)))

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
        ctx = {'score_pred': _clean(rec), 'team_stats': tp_by_fix.get(int(fid), {})}
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


async def load_bundles_for_players(player_ids):
    """All bundle rows (players + context) for every fixture ANY of the given
    players appears in. Returns (frame_df, score_preds_df, team_stats_by_fixture)."""
    if not player_ids:
        return None, None, None
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            ph_p = ",".join(["%s"] * len(player_ids))
            await cur.execute(
                f"SELECT DISTINCT fixture_id FROM fpl_assembly_bundles WHERE player_id IN ({ph_p})",
                tuple(int(p) for p in player_ids),
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


async def load_all_dials_and_bands():
    """ALL dial rows + model bands, keyed by player_id. Recalc applies every
    dial present in the loaded frame — never just the requested players —
    so updating full fixture casts can't regress another dialed player to
    model values."""
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT player_id, p_play, p60, p90, goal_share, assist_share, defcon_share FROM fpl_player_dials"
            )
            dials = {int(r[0]): r[1:] for r in await cur.fetchall()}
            await cur.execute("SELECT player_id, p_play, p60, p90 FROM fpl_player_bands")
            bands = {int(r[0]): r[1:] for r in await cur.fetchall()}
    finally:
        if _db.pool:
            _db.pool.release(conn)
    return dials, bands


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


async def load_existing_fpl_pairs(player_ids):
    """(player_id, fixture_id) pairs that actually exist in fpl_projections
    for these players — the fantasy-scoped truth the panel reads."""
    if not player_ids:
        return set()
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            ph = ",".join(["%s"] * len(player_ids))
            await cur.execute(
                f"SELECT player_id, fixture_id FROM fpl_projections WHERE player_id IN ({ph})",
                tuple(int(p) for p in player_ids),
            )
            return {(int(r[0]), int(r[1])) for r in await cur.fetchall()}
    finally:
        if _db.pool:
            _db.pool.release(conn)


async def load_existing_bonus(fixture_ids):
    """Bonus already stored in fpl_projections for these fixtures.

    The recalc path no longer recomputes bonus (George, 2026-08-04): bonus is a
    RANK, so changing one player forces re-simulating every fixture he appears
    in, which took the panel's Update past its 60s timeout. Points update
    instantly; bonus is carried through unchanged and refreshes on the next run.

    Carrying it is not optional — FPL Points = PTS + Bonus, so returning zero
    would silently strip bonus from every dialled player.

    Returns a DataFrame [fixture_id, player_id, 'Bonus Points'].
    """
    if not fixture_ids:
        return pd.DataFrame(columns=['fixture_id', 'player_id', 'Bonus Points'])
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            ph = ",".join(["%s"] * len(fixture_ids))
            await cur.execute(
                f"SELECT fixture_id, player_id, bonus FROM fpl_projections WHERE fixture_id IN ({ph})",
                tuple(int(f) for f in fixture_ids),
            )
            rows = await cur.fetchall()
    finally:
        if _db.pool:
            _db.pool.release(conn)
    return pd.DataFrame(
        [{'fixture_id': int(r[0]), 'player_id': int(r[1]),
          'Bonus Points': float(r[2]) if r[2] is not None else 0.0} for r in rows],
        columns=['fixture_id', 'player_id', 'Bonus Points'],
    )
