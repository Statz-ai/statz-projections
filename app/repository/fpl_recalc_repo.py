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

# Mirrors data_loader.FPL_SNAPSHOT_MIN_PLAYERS — a snapshot date is only a
# complete bootstrap above this many distinct players.
FPL_SNAPSHOT_MIN_PLAYERS = 400

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
    # Team-down defensive projections the bonus simulator now scores BPS from
    # (replacing the "* Average" career means). Without these in the bundle the
    # recalc path would fall back to the Averages and score bonus off different
    # numbers than the run did.
    'Tackles Won', 'Ball Recovery', 'Clearances Blocks Interceptions (FPL)',
    # Penalty split. Both are needed: the points path scores non-penalty and
    # penalty goals at the same 4 points but charges misses, and the bonus
    # simulator scores penalties at 12 BPS flat instead of 12/18/24. Absent
    # from the bundle, recalc falls back to the undivided 'Goals' and quietly
    # awards every taker his position rate on spot-kicks — so a recalc would
    # disagree with the run that produced it.
    'Non-Penalty Goals', 'Penalties Scored',
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
        'Tackles Won', 'Ball Recovery', 'Clearances Blocks Interceptions (FPL)',
        'Non-Penalty Goals', 'Penalties Scored',
    )
]

SCORE_PRED_COLS = ['id', 'Home Team', 'Away Team', 'Home Goals', 'Away Goals',
                   'Home Clean Sheet %', 'Away Clean Sheet %']


async def save_assembly_bundles(frame, score_preds, team_predictions):
    """Replace THIS RUN'S fixtures in the bundle snapshot.

    Scoped to the fixtures the run actually wrote, NOT a full table wipe.

    It used to be `DELETE FROM fpl_assembly_bundles`, justified as "PL-only
    feature, so a full swap is correct". That holds for a full run and fails
    completely for a partial one. On 2026-08-21 a single-fixture re-projection
    fired when confirmed lineups landed for the 19:00 kick-off, and it deleted
    all 11,740 bundle rows and replaced them with 60 — one fixture.

    Two things broke, one of them silently:
      - the admin dials panel reads its team projections (tg/ta/tcbit/trec)
        from the player_id=0 context rows, so every ghost line under the
        sliders went blank;
      - RECALC loads from bundles, so ~495 of 554 players would have come back
        "no_bundle" on the next Update.

    Deleting per fixture keeps a partial run honest: it replaces its own
    fixtures and leaves every other fixture's snapshot alone.
    """
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

    # Fixtures this run is about to write. Anything else in the table belongs
    # to a fixture this run did not touch and must survive.
    _fixture_ids = sorted({int(r[0]) for r in rows})
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            for _i in range(0, len(_fixture_ids), 500):
                _chunk = _fixture_ids[_i:_i + 500]
                _ph = ",".join(["%s"] * len(_chunk))
                await cur.execute(
                    f"DELETE FROM fpl_assembly_bundles WHERE fixture_id IN ({_ph})",
                    tuple(_chunk),
                )
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
    logger.info(f"[fpl_assembly_bundles] snapshotted {len(rows)} rows "
                f"across {len(_fixture_ids)} fixtures (other fixtures untouched)")
    return affected


async def load_penalty_orders():
    """{player_id: (rank, weight)} for every designated penalty taker.

    Mirrors the resolution in data_loader's fpl_player_mappings query — admin
    overrides replace FPL's list ALL-OR-NOTHING per club — so that a recalc and
    a full run agree on who takes penalties. If you change the rule in one
    place, change it in the other; the two are deliberately kept identical
    rather than abstracted, because the loader needs it inline in a much larger
    SELECT.

    Weight is NULL unless an admin has split a tier: FPL never shares a rank.
    """
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT m.player_id,
                       CASE WHEN oc.team_id IS NOT NULL
                            THEN o.penalty_rank ELSE s.penalties_order END AS pen_rank,
                       CASE WHEN oc.team_id IS NOT NULL
                            THEN o.weight ELSE NULL END AS pen_weight
                FROM fpl_player_mappings m
                LEFT JOIN fpl_team_mappings ftm ON ftm.fpl_id = m.fpl_team_id
                JOIN fpl_player_snapshots s
                  ON s.player_id = m.player_id
                 AND s.snapshot_date = (
                        SELECT snapshot_date FROM fpl_player_snapshots
                        GROUP BY snapshot_date
                        HAVING COUNT(DISTINCT player_id) >= %s
                        ORDER BY snapshot_date DESC
                        LIMIT 1
                    )
                LEFT JOIN fpl_penalty_orders o ON o.player_id = m.player_id
                LEFT JOIN (
                    SELECT DISTINCT team_id FROM fpl_penalty_orders
                ) oc ON oc.team_id = ftm.team_id
                """,
                (FPL_SNAPSHOT_MIN_PLAYERS,),
            )
            out = {}
            for pid, rank, weight in await cur.fetchall():
                if rank is None:
                    continue
                out[int(pid)] = (
                    int(rank), None if weight is None else float(weight)
                )
    finally:
        if _db.pool:
            _db.pool.release(conn)
    return out


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
