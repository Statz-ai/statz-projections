import logging
import app.database as _db
from app.repository.db_utils import execute_chunked

logger = logging.getLogger("fpl_repo")


async def insert_fpl_projections_async(data_list):
    if len(data_list) == 0:
        return

    df = data_list.copy()
    # Only the DataFrame columns whose names don't already match the DB
    # column need renaming. def_con_pct is already snake_cased in
    # projection_service.py so no entry here.
    df = df.rename(columns={
        "FPL Points": "fpl_points",
        "Venue": "venue",
        "Gameweek": "gameweek_id",
        "Bonus Points": "bonus",
    })

    if hasattr(df['kickoff_datetime'].iloc[0], 'strftime'):
        df['kickoff_datetime'] = df['kickoff_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # gameweek_id / team_id / opponent_id are optional non-key columns —
    # kept nullable in the DB so older callers (and any older
    # fpl_projections rows) survive. Coerce NaN/None safely.
    # bonus + def_con_pct are Phase 2 additions; same nullable contract.
    has_gw = "gameweek_id" in df.columns
    has_team = "team_id" in df.columns
    has_opp = "opponent_id" in df.columns
    has_bonus = "bonus" in df.columns
    has_def_con = "def_con_pct" in df.columns
    has_xmin = "expected_minutes" in df.columns

    def _int_or_none(v):
        if v is None:
            return None
        try:
            if v != v:  # NaN
                return None
        except Exception:
            pass
        try:
            return int(v)
        except Exception:
            return None

    def _float_or_none(v):
        if v is None:
            return None
        try:
            if v != v:  # NaN
                return None
        except Exception:
            pass
        try:
            return float(v)
        except Exception:
            return None

    values = [
        (
            row.get("fixture_id"),
            row.get("player_id"),
            row.get("kickoff_datetime"),
            row.get("venue"),
            row.get("fpl_points"),
            _float_or_none(row.get("bonus")) if has_bonus else None,
            _float_or_none(row.get("def_con_pct")) if has_def_con else None,
            _float_or_none(row.get("expected_minutes")) if has_xmin else None,
            _int_or_none(row.get("gameweek_id")) if has_gw else None,
            _int_or_none(row.get("team_id")) if has_team else None,
            _int_or_none(row.get("opponent_id")) if has_opp else None,
        )
        for _, row in df.iterrows()
    ]

    sql = """
    INSERT INTO fpl_projections (
        fixture_id, player_id, kickoff_datetime, venue, fpl_points,
        bonus, def_con_pct, expected_minutes,
        gameweek_id, team_id, opponent_id,
        created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE
        fpl_points = new.fpl_points,
        bonus = new.bonus,
        def_con_pct = new.def_con_pct,
        expected_minutes = new.expected_minutes,
        gameweek_id = new.gameweek_id,
        team_id = new.team_id,
        opponent_id = new.opponent_id,
        updated_at = NOW()
    """
    return await execute_chunked(sql, values, label="[fpl_projections]")


async def cleanup_fpl_projections_async(gameweek_ids, keep_player_ids):
    """Membership semantics for the covered gameweeks: after the upsert,
    delete rows for players NOT in this run's kept frame. The insert alone
    can never REMOVE a player — one newly red-flagged (J.Timber's run-7 rows
    survived his 'i' flag landing) or dropped from the pool keeps his
    previous run's rows forever. Returns rows deleted."""
    gw_ids = [int(g) for g in gameweek_ids if g is not None]
    keep = [int(p) for p in keep_player_ids if p is not None]
    if not gw_ids or not keep:
        return 0
    conn = await _db.get_connection()
    try:
        async with conn.cursor() as cur:
            gw_ph = ",".join(["%s"] * len(gw_ids))
            keep_ph = ",".join(["%s"] * len(keep))
            await cur.execute(
                f"DELETE FROM fpl_projections WHERE gameweek_id IN ({gw_ph}) AND player_id NOT IN ({keep_ph})",
                tuple(gw_ids) + tuple(keep),
            )
            deleted = cur.rowcount
        await conn.commit()
        return deleted
    finally:
        _db.pool.release(conn)
