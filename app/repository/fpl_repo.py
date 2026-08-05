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


async def cleanup_fpl_projections_async(gameweek_ids, keep_pairs):
    """Membership semantics for the covered gameweeks: after the upsert, delete
    every row this run did NOT produce. The insert can only add or update, so
    anything it did not write survives forever.

    keep_pairs: iterable of (fixture_id, player_id) the run actually wrote.

    Keyed on the PAIR, not on player_id alone. Player-only membership missed a
    transfer entirely: Elliot Anderson moved Forest -> Man City, the run
    correctly wrote 19 Man City rows, and because he was still "kept" his 18
    Forest rows stayed — he projected at both clubs at once, and would have
    shown twice on the site. Lacroix was the same Palace -> Chelsea. The old
    club's rows hang off the old club's FIXTURES, so no upsert ever touches
    them. (2026-08-05, found the day the transfer backfill first moved anyone.)

    Still covers the original case — a player dropped from the pool or newly
    flagged contributes no pairs, so all his rows go (J.Timber kept run-7 rows
    after his 'i' flag landed).

    Chunked per gameweek to keep the row-constructor IN list to roughly a
    squad-round in size rather than the whole horizon.
    """
    gw_ids = [int(g) for g in gameweek_ids if g is not None]
    pairs = {(int(f), int(p)) for f, p in keep_pairs if f is not None and p is not None}
    if not gw_ids or not pairs:
        return 0
    conn = await _db.get_connection()
    deleted = 0
    try:
        async with conn.cursor() as cur:
            for gw in gw_ids:
                # A fixture belongs to exactly one gameweek, so a pair from
                # another gameweek can never appear in `existing` here — the
                # global keep-set is safe to diff against per gameweek.
                await cur.execute(
                    "SELECT fixture_id, player_id FROM fpl_projections WHERE gameweek_id = %s",
                    (gw,),
                )
                existing = {(int(f), int(p)) for f, p in await cur.fetchall()}
                stale = existing - pairs
                if not stale:
                    continue
                stale = list(stale)
                for i in range(0, len(stale), 1000):
                    chunk = stale[i:i + 1000]
                    ph = ",".join(["(%s,%s)"] * len(chunk))
                    await cur.execute(
                        f"DELETE FROM fpl_projections WHERE gameweek_id = %s "
                        f"AND (fixture_id, player_id) IN ({ph})",
                        (gw,) + tuple(v for pair in chunk for v in pair),
                    )
                    deleted += cur.rowcount
        await conn.commit()
        return deleted
    finally:
        _db.pool.release(conn)


async def prune_stale_fpl_rows(conn=None):
    """Delete fpl_projections rows whose gameweek belongs to a non-current
    season (2026-07-30: 363 May relics survived because upserts never
    delete — they polluted raw per-player sums and debugging). Called after
    each insert; season-scoped so current-season history is untouched."""
    from app.database import get_connection
    import app.database as _db
    own = conn is None
    if own:
        conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """DELETE f FROM fpl_projections f
                   JOIN gameweeks g ON g.id = f.gameweek_id
                   WHERE g.season_id != (
                       SELECT id FROM seasons
                       WHERE competition_id = 8 AND is_current = 1 LIMIT 1
                   )"""
            )
            n = cur.rowcount
        await conn.commit()
        if n:
            import logging
            logging.getLogger("fpl_repo").info(f"[fpl_projections] pruned {n} old-season rows")
        return n
    finally:
        if own and _db.pool:
            _db.pool.release(conn)
