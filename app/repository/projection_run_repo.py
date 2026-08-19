import asyncio
import logging
from datetime import datetime, timezone

import app.database as _db
from app.database import get_connection

logger = logging.getLogger("projection_run_repo")


def _to_mysql_datetime(value):
    """
    Coerce ISO-format strings (what routes._run_single_league passes) to
    MySQL-friendly datetime objects. Returns None for falsy input.
    aiomysql binds datetime.datetime as DATETIME natively.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(value)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


async def touch_all_running():
    """Bump started_at to NOW on every 'running' row in projections_runs.

    Called at the top of each league's iteration in the all-teams loop.
    The Laravel triggerRunAll pre-create stamps all 24 running rows
    with the click-time, but the Python loop processes sequentially at
    ~5 min per league — so rows for leagues later in the queue cross
    mark-stuck's 30-min threshold while still legitimately queued.

    Touching ALL running rows (not just the current league's) keeps the
    whole batch fresh. Any row we bump will continue to live; if the
    container truly wedges (OOM, network, worker hang), the Python loop
    stops executing and no rows get touched — mark-stuck then correctly
    catches them after 30 min.

    Safe if other single-league runs are concurrently in progress: their
    rows get touched too, which only EXTENDS the time before mark-stuck
    would flip them. Any genuinely-stuck run would still eventually
    surface because nothing's refreshing its started_at.
    """
    conn = None
    try:
        conn = await asyncio.wait_for(get_connection(), timeout=30)
        async with conn.cursor() as cursor:
            await cursor.execute(
                "UPDATE projections_runs SET started_at = NOW() "
                "WHERE status = 'running'"
            )
            await conn.commit()
    except Exception as e:
        logger.error(f"[projections_runs] touch_all_running failed: {e}")
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)


async def upsert_run_complete(
    competition_id: str,
    status: str,
    started_at: str,
    finished_at: str,
    exit_code: int = None,
    stdout: str = None,
    stderr: str = None,
):
    """
    Replacement for the HTTP status callback. Writes projection run
    completion state directly to the projections_runs table instead of
    POSTing to Laravel's /api/internal/projections/status endpoint.

    Mirrors the logic in ProjectionsAdminController::reportStatus — find
    the latest 'running' row for this competition_id and update it; if
    none exists (e.g. the run was never pre-registered), insert a complete
    row.

    Does NOT raise on DB errors — mark-stuck (runs every 5 min on the
    Laravel side) is the safety net. Better to log + move on than block
    the projection lock release.
    """
    stdout_snippet = (stdout or '')[:500]
    stderr_snippet = (stderr or '')[:500]
    started_at_dt = _to_mysql_datetime(started_at)
    finished_at_dt = _to_mysql_datetime(finished_at)

    conn = None
    try:
        conn = await asyncio.wait_for(get_connection(), timeout=30)
        async with conn.cursor() as cursor:
            await cursor.execute(
                "SELECT id FROM projections_runs "
                "WHERE competition_id = %s AND status = 'running' "
                "ORDER BY started_at DESC LIMIT 1",
                (competition_id,),
            )
            row = await cursor.fetchone()
            if row:
                run_id = row[0]
                # started_at is rewritten from OUR clock, not left as Laravel
                # wrote it, so both ends of the duration come from one writer.
                #
                # These are TIMESTAMP columns, so MySQL converts using the
                # SESSION timezone — and the two sides disagree. This service
                # pins its session to UTC; Laravel leaves it at SYSTEM, which
                # is BST half the year. A row created by Laravel at start and
                # finished by us therefore had one end shifted and the other
                # not, adding exactly 60 minutes to every scheduled run.
                #
                # It made run-duration monitoring useless: every league read
                # as 61-64 minutes regardless of size, with near-zero variance
                # (Bundesliga 61/61/61), because that was an offset and not a
                # duration. League One's failed run on 2026-08-18 looked like
                # 65 minutes and was really about 5. A genuinely slow run
                # would have been invisible in that noise.
                #
                # Not fixed by aligning the session timezones: Laravel writes
                # UTC values over a BST session, so its timestamps round-trip
                # correctly while being stored an hour early. Changing that
                # shifts every historical created_at in the whole app, and the
                # skew disappears by itself when the box returns to GMT.
                # Owning both ends here costs nothing and needs no migration.
                await cursor.execute(
                    "UPDATE projections_runs SET "
                    "status = %s, started_at = %s, finished_at = %s, "
                    "exit_code = %s, "
                    "stdout_snippet = %s, stderr_snippet = %s "
                    "WHERE id = %s",
                    (status, started_at_dt, finished_at_dt, exit_code,
                     stdout_snippet, stderr_snippet, run_id),
                )
                await conn.commit()
                logger.info(
                    f"[projections_runs] {competition_id}: updated run {run_id} -> {status}"
                )
            else:
                await cursor.execute(
                    "INSERT INTO projections_runs "
                    "(competition_id, started_at, finished_at, status, "
                    "exit_code, stdout_snippet, stderr_snippet, "
                    "triggered_by, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, 'schedule', NOW())",
                    (competition_id, started_at_dt, finished_at_dt, status,
                     exit_code, stdout_snippet, stderr_snippet),
                )
                await conn.commit()
                logger.info(
                    f"[projections_runs] {competition_id}: no running row — "
                    f"inserted complete row as {status}"
                )
    except Exception as e:
        logger.error(
            f"[projections_runs] {competition_id}: DB write failed: {e}",
            exc_info=True,
        )
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)
