import fcntl
import logging
import os
import time
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from app.repository.projection_run_repo import upsert_run_complete

class AllLeaguesRequest(BaseModel):
    leagues: Optional[List[str]] = None


class PromoteModelRequest(BaseModel):
    reason: Optional[str] = "manual promotion via admin panel"


class PartialRetrainRequest(BaseModel):
    competition_id: Optional[int] = None  # None → All Leagues fallback scope
    stats: List[str]  # e.g. ["Interceptions", "Offsides"]
    promote: bool = False  # True → run the Phase 5 guardrail and auto-promote on pass


class RetrainRequest(BaseModel):
    """Body for /api/projections/retrain.

    promote=False (default): Phase 4 dry-run — train models, log
        would-promote decisions, leave is_active untouched.
    promote=True: Phase 5 — train, then for each model where the
        guardrail passes (≥5% improvement OR initial baseline),
        atomically demote incumbent + activate new. Models that
        regress >15% are rejected.
    """
    promote: Optional[bool] = False

from app.services.projection_service import ProjectionService
from app.services.euro_comp_projection_service import EuroCompProjectionService
from app.services.international_projection_service import InternationalProjectionService
from app.models.requests.league_request import LeagueRequest
from app.services.projection_all_teams_service import ProjectionAllTeams

router = APIRouter(prefix="/api/projections", tags=["API"])
logger = logging.getLogger("routes")

projection_service = ProjectionService()
euro_comp_service = EuroCompProjectionService()
international_projection_service = InternationalProjectionService()
projection_all_teams_service = ProjectionAllTeams()

# Two-tier lock: an OS file-lock for cross-worker serialisation + an
# in-process boolean for cross-coroutine serialisation within a single
# worker.
#
# Why both: Linux flock(2) is per-open-file-description, NOT per-process.
# So two open() calls in the same process produce different OFDs, and
# flock() on each succeeds — within one worker, multiple concurrent
# coroutines could each open the file, lock it, and proceed in parallel.
# Surfaced 2026-04-30 evening when 9 rapid-fire triggerCompetition calls
# resulted in 3 leagues running concurrently with deadlock retries on
# shared player_projections / model_dataset writes.
#
# Originally the code used a Python module global `_projection_running`,
# replaced 2026-04-24 with the file lock alone after a 2-worker incident
# corrupted fixture_team_stats.csv. The replacement was an over-correction
# — the file lock handles cross-worker, but you ALSO need an in-process
# guard. Restored 2026-04-30.
#
# flock is held by the kernel for the FD that owns the file handle, so
# it's truly cross-process. Released automatically if the worker crashes
# (kernel closes FD), so no zombie-lock risk. The in-process boolean is
# safe because asyncio coroutines only context-switch at await points,
# and _try_acquire_lock has no awaits — so the check + set is atomic.
_LOCK_PATH = "/tmp/_statz_projection.lock"
_lock_fh = None
_in_process_running = False

def _try_acquire_lock() -> bool:
    """Non-blocking acquire. Returns True on success, False if another
    worker/process/coroutine already holds the lock."""
    global _lock_fh, _in_process_running
    if _in_process_running:
        return False
    try:
        _lock_fh = open(_LOCK_PATH, "w")
        fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fh.write(str(os.getpid()))
        _lock_fh.flush()
        _in_process_running = True
        return True
    except (IOError, OSError):
        if _lock_fh:
            try:
                _lock_fh.close()
            except Exception:
                pass
            _lock_fh = None
        return False

def _release_lock() -> None:
    global _lock_fh, _in_process_running
    if _lock_fh is not None:
        try:
            fcntl.flock(_lock_fh, fcntl.LOCK_UN)
            _lock_fh.close()
        except Exception:
            pass
        _lock_fh = None
    _in_process_running = False


async def _report_status(competition_id: str, status: str, started_at: str, finished_at: str = None, exit_code: int = None, stdout: str = None, stderr: str = None):
    """
    Report projection run status back to the Statz admin dashboard.

    Writes directly to the projections_runs table via the DB pool instead
    of POSTing over HTTP. Fixes the deploy-window callback-loss issue
    where Laravel restarts during a deploy would drop incoming HTTP
    requests and leave runs marked 'running' until mark-stuck timed them
    out 30 min later.
    """
    await upsert_run_complete(
        competition_id=competition_id,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
    )


def _league_to_competition_id(league: str) -> str:
    """Convert league name to competition ID (slug format)."""
    return league.lower().replace(' ', '-').replace('.', '')


async def _run_all_leagues(leagues=None, **_unused):
    try:
        await projection_all_teams_service.projectionAllTeams(leagues=leagues)
    except Exception as e:
        logger.error(f"All-leagues projection FAILED: {e}", exc_info=True)
    finally:
        _release_lock()
        logger.info("Projection lock released.")


async def _run_single_league(request):
    competition_id = _league_to_competition_id(request.league)
    started_at = datetime.now(timezone.utc).isoformat()
    # A partial run deliberately leaves downstream tables holding the previous
    # run's numbers, so it must not report itself as a healthy run: a 'success'
    # row would clear a failed-comp alert and feed the pipeline-dead canary in
    # ProjectionsFreshnessCheck. Nothing scheduled sets these flags.
    _is_partial = bool(getattr(request, 'stop_after', None)
                       or getattr(request, 'skip_datasets', False))
    try:
        if InternationalProjectionService.is_international_comp(request.league):
            await international_projection_service.projections(request)
        elif EuroCompProjectionService.handles(request.league):
            await euro_comp_service.projections(request)
        else:
            await projection_service.projections(request)
        finished_at = datetime.now(timezone.utc).isoformat()
        if _is_partial:
            logger.info(f"[{request.league}] partial run finished — no projections_runs row written")
        else:
            await _report_status(competition_id, "success", started_at, finished_at, exit_code=0)
    except Exception as e:
        finished_at = datetime.now(timezone.utc).isoformat()
        logger.error(f"[{request.league}] projection FAILED: {e}", exc_info=True)
        if _is_partial:
            logger.error(f"[{request.league}] partial run failed — no projections_runs row written")
        else:
            await _report_status(competition_id, "failed", started_at, finished_at, exit_code=1, stderr=str(e)[:500])
    finally:
        _release_lock()
        logger.info("Projection lock released.")


@router.post("")
async def projections(request: LeagueRequest, background_tasks: BackgroundTasks):
    """Start league projection in background - returns immediately, no timeout.

    `stop_after` / `skip_datasets` make this a PARTIAL run — see
    LeagueRequest. Validated here rather than in the background task so a
    typo comes back as a 400 instead of dying silently after the lock is
    taken.
    """
    if request.stop_after and request.stop_after not in ProjectionService.STAGES:
        raise HTTPException(
            status_code=400,
            detail=f"stop_after must be one of {list(ProjectionService.STAGES)}, got {request.stop_after!r}",
        )
    if not _try_acquire_lock():
        return {"status": "busy", "message": "A projection is already running. Wait for it to finish."}
    background_tasks.add_task(_run_single_league, request)
    resp = {"status": "started", "league": request.league}
    if request.stop_after or request.skip_datasets:
        resp["partial"] = {"stop_after": request.stop_after, "skip_datasets": bool(request.skip_datasets)}
    return resp


class FixtureProjectionRequest(BaseModel):
    fixture_id: int


@router.post("/fixture")
async def project_fixture(request: FixtureProjectionRequest, background_tasks: BackgroundTasks):
    """Re-project a single fixture (typically triggered when a confirmed
    lineup arrives). Reuses the standard projection pipeline with a
    `fixture_ids` filter so all per-fixture rows (fixture_projections,
    team_projections, player_projections, player_prop_projections, and
    wc_fantasy_projections for WC) get refreshed with current odds and
    the latest lineup data.

    Guards:
      - Fixture must be in our projection scope (row exists in
        fixture_projections). Prevents re-projecting fixtures we don't
        normally cover.
      - Kickoff must be > 5 minutes in the future. Buffer accounts for
        queue lag + projection runtime; closer to kickoff isn't worth
        the race against the whistle.
    """
    from app.source_database import get_source_connection, release_source_connection

    fid = request.fixture_id
    conn = await get_source_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT f.id, f.kickoff_datetime, c.name AS comp_name,
                       EXISTS(SELECT 1 FROM fixture_projections WHERE fixture_id = f.id) AS already_projected
                FROM fixtures f
                JOIN competitions c ON c.id = f.competition_id
                WHERE f.id = %s
                """,
                (fid,),
            )
            row = await cur.fetchone()
    finally:
        release_source_connection(conn)

    if not row:
        return {"status": "skipped", "reason": "fixture not found", "fixture_id": fid}

    _, kickoff_dt, comp_name, already_projected = row
    if not already_projected:
        return {"status": "skipped", "reason": "fixture not previously projected", "fixture_id": fid}

    now_utc = datetime.now(timezone.utc)
    # kickoff_datetime stored UTC-naive in MySQL — attach UTC for the compare
    if kickoff_dt.tzinfo is None:
        kickoff_dt = kickoff_dt.replace(tzinfo=timezone.utc)
    if (kickoff_dt - now_utc).total_seconds() < 300:
        return {"status": "skipped", "reason": "kickoff within 5 minutes", "fixture_id": fid}

    if not _try_acquire_lock():
        return {"status": "busy", "message": "A projection is already running. Wait for it to finish."}

    league_request = LeagueRequest(league=comp_name, fixture_ids=[fid])
    background_tasks.add_task(_run_single_league, league_request)
    return {"status": "started", "fixture_id": fid, "competition": comp_name}


@router.post("/all-leagues")
async def all_leagues(background_tasks: BackgroundTasks, request: AllLeaguesRequest = None):
    """Start all-leagues projection in background. Optionally pass 'leagues' list to run only specific leagues."""
    if not _try_acquire_lock():
        return {"status": "busy", "message": "A projection is already running. Wait for it to finish."}
    leagues = request.leagues if request else None
    background_tasks.add_task(_run_all_leagues, leagues)
    msg = f"leagues: {leagues}" if leagues else "all leagues"
    return {"status": "started", "message": f"Projection started ({msg})"}


@router.post("/retrain")
async def retrain(background_tasks: BackgroundTasks, request: RetrainRequest = None):
    """Kick off model retraining in the background.

    Body: {"promote": true|false}  (default false = dry-run)

    Returns immediately; the full retrain takes ~30 min with the trimmed
    grid (commit 0afcca4) on current data volumes.

    Shares the _projection_running lock with the projection endpoints —
    training 132+ PoissonRegressor models + grid searches is memory-heavy
    and can OOM if a projection is running concurrently. Returns 'busy'
    if a projection is already in progress; caller should retry later.

    Phase 4 (dry-run) inserts new models with is_active=0 and only logs
    promotion decisions. Phase 5 (promote=true) atomically demotes the
    incumbent and activates the new model when the guardrail passes
    (≥5% improvement OR no incumbent). Models that regress >15% are
    rejected and incumbent stays.
    """
    from app.services.retrain_service import retrain_all_models

    if not _try_acquire_lock():
        return {"status": "busy", "message": "A projection or retrain is already running. Wait for it to finish."}

    promote = bool(request.promote) if request else False
    dry_run = not promote

    async def _run():
        try:
            await retrain_all_models(dry_run=dry_run)
        except Exception as e:
            logger.error(f"retrain FAILED: {e}", exc_info=True)
        finally:
            _release_lock()
            logger.info("Retrain lock released.")

    background_tasks.add_task(_run)
    mode = "promote" if promote else "dry-run"
    return {"status": "started", "mode": mode, "message": f"Retraining started in background ({mode}); check projection.log for per-(league,stat) output"}


@router.post("/retrain/partial")
async def retrain_partial_endpoint(request: PartialRetrainRequest, background_tasks: BackgroundTasks):
    """Retrain a subset of (competition, stat) pairs — fills gaps left by
    a partial-success full retrain without re-doing the per-league work
    that already completed.

    Example: after a full retrain OOM'd during All Leagues Interceptions,
    POST {"competition_id": null, "stats": ["Interceptions", "Offsides"]}
    trains just those 2 stats against the top-5 fallback dataset.
    """
    from app.services.retrain_service import retrain_partial

    if not _try_acquire_lock():
        return {"status": "busy", "message": "A projection or retrain is already running. Wait for it to finish."}

    dry_run = not bool(request.promote)

    async def _run():
        try:
            await retrain_partial(request.competition_id, request.stats, dry_run=dry_run)
        except Exception as e:
            logger.error(f"retrain partial FAILED: {e}", exc_info=True)
        finally:
            _release_lock()
            logger.info("Retrain (partial) lock released.")

    background_tasks.add_task(_run)
    scope = "All Leagues" if request.competition_id is None else f"competition_id={request.competition_id}"
    mode = "promote" if request.promote else "dry-run"
    return {"status": "started", "mode": mode, "scope": scope, "stats": request.stats}


@router.post("/models/{model_id}/promote")
async def promote_model_endpoint(model_id: int, request: PromoteModelRequest = None):
    """Manually promote a specific projection_models row to is_active=1.
    The per-(competition_id, stat_name) invariant is maintained atomically
    by promote_model() — demotes the incumbent in the same transaction.

    Phase 4 is dry-run (auto-retrain doesn't flip is_active). Manual
    promotion via this endpoint is the Phase 6 admin override — user
    explicitly chose to replace the incumbent, so no guardrail check.
    Also covers rollback: promoting an older version restores it as
    active, demoting the current one.
    """
    from app.repository.projection_model_repo import promote_model

    reason = (request.reason if request else None) or "manual promotion via admin panel"
    try:
        await promote_model(model_id, reason)
        return {"status": "ok", "model_id": model_id, "reason": reason}
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"promote_model({model_id}) failed: {e}", exc_info=True)
        return {"status": "error", "message": f"Internal error: {type(e).__name__}"}


# /fetch-data endpoint removed 2026-05-11 in Phase 7.2 cleanup. Was the
# legacy CSV-cache primer from before the 2026-04-28 DB-loader migration.
# All callers gone: admin panel "Run Now"/"Run All" stopped passing
# fetch_first (commit 7.1b), schedule-check stopped calling it (commit
# 0b5e8d65), admin endpoint removed (commit 7.1).


class FplRecalcRequest(BaseModel):
    player_id: int = None
    player_ids: list = None


@router.post("/fpl/recalc-player")
async def fpl_recalc_player(request: FplRecalcRequest):
    """Instant per-player FPL recalc after an admin dial edit (George,
    2026-07-30). Reassembles the player's points from the last run's
    persisted scoring snapshot — no global lock: a single-player UPDATE is
    safe alongside a full run (last-writer-wins; the run reads the same
    dials, so they converge)."""
    from app.services.fpl_recalc_service import recalc_fpl_players
    ids = request.player_ids or ([request.player_id] if request.player_id else [])
    try:
        result = await recalc_fpl_players(ids)
    except Exception as err:
        logger.exception("fpl recalc failed")
        return {"ok": False, "error": str(err)}
    return result
