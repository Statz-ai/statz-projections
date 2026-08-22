import gc
import logging
import resource
import time
from datetime import datetime, timezone
from app.repository.projection_run_repo import touch_all_running, upsert_run_complete
from app.services.projection_service import ProjectionService
from app.services.multi_league_projection_service import MultiLeagueProjectionService
from app.services.international_projection_service import InternationalProjectionService
from app.models.requests.league_request import LeagueRequest
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


logger = logging.getLogger("projection")

class ProjectionAllTeams:
    # Backstop in case the DB lookup below fails for any reason. The actual
    # default at runtime comes from `competitions WHERE is_projected = 1`.
    DEFAULT_LEAGUES = [
        "Championship",
        "Premier League",
        "La Liga",
        "Serie A",
        "Campeonato Brasileiro",
        "League One",
        "League Two",
        "Ligue 1",
        "Bundesliga",
        "Champions League",
        "Europa League",
    ]

    @staticmethod
    async def _resolve_default_leagues() -> list[str]:
        """Fetch all is_projected=true competition names from the source DB.

        Mirrors Laravel's `triggerRunAll()` which already queries
        `Competition::where('is_projected', true)`. When the /api/projections/all-leagues
        endpoint is hit without an explicit `leagues` list (e.g. direct curl,
        ad-hoc trigger), this is the source of truth — not the static
        DEFAULT_LEAGUES list, which goes stale every time a new comp is
        added via the admin panel.
        """
        from app.source_database import get_source_connection, release_source_connection
        conn = await get_source_connection()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT name FROM competitions "
                    "WHERE is_projected = 1 "
                    "ORDER BY `order` IS NULL, `order`, name"
                )
                rows = await cur.fetchall()
            return [r[0] for r in rows]
        finally:
            release_source_connection(conn)

    async def projectionAllTeams(self, leagues=None):
        if leagues is None:
            try:
                leagues = await ProjectionAllTeams._resolve_default_leagues()
                logger.info(f"All-leagues: resolved {len(leagues)} leagues from DB (is_projected=true)")
            except Exception as e:
                logger.warning(f"All-leagues: DB lookup failed ({e}); falling back to hardcoded DEFAULT_LEAGUES")
                leagues = ProjectionAllTeams.DEFAULT_LEAGUES

        _total_start = time.time()
        _league_times = {}

        _euro_comp_service = MultiLeagueProjectionService()
        _intl_service = InternationalProjectionService()
        _domestic_service = ProjectionService()

        for league in leagues:
            # Slug and start-time for projections_runs status write.
            # Matches routes._league_to_competition_id() so the completion
            # row lines up with any pre-created 'running' row (inserted by
            # Laravel's triggerRunAll) keyed on the same slug.
            _league_slug = league.lower().replace(' ', '-').replace('.', '')
            _league_started_iso = datetime.now(timezone.utc).isoformat()
            # Bump started_at to NOW on every 'running' row in projections_runs
            # to keep the whole queue fresh against mark-stuck's 30-min threshold.
            # Laravel's pre-create stamps all 24 rows with the click-time, so
            # without this, leagues still queued past minute ~30 would false-stick.
            # Touching ALL running rows (not just this league's) protects every
            # row that's still waiting its turn — the loop processes at ~5 min
            # per league, so every remaining row gets its started_at refreshed
            # every iteration.
            await touch_all_running()
            try:
                # Delegate euro comps to dedicated service
                if MultiLeagueProjectionService.handles(league):
                    logger.info(f"[{league}] Delegating to MultiLeagueProjectionService")
                    _start_time = time.time()
                    request = LeagueRequest(league=league)
                    await _euro_comp_service.projections(request)
                    _elapsed_s = time.time() - _start_time
                    _league_times[league] = _elapsed_s / 60
                    logger.info(f"[{league}] DONE in {_elapsed_s:.1f}s")
                    await upsert_run_complete(
                        competition_id=_league_slug,
                        status='success',
                        started_at=_league_started_iso,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        exit_code=0,
                    )
                    continue

                # Delegate international comps (WC, Friendly Intl, future
                # Euros / Copa / AFCON / Nations League etc.) to the
                # dedicated international projection pipeline. These don't
                # fit the domestic league-table mental model and have
                # their own rating / squad / fixture flow.
                if InternationalProjectionService.is_international_comp(league):
                    logger.info(f"[{league}] Delegating to InternationalProjectionService")
                    _start_time = time.time()
                    request = LeagueRequest(league=league)
                    await _intl_service.projections(request)
                    _elapsed_s = time.time() - _start_time
                    _league_times[league] = _elapsed_s / 60
                    logger.info(f"[{league}] DONE in {_elapsed_s:.1f}s")
                    await upsert_run_complete(
                        competition_id=_league_slug,
                        status='success',
                        started_at=_league_started_iso,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        exit_code=0,
                    )
                    continue

                # projections() logs its own START / COMPLETE lines; this
                # timer exists only to feed the end-of-run summary table.
                _start_time = time.time()

                # Delegate to the domestic pipeline, exactly as the euro and
                # international branches above delegate to theirs. This loop
                # is orchestration only — it owns the league list and the
                # projections_runs bookkeeping, not the maths.
                #
                # It used to inline its own ~1,900-line copy of the domestic
                # pipeline, which had silently drifted: the pre-guardrail
                # binary odds gate, no goals over/under cascade, no
                # Dixon-Coles, no team dials, no FPL bonus simulation. Run All
                # therefore overwrote every league's published numbers with
                # older maths. One implementation removes that whole class of
                # bug — see docs/projection-services-design-scope.md.
                #
                # The original batch existed to amortise a shared DataCache
                # pre-load across leagues. That cache was deleted in the
                # direct-DB migration, so there is nothing left to amortise:
                # projections() builds its own per-league LeagueDataLoader,
                # which is what this loop was doing anyway.
                await _domestic_service.projections(LeagueRequest(league=league))

                _league_times[league] = (time.time() - _start_time) / 60
                await upsert_run_complete(
                    competition_id=_league_slug,
                    status='success',
                    started_at=_league_started_iso,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    exit_code=0,
                )
            except Exception as e:
                logger.error(f"[{league}] FAILED - skipping: {e}", exc_info=True)
                _league_times[league] = "FAILED"
                try:
                    await upsert_run_complete(
                        competition_id=_league_slug,
                        status='failed',
                        started_at=_league_started_iso,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        exit_code=1,
                        stderr=str(e)[:500],
                    )
                except Exception as _status_err:
                    logger.error(f"[{league}] failed to write status row: {_status_err}")
            finally:
                # Memory hygiene between leagues. The all-leagues batch has
                # OOM-killed the gunicorn worker mid-run multiple times
                # (Apr 22 retrain, 2026-05-05 Saudi Pro mid-batch). Each
                # league loads its own LeagueDataLoader (50k+ team rows,
                # 200k+ player rows for euro comps) plus model dicts plus
                # per-player projection DataFrames, and the loader stays
                # pinned by the ProjectionService._current_source class attr
                # after the call returns. Nulling it + gc.collect forces the
                # release between iterations; the RSS line shows whether the
                # leak stays bounded.
                ProjectionService._current_source = None
                gc.collect()
                _rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                logger.info(f"[{league}] memory after cleanup: peak RSS={_rss_mb:.0f} MB")

        _total_elapsed = (time.time() - _total_start) / 60
        logger.info("=" * 60)
        logger.info("ALL LEAGUES COMPLETE - SUMMARY:")
        for _l, _t in _league_times.items():
            _t_str = f"{_t:.1f} min" if isinstance(_t, float) else _t
            logger.info(f"  {_l:<30} {_t_str}")
        logger.info(f"  {'TOTAL':<30} {_total_elapsed:.1f} min")
        logger.info("=" * 60)
