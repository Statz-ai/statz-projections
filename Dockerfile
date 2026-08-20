FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p app/projection-outputs

EXPOSE 8000

# Two workers, restored 2026-08-20.
#
# WHY IT IS SAFE NOW — all three historical objections have expired, and each
# was checked separately rather than assumed:
#
#  1. Serialisation. The 2026-04-24 incident was a Python module global
#     (_projection_running) being PER-WORKER, so two projections ran at once.
#     The cross-process flock in routes.py replaced it and still guards this.
#     _in_process_running now only has to catch two coroutines inside one
#     worker, which it still does.
#     DO NOT read the history as a track record: the worker count has
#     flip-flopped four times, and fafcf66 (24 Apr 13:55) set two workers only
#     to be reverted by a32784d the SAME AFTERNOON (16:16) when the second
#     worker's DataCache load OOM'd on a CL spot-check against 4.7GB free.
#     Two workers have never run here for long. The safety case rests on
#     point 3 below, not on precedent.
#  2. Data corruption. That incident corrupted fixture_team_stats.csv via
#     last-writer-wins. Every to_csv is now commented out. Live to_parquet
#     writes DO remain — data_loader.py:1703-1721 (a diagnostic scope dump)
#     and projection_service.py:293 (a one-time xlsx->parquet migration
#     reached only when the parquet is absent, where two workers would write
#     identical content from the same source). Neither sits on the projection
#     write path, which goes to the database.
#  3. Memory. The DataCache class this comment used to describe no longer
#     exists; LeagueDataLoader is per-league and per-request. Measured across
#     a real run 2026-08-20: working worker 0.42GB peak, idle worker 0.12GB,
#     host free 6GB and unchanged. Note peak is (one run + one idle worker),
#     NOT double a run — the flock means only one worker ever loads a league,
#     so the idle one never builds a loader. The April failure was two workers
#     EAGERLY loading DataCache at startup, a mechanism that no longer exists.
#
# WHY TWO — a projection blocks its worker's event loop for long stretches
# (71% of probes failed during a run, 0% outside it), so a single worker goes
# deaf and every caller times out: the FPL Optimise button, ProjectFixtureJob,
# the admin trigger, even the server's own "busy" reply. A second worker can
# answer while the first is computing.
#
# VERIFIED 2026-08-20 under a real Ligue 1 projection: 25 of 25 probes to
# /openapi.json returned 200, all under 14ms, against a 71% failure rate on one
# worker. The run itself completed clean.
#
# HONEST LIMIT — workers share a listening socket, so a request accepted by the
# busy worker can still wait. Not observed in the verification above, but the
# mechanism is real. If it ever proves insufficient, move the heavy work off
# the loop entirely — see docs/event-loop-blocking-scope.md.
#
# STILL UNTESTED — the 02:00 full run, which adds the accuracy gap-fill and
# metrics, and includes the Premier League (~33 min, 200 fixtures, all the FPL
# work). The 0.42GB figure above is a 3.5-minute refresh.
CMD ["gunicorn", "app.main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "1800"]
