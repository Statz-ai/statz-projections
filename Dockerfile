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
#     The cross-process flock in routes.py replaced it, and that same commit
#     already ran two workers successfully. _in_process_running now only has
#     to catch two coroutines inside one worker, which it still does.
#  2. Data corruption. That incident corrupted fixture_team_stats.csv via
#     last-writer-wins. Every to_csv in the codebase is now commented out —
#     the pipeline writes to the database, so there is no shared file to race.
#  3. Memory. The 15M-row/4GB DataCache this comment used to describe was
#     replaced by the per-league LeagueDataLoader. Measured 2026-08-20 across
#     the 13:35 refresh: idle 0.62GB, peak 0.78GB, against ~7GB free.
#
# WHY TWO — a projection blocks its worker's event loop for long stretches
# (71% of probes failed during a run, 0% outside it), so a single worker goes
# deaf and every caller times out: the FPL Optimise button, ProjectFixtureJob,
# the admin trigger, even the server's own "busy" reply. A second worker can
# answer while the first is computing.
#
# HONEST LIMIT — workers share a listening socket, so a request accepted by the
# busy worker still waits. This improves the odds substantially; it does not
# guarantee a response. If it proves insufficient, the fix is to move the heavy
# work off the loop entirely — see docs/event-loop-blocking-scope.md.
CMD ["gunicorn", "app.main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "1800"]
