# The projection server goes deaf during a run

Scoping note, 2026-08-20. Not a plan to execute — options, evidence and the
one question that has to be answered before choosing between them.

---

## The problem, measured

While a projection is running, the server answers **nothing**.

`/openapi.json` is a static document FastAPI serves from memory. It needs no
database, no lock, no application code — only a free event loop. During a run
it timed out **nine times consecutively at 20–25s**. A missing route would 404
instantly; hanging means the loop itself is blocked.

**Consequences today**

| Caller | What happens |
|---|---|
| FPL Optimise (a real user, clicking a button) | Times out at 20s |
| `ProjectFixtureJob` | Times out, retries, eventually succeeds |
| Admin "run projections" | Times out at 15s though the run has started |
| The server's own `busy` reply | Never sent — that path needs the loop too |

The Laravel side now tells FPL users "projections are updating, try again in a
few minutes" rather than "no plan available", so the visible damage is
contained. Nothing has been fixed on the server.

## Why the loop blocks at all

The pipeline is genuinely async — 109 `await`s in `projection_service.py`, real
async DB I/O. It is not `async` decoration over sync code.

But between those awaits sit long synchronous stretches of numpy/pandas
modelling, and an `await` only yields at the await. A stretch that runs for
twenty seconds blocks everything for twenty seconds.

**This has not been profiled.** Which stretches dominate is unknown, and that
matters for option B below.

---

## ⚠️ Check this first — it may make the whole thing cheap

`Dockerfile` pins `--workers 1`, and the comment explains why:

> *"With the 2-season fetch, fixture_player_stats.csv has 15M rows (~4GB in
> pandas). 2 workers each holding their own DataCache doubled memory pressure
> and OOM'd on the 2nd worker's cache load."*

**That describes an architecture that no longer exists.** The CSV/DataCache path
was replaced by `LeagueDataLoader` — per-league, scoped, DB-direct. The only
remaining `read_csv` in the codebase is inside `data_loader.py` itself, and
`projection_service.py` loads exclusively through `LeagueDataLoader`.

Measured on the box while idle: gunicorn RSS **650 MB**, against 15 GB total
with ~7 GB available.

So the reason for a single worker may simply be out of date.

### ✅ MEASURED 2026-08-20, during the 13:35 refresh

150 samples at 12-second intervals, each recording gunicorn RSS alongside an
`/openapi.json` probe on a 4-second timeout.

| | |
|---|---|
| Idle RSS | **0.62 GB** |
| Peak RSS during the run | **0.78 GB** |
| Box total / available | 15 GB / ~7 GB |
| Blocked probes inside the run window (13:37–13:52) | **41 of 57 — 71%** |
| Blocked probes outside it | **0 of 93** |

**Memory is not the constraint, and is not close to being one.** A run costs
about 150 MB above idle. The Dockerfile's fear of 4 GB per worker belongs to
the CSV architecture that no longer exists; a second worker costs well under a
gigabyte against seven available.

**And the blocking is real but not total.** 71% of probes failed during the
run, 0% outside it — so the loop does surface between synchronous stretches,
just not often enough or long enough for a caller to rely on. That is the
shape you would expect from long numpy sections between genuine awaits, and it
is why a second worker helps: the free worker answers while the busy one is in
a stretch.

The cross-worker safety already exists — the flock in `routes.py` is held by
the kernel per file descriptor, so it is genuinely cross-process.

**Honest caveat, so this is not oversold:** a second worker is a large
improvement, not a guarantee. Workers share a listening socket, so a connection
accepted by the blocked worker still waits. It changes "always deaf" into
"usually answers", which for a user clicking Optimise is the difference that
matters — but it is not the same as never blocking.

---

## Options

### A. Second gunicorn worker
**Work:** one number. **Risk:** memory, unquantified until measured.
Cheapest by a distance. Probabilistic rather than absolute (see caveat above).
The in-process `_in_process_running` guard becomes per-worker, which is fine —
the flock is the real cross-process lock and the boolean was always
belt-and-braces.

### B. Run the pipeline in a thread
**Work:** moderate. **Risk:** high, and the risk is not obvious.
The pipeline is async, so a thread needs its own event loop (`asyncio.run`
inside the thread). The aiomysql pools are bound to the loop that created them
and cannot be shared across loops, so this means a second pool with its own
lifecycle. It also breaks the reasoning behind the in-process lock, which is
documented as safe *because* coroutines only switch at awaits.

### C. Split API and worker into separate services
**Work:** largest. **Risk:** lowest once done.
A light API container that never runs projections, and a worker that does.
The API stays responsive by construction rather than by scheduling luck.
Needs a queue or an RPC hop, and a second container's memory.

### D. Run each projection as a subprocess
**Work:** moderate. **Risk:** low.
The endpoint spawns `python -m app.run_projection --league X` and returns
immediately. The API process never does heavy work, so the loop never blocks.
No CLI entry point exists today — that is the bulk of the work. The flock keeps
coordinating exactly as it does now, and memory is transient per run rather
than resident.

Worth noting `cron_projections.sh` already treats the API as a trigger and
nothing more, so a subprocess model fits the existing shape.

---

## ✅ SHIPPED AND VERIFIED 2026-08-20

Two workers deployed ~13:47 UTC (`049d1df`). Probed throughout a real Ligue 1
projection: **25 of 25 probes returned 200, all under 14ms**, against a 71%
failure rate on one worker. The run completed clean — 9 fixtures, 3.5 minutes,
no errors.

Memory came in cheaper than estimated: working worker **0.42 GB** peak, idle
worker **0.12 GB**, host free 6 GB and unchanged. The reason is worth keeping:
the flock means only one worker ever loads a league, so peak is *(one run + one
idle worker)*, not double a run. That is also why the April failure mode cannot
recur — that was two workers **eagerly** loading a DataCache at startup, a
class that no longer exists.

**Two corrections to the original safety case**, recorded because a future
reader deserves the accurate version:

1. The claim that two workers "already ran successfully in production" was
   wrong. The worker count has flip-flopped four times, and `fafcf66` was
   reverted by `a32784d` **the same afternoon**, 2h21m later, on an OOM. The
   safety case rests entirely on the DataCache being gone — verified — and not
   on precedent.
2. The `to_csv` sweep missed live `to_parquet` writes at
   `data_loader.py:1703-1721` and `projection_service.py:293`. Both were
   checked and neither is on the projection write path, so the conclusion
   stands, but the sweep should have caught them.

### ✅ Full run confirmed 2026-08-21

300 samples across 01:55–02:55, covering twelve competitions run back to back
— Championship (9.3 min), La Liga, Serie A, Bundesliga, League One, Scottish
Premiership, MLS, Campeonato Brasileiro, Süper Lig, Saudi Pro League,
Eredivisie, Liga Portugal. All completed successfully.

| | |
|---|---|
| Blocked probes | **0 of 300** |
| Peak single worker | 0.65 GB |
| Peak combined | **1.11 GB** against ~7 GB available |

An hour of continuous projections without the server going deaf once, against
71% blocked on a single worker. Memory sits at roughly a sixth of headroom.

**The one gap:** the Premier League ran at 14:54, outside the capture window,
so the single heaviest league (15.3 min) has not been directly probed. The
mechanism does not depend on which league — a longer run is more of the same,
not different in kind — and a 9.3-minute Championship run inside the window
blocked nothing. Worth a confirming sample if anyone happens to be watching
during a PL run, but not worth arranging.

---

## Recommendation

**Do A.** The measurement removed the only reason not to: memory was the
objection, and a run peaks at 0.78 GB against ~7 GB available. It is one number
in the Dockerfile plus a rebuild, and the flock already makes it safe across
processes.

Then re-run the probe during the next run. If blocked probes fall from 71% to
near zero, stop — the problem is solved to the standard that matters.

**Remaining honesty about A:** workers share a listening socket, so a request
accepted by the busy worker still waits. Expect a large improvement, not a
guarantee. If the probe shows it is not enough, **do D** — it fixes the cause
rather than improving the odds, and fits how the server is already used.

**Do not do B.** It is the middle option that carries the most risk for the
least structural gain.

**One caveat on the measurement:** this was the 13:35 *refresh*, which skips the
accuracy-dataset gap-fill and metrics that 02:00 does. The full run will use
more. But the gap between 0.78 GB and a problem is roughly ninefold, so it
would take a very large surprise to change the conclusion — worth a confirming
sample at 02:00 rather than a reason to wait.

## How to know it worked

The same probe that found it:

```bash
# during a run, from the app box
for i in $(seq 5); do
  curl -s -o /dev/null -m 20 -w "%{http_code} %{time_total}s\n" \
    http://127.0.0.1:8000/openapi.json
done
```

Nine timeouts became the evidence. Five fast 200s would be the fix.

## Not worth doing

- **Raising client timeouts.** Already done for `ProjectFixtureJob` (15s → 30s)
  and it treats the symptom. A run lasts minutes; no client timeout is long
  enough without making a genuine failure take minutes to surface.
- **Client-side retries.** Same reason — retrying into a blocked server just
  doubles the wait.
