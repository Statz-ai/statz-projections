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

**The question to answer: what is peak RSS during a real run under the current
loader?** If it is comfortably under half of available memory, a second worker
becomes viable, and the cross-worker safety already exists — the flock in
`routes.py` is held by the kernel per file descriptor, so it is genuinely
cross-process.

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

## Recommendation

1. **Measure peak RSS during a run.** One command while a projection is going.
   It decides whether A is available, and A is an order of magnitude cheaper
   than the alternatives.
2. **If memory allows, do A** and confirm with the same `/openapi.json` probe
   that produced the evidence above. If the server answers during a run, stop —
   the problem is solved to the standard that matters.
3. **If memory does not allow, do D.** It fixes the cause rather than improving
   the odds, and it fits how the server is already used.
4. **Do not do B.** It is the middle option that carries the most risk for the
   least structural gain.

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
