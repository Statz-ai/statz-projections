# Projection services — design scope

Written 2026-08-20. **Phases 0 and 1 are SHIPPED and deployed** (commits
`9df184b`, `53dbc05`, `2026-08-20`) — see the status notes on each phase below.
Phases 2–4 remain proposals.

## Why now

Four services compute per-fixture expected goals for overlapping sets of
fixtures, with separately-maintained copies of the same maths:

| file | lines |
|---|---|
| `app/services/projection_service.py` | 6,346 |
| `app/services/projection_all_teams_service.py` | 2,056 |
| `app/services/international_projection_service.py` | 1,049 |
| `app/services/euro_comp_projection_service.py` | 824 |

Inside `projection_service.py` alone the blend is written out **six times** —
`projections()`, `fixtures()`, `predicted_table()`, `teams()`, `players()`,
`player_props()` each call `_setup_league` + `_prepare_league` and then
re-implement the per-fixture maths. `projection_all_teams_service` is a
seventh copy. Shipping the odds guardrail on 2026-08-20 meant editing six
sites; the seventh was missed, and nobody noticed because the seventh only
runs when a human presses a button.

That is the actual cost: **a change is not "done" until it is done seven
times, and there is no signal when it isn't.**

---

## Part 1 — what each service is actually for

The point of this section is to separate *genuine domain difference* from
*accidental duplication*. Only the second kind should be merged.

### 1. `projection_service.projections()` — KEEP (reference implementation)

The domestic league pipeline. Its domain assumptions are specific and real:
a fixed league table, promotion/relegation, home advantage measured from
standings, per-league goal averages, promoted-team priors, and a unified
rating built from form + market value + odds.

Triggered by `projections:schedule-check` (full pass + refresh pass) via
`/api/projections`. **This is the canonical path** — everything the site
publishes on a normal day comes from here.

### 2. `projection_service`'s five other public methods — DELETE

`fixtures()`, `predicted_table()`, `teams()`, `players()`, `player_props()`
(lines 3061–6346, ~3,285 lines) exist to run one stage of the pipeline in
isolation. Their endpoints — `/api/projections/fixtures`,
`/predicted-tables`, `/teams`, `/players`, `/player-props` — have **zero
callers**: not Laravel (all 11 `callProjectionsApi` sites checked), not the
scheduler, not any script in either repo.

They are five of the seven copies of the blend, kept alive by nothing.

> Note: Laravel serves its own public read-API at `statz.ai/api/projections/
> fixtures`. Same path, different server, unrelated. Don't confuse them when
> grepping.

### 3. `projection_all_teams_service` — GUT to an orchestrator

Its **legitimate** job is orchestration: loop every `is_projected`
competition, keep `projections_runs` rows warm (`touch_all_running`), and
write a completion row per league (`upsert_run_complete`). ~150 lines of
real work.

Its **illegitimate** job is the other ~1,900 lines: an inlined copy of the
domestic pipeline. There is no domain reason for it. The file already
delegates two of its three cases correctly — euro comps to
`EuroCompProjectionService.projections()` and internationals to
`InternationalProjectionService.projections()` (lines 118–152). Only the
domestic branch inlines.

The original justification is gone, and the code says so. Line 94:

> *"Per-league LeagueDataLoader (created in the loop below) reads from the
> source DB directly. No shared cache pre-load needed."*

It was built to amortise one big `DataCache` load across leagues. That cache
was deleted in the direct-DB migration. What's left is a loop that happens
to have grown its own maths.

### 4. `euro_comp_projection_service` — KEEP (genuine domain difference)

CL/EL/ECL teams come from ~20 different domestic leagues, so ratings must be
normalised onto a common scale (`LEAGUE_COUNTRY_DICT`, `TOP_5_LEAGUE_IDS`
goal-average pool) and the data load must be scoped narrowly
(`restrict_team_ids`, `comp_fixture_ids` — the change that cut finals runs
432s → 20s). There is no single league table to simulate.

It already reuses `ProjectionService._resolve_league_id_db`, the shared
loader and `compute_final_goals_and_probs`. That is the right relationship.

### 5. `international_projection_service` — KEEP, **and copy its design**

National teams: no club form, no market values, ratings from the
international algorithm, squads instead of rosters, neutral venues, a
friendlies venue classifier, and a knockout bracket simulator.

More importantly, **this service is already built the way the domestic one
should be.** It has:

- **Named, numbered stages** — Step 1 ratings, 2 fixtures/1X2, 3 bracket sim,
  3b goals re-sync, 4 team stats, 5 player stats, 6 WC fantasy, 6b FanTeam,
  7 tournament totals.
- **A declarative per-comp scope** — `IntlProjectionScope` (frozen dataclass)
  with `bracket_config`, `fantasy_rules`, `has_squad_source`,
  `require_result_odds`, `skip_both_below_rating`, `odds_beta`. A `None`
  turns a stage off. Adding a competition = one registry entry + flipping
  `is_projected`.
- **Modes that skip stages** — `lean` runs only 1/2/3/7 (~15–20s vs ~3.5min);
  `ratings_only` runs 1 and returns; `fixture_ids` scopes to per-fixture.

That is exactly the shape the domestic pipeline needs. The target design
below is not speculative architecture — it is "make the oldest service look
like the newest one".

### 6. Stage services — KEEP as-is

`tournament_simulation_service`, `wc_player_stat_service`,
`wc_fantasy_points_service`, `fanteam_wc_projection_service`,
`wc_tournament_player_projection_service`, `international_team_stat_service`,
`fpl_recalc_service`, `retrain_service`. Each owns one stage and is called by
a pipeline. Already correct.

### 7. Shared maths — KEEP as-is

`odds_blend`, `statz_functions`, `unified_ratings`, `team_strength`,
`fixture_strengths`, `xminutes`, `fpl_bonus_sim`, `fpl_penalties`,
`fpl_bps`. Libraries, no orchestration. Already correct.

---

## Part 2 — measured drift (what the status quo actually costs)

Verified in code 2026-08-20. `projection_all_teams_service` vs
`projection_service`, both projecting the same domestic fixtures:

| # | divergence | evidence |
|---|---|---|
| 1 | Old binary odds gate still live | `_fx_beta = 0.0 if _both else odds_beta`, all-teams line 887. Deleted from the six `projection_service` sites; never removed here |
| 2 | No goals O/U cascade | `compute_final_goals_and_probs` never called; still on the legacy probability-space path via `find_inputs_for_probs`. Misses paths 1–3 |
| 3 | No odds guardrail | consequence of #2 |
| 4 | No Dixon-Coles | zero mentions of `rho` in the file; `get_result_probs(h, a, boost)` called without it → flat draw boost, including for the rho values retuned 2026-08-20 |
| 5 | **Team dials ignored** | `apply_team_dials_to_ratings`: 2 call sites in `projection_service`, **0** in all-teams. George's manual admin dials are silently dropped |
| 6 | **FPL bonus sim skipped** | `simulate_bonus_for_frame`: 2 sites vs **0**. Bonus falls back to the pre-2026-08-03 proxy |
| 7 | Stale FPL rows left behind | `cleanup_fpl_projections_async` absent |
| 8 | No `mode` / `lean` / `ratings_only` / `fixture_ids` | the button is full-run-only |

Both services write **the same tables** — `fixture_projections`,
`team_projections`, player + prop projections, `predicted_table`,
`league_position_probabilities`, all five fantasy tables, and the model +
accuracy datasets. It is last-writer-wins. Pressing Run All Leagues
overwrites every league's published numbers with the older maths.

> Corrects an earlier note in memory that all-teams "owns"
> `league_position_probabilities`. It doesn't — both write it. The only
> difference is which trigger started the run.

---

## Part 3 — target design

Three layers. Layer 1 already exists; Layer 3 already exists for
internationals.

### Layer 1 — the spine (exists, unchanged)

```
_setup_league(league)   -> ctx     # config, loader, seasons, weightings, betas
_prepare_league(ctx...) -> ratings # models, accuracy datasets, unified ratings, dials
```

Both already shared by all six `projection_service` methods. The all-teams
copy re-implements them inline (its own comments at lines 362 and 711 say
"mirrors `projection_service._prepare_league`").

### Layer 2 — stages (new: extract from `projections()`)

`projections()` is ~2,070 lines (991–3060) of straight-line code. It already
has natural seams at each write. Extract to named functions, each taking
`(ctx, ratings, ...)` and returning a frame:

| stage | today | writes |
|---|---|---|
| `stage_fixtures` | 1090–1257 | `fixture_projections` |
| `stage_table` | ~1300–1442 | `predicted_table`, `league_position_probabilities` |
| `stage_model_dataset` | ~1512 | model dataset |
| `stage_teams` | ~1600–1842 | `team_projections` |
| `stage_accuracy` | ~1872 | accuracy dataset |
| `stage_players` | ~1900–2033 | player projections, player stat probs |
| `stage_props` | 5405–6346 equivalent | prop projections |
| `stage_fantasy` | ~2100–3060 | 5 fantasy tables, xMins, bonus, dials |

**The one rule that makes this stick:** the blend lives in exactly one
function, `compute_final_goals_and_probs`. Every pipeline calls it and passes
its own weight policy (`guardrail=True` domestic, flat `odds_beta` for euro
and intl). Three of the four already do. No pipeline gets its own copy —
ever.

### Layer 3 — pipelines (thin composition)

```
DomesticPipeline  = spine + all stages
EuroPipeline      = own cross-league ratings + fixtures/teams/players (no table sim)
IntlPipeline      = own intl ratings + stages + bracket sim + WC fantasy   [exists]
run_all(leagues)  = for league in leagues: pipeline_for(league).run()      [orchestration only]
```

Scope config: internationals use the `IntlProjectionScope` dataclass because
their per-comp differences are structural. Domestic competitions already have
their equivalent in the **database** — `competition_projection_config` holds
`odds_beta`, `mv_beta`, `dixon_coles_rho`, `boost` per league. Domestic
doesn't need a new dataclass; it needs the stage toggles (`lean`,
`ratings_only`) that `LeagueRequest` already declares but only the intl
service honours.

---

## Part 4 — phases

Each is independently shippable. Only Phase 1 changes published numbers.

### Phase 0 — delete the five dead endpoints — ✅ SHIPPED 2026-08-20
Remove `fixtures()`, `predicted_table()`, `teams()`, `players()`,
`player_props()` from `projection_service.py` and their five routes.

- **~3,285 lines deleted; 7 copies of the blend → 2.**
- Output impact: **none.** Nothing calls them.
- Confirmed unused before deleting, by measurement not inspection: all six
  methods log `_prepare_league mode=` and only `projections()` writes a
  `projections_runs` row, so a call to any of the five would log without a
  matching row. Over 14 days of production traffic: **743 log lines, 743
  rows.** Post-deploy all five return 404; `/api/projections/fixture` still
  answers.
- Result: `projection_service.py` 6,346 → 3,060 lines.

### Phase 1 — make the all-leagues loop delegate — ✅ SHIPPED 2026-08-20
In the domestic branch of `projectionAllTeams`, do what the euro and intl
branches already do:

```python
await projection_service.projections(LeagueRequest(league=league))
```

Keep `touch_all_running` / `upsert_run_complete` around the call. Delete the
~1,900 inlined lines. `projection_all_teams_service.py` ends up ~150 lines.

- **2 copies → 1. The drift class is gone permanently.**
- Output impact: **yes, by design.** Run All Leagues starts producing the
  same numbers as the nightly — gaining the guardrail, the goals cascade,
  Dixon-Coles, team dials, the bonus simulator and FPL stale-row cleanup.
  This is the sign-off item.
- Verified on prod: League Two through the button logged
  `team dials applied: Exeter City…` and `24 team(s) carry odds in their
  rating — guardrail moderates the overlap by distance`. Neither line could
  appear on the old path. 6.2 min vs 5.7–7.5 min for the same league on the
  nightly. `projections_runs` row written as success.
- Result: `projection_all_teams_service.py` 2,056 → 199 lines.

### Phase 2 — extract the stages
Pure refactor of `projections()` into the Layer 2 functions. No logic change.

- Verify: byte-identical output on a full run against a pre-refactor
  baseline, league by league.
- Unlocks: `lean` / `ratings_only` for domestic comps (currently intl-only),
  which is what makes cheap partial re-runs possible.
- Effort: ~2 days, best done one stage per commit.

### Phase 3 — converge euro on the shared stages (optional)
Once stages exist, `euro_comp_projection_service` keeps its own ratings
builder and scoping but composes the shared stages for fixtures/teams/
players. Only worth doing if Phase 2 lands cleanly.

### Phase 4 — housekeeping
`cron_projections.sh` (repo root) posts to `/api/projections/fetch-data`,
an endpoint deleted in the Phase 7 direct-DB cleanup. If it is still in the
server's crontab it fails at step 1 every night and never reaches its
`/all-leagues` call. Confirm it is uninstalled, then delete the file. It is
also the only other caller of `/all-leagues` besides the admin button.

---

## Part 5 — risks and things that must not be lost

- **Class-level globals.** `ProjectionService._current_source` and
  `._strength_inputs` are set on the class, not the instance. Sequential runs
  are safe (the server holds a global projection lock) but stage extraction
  must either keep them or thread them through `ctx`. Do not make stages
  concurrent.
- **Run-status bookkeeping.** The all-teams loop refreshes `started_at` on
  every queued `projections_runs` row each iteration, so leagues waiting past
  30 minutes don't trip `projections:mark-stuck`. `projection_service` does
  not do this — `routes.py` does it for single runs. Phase 1 must keep the
  loop's version.
- **Don't over-merge.** Euro and international pipelines differ for real
  domain reasons (cross-league rating scale; national teams, squads, neutral
  venues, brackets). The goal is one copy of each *shared* stage, not one
  pipeline for everything.
- **Deploy.** The projection server has no auto-deploy: manual
  `git pull && docker compose up --build -d` after every push.

---

## Part 6 — decisions

1. ~~**Phase 1 output change.**~~ Agreed by George 2026-08-20 and shipped.
   Run All Leagues now matches the nightly.
2. ~~**Phase 0 deletion.**~~ Deleted outright, after the 743/743 measurement
   above.
3. **Open: appetite for Phase 2.** The ~2-day tidy that extracts named stages
   from the 2,070-line `projections()` and gives domestic comps the
   `lean` / `ratings_only` partial re-runs internationals already have.
   Nothing depends on it — Phases 0+1 already killed the drift.
