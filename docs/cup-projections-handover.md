# Domestic cup projections — handover

Written 2026-08-20 for a fresh session. **Nothing is built.** This is the
groundwork: what's true today, what to copy, and the decisions to make first.

Goal: project **Carabao Cup** fixtures, then **FA Cup** later in the season on
the same logic (George, 2026-08-20). The hard part is cross-league team
strength — cup fixtures pair clubs from four different tiers.

---

## 1. Start with Carabao. The FA Cup is a January problem.

Measured on prod 2026-08-20:

| | comp id | upcoming fixtures | distinct teams | teams with `team_ratings` | fixtures with bet365 odds |
|---|---|---|---|---|---|
| Carabao Cup | **27** | 23 | 46 | **46 / 46** | **23 / 23** |
| FA Cup | **24** | 137 | 273 | **1 / 273** | 0 |

**Carabao is fully tractable today** — every team already carries a rating,
every fixture already carries a price. First upcoming fixture 2026-08-25.

**The FA Cup is not, and it isn't a code problem.** Its 137 upcoming fixtures
are qualifying rounds between non-league clubs — Hungerford Town vs Wokingham
Town, Jersey Bulls vs Sheerwater. We hold no ratings and no odds for them and
never will. The FA Cup only enters scope at the **third round proper (January)**
when Premier League and Championship clubs join, and even then the logic should
skip any tie whose teams aren't rated. Build for Carabao; the FA Cup switch-on
is later and mostly a scope question, not new logic.

Both comps are `is_projected = 0` today. Note the Carabao odds only exist
because comp 27 was wired into four books on 2026-08-20 — see
`memory/carabao_cup_odds_wiring.md`.

---

## 2. Copy the euro-comp pattern (George's call, and it's the right one)

`app/services/euro_comp_projection_service.py` already solves exactly this
problem — a competition whose teams come from many different leagues — and it
solves it by **not recomputing anything**:

- **`LEAGUE_COUNTRY_DICT`** — the domestic leagues in scope. Every entry MUST
  have `team_ratings` rows and a `competition_projection_config` row.
- **Cached-ratings path (2026-05-27)** — it reads the latest `team_ratings` row
  per (competition, team) straight from the DB rather than recomputing inner-
  league ratings. Those rows are already post-MV, post-dial and rescaled to
  mean 100 by the nightly domestic run. Recomputing was ~25s × 15 leagues of
  wasted work.
- **A per-league multiplier on top** — the UEFA coefficient
  (`competitions.uefa_coefficient_index`) is what makes ratings from different
  leagues comparable.
- **`TOP_5_LEAGUE_IDS`** — the Poisson goal-average baseline comes from the top
  5 leagues only, not the whole pool. Averaging everything dragged the baseline
  toward weaker leagues' scoring rates (decision 2026-05-27).
- **Narrow scope** — `restrict_team_ids` + `comp_fixture_ids` cut finals runs
  from 432s to 20s. Cups have the same shape (few fixtures, many leagues), so
  wire this in from the start, not later.

The cup equivalent is nearly a rename: `CUP_COMPS = ['Carabao Cup', 'FA Cup']`,
and a `LEAGUE_DICT` of the four EFL tiers instead of 15 European ones.

---

## 3. The one real design decision: what replaces the UEFA coefficient?

Cross-league comparability is the whole problem. Three candidates, and this is
the thing to settle before writing code.

**(a) The cross-tier weights already in the DB.** `competition_projection_config`
carries them, and they already encode our view of the gap between tiers:

| comp | above | below | above_attack | above_defense | below_attack | below_defense |
|---|---|---|---|---|---|---|
| 8 Premier League | — | 9 | 1.000 | 1.000 | 0.600 | 0.600 |
| 9 Championship | 8 | 12 | 1.600 | 1.700 | 0.700 | 0.700 |
| 12 League One | 9 | 14 | 1.350 | 1.300 | 0.700 | 0.750 |
| 14 League Two | 12 | — | 1.250 | 1.200 | 0.700 | 0.750 |

Free, already tuned, already used to scale a promoted club's history. **But
they're pairwise (adjacent tiers only)** — chaining them to compare a PL club
with a League Two club compounds four hand-set numbers, which is exactly the
compounding problem that bit us on Ligue 1 lambdas. Treat with suspicion.

**(b) Derive it from cup fixture odds — George's suggestion.** *"Now we have
fixture odds we can use power rankings for the 4 EFLs and go from there."*
Cross-tier cup ties are priced by bookmakers directly, so each one is a
market-implied measurement of the gap between two specific clubs in two
different tiers. Fit a per-tier offset that best reconciles our mean-100
ratings with those prices. **This is the most principled option** and the
reason it's newly possible is that comp 27 got odds today.

Caveat: 23 fixtures is a thin sample for fitting four tier offsets, and
early-round Carabao ties are heavily rotated squads — the market is pricing a
weakened PL side, not the PL side our rating describes. See §4.

**(c) A hand-set cup coefficient**, mirroring `uefa_coefficient_index`. Simplest
to ship, easiest to reason about, and George owns the numbers (consistent with
how promoted-team ratings work). Reasonable as a v1 with (b) as the follow-up.

**Recommendation:** ship (c) or (a) to get fixtures on the board, and treat (b)
as the calibration pass once a round or two of results exist to validate
against. Don't block the first projection on the fitting problem.

---

## 4. Rotation is the real modelling risk, and it's cup-specific

A Carabao second-round tie is not the same team the league rating describes.
Premier League clubs rotate heavily in early rounds; League Two clubs often
don't. So the strength gap in a cup tie is systematically **narrower** than the
league ratings imply, and asymmetrically so.

Our confirmed-lineup machinery (`load_confirmed_lineups` in `odds_blend`)
only helps within an hour or two of kickoff. Before that, the market price is
the only thing that knows. Two implications:

- Expect our raw model to over-favour the higher-tier side. Don't read that as
  a ratings bug — it's a rotation effect the ratings can't see.
- This is an argument for a **high `odds_beta` for cups** — hug the book, as
  the international pipeline already does for friendlies (`odds_beta=0.7`,
  "the model is noisier on thin-data nations, so hug the book more and stop
  surfacing model error as inflated edges"). Same reasoning applies here.

---

## 5. Concrete first steps

1. **Read `euro_comp_projection_service.projections()` end to end.** It is ~820
   lines and is the template. Note how it delegates to `ProjectionService` for
   the shared helpers rather than reimplementing them.
2. **Decide the coefficient question (§3).** Everything else is mechanical.
3. **New service or a scope on the euro one?** The euro service is already
   "competition whose teams span leagues". A `CUP_SCOPES` registry in the style
   of `INTL_SCOPES` (frozen dataclass, per-comp config, `None` switches a stage
   off) is the better-aged pattern — see
   `international_projection_service.py`, and
   `docs/projection-services-design-scope.md` §1.5 for why that design is the
   one worth copying.
4. **Set `odds_beta` high** for cups (start 0.7, per §4).
5. **Flip `is_projected = 1`** on comp 27 — that's the switch
   `ScheduleProjections` and the admin Run All both read.
6. **Verify with `stop_after=fixtures`** (shipped 2026-08-20): a fixtures-only
   run is ~1-3 min instead of a full run. See `memory/partial_league_reruns.md`.

---

## 6. Gotchas that will cost you a session each

- **No auto-deploy on the projection server.** `git pull && docker compose up
  -d --build` after every push, and avoid the ~01:00-02:30 and ~12:35-14:05 UTC
  run windows — a rebuild kills an in-flight run.
- **Qualifying fixtures are filtered** by `->mainStage()` on the Laravel side
  (`memory/qualifying_fixtures_exclusion.md`). Check what that does to FA Cup
  qualifying rounds before assuming they're absent — and `Fixture::getIds()` is
  hub-semantics (already filtered).
- **The tiered gameweek horizon is Premier League only.** Cups take the plain
  date window, so `PROJECTION_DAYS` governs how far ahead they project — a cup
  round more than 5 days out won't be picked up. Likely needs widening for
  cups, where rounds are weeks apart.
- **Team ratings are per (competition, team, date)** and overwritten in place.
  A cup club's rating lives under its DOMESTIC competition id, not the cup's —
  read accordingly.
- **`team_projections.goals` is a copy of `fixture_projections` goals**
  (`memory/team_goals_copy_of_fixture.md`).
- **One pipeline now.** `projection_all_teams_service` no longer computes
  anything (2026-08-20) — don't add cup logic there.
- **Projection logic changes need George's sign-off** before shipping.

## 7. Open questions for George

1. Which coefficient approach (§3) — hand-set to start, or fit from odds?
2. How far ahead should cups project? Rounds are weeks apart; the 5-day
   domestic window won't reach them.
3. Do cups need player projections and props, or fixture-level only to start?
   Player-level roughly triples the work and is where rotation hurts most.
