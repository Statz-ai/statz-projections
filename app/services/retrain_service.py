"""
Phase 4: weekly model retraining — dry-run mode.

Iterates every (league, stat) pair, trains a fresh PoissonRegressor on
the DB-sourced training data (projection_model_dataset, populated by
Phase 2 dual-write + Phase 3 reads), writes the .sav to a versioned path,
and INSERTs a row in projection_models with is_active=0.

For each (league, stat) ALSO logs what the promotion decision WOULD be
against the current incumbent, using the guardrail in
_decide_promotion(). Phase 4 does NOT actually promote — that's Phase 5.
Purpose of dry-run: observe week-over-week MAE variance for 3–4 cycles
so the 2%/10% thresholds can be tuned empirically before promotion goes
live.

Also trains the "All Leagues" fallback models (competition_id=NULL rows)
by training on every row in projection_model_dataset, regardless of
league.

Called from /api/retrain endpoint (routes.py). Not yet scheduled — Phase
5 adds the weekly cron on Laravel's Kernel.
"""
import gc
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from app.repository.projection_dataset_repo import load_model_dataset_async
from app.repository.projection_model_repo import fetch_active_model, insert_projection_model, promote_model
from app.services.statz_functions import get_stat_list, get_trainable_stat_list

logger = logging.getLogger("retrain")

# Minimum training rows to attempt a per-league model. Below this, we skip
# the league and let it fall back to the All Leagues pool at projection
# time. Current Premier League has ~1,600 rows, Liga Portugal ~1,400,
# smaller comps fewer — 100 is a defensive floor, NOT a quality threshold.
MIN_TRAINING_ROWS = 100

# Leagues whose fixtures train the "All Leagues" fallback model pool.
# This fallback is used at projection time for:
#   - Euro comps (Champions / Europa / Conference) which draw teams from
#     many domestic leagues and have no single-league training source
#   - Newly-added comps with <MIN_TRAINING_ROWS data of their own
#
# Previously the fallback trained on the union of ALL leagues (~11,600
# rows) which OOM'd the gunicorn worker on 2026-04-23 while grid-searching
# Tackles. Top-5 union is ~6,000 rows — fits in memory, represents the
# football styles of the teams the fallback actually serves (euro-comp
# teams mostly come from top-5 leagues), and avoids polluting the model
# with sparse data from newly-onboarded leagues.
TOP_5_LEAGUE_NAMES = ("Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1")

# When training the All Leagues fallback, cap per-league rows to the most
# recent N fixtures. Recent data reflects current tactics / players /
# refereeing trends — older fixtures drift from reality. 1000 rows per
# league ≈ 1.3 seasons of recent data. 5 leagues × 1000 = 5,000 total —
# comfortably within worker memory AND a reasonable CV fold size
# (3-fold = ~1,666 rows per fold, healthy for Poisson regression).
FALLBACK_RECENT_ROWS_PER_LEAGUE = 1000

# Per-design: fit_model (direct PoissonRegressor newton-cholesky fit) for
# the two Passes stats because grid_search was historically slow + offered
# little improvement on the wider Passes distributions. grid_search for
# everything else.
_FIT_MODEL_STATS = ("Passes", "Successful Passes")


def _snake(stat_name: str) -> str:
    """'Shots On Target' → 'shots_on_target'."""
    return stat_name.lower().replace(" ", "_")


def _predictor_columns(stat_name: str) -> tuple:
    """Parquet-format column names returned by load_model_dataset_async."""
    return (f"Team {stat_name} History", f"Opponent {stat_name} History Against")


def _target_column(stat_name: str) -> str:
    return f"Team {stat_name}"


def _model_build_path(league_dir_name: str, stat_name: str) -> str:
    """Timestamped file path under /app/app/model-builds/{league_dir}/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Keep the existing directory naming convention:
    # - Per-league: /app/app/model-builds/{Premier League}/
    # - Fallback:   /app/app/model-builds/All Leagues/
    base = Path("/app/app/model-builds") / league_dir_name
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"{stat_name}_v{ts}.sav")


def _fit_model(X_train, y_train) -> tuple:
    """Direct newton-cholesky fit. Returns (model, algorithm_name, hyperparams_dict)."""
    model = PoissonRegressor(solver="newton-cholesky")
    model.fit(X_train, y_train)
    return model, "fit_model", {"solver": "newton-cholesky"}


def _grid_search(X_train, y_train) -> tuple:
    """CV over alpha × fit_intercept. Returns (model, algorithm_name, best_params).

    Returns best_estimator_ only — NOT the GridSearchCV wrapper. cv_results_
    holds every fold's fitted sub-model and is the primary driver of the
    memory bloat we hit on 2026-04-22 (gunicorn OOM after ~12 grid searches).
    best_estimator_ is a PoissonRegressor; same .predict() contract so
    downstream pickle-load + .predict() at projection time is unchanged.

    Trimmed grid (2026-04-23):
      - alpha: 5 values (step 0.2) covers [0, 1.0]; 10-step was noise exploration
      - max_iter fixed at 200 (Poisson on ~1k-2k rows converges well under that)
      - fit_intercept: [True, False] retained
      - cv folds: 5 → 3 (still ~400 rows/fold on smallest league)
    Total fits per stat: 5 × 2 × 3 = 30 (was 5 × 3 × 2 × 5 = 300). ~10× faster.
    Projected full retrain: ~30 min (was ~5h). Alpha is the hyperparam that
    actually moves MAE for this model class; the cuts trim noise sweep not
    signal.
    """
    param_grid = {
        "alpha": np.arange(0, 1.01, 0.2),
        "max_iter": [200],
        "fit_intercept": [True, False],
    }
    pr = PoissonRegressor()
    gs = GridSearchCV(pr, param_grid, cv=3, scoring="neg_mean_squared_error")
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    best_params = dict(gs.best_params_)
    # Drop the CV wrapper + its results so they can be GC'd before next iter
    del gs
    return best, "grid_search", best_params


def _decide_promotion(incumbent: Optional[dict], new_mae: float) -> tuple:
    """Returns (should_promote, reason) — used by Phase 5; Phase 4 logs only.

    Conservative initial thresholds for first auto-promote production
    rollout (2026-05-05): require meaningful improvement to promote,
    fail-loud on real regressions.
      - Promote if new MAE ≤ 5% worse than incumbent (allows session noise +
        small trim-induced drift up to that bound — see GridSearchCV trim
        verification 2026-05-05: 5/11 stats within 2%, 5 within 5%, 1 at 5.7%)
      - Hold if 5–15% worse (no promotion, no rejection — keeps incumbent)
      - Reject + alert if >15% worse (real regression)
    Wider than the original 2%/10% draft from model_retraining_design.md
    open decision #6, intentionally — first auto-promote run shouldn't
    flip on borderline noise. Tighten after a few cycles of observation
    if false-promotes turn out to be common.
    """
    if incumbent is None or incumbent.get("holdout_mae") is None:
        return True, "initial (no incumbent or no baseline MAE)"

    incumbent_mae = incumbent["holdout_mae"]
    ratio = new_mae / incumbent_mae

    if ratio <= 1.05:
        pct = (incumbent_mae - new_mae) / incumbent_mae * 100
        sign = "improved" if pct >= 0 else "within noise"
        return True, f"MAE {sign} {abs(pct):.1f}%"

    if ratio >= 1.15:
        pct = (ratio - 1) * 100
        return False, f"MAE degraded {pct:.1f}% — REJECTED (>15% worse)"

    pct = (ratio - 1) * 100
    return False, f"MAE worse ({pct:.1f}%) — holding incumbent"


async def _train_one(
    df: pd.DataFrame,
    stat_name: str,
    dry_run: bool = True,
) -> Optional[dict]:
    """Train one global model. Returns summary dict or None if
    insufficient data / training fails.

    Per-league training was retired 2026-05-21 — every league now uses
    the single "All Leagues" global model. Caller no longer passes a
    competition_id; rows are always inserted with NULL.

    dry_run=True: Phase 4 — log promotion decision only.
    dry_run=False: Phase 5 — if guardrail passes, call promote_model() to
    flip is_active to the new row and demote the incumbent.
    """
    label = "All Leagues"
    predictors = _predictor_columns(stat_name)
    target = _target_column(stat_name)

    needed = list(predictors) + [target]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        logger.warning(f"[retrain {label}] {stat_name}: missing columns {missing} — skipping")
        return None

    train_df = df.dropna(subset=needed)
    if len(train_df) < MIN_TRAINING_ROWS:
        logger.info(
            f"[retrain {label}] {stat_name}: only {len(train_df)} rows "
            f"(< {MIN_TRAINING_ROWS}) — skipping"
        )
        return None

    X = train_df[list(predictors)]
    y = train_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    n_train_rows = len(X_train)

    if stat_name in _FIT_MODEL_STATS:
        model, algo, hyperparams = _fit_model(X_train, y_train)
    else:
        model, algo, hyperparams = _grid_search(X_train, y_train)

    y_pred = model.predict(X_test)
    holdout_mae = float(mean_absolute_error(y_test, y_pred))
    holdout_r2 = float(r2_score(y_test, y_pred))

    # Save .sav under the global "All Leagues" directory
    file_path = _model_build_path(label, stat_name)
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

    new_id = await insert_projection_model(
        competition_id=None,
        stat_name=stat_name,
        algorithm=algo,
        hyperparams=hyperparams,
        trained_on_n_rows=n_train_rows,
        holdout_mae=holdout_mae,
        holdout_r2=holdout_r2,
        file_path=file_path,
    )

    incumbent = await fetch_active_model(None, stat_name)
    would_promote, reason = _decide_promotion(incumbent, holdout_mae)
    incumbent_mae = incumbent.get("holdout_mae") if incumbent else None

    # Phase 5: when dry_run=False AND the guardrail passes, actually flip
    # is_active to the new row. promote_model() does the demote-incumbent +
    # activate-new in a single transaction so projections always see exactly
    # one active model per (comp, stat). Failures are logged + swallowed —
    # we'd rather lose a single promotion than abort the whole retrain run.
    promoted = False
    if not dry_run and would_promote:
        try:
            await promote_model(new_id, reason)
            promoted = True
        except Exception as promo_err:
            logger.warning(
                f"[retrain {label}] {stat_name}: promote_model({new_id}) failed — "
                f"keeping incumbent active: {promo_err}",
                exc_info=True,
            )

    # Free the bulky sklearn objects before the next iteration.
    # Previous runs OOM'd around the 12th-13th grid_search because
    # GridSearchCV retains every CV-trained sub-model in .cv_results_
    # plus all the fold predictions. gunicorn's worker grew until
    # kernel SIGKILL'd it mid-training. Explicit del + gc.collect()
    # knocks the working set back down between iterations.
    del model, X, y, X_train, X_test, y_train, y_test, y_pred, train_df
    gc.collect()

    action = "PROMOTED" if promoted else ("would_promote" if would_promote else "NOT promoting")
    logger.info(
        f"[retrain {label}] {stat_name}: algo={algo} rows={n_train_rows} "
        f"holdout_mae={holdout_mae:.4f} r2={holdout_r2:.4f} "
        f"incumbent_mae={incumbent_mae} {action} ({reason}) "
        f"new_id={new_id}"
    )

    return {
        "stat": stat_name,
        "algorithm": algo,
        "n_rows": n_train_rows,
        "holdout_mae": holdout_mae,
        "holdout_r2": holdout_r2,
        "incumbent_mae": incumbent_mae,
        "would_promote": would_promote,
        "promoted": promoted,
        "reason": reason,
        "new_id": new_id,
    }


async def retrain_all_models(dry_run: bool = False) -> dict:
    """Main entry point. Returns a summary dict with per-(league, stat) results.

    dry_run=False (Phase 5 default, since 2026-04-28): after each model
    trains, if the guardrail in _decide_promotion() passes (new MAE ≤ 2%
    worse than incumbent, or incumbent has no baseline), call
    promote_model() to flip is_active. Models that degrade more than 10%
    are rejected; 2-10% worse held without promotion.

    dry_run=True: original Phase 4 behaviour — trains + inserts rows with
    is_active=0, logs would-be promotion decisions, doesn't actually
    promote. Useful for one-off retrains where you want to inspect the
    holdout MAE without affecting live projections.
    """
    t_start = time.time()
    logger.info(f"[retrain] START dry_run={dry_run}")

    # Pull the full dataset once; filter per-league in memory.
    all_df = await load_model_dataset_async()
    logger.info(f"[retrain] loaded {len(all_df)} total rows from projection_model_dataset")

    # Trainable set, not get_stat_list(): the PL-only stats (Key Passes,
    # Big Chances Created) need models built from the FULL top-5 training
    # set even though only the PL run consumes them.
    stat_list = [s for s in get_trainable_stat_list() if s != "Goals"]
    # `per_league` / `skipped_leagues` kept in the response shape so admin
    # / API consumers don't break, but are always empty now — per-league
    # models were retired 2026-05-21 (every league reads the same global
    # model via load_model() in statz_functions). See the load_model
    # docstring for the Superliga case that drove the change.
    results = {"per_league": {}, "all_leagues": {}, "skipped_leagues": []}

    # Single global model per stat, trained on the top-5 + recent-rows
    # fallback dataset. This is now the ONLY model the projection pipeline
    # reads.
    fallback_df = await _build_all_leagues_fallback_df(all_df)
    all_results = []
    for stat in stat_list:
        r = await _train_one(fallback_df, stat, dry_run=dry_run)
        if r is not None:
            all_results.append(r)
    results["all_leagues"] = all_results

    elapsed = (time.time() - t_start) / 60
    logger.info(
        f"[retrain] COMPLETE dry_run={dry_run} — {elapsed:.1f} min, "
        f"{len(all_results)} global models trained "
        f"(per-league training retired 2026-05-21)"
    )
    return results


async def _build_all_leagues_fallback_df(all_df: pd.DataFrame) -> pd.DataFrame:
    """Construct the training set used by the 'All Leagues' fallback models.

    Scoped to the top-5 European leagues and capped to the most recent
    N rows per league. Previously trained on the full union (~11.6k
    rows) which OOM'd the gunicorn worker on 2026-04-23 around the 9th
    grid-search. Top-5 + row cap = ~5k rows, comfortable headroom.

    Kept separate from retrain_all_models so both the full weekly cycle
    and retrain_partial() build the identical fallback set — preventing
    drift between the two code paths.
    """
    top5_ids = await _lookup_competition_ids(TOP_5_LEAGUE_NAMES)
    if not top5_ids:
        logger.warning("[retrain All Leagues] no top-5 league IDs resolved — using full union")
        fallback_df = all_df
    else:
        top5_df = all_df[all_df["comp_id"].isin(top5_ids)]
        if "kickoff_datetime" in top5_df.columns:
            fallback_df = (
                top5_df.sort_values("kickoff_datetime", ascending=False)
                .groupby("comp_id", sort=False)
                .head(FALLBACK_RECENT_ROWS_PER_LEAGUE)
            )
        else:
            logger.warning(
                "[retrain All Leagues] kickoff_datetime column missing — "
                "cannot apply recent-rows cap; using full top-5 union"
            )
            fallback_df = top5_df
    logger.info(
        f"[retrain All Leagues] training on {len(fallback_df)} rows from "
        f"{len(top5_ids)} top-5 leagues "
        f"(cap={FALLBACK_RECENT_ROWS_PER_LEAGUE}/league; was {len(all_df)} across all leagues)"
    )
    return fallback_df


async def retrain_partial(competition_id: Optional[int], stats: list, dry_run: bool = False) -> dict:
    """Train a subset of global models — used to fill gaps left by a
    partial-success full retrain without re-doing the work that already
    completed.

    Per-league training was retired 2026-05-21; the `competition_id`
    parameter is retained at the API surface (PartialRetrainRequest)
    for callers built against the older contract, but is ignored —
    every retrain runs against the "All Leagues" fallback scope
    (top-5 + row cap) and writes a single global row.

    stats is a list of canonical stat names (e.g. ["Interceptions",
    "Offsides"]) matching get_stat_list() output. Unknown stats are
    skipped with a warning.

    dry_run defaults to False (Phase 5) — pass True for an inspection-only
    run that doesn't flip is_active.
    """
    if not stats:
        return {"scope": "n/a", "results": [], "skipped": [], "message": "no stats specified"}

    if competition_id is not None:
        logger.warning(
            f"[retrain partial] ignoring competition_id={competition_id} — "
            f"per-league training retired 2026-05-21; running All Leagues"
        )

    t_start = time.time()
    scope = "All Leagues"
    logger.info(f"[retrain partial] START scope={scope} stats={stats}")

    valid_stats = [s for s in get_trainable_stat_list() if s != "Goals"]
    unknown = [s for s in stats if s not in valid_stats]
    if unknown:
        logger.warning(f"[retrain partial] unknown stats (skipped): {unknown}")
    target_stats = [s for s in stats if s in valid_stats]
    if not target_stats:
        return {"scope": scope, "results": [], "skipped": unknown, "message": "no valid stats after filtering"}

    all_df = await load_model_dataset_async()
    logger.info(f"[retrain partial] loaded {len(all_df)} total rows from projection_model_dataset")
    df = await _build_all_leagues_fallback_df(all_df)

    results = []
    for stat in target_stats:
        r = await _train_one(df, stat, dry_run=dry_run)
        if r is not None:
            results.append(r)

    elapsed = (time.time() - t_start) / 60
    logger.info(f"[retrain partial] COMPLETE scope={scope} — {elapsed:.1f} min, {len(results)} models trained")
    return {"scope": scope, "results": results, "skipped": unknown}


async def _lookup_competition_ids(names: tuple) -> list:
    """Resolve a tuple of competition names → list of IDs via the competitions
    table. Returns ids in the same order as names; names not matched are
    skipped silently (a warning is logged)."""
    import app.database as _db
    from app.database import get_connection
    if not names:
        return []
    conn = None
    try:
        conn = await get_connection()
        async with conn.cursor() as cursor:
            placeholders = ", ".join(["%s"] * len(names))
            await cursor.execute(
                f"SELECT id, name FROM competitions WHERE name IN ({placeholders})",
                tuple(names),
            )
            rows = await cursor.fetchall()
            found = {r[1]: int(r[0]) for r in rows}
            missing = [n for n in names if n not in found]
            if missing:
                logger.warning(f"_lookup_competition_ids: no match for {missing}")
            return [found[n] for n in names if n in found]
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)
