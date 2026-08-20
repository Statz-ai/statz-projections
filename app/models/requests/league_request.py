from typing import List, Optional
from pydantic import BaseModel

class LeagueRequest(BaseModel):
    league: str
    fixture_ids: Optional[List[int]] = None
    # "full" (default) runs the full projection pipeline including accuracy
    # dataset gap-fill + metrics calculation. "refresh" skips those analysis
    # blocks since the 1:35pm scheduled refresh run doesn't need to rebuild
    # accuracy numbers that were already computed by the morning 2am run.
    # See memory/projections_system.md for the full justification.
    mode: Optional[str] = "full"
    # Lean tournament-only re-sim. When True (and no fixture_ids), the run
    # executes ONLY the bracket-wide steps — ratings (1), 1X2 (2), tournament
    # sim (3) and tournament-player/top-scorer totals (7) — and SKIPS the
    # expensive per-fixture stat/fantasy steps (4/5/6/6b). ~15-20s vs ~3.5min.
    # Used by the kickoff-timed post-match finalizer to refresh advancement %,
    # power ratings and the scorer race right after a result lands, without
    # re-deriving every player's per-fixture lines (those refresh on the
    # twice-daily full passes). Must be paired with an empty fixture list.
    lean: Optional[bool] = False
    # Ratings-only run: recompute + write team_ratings (Step 1) and return —
    # skip the sim, per-fixture stats, fantasy and top-scorer. Used by the
    # WC xG-landed trigger: team xG (stat 5304) arrives ~1h after a match
    # finalises, so the post-match lean re-sim fires too early to capture it;
    # this re-runs JUST the ratings once the xG lands. Forces an empty fixture
    # list (ratings are bracket-wide).
    ratings_only: Optional[bool] = False
    # Stop the DOMESTIC pipeline after a named stage instead of running it to
    # the end. Testing tool: the fixture stage — where the odds blend, the
    # guardrail and Dixon-Coles all live — takes about a second, but a full
    # Premier League run costs ~13.7 minutes, nearly all of it in the season
    # simulation, the model dataset and the player/fantasy stages downstream
    # of it. `stop_after="fixtures"` gets that loop down to ~2 minutes, which
    # is data load plus ratings and almost nothing else.
    #
    # Stages are a strict prefix chain, so this is "stop here", not "pick
    # some": each stage consumes the one before it.
    #
    #   fixtures  -> fixture_projections                       (~2m10s on PL)
    #   table     -> + predicted_table, league_position_probabilities (~3m45s)
    #   teams     -> + team_projections                              (~6m30s)
    #   players   -> + player_projections, player stat probs         (~11m)
    #   None      -> everything, including props and the 5 fantasy tables
    #
    # A partial run leaves every downstream table holding the PREVIOUS run's
    # numbers — that is the point, but it means a partial must never be
    # mistaken for a healthy full run. `_run_single_league` therefore writes
    # NO projections_runs row when this is set, so a partial can't clear a
    # failed-comp alert or feed the pipeline-dead canary in
    # ProjectionsFreshnessCheck. Nothing scheduled sets it.
    stop_after: Optional[str] = None
    # Skip the two analytics dataset writes (model dataset ~2m02s, accuracy
    # dataset ~47s on PL). They feed retraining and accuracy tracking; nothing
    # downstream in the same run reads them, so dropping them is safe for a
    # test run and saves ~2m49s. Leave False for anything whose numbers you
    # intend to keep.
    skip_datasets: Optional[bool] = False
