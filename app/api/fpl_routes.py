"""
FPL optimiser — on-demand solve endpoints.

Pure, synchronous, DB-free: Laravel builds the full input (player pool +
projections + eligibility, and for transfers the owned squad + sell prices +
bank), POSTs it here, and gets the solved plan/draft straight back. The solve
is HiGHS ILP over ~270 players, ~0.1-0.3s — well inside the request, so no
background task / lock / callback (unlike the projection pipeline). Does NOT
touch the DataCache or the projection file-lock; it's independent of the
projection run machinery.

  POST /api/fpl/solve-transfer  -> best transfers from an existing squad
  POST /api/fpl/solve-build     -> best squad from scratch (wildcard/freehit)
"""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.fpl.solver import solve_transfer, solve_build, NoFeasibleSolution

logger = logging.getLogger("fpl")
router = APIRouter(prefix="/api/fpl", tags=["FPL"])


class TransferRequest(BaseModel):
    players: List[Dict[str, Any]]
    season_id: int = 0
    from_gameweek: int = 1
    horizon: Optional[int] = None
    bank: float = 0.0
    free_transfers: int = 1


class BuildRequest(BaseModel):
    players: List[Dict[str, Any]]
    season_id: int = 0
    from_gameweek: int = 1
    horizon: Optional[int] = None
    budget: float = 100.0
    scope: str = "wildcard"


@router.post("/solve-transfer")
async def solve_transfer_endpoint(request: TransferRequest):
    data = request.model_dump()
    try:
        plan = solve_transfer(data)
        return {"status": "ok", "plan": plan}
    except NoFeasibleSolution as e:
        # An ANSWER, not a failure: the constraints admit no legal squad.
        # Logged at WARNING with the inputs, never as an exception — a
        # traceback here says "we broke" and sends whoever reads the health
        # digest looking for a bug that is not there.
        #
        # Returned as its own status so the caller can say "no legal plan"
        # instead of "try again in a few minutes", which for an infeasible
        # squad is advice that can never work. Three failures nine minutes
        # apart on 2026-08-21 look exactly like someone taking that advice.
        logger.warning("solve-transfer infeasible: %s", e)
        return {"status": "infeasible", "message": str(e)}
    except Exception as e:
        logger.exception("solve-transfer failed")
        return {"status": "error", "message": str(e)}


@router.post("/solve-build")
async def solve_build_endpoint(request: BuildRequest):
    try:
        horizon = request.horizon or len(request.players[0]["xpts"])
        draft = solve_build(
            request.players, horizon, request.from_gameweek,
            request.season_id, budget=request.budget, scope=request.scope,
        )
        return {"status": "ok", "draft": draft}
    except NoFeasibleSolution as e:
        # An ANSWER, not a failure: the constraints admit no legal squad.
        # Logged at WARNING with the inputs, never as an exception — a
        # traceback here says "we broke" and sends whoever reads the health
        # digest looking for a bug that is not there.
        #
        # Returned as its own status so the caller can say "no legal plan"
        # instead of "try again in a few minutes", which for an infeasible
        # squad is advice that can never work. Three failures nine minutes
        # apart on 2026-08-21 look exactly like someone taking that advice.
        logger.warning("solve-build infeasible: %s", e)
        return {"status": "infeasible", "message": str(e)}
    except Exception as e:
        logger.exception("solve-build failed")
        return {"status": "error", "message": str(e)}
