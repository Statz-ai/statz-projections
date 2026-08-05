"""
Repo for the projections_team_dials table — per-team Attack / Defence
overrides set by an admin operator in the Projections Admin Console.

Each row is a (competition_id, team_id) override. Values are signed
percentage adjustments (-50..+50) applied to the team's Attack and/or
Defence rating during projection. A row with both adjustments at 0 is
deleted by the Laravel controller, so any row returned here is by
definition "active".

Applied after the market-value adjustment but before the rescale-to-
mean=100 step, so dialled teams shift the league mean and other teams'
indexed values drift naturally. Two call sites today:

  - projection_service._prepare_league() for single-league projections.
  - euro_comp_projection_service._project() per inner domestic league
    in the cross-league rating set — propagates a team's dial to its
    appearances in CL / Europa / Conf League ratings too.
"""
import logging

import app.database as _db
from app.database import get_connection

logger = logging.getLogger("team_dials_repo")


async def load_team_dials(competition_id: int) -> dict:
    """Return {team_id: (attack_pct, defense_pct)} for a competition.

    Empty dict if no dials are set. Zero is filtered out at write time, so
    any value here is a real override.

    Cast to FLOAT deliberately. The column is decimal, so the driver hands
    back Decimal — and Decimal / int stays Decimal, which multiplied into a
    pandas float column either raises or silently turns it to object dtype.
    That is the same failure that killed the promoted-ratings path.
    """
    if not competition_id:
        return {}

    conn = None
    try:
        conn = await get_connection()
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT team_id, attack_adjustment, defense_adjustment
                FROM projections_team_dials
                WHERE competition_id = %s
                  AND (attack_adjustment != 0 OR defense_adjustment != 0)
                """,
                (int(competition_id),),
            )
            rows = await cur.fetchall()
        return {int(tid): (float(atk), float(dfn)) for tid, atk, dfn in rows}
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)


async def apply_team_dials_to_ratings(ratings, competition_id, teams, league_label):
    """Multiply Attack / Defence values on the ratings DataFrame by each
    team's dialled percent (1 + pct/100). In-place mutation. Logs the
    teams touched. Safe to call when no dials exist — early-returns.

    Caller provides:
      ratings — DataFrame with 'Team' and 'Attack'/'Defense' columns.
      competition_id — the competition whose dials to load.
      teams — full teams DataFrame, used for team_id → name lookup.
      league_label — log prefix, e.g. "Premier League" or
        "Champions League / Ligue 1".
    """
    dials = await load_team_dials(competition_id)
    if not dials:
        return

    if teams is None or teams.empty:
        logger.warning(f"[{league_label}] team dials skipped — empty teams DataFrame")
        return

    id_to_name = teams.set_index('id')['name'].to_dict()
    touched = []
    for team_id, (atk_pct, def_pct) in dials.items():
        team_name = id_to_name.get(int(team_id))
        if not team_name:
            logger.warning(f"[{league_label}] team_dial team_id={team_id} not in teams DataFrame — skipping")
            continue
        mask = ratings['Team'] == team_name
        if not mask.any():
            logger.warning(f"[{league_label}] team_dial '{team_name}' not in ratings — skipping")
            continue
        if atk_pct:
            ratings.loc[mask, 'Attack'] = ratings.loc[mask, 'Attack'] * (1 + atk_pct / 100)
        if def_pct:
            ratings.loc[mask, 'Defense'] = ratings.loc[mask, 'Defense'] * (1 + def_pct / 100)
        touched.append(f"{team_name}({atk_pct:+.1f}A/{def_pct:+.1f}D)")
    if touched:
        logger.info(f"[{league_label}] team dials applied: {', '.join(touched)}")
