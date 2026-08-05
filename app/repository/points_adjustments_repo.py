"""Manual points adjustments for the season simulation.

Sportmonks carries points deductions, but folded into `points` with no field
naming them, and only once they take effect. Southampton's -4 for 26/27 was
announced while the API still had them on 0 (checked 2026-08-05), so every
outright number treated them as starting level.

This lets an operator enter the deduction in the admin panel and have the
simulation start that team on -4 until the real standings catch up.

The hard part is knowing when to STOP applying it. Once Sportmonks folds the
deduction into `points`, adding ours again double-counts it — Southampton
would start on -8. So before applying, check whether the standings already
show it: in a normal league, points should equal 3*won + drawn, and any gap
is a deduction already in the data.

That test does NOT hold in leagues that halve points at the split (Austria,
Belgium, and the Scottish Premiership among ours), where every team shows a
large arithmetic gap for structural reasons. Those are logged and skipped
rather than guessed at — a deduction there is rare enough to handle by hand.
"""
import logging

import aiomysql

from app.database import get_connection

logger = logging.getLogger("projection")

# Competitions whose points are halved or carried at a split, so
# "points != 3*won + drawn" says nothing about deductions.
SPLIT_POINT_COMPETITIONS = {
    181,  # Austrian Bundesliga
    208,  # Belgian Pro League
    501,  # Scottish Premiership
}


async def load_points_adjustments(competition_id: int, season_id: int) -> dict:
    """Return {team_id: (points, note)} for a competition + season."""
    if not competition_id or not season_id:
        return {}

    conn = None
    try:
        conn = await get_connection()
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT team_id, points, note
                FROM projections_points_adjustments
                WHERE competition_id = %s AND season_id = %s AND points <> 0
                """,
                (int(competition_id), int(season_id)),
            )
            rows = await cur.fetchall()
        return {int(tid): (int(pts), note) for tid, pts, note in rows}
    except Exception as err:
        logger.warning(f"points adjustments unavailable ({err}) — continuing without")
        return {}
    finally:
        import app.database as _db
        if conn is not None and _db.pool:
            _db.pool.release(conn)


def _already_in_standings(row, adjustment: int, split_league: bool) -> bool:
    """Do the stored standings already reflect this adjustment?

    `points - (3*won + drawn)` is the deduction the league has actually
    applied. If that already matches what the operator entered, applying
    ours on top would double it.

    Returns False for split-point leagues, where the arithmetic is
    meaningless — better to apply the operator's number than to silently
    drop it on a bad test.
    """
    if split_league:
        return False
    try:
        implied = int(row['points']) - (3 * int(row['won']) + int(row['drawn']))
    except (KeyError, TypeError, ValueError):
        return False
    if implied == 0:
        return False
    # Same direction and at least as large — the league has applied it.
    return (implied <= adjustment) if adjustment < 0 else (implied >= adjustment)


async def apply_points_adjustments(league_table: dict, standings_rows, competition_id: int,
                                   season_id: int, teams, league_label: str = "") -> dict:
    """Seed the season simulation with manual points adjustments.

    league_table    {team_name: {'Points': int, 'Goals For': ..., ...}} — the
                    dict handed to sim_multiple_seasons. Mutated and returned.
    standings_rows  the standings DataFrame for this season, used to tell
                    whether the deduction is already in the data.
    teams           teams DataFrame, for team_id -> name.

    Never raises: a failure here must not cost a projection run.
    """
    try:
        adjustments = await load_points_adjustments(competition_id, season_id)
        if not adjustments:
            return league_table

        id_to_name = teams.set_index('id')['name'].to_dict()
        split = int(competition_id) in SPLIT_POINT_COMPETITIONS
        by_team = {}
        if standings_rows is not None and not standings_rows.empty:
            by_team = {int(r['team_id']): r for _, r in standings_rows.iterrows()}

        applied, absorbed = [], []
        for team_id, (points, note) in adjustments.items():
            name = id_to_name.get(int(team_id))
            if not name or name not in league_table:
                logger.warning(
                    f"[{league_label}] points adjustment for team_id={team_id} "
                    f"has no row in the league table — skipping")
                continue

            row = by_team.get(int(team_id))
            if row is not None and _already_in_standings(row, points, split):
                absorbed.append(f"{name}({points:+d})")
                continue

            league_table[name]['Points'] = int(league_table[name].get('Points', 0)) + int(points)
            applied.append(f"{name}({points:+d}{' — ' + note if note else ''})")

        if applied:
            logger.info(f"[{league_label}] points adjustments applied: {', '.join(applied)}")
        if absorbed:
            logger.info(
                f"[{league_label}] points adjustments already in the standings, "
                f"not re-applied: {', '.join(absorbed)}")
    except Exception as err:
        logger.warning(f"[{league_label}] points adjustments failed ({err}) — continuing without")
    return league_table
