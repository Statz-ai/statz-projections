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
            try:
                await cur.execute(
                    """
                    SELECT team_id, attack_offset, defense_offset
                    FROM projections_team_dials
                    WHERE competition_id = %s
                      AND (attack_offset != 0 OR defense_offset != 0)
                    """,
                    (int(competition_id),),
                )
            except Exception:
                # Offset columns not migrated yet. Laravel and this service
                # deploy separately, so during that window fall back to the
                # old percentage columns and let the caller keep the previous
                # behaviour rather than silently dropping every dial.
                logger.warning(
                    "projections_team_dials has no offset columns yet — "
                    "falling back to the legacy percentage path")
                return {}
            rows = await cur.fetchall()
        return {int(tid): (float(atk), float(dfn)) for tid, atk, dfn in rows}
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)


async def apply_team_dials_to_ratings(ratings, competition_id, teams, league_label):
    """Apply each team's dialled OFFSET, in index points, to the ratings frame.

    The dial used to be a percentage of the model rating, which meant the
    operator's adjustment scaled with the model: push a team to 150 Attack,
    they play badly, the model drops them, and a +14 opinion silently became
    +12. An offset instead says "the model is under-rating them by 14" — the
    team still moves on results, the disagreement stays the size it was set.

    THE SOLVE. Offsets are expressed on the post-rescale index (mean = 100),
    but this runs BEFORE the rescale — deliberately, so the dial also lands in
    the xG-per-game snapshot the caller takes immediately after. So rather
    than writing the target directly, write the raw value that COMES OUT as
    the target once the caller rescales.

    With raw values r, n teams, and dialled targets T_j:

        index_j = 100 * X_j / mean(X)  ->  X_j = T_j * M / 100
        M = (S + sum_j X_j) / n,  S = sum of un-dialled raw
        =>  M = S / (n - sum_j T_j / 100)

    Closed form, no iteration, and it holds for any number of dialled teams
    at once. The league mean still comes out at 100, so every other team's
    rating reads normally — which the old percentage path did not guarantee.

    In-place mutation. Safe to call when no dials exist — early-returns.

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

    # Resolve dials to row masks once, so a name that does not match is
    # reported a single time rather than per column.
    resolved = []
    for team_id, (atk_off, def_off) in dials.items():
        team_name = id_to_name.get(int(team_id))
        if not team_name:
            logger.warning(f"[{league_label}] team_dial team_id={team_id} not in teams DataFrame — skipping")
            continue
        if not (ratings['Team'] == team_name).any():
            logger.warning(f"[{league_label}] team_dial '{team_name}' not in ratings — skipping")
            continue
        resolved.append((team_name, float(atk_off), float(def_off)))

    if not resolved:
        return

    n = len(ratings)
    touched = []

    for col, offset_index in (('Attack', 1), ('Defense', 2)):
        raw = ratings[col].astype(float)
        mean_raw = raw.mean()
        if not mean_raw or mean_raw <= 0:
            logger.warning(f"[{league_label}] team dials skipped for {col} — mean is not usable")
            continue

        # Target index for each dialled team = its model index + the offset.
        targets = {}
        for entry in resolved:
            team_name, offset = entry[0], entry[offset_index]
            if not offset:
                continue
            model_raw = float(raw[ratings['Team'] == team_name].iloc[0])
            model_index = 100.0 * model_raw / mean_raw
            targets[team_name] = model_index + offset

        if not targets:
            continue

        dialled_names = set(targets)
        undialled_sum = float(raw[~ratings['Team'].isin(dialled_names)].sum())
        denominator = n - sum(targets.values()) / 100.0

        # Degenerate only if the dialled targets are so large they consume the
        # whole league mean — e.g. every team pinned far above 100. Leave the
        # ratings untouched rather than produce a negative or exploded scale.
        if denominator <= 1e-6:
            logger.warning(
                f"[{league_label}] team dials skipped for {col} — dialled targets "
                f"({sum(targets.values()):.0f} across {len(targets)} teams) leave no "
                f"room for the league mean")
            continue

        solved_mean = undialled_sum / denominator
        for team_name, target in targets.items():
            ratings.loc[ratings['Team'] == team_name, col] = target * solved_mean / 100.0
            touched.append(f"{team_name} {col[:3]}->{target:.1f}")

    if touched:
        logger.info(f"[{league_label}] team dials applied: {', '.join(touched)}")
