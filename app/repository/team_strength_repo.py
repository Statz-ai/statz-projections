import logging

import pandas as pd

from app.repository.db_utils import execute_chunked

logger = logging.getLogger("team_strength_repo")


async def insert_team_strength_async(strength_df, competition_id, season_id):
    """Persist one run's Team Strength blend to team_strength_ratings.

    Stores the three component z-scores and the weights alongside the blended
    output, so any team's strength number can be decomposed to its inputs
    without re-running the pipeline — the audit trail that makes the blend
    arguable rather than oracular.

    Keyed (competition_id, team_id, date): re-running a league on the same day
    overwrites rather than duplicating.
    """
    if strength_df is None or len(strength_df) == 0:
        return 0

    df = strength_df.copy()
    if 'team_id' not in df.columns:
        logger.warning("[team_strength_ratings] no team_id column — skipping write")
        return 0
    df = df[df['team_id'].notna()]
    if df.empty:
        return 0

    today = pd.Timestamp('today').date()

    def _f(value):
        if value is None:
            return None
        try:
            if value != value:  # NaN
                return None
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return None

    values = [
        (
            int(competition_id),
            int(row.get('team_id')),
            int(season_id) if season_id is not None else None,
            today,
            _f(row.get('base_z')),
            _f(row.get('market_z')),
            _f(row.get('mv_z')),
            _f(row.get('blended_z')),
            _f(row.get('market_position')),
            _f(row.get('base_overall')),
            _f(row.get('blended_overall')),
            _f(row.get('Attack')),
            _f(row.get('Defense')),
            _f(row.get('w_base')),
            _f(row.get('w_market')),
            _f(row.get('w_mv')),
            int(row.get('matches_played') or 0),
            bool(row.get('is_promoted')),
        )
        for _, row in df.iterrows()
    ]

    sql = """
    INSERT INTO team_strength_ratings (
        competition_id, team_id, season_id, date,
        base_z, market_z, mv_z, blended_z,
        market_position, base_overall, blended_overall,
        attack, defense,
        w_base, w_market, w_mv,
        matches_played, is_promoted,
        created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE
        season_id = new.season_id,
        base_z = new.base_z,
        market_z = new.market_z,
        mv_z = new.mv_z,
        blended_z = new.blended_z,
        market_position = new.market_position,
        base_overall = new.base_overall,
        blended_overall = new.blended_overall,
        attack = new.attack,
        defense = new.defense,
        w_base = new.w_base,
        w_market = new.w_market,
        w_mv = new.w_mv,
        matches_played = new.matches_played,
        is_promoted = new.is_promoted,
        updated_at = NOW()
    """
    return await execute_chunked(sql, values, label="[team_strength_ratings]")
