import logging
import asyncio
from datetime import datetime
import aiomysql
import pandas as pd
from app.database import get_connection
from app.repository.db_utils import execute_chunked

logger = logging.getLogger("transfermarkt_mv_repo")


async def insert_market_value_snapshots_async(df: pd.DataFrame, league_dashed: str) -> int:
    """Bank today's Transfermarkt scrape into transfermarkt_market_value_snapshots.

    Unique key is (league_dashed, team_name, snapshot_date), so each run
    adds a row and the series accumulates; a retry or a second run on the
    same day updates that day's row instead of minting a near-duplicate.

    It used to key on (league_dashed, team_name), which meant every run
    overwrote the last — the weekly scrape had been running since mid-July
    and left exactly one snapshot behind. The history matters: calibrating
    what squad value is worth needs a season's OPENING values against that
    season's results, and with a single snapshot the only available test
    was today's squads against last season's table.
    """
    if df is None or len(df) == 0:
        return 0

    values = []
    for _, row in df.iterrows():
        team = row.get('Team')
        mv = row.get('Market Value')
        if team is None or mv is None:
            continue
        values.append((league_dashed, str(team).strip(), str(mv).strip()))

    if not values:
        return 0

    sql = """
    INSERT INTO transfermarkt_market_value_snapshots (
        league_dashed, team_name, market_value, scraped_at, snapshot_date,
        created_at, updated_at
    ) VALUES (%s, %s, %s, NOW(), CURDATE(), NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE
        market_value = new.market_value,
        scraped_at = NOW(),
        updated_at = NOW()
    """
    return await execute_chunked(sql, values, label=f"[transfermarkt_mv_snapshots:{league_dashed}]")


async def read_latest_market_values_async(league_dashed: str) -> pd.DataFrame:
    """Read the most recent snapshot for league_dashed.

    Returned df has the same shape as a successful get_market_value scrape:
    columns ['Team', 'Market Value']. Empty df if there's no cached data
    yet (first-time scrape failure on a league we've never seen).
    """
    conn = await asyncio.wait_for(get_connection(), timeout=10)
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT team_name AS Team, market_value AS `Market Value`, scraped_at "
                "FROM transfermarkt_market_value_snapshots "
                "WHERE league_dashed = %s "
                "ORDER BY scraped_at DESC",
                (league_dashed,),
            )
            rows = await cur.fetchall()
    finally:
        import app.database as _db
        if _db.pool:
            _db.pool.release(conn)

    if not rows:
        return pd.DataFrame(columns=['Team', 'Market Value'])

    df = pd.DataFrame(rows)
    most_recent_at = df['scraped_at'].iloc[0]
    # Only the newest batch: rows upsert per (league, team), so teams that
    # left the league (relegation/season turnover) linger with an old
    # scraped_at. Without this filter the reader served current teams
    # PLUS last season's relegated ones, skewing the MV index and keeping
    # phantom "unmapped team" warnings alive.
    df = df[df['scraped_at'] == most_recent_at]
    # Snapshots are the PRIMARY MV source (weekly out-of-band refresh) —
    # nudge when the weekly cadence has slipped, stay quiet otherwise.
    age_days = (datetime.utcnow() - most_recent_at).days if isinstance(most_recent_at, datetime) else None
    if age_days is not None and age_days > 14:
        logger.warning(
            f"[transfermarkt_mv_snapshots:{league_dashed}] Snapshot is {age_days} days old "
            f"({most_recent_at}) — run the weekly MV scrape (scripts/scrape_market_values.py)."
        )
    else:
        logger.info(
            f"[transfermarkt_mv_snapshots:{league_dashed}] "
            f"Serving {len(df)} MVs from snapshot {most_recent_at}"
        )
    return df[['Team', 'Market Value']]
