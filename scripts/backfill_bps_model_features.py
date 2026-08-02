"""One-off: backfill Key Passes + Big Chances Created history features.

Context: statz `docs/fpl-bps-rebuild-spec.md`. Both stats were added to
projection_model_dataset (statz migration 2026_08_02_120000) but every existing
row has NULL for them, so there is nothing to train on. History features are
normally computed at prediction time for UPCOMING fixtures; this replays them
for fixtures already played.

POINT-IN-TIME. Each row's features are computed with `as_of` pinned to that
fixture's own kickoff, so the history contains only matches that had actually
been played at the time. Computing them with today's data would put the result
being predicted inside its own feature and produce a model that scores
beautifully and is worthless.

Fidelity: calls `get_team_stat_histories` — the same function the live
projection path uses — rather than reimplementing the weighted average.

Scope: the top 5 leagues, because the models train on that pool even though
only the PL projects these stats.

Run:  python scripts/backfill_bps_model_features.py [--dry-run] [--limit N]
"""
import argparse
import asyncio
import logging
import sys
import time

sys.path.insert(0, "/app")

import pandas as pd

from app.database import get_connection, init_db_pool, close_db_pool
import app.database as _db
from app.repository.db_utils import execute_chunked
from app.services.statz_functions import get_team_stat_histories

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bps_backfill")

# competition_id -> label. Models train on this pool (see get_trainable_stat_list).
TOP5 = {8: "Premier League", 82: "Bundesliga", 301: "Ligue 1", 384: "Serie A", 564: "La Liga"}

# (stat name in stats_types, snake_case column suffix)
STATS = [("Key Passes", "key_passes"), ("Big Chances Created", "big_chances_created")]

# Matches the live call in projection_service.get_team_round_predictions.
GAMES = 50

# Actuals. Two statements per stat because Sportmonks stores no zero rows: a
# match where neither side created a big chance has no row at all, and leaving
# those NULL would drop exactly the low-event fixtures from the training sample.
# `team_stats_imported = 1` is the guard that separates "genuinely zero" from
# "not ingested yet".
ACTUALS_COPY_SQL = """
UPDATE projection_model_dataset d
  JOIN fixture_team_stats s
    ON s.fixture_id = d.fixture_id AND s.team_id = d.team_id AND s.stats_type_id = %s
   SET d.team_{col} = s.value, d.updated_at = NOW()
 WHERE d.competition_id = %s AND d.team_{col} IS NULL
"""

ACTUALS_ZERO_SQL = """
UPDATE projection_model_dataset d
  JOIN fixtures f ON f.id = d.fixture_id
   SET d.team_{col} = 0, d.updated_at = NOW()
 WHERE d.competition_id = %s
   AND d.team_{col} IS NULL
   AND f.state_id = 5
   AND f.team_stats_imported = 1
"""

UPDATE_SQL_TEMPLATE = """
UPDATE projection_model_dataset
   SET team_{col}_history = %s,
       opponent_{col}_history_against = %s,
       updated_at = NOW()
 WHERE fixture_id = %s AND team_id = %s
"""


async def _fetch(sql, params=()):
    conn = None
    try:
        conn = await asyncio.wait_for(get_connection(), timeout=60)
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)


async def load_league(comp_id):
    """fixtures / team_stats / teams / stats_types / comp_teams for one league.

    Only the two stats we are backfilling are pulled into team_stats — the
    history calc for a stat never reads any other stat's rows.
    """
    fixtures = await _fetch(
        """SELECT id, home_team_id, away_team_id, kickoff_datetime, season_id,
                  competition_id, stats_imported, state_id, round_id,
                  home_team_goals, away_team_goals
             FROM fixtures WHERE competition_id = %s""",
        (comp_id,),
    )
    stats_types = await _fetch(
        "SELECT id, name FROM stats_types WHERE name IN (%s, %s)",
        tuple(s for s, _ in STATS),
    )
    if len(stats_types) < len(STATS):
        raise RuntimeError(f"stats_types missing one of {[s for s, _ in STATS]}")

    fx_ids = tuple(int(x) for x in fixtures["id"].tolist())
    st_ids = tuple(int(x) for x in stats_types["id"].tolist())
    team_stats = await _fetch(
        f"""SELECT fixture_id, stats_type_id, team_id, value
              FROM fixture_team_stats
             WHERE stats_type_id IN ({','.join(['%s'] * len(st_ids))})
               AND fixture_id IN ({','.join(['%s'] * len(fx_ids))})""",
        st_ids + fx_ids,
    )
    teams = await _fetch("SELECT id, name FROM teams")
    # competition_season_teams — same table LeagueDataLoader reads into
    # ctx.comp_teams. get_team_id() scopes name lookups through it.
    comp_teams = await _fetch(
        "SELECT * FROM competition_season_teams WHERE competition_id = %s",
        (comp_id,),
    )
    return fixtures, team_stats, teams, stats_types, comp_teams


async def _execute(sql, params):
    conn = None
    try:
        conn = await asyncio.wait_for(get_connection(), timeout=120)
        async with conn.cursor() as cur:
            n = await cur.execute(sql, params)
        await conn.commit()
        return n
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)


async def backfill_actuals(comp_id, label, stat_ids, dry_run=False):
    """Copy the played values in. No point-in-time concern — an actual is just
    what happened."""
    for stat, col in STATS:
        sid = int(stat_ids[stat])
        if dry_run:
            probe = await _fetch(
                f"""SELECT COUNT(*) n FROM projection_model_dataset d
                      JOIN fixture_team_stats s ON s.fixture_id = d.fixture_id
                       AND s.team_id = d.team_id AND s.stats_type_id = %s
                     WHERE d.competition_id = %s AND d.team_{col} IS NULL""",
                (sid, comp_id))
            logger.info(f"[{label}] DRY RUN actuals {col}: would copy {int(probe['n'].iloc[0])}")
            continue
        copied = await _execute(ACTUALS_COPY_SQL.format(col=col), (sid, comp_id))
        zeroed = await _execute(ACTUALS_ZERO_SQL.format(col=col), (comp_id,))
        logger.info(f"[{label}] actuals {col}: copied={copied} zeroed={zeroed}")


async def backfill_league(comp_id, label, dry_run=False, limit=None):
    t0 = time.time()
    fixtures, team_stats, teams, stats_types, comp_teams = await load_league(comp_id)
    logger.info(f"[{label}] fixtures={len(fixtures)} team_stat_rows={len(team_stats)}")

    stat_ids = dict(zip(stats_types["name"], stats_types["id"]))
    await backfill_actuals(comp_id, label, stat_ids, dry_run=dry_run)

    # Rows needing features. Restricting to played fixtures: an unplayed row's
    # features get written by the next projection run anyway.
    targets = await _fetch(
        """SELECT d.fixture_id, d.team_id, d.venue, d.season_id, d.team_name, d.opponent_name,
                  f.kickoff_datetime
             FROM projection_model_dataset d
             JOIN fixtures f ON f.id = d.fixture_id
            WHERE d.competition_id = %s
              AND f.state_id = 5
              AND (d.team_key_passes_history IS NULL
                   OR d.team_big_chances_created_history IS NULL)
            ORDER BY f.kickoff_datetime""",
        (comp_id,),
    )
    if limit:
        targets = targets.head(limit)
    if targets.empty:
        logger.info(f"[{label}] nothing to do")
        return 0

    seasons = sorted({int(s) for s in fixtures["season_id"].dropna().unique()}, reverse=True)
    # season_id shape mirrors the live call: [current, previous, above, below].
    # Only the first two are meaningful here — the promoted/relegated league
    # weightings never apply to these stats (they are not in the six-stat list
    # inside get_team_weighted_average), so ratings/league_weightings are None.
    season_arg = [seasons[0] if seasons else None,
                  seasons[1] if len(seasons) > 1 else None, None, None]

    updates = {col: [] for _, col in STATS}
    skipped = 0
    for n, row in enumerate(targets.itertuples(index=False), 1):
        team, opp = row.team_name, row.opponent_name
        if not team or not opp:
            skipped += 1
            continue
        for stat, col in STATS:
            try:
                th, oh = get_team_stat_histories(
                    team, opp, fixtures.copy(), stat, team_stats.copy(), teams, stats_types,
                    ratings=None, venue=row.venue, comp_id=[comp_id], league_weightings=None,
                    season_id=season_arg, games=GAMES, comp_teams=comp_teams,
                    as_of=row.kickoff_datetime,
                )
            except Exception as exc:  # noqa: BLE001 — one bad row must not kill the run
                logger.warning(f"[{label}] fixture={row.fixture_id} team={team} {stat}: {exc}")
                skipped += 1
                continue
            if pd.isna(th) or pd.isna(oh):
                skipped += 1
                continue
            updates[col].append((float(th), float(oh), int(row.fixture_id), int(row.team_id)))
        if n % 250 == 0:
            logger.info(f"[{label}] {n}/{len(targets)} rows ({time.time()-t0:.0f}s)")

    total = sum(len(v) for v in updates.values())
    logger.info(f"[{label}] computed {total} feature pairs, skipped {skipped}, {time.time()-t0:.0f}s")
    if dry_run:
        for _, col in STATS:
            sample = updates[col][:3]
            logger.info(f"[{label}] DRY RUN {col}: {sample}")
        return total

    for _, col in STATS:
        if updates[col]:
            await execute_chunked(UPDATE_SQL_TEMPLATE.format(col=col), updates[col],
                                  label=f"[bps_backfill {label} {col}]")
    return total


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="max rows per league")
    ap.add_argument("--comp", type=int, default=None, help="single competition_id")
    args = ap.parse_args()

    leagues = {args.comp: TOP5.get(args.comp, str(args.comp))} if args.comp else TOP5
    await init_db_pool()
    try:
        grand = 0
        for cid, label in leagues.items():
            grand += await backfill_league(cid, label, dry_run=args.dry_run, limit=args.limit)
        logger.info(f"TOTAL feature pairs: {grand}{' (dry run — nothing written)' if args.dry_run else ''}")
    finally:
        await close_db_pool()


if __name__ == "__main__":
    asyncio.run(main())
