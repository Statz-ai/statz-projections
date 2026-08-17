import logging
import math
from datetime import datetime
import app.database as _db
from app.repository.db_utils import execute_chunked, resolve_team_id

logger = logging.getLogger("player_stat_repo")

STATUS_TYPES = {
    "Shots Total": 42,
    "Offsides": 51,
    "Goals": 52,
    "Fouls": 56,
    "Saves": 57,
    "Tackles": 78,
    "Assists": 79,
    "Passes": 80,
    "Yellow Cards": 84,
    "Shots On Target": 86,
    "Fouls Drawn": 96,
    "Total Crosses": 98,
    "Interceptions": 100,
    "Accurate Passes": 116,
    "Key Passes": 117,
    "Fouls Committed": 56,
}


async def insert_players_stats_async(data_list, teams=None, competition_id=None, comp_teams=None):
    if len(data_list) == 0:
        return

    api_pl_projections = data_list.copy()
    api_pl_projections = api_pl_projections.rename(columns={
        "Player": "player_name",
        "Position": "position",
        "Team": "team",
        "Opponent": "opponent",
        "Venue": "venue",
        "Market": "market_name",
        "Prop": "prop",
        "Projection %": "projection_percent"
    })

    api_pl_projections['kickoff_datetime'] = api_pl_projections['kickoff_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    def _parse_pct(x):
        # Strings like "23.45%" → 23.45; NaN/None/empty → None; numeric pass-through.
        # Previous version didn't reject NaN floats: str(nan) == 'nan',
        # float('nan') is still NaN, and the DB column is NOT NULL → insert
        # fail. Seen on players with no history whose projections became NaN
        # after the 2026-04-24 CSV restore surfaced all the downstream code
        # paths hidden behind earlier parquet crashes.
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        s = str(x).replace('%', '').strip()
        if s == '' or s.lower() == 'nan':
            return None
        try:
            v = float(s)
        except ValueError:
            return None
        return None if math.isnan(v) else v

    api_pl_projections['projection_percent'] = api_pl_projections['projection_percent'].apply(_parse_pct)

    # Rows with NULL projection_percent can't land (DB NOT NULL) and aren't
    # useful anyway — drop them with a count log so we can watch for upstream
    # regressions that produce widespread NaNs.
    _before = len(api_pl_projections)
    api_pl_projections = api_pl_projections[api_pl_projections['projection_percent'].notna()]
    _dropped = _before - len(api_pl_projections)
    if _dropped > 0:
        logger.warning(f"[player_prop_projections] dropped {_dropped}/{_before} rows with NULL projection_percent")

    # Note: player_name / team / opponent strings are no longer written to
    # the DB — team_id / opponent_id / player_id FKs replace them. See
    # nullable migration 2026_04_17_120000.
    # Build the insert tuples WITHOUT iterrows (Series-per-row is the slowest
    # pandas iteration) and WITHOUT re-resolving the same ~40 team names ~49k×.
    # Column-array iteration + a memoised resolver — byte-identical output:
    # same per-row logic, same STATUS_TYPES lookup, same resolve_team_id result
    # per name (a deterministic lookup), same None/NaN handling (execute_chunked
    # still cleans NaN→None downstream). row.get(col) → None for an absent
    # column is preserved by pre-filling missing columns with None.
    _cols = ['fixture_id', 'player_id', 'position', 'team', 'opponent',
             'venue', 'market_name', 'prop', 'projection_percent', 'kickoff_datetime']
    for _c in _cols:
        if _c not in api_pl_projections.columns:
            api_pl_projections[_c] = None

    _tid_cache = {}
    def _resolve(name):
        if teams is None:
            return None
        # Key NaN separately so a genuine NaN name is still resolved once,
        # identically to the original per-row resolve_team_id(nan, ...).
        key = '\x00NAN' if isinstance(name, float) and math.isnan(name) else name
        if key not in _tid_cache:
            _tid_cache[key] = resolve_team_id(name, teams, competition_id, comp_teams)
        return _tid_cache[key]

    values = []
    # (fixture_id, player_id) this run actually produced, for the membership
    # cleanup after the upsert. Collected here so it costs nothing extra.
    written_pairs = set()
    for (fid, pid, pos, team, opp, ven, mkt, prop, pct, ko) in zip(
        api_pl_projections['fixture_id'], api_pl_projections['player_id'],
        api_pl_projections['position'], api_pl_projections['team'],
        api_pl_projections['opponent'], api_pl_projections['venue'],
        api_pl_projections['market_name'], api_pl_projections['prop'],
        api_pl_projections['projection_percent'], api_pl_projections['kickoff_datetime'],
    ):
        values.append((
            fid, pid, pos,
            _resolve(team), _resolve(opp),
            ven, mkt, STATUS_TYPES.get(mkt, 0), prop, pct, ko,
        ))
        if fid is not None and pid is not None:
            written_pairs.add((int(fid), int(pid)))

    sql = """
    INSERT INTO player_prop_projections (
        fixture_id, player_id, position,
        team_id, opponent_id,
        venue, market_name, stats_type_id, prop, projection_percent,
        kickoff_datetime, created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    ON DUPLICATE KEY UPDATE
        position = VALUES(position),
        team_id = VALUES(team_id),
        opponent_id = VALUES(opponent_id),
        venue = VALUES(venue),
        stats_type_id = VALUES(stats_type_id),
        projection_percent = VALUES(projection_percent),
        kickoff_datetime = VALUES(kickoff_datetime),
        updated_at = NOW()
    """
    result = await execute_chunked(sql, values, label="[player_prop_projections]")
    await cleanup_player_prop_projections_async(written_pairs)
    return result


async def cleanup_player_prop_projections_async(written_pairs):
    """Delete prop rows this run did not produce, for the fixtures it covered.

    The insert is INSERT ... ON DUPLICATE KEY UPDATE keyed on
    (fixture_id, player_id, stats_type_id, prop), so it can only add or update.
    Anything it does not write survives forever — and a transfer changes the
    key, because the old club's rows hang off the OLD CLUB'S FIXTURES. No
    upsert can ever reach them. Measured 2026-08-17: 18,942 future rows across
    58 players sitting at clubs they had left, being served on prop picks, SGP,
    form edges and the /players cards. The equivalent hole in fpl_projections
    was closed in Aug 2026; this table never got the same treatment.

    Two deliberate scoping choices, both there to make this unable to delete
    anything legitimate:

    1. ONLY fixtures this run wrote at least one row for. A run restricted to a
       subset of fixtures (euro comps use restrict_team_ids / comp_fixture_ids)
       therefore leaves every other fixture completely alone, and a run that
       fails before writing a fixture cannot prune it.

    2. Membership on (fixture_id, player_id), NOT on the full four-part unique
       key. Runs legitimately differ in which markets they produce, so keying
       on the full row would delete a market this run happened not to generate.
       Player-level membership targets exactly the case that matters: someone
       who should no longer appear in this fixture at all.

    A player the model has stopped projecting loses his rows here, which is
    intended and matches fpl_repo.cleanup_fpl_projections_async. Rows are
    regenerated by the next good run, so the blast radius of a bad run is
    "some players missing props until the next one", not permanent loss.
    """
    pairs = {(int(f), int(p)) for f, p in written_pairs if f is not None and p is not None}
    if not pairs:
        return 0

    fixture_ids = {f for f, _ in pairs}
    conn = await _db.get_connection()
    deleted = 0
    try:
        async with conn.cursor() as cur:
            # Chunk the fixture list so the IN clause stays a sane size on a
            # full-league run.
            fids = list(fixture_ids)
            for i in range(0, len(fids), 200):
                chunk = fids[i:i + 200]
                ph = ','.join(['%s'] * len(chunk))
                await cur.execute(
                    f"SELECT DISTINCT fixture_id, player_id FROM player_prop_projections "
                    f"WHERE fixture_id IN ({ph})",
                    tuple(chunk),
                )
                existing = {(int(f), int(p)) for f, p in await cur.fetchall()}
                stale = list(existing - pairs)
                for j in range(0, len(stale), 1000):
                    sub = stale[j:j + 1000]
                    cond = ' OR '.join(['(fixture_id = %s AND player_id = %s)'] * len(sub))
                    params = tuple(v for pair in sub for v in pair)
                    await cur.execute(
                        f"DELETE FROM player_prop_projections WHERE {cond}", params
                    )
                    deleted += cur.rowcount or 0
            await conn.commit()
        if deleted:
            logger.info(f"[player_prop_projections] removed {deleted} stale rows "
                        f"across {len(fixture_ids)} fixtures (players no longer in the fixture)")
    except Exception as err:
        logger.warning(f"[player_prop_projections] cleanup failed ({err}) — stale rows left in place")
    finally:
        if conn is not None and _db.pool:
            _db.pool.release(conn)
    return deleted
