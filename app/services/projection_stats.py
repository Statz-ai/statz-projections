"""
Single source of truth for which stat_types the projection pipeline reads
from `fixture_team_stats` and `fixture_player_stats`.

Why this module exists
----------------------
Before 2026-04-30, the LeagueDataLoader pulled every row from those tables
(all ~1,116 stat_types) and the projection code filtered at use time.
That wasted ~70% of the loaded volume. Worse, `team_stats` was scoped to
in-league fixtures only, which broke share calculations for transferred
players whose history is at out-of-scope clubs (Souza-style NaN-guard).

This file is the contract: every stat name listed here gets pulled from
the DB. If projection code references a stat NOT listed here, the loader
won't fetch its rows and the share calc will silently return 0 — exactly
the behaviour Flavour B teams (Barcelona/Lecce/etc.) were exhibiting.

How to add a stat
-----------------
1. Add the human-readable name to TEAM_STAT_NAMES and/or PLAYER_STAT_NAMES.
   Names must match `stats_types.name` exactly (case-sensitive).
2. Add the corresponding code that consumes it (the get_stat_id() lookups
   in statz_functions.py, etc.).
3. Redeploy. Loader picks up the new entry on next projection run.

How to remove a stat
--------------------
1. Remove all code references first (search the codebase for the stat name).
2. Then remove from this list. Order matters — if you remove from the list
   first, code still referencing it falls through to fillna(0) and projects
   zeros silently.

Fail-fast guard
---------------
resolve_stat_ids() raises if any name doesn't match a `stats_types` row.
The loader calls it at startup, so typos and DB drift surface immediately
instead of producing zero-filled projections.
"""

# Stats queried from fixture_team_stats.
# Used by: get_team_stats, get_opp_stats, get_ratings (xG + Goals),
# get_average_goals / get_home_advantage / get_home_goal_avg / get_away_goal_avg
# (xG + Goals), and the per-row team-side merge inside get_player_stats.
#
# Notes on aliases:
#   - "Fouls Drawn" looks up the OPPONENT's "Fouls" stat (no separate column).
#   - "Accurate Passes" alias resolves to "Successful Passes" for team lookups.
# So we only need the canonical names below.
TEAM_STAT_NAMES = (
    "Goals",
    "Expected Goals (xG)",
    "Shots Total",
    "Shots On Target",
    "Corners",
    "Fouls",
    "Yellowcards",
    "Tackles",
    "Passes",
    "Successful Passes",
    "Total Crosses",
    "Interceptions",
    "Offsides",
    # Assists is a computed column added to team_predictions downstream of
    # get_team_round_predictions (Assists = Goals × 0.82).
    # distribute_team_predictions_to_players iterates it as part of stat_list,
    # calling get_player_weighted_average which hits team_stats per row by
    # stat_type_id. Without it in the loaded set, the merge fillna's to 0 →
    # denominator collapses → NaN-guard fires for every PL player.
    "Assists",
    # Key Passes + Big Chances Created are PROJECTED team stats for the PL
    # only (statz_functions.PL_ONLY_STATS) — they feed FPL bonus. Key Passes
    # was previously derived as Shots Total × 0.75 everywhere; that fallback
    # is still used for non-PL leagues. Loaded for ALL leagues regardless,
    # because the models are trained on the full top-5 set. Coverage checked
    # 2026-08-02 across the top 5: Key Passes 100%, Big Chances 98-99%.
    "Key Passes",
    "Big Chances Created",
    # Expected Assists (xA) — FPL-only synthetic stat. No Sportmonks rows
    # exist; LeagueDataLoader._overlay_fpl_stats injects per-fixture team
    # totals (sum of FPL player xA) into team_stats in-memory for PL only.
    # Used by distribute_team_predictions_to_players to blend into the
    # Assists projection (mirrors Goals/xG blend).
    "Expected Assists (xA)",
    # Defensive stats — required at TEAM level for the team-down CBIT
    # projection (project team total per fixture × player share %).
    # 'Tackles' is already above. 'Ball Recovery' added here because
    # team_stats didn't previously load it. 'Clearances Blocks
    # Interceptions (FPL)' is FPL-synthetic (id 999002): SM has 3
    # separate components (Clearances/Blocked Shots/Interceptions) but
    # no combined total — _overlay_fpl_stats injects the FPL combined
    # value as a fresh team_stats row.
    "Ball Recovery",
    "Clearances Blocks Interceptions (FPL)",
)

# Stats queried from fixture_player_stats.
# Used by: get_player_stats (per (player, stat) share calculation),
# get_player_cbit_weighted_average + get_extra_stats (FPL CBIT bonus),
# player_criteria (45-min filter via Minutes Played),
# bonus_points_score (FPL bonus components like Key Passes / Big Chances).
PLAYER_STAT_NAMES = (
    # Filter / criteria
    "Minutes Played",
    # Per-stat share denominator (mirror of TEAM_STAT_NAMES above)
    "Goals",
    "Expected Goals (xG)",
    "Shots Total",
    "Shots On Target",
    "Corners",
    "Fouls",
    "Yellowcards",
    "Tackles",
    "Passes",
    "Successful Passes",
    "Total Crosses",
    "Interceptions",
    "Offsides",
    # Fouls Drawn IS queried as a per-player stat. distribute_team_predictions
    # iterates 'Fouls Drawn' (computed column from get_team_round_predictions)
    # → calls get_player_weighted_average('Fouls Drawn') → get_player_stats
    # filters player rows to name == 'Fouls Drawn'. Without this entry the
    # filter returns 0 rows → weighted_sum=0 → early return → projection
    # forced to 0 (silent — doesn't hit NaN-guard). Team-side is fine: the
    # special-case in get_player_stats looks up opponent's 'Fouls' which is
    # already in TEAM_STAT_NAMES.
    "Fouls Drawn",
    # Player-only outputs
    "Assists",
    "Saves",
    "Accurate Passes",
    # FPL CBIT (Clearances/Blocks/Interceptions/Tackles + Recoveries)
    "Clearances",
    "Blocked Shots",
    "Tackles Won",
    "Ball Recovery",
    # FPL bonus chance creation
    "Key Passes",
    "Big Chances Created",
    # Expected Assists (xA) — FPL-only synthetic stat. Player rows are
    # injected by LeagueDataLoader._overlay_fpl_stats for PL only.
    # Used by get_player_weighted_average inside the Assists/xA blend.
    "Expected Assists (xA)",
    # Clearances Blocks Interceptions (FPL) — synthetic stat (id 999002),
    # FPL-only. Sportmonks tracks the 3 components separately (Clearances,
    # Blocked Shots, Interceptions — all already in this list above) but
    # no combined total. _overlay_fpl_stats injects per-player FPL CBI
    # rows for PL fixtures. Required so the team-down CBIT projection
    # can compute the player's CBI share of the team CBI total.
    "Clearances Blocks Interceptions (FPL)",
    # BPS static-rate stats (fpl_static_rates.py). Player traits rated per-90
    # from history rather than projected team-down, so they need PLAYER rows
    # only — deliberately absent from TEAM_STAT_NAMES.
    "Successful Dribbles",
    "Big Chances Missed",
)


def resolve_stat_ids(names, stats_types_df):
    """Resolve a sequence of stat names to a list of stats_type_ids.

    Args:
        names: Iterable of stat names (must match `stats_types.name` exactly).
        stats_types_df: Loader's pre-loaded `stats_types` DataFrame
            (columns include 'id', 'name').

    Returns:
        List of integer stat_type_ids, in the same order names were supplied
        (deduplicated).

    Raises:
        RuntimeError: If any name fails to match a row. Surfaces typos and
            DB drift at loader-init time rather than producing silent zeros
            in downstream projections.
    """
    requested = list(dict.fromkeys(names))  # de-dup, preserve order
    matched = stats_types_df[stats_types_df["name"].isin(requested)]
    found = set(matched["name"].tolist())
    missing = [n for n in requested if n not in found]
    if missing:
        raise RuntimeError(
            f"projection_stats.resolve_stat_ids: {len(missing)} stat name(s) "
            f"not found in stats_types: {missing}. "
            f"Check spelling against `stats_types.name` or remove from list."
        )
    name_to_id = dict(zip(matched["name"].tolist(), matched["id"].tolist()))
    return [int(name_to_id[n]) for n in requested]
