import logging
import os
import pickle
import re
from typing import Any

_PROJ_LOGGER = logging.getLogger("projection")

# Sportmonks pre-creates knockout-final fixture rows with placeholder team
# names like 'TBC', 'Winner Semi-final 1', 'Winner Match 73' (see e.g. EFL
# playoff finals before semis resolve, or WC/Euros brackets pre-draw — also
# covered in sportmonks_tournament_placeholders.md). These rows have real
# teams.id values but no rating / no league-table presence, so projecting
# them produces junk numbers and noisy 'Team missing/NaN in ratings' warnings.
_PLACEHOLDER_TEAM_RE = re.compile(
    r"^(TBC|To\s*[Bb]e\s*[Cc]onfirmed|"
    r"(Winner|Loser|Runner[\s\-]up)\b.*|"
    r"Group\s+[A-Z]\s+(Winner|Runner[\s\-]up).*)$",
    re.IGNORECASE,
)


def is_placeholder_team_name(name) -> bool:
    """True if a team name looks like a tournament-bracket placeholder."""
    if name is None:
        return False
    return bool(_PLACEHOLDER_TEAM_RE.match(str(name).strip()))


def drop_placeholder_fixtures(next_fix, league: str):
    """Filter out fixtures whose home_team or away_team is a bracket placeholder.

    Logs one info line per dropped fixture so the skip is visible without
    spamming WARNING/ERROR (which would page the freshness digest).
    """
    if next_fix is None or len(next_fix) == 0:
        return next_fix
    if 'home_team' not in next_fix.columns or 'away_team' not in next_fix.columns:
        return next_fix
    mask = (
        next_fix['home_team'].astype(str).map(is_placeholder_team_name)
        | next_fix['away_team'].astype(str).map(is_placeholder_team_name)
    )
    if mask.any():
        for _, row in next_fix[mask].iterrows():
            _PROJ_LOGGER.info(
                f"[{league}] Skipping placeholder fixture (bracket not resolved): "
                f"{row.get('home_team', '?')} v {row.get('away_team', '?')} "
                f"(kickoff {row.get('kickoff_datetime', '?')})"
            )
        next_fix = next_fix[~mask].reset_index(drop=True)
    return next_fix


def get_league_id(league_name, comps):
    if league_name == 'Brazil Serie A':
        return 648
    matches = comps[comps['name'] == league_name]['id'].values
    if len(matches) == 0:
        raise ValueError(f"League '{league_name}' not found in competitions table")
    return matches[0]


def get_season_id(comp_id, seasons, previous=False):
    """Return the (current / previous) season id for a competition.

    Returns None when no matching season exists — caller should check and
    bail out of the projection. Historically this was `.values[0]` which
    threw IndexError, exposing every league with a recently-ended season
    + no replacement created yet (e.g. Bundesliga / Ligue 1 in late May
    2026 — 2024-25 ended, Sportmonks hadn't created 2025-26 rows).
    """
    import pandas as pd
    comp_seasons = seasons[seasons['competition_id'] == comp_id]
    current_season = comp_seasons[comp_seasons['is_current'] == 1]
    if previous:
        comp_seasons = comp_seasons.drop(current_season.index)
        if comp_seasons.empty:
            return None
        comp_seasons['end_date'] = pd.to_datetime(comp_seasons['end_date'])
        previous_season = comp_seasons[comp_seasons['end_date'] < pd.to_datetime('today')]
        if previous_season.empty:
            return None
        previous_season = previous_season.sort_values('end_date', ascending=False).iloc[[0]]
        return previous_season['id'].values[0]
    else:
        if current_season.empty:
            return None
        return current_season['id'].values[0]


# Competition id for the Premier League. The two stats below are projected
# for PL ONLY — they exist purely to feed FPL bonus (BPS), so there is no
# reason to spend run time on them for the other leagues.
PL_COMPETITION_ID = 8

# PL-only projected team stats (George, 2026-08-02).
#   'Key Passes'          — was derived as Shots Total x 0.75 in
#                           projection_service. Measured across the top 5 the
#                           real ratio is 0.72-0.74, so the constant was
#                           consistently over-projecting.
#   'Big Chances Created' — never projected; BPS proxied it as Key Passes x 0.2.
# Both are worth 1 and 3 BPS respectively in the published 26/27 table.
# MODELS ARE TRAINED ON ALL TOP-5 LEAGUES (see get_trainable_stat_list) and
# only USED for PL — training on PL rows alone would be ~760 rows/season,
# the small-sample regime that caused the per-league model drift retired in
# May 2026 (see load_model's docstring).
PL_ONLY_STATS = ('Key Passes', 'Big Chances Created')


def get_stat_list(comp_id=None):
    """Team stats projected for `comp_id`.

    Returns a fresh list every call — callers mutate it (get_team_round_predictions
    does stat_list.remove('Goals')).
    """
    stats = ['Goals', 'Shots Total', 'Shots On Target', 'Corners', 'Fouls', 'Yellowcards', 'Tackles', 'Passes',
             'Successful Passes', 'Total Crosses', 'Interceptions', 'Offsides']
    if comp_id is not None and int(comp_id) == PL_COMPETITION_ID:
        stats.extend(PL_ONLY_STATS)
    return stats


def get_trainable_stat_list():
    """Every stat that needs a model file — the union across competitions.

    The retrainer uses this rather than get_stat_list() so the PL-only stats
    get models built from the full top-5 training set.
    """
    return get_stat_list() + list(PL_ONLY_STATS)


TEAM_NAME_FIXES = {
    "Milan": "AC Milan",
}

# Per-run accumulator for the NaN-guard inside get_player_weighted_average.
# Populated when team_weighted_sum collapses to 0 (cross-club share inflation
# or team_stats data gaps). Reset at start of distribute_team_predictions_to_players,
# summary line emitted at end. Module-level state is safe because projection
# runs are serialised by the cross-worker file lock — only one runs at a time.
# See nan_guard_share_inflation.md memory for full context.
_NAN_GUARD_HITS: list = []

# Per-run dedup set for stat-coverage warnings (stat, scope) pairs.
# Fires when filter-by-stat-name returns 0 rows but parent df is populated —
# strong signal that the loader's TEAM_STAT_NAMES / PLAYER_STAT_NAMES list
# in projection_stats.py is missing the stat. Without that entry, the share
# calc silently returns 0 across every player. Reset alongside NaN-guard.
_STAT_COVERAGE_WARNINGS: set = set()


def _warn_stat_coverage_miss(stat, scope):
    """Emit at most one WARNING per (stat, scope) per run.

    Caught patterns this protects against:
      * 'Assists' / 'Key Passes' missing from TEAM_STAT_NAMES (caught after
        regression — Sep-style would be 587 NaN-guard hits before the warning).
      * 'Fouls Drawn' missing from PLAYER_STAT_NAMES (caught after silent
        all-zero-projection regression — every player projecting 0).
    """
    key = (stat, scope)
    if key in _STAT_COVERAGE_WARNINGS:
        return
    _STAT_COVERAGE_WARNINGS.add(key)
    import logging as _logging
    list_name = "PLAYER_STAT_NAMES" if scope == "player" else "TEAM_STAT_NAMES"
    _logging.getLogger("projection").warning(
        f"[stat-coverage] '{stat}' has 0 rows in loaded {scope}_stats — "
        f"likely missing from {list_name} in app/services/projection_stats.py. "
        f"Without it, '{stat}' projections silently zero out across every player."
    )


def _log_nan_guard_summary(competition_id, comps, teams):
    """Emit a single WARNING line summarising NaN-guard hits for the run.

    Per-hit detail is at DEBUG level inside get_player_weighted_average; this
    aggregates them into one observable summary so the daily digest counts
    1 per run (not 50-100). Top affected teams + stats included so the
    summary is actionable without needing to dig.
    """
    if not _NAN_GUARD_HITS:
        return
    import logging
    from collections import Counter
    _log = logging.getLogger("projection")

    n = len(_NAN_GUARD_HITS)
    unique_pairs = len({(h['player_id'], h['stat']) for h in _NAN_GUARD_HITS})

    league = "?"
    try:
        if competition_id is not None and comps is not None:
            cid = competition_id[0] if isinstance(competition_id, (list, tuple)) else competition_id
            row = comps[comps['id'] == cid]
            if not row.empty:
                league = str(row['name'].iloc[0])
    except Exception:
        pass

    team_counter = Counter(h['team_id'] for h in _NAN_GUARD_HITS)
    stat_counter = Counter(h['stat'] for h in _NAN_GUARD_HITS)

    def _team_name(tid):
        try:
            r = teams[teams['id'] == tid]
            return str(r['name'].iloc[0]) if not r.empty else f"team_id={tid}"
        except Exception:
            return f"team_id={tid}"

    top_teams = ", ".join(f"{_team_name(tid)}×{c}" for tid, c in team_counter.most_common(5))
    top_stats = ", ".join(f"{stat}×{c}" for stat, c in stat_counter.most_common(5))

    _log.warning(
        f"[{league}] NaN-guard summary: {n} hits across {unique_pairs} (player,stat) pairs — "
        f"those projections forced to 0. "
        f"Top teams: {top_teams}. Top stats: {top_stats}. "
        f"DEBUG-level per-hit detail available; see nan_guard_share_inflation.md."
    )
    _NAN_GUARD_HITS.clear()

def get_team_id(team_name, teams, competition_id=None, comp_teams=None):
    """
    Resolve a team name to its id. When competition_id + comp_teams are
    provided, the lookup is restricted to teams registered in that comp via
    the competition_season_teams mapping — fixes the duplicate-name bug
    where "Nacional" can mean either CD Nacional (Portugal id=7035) or
    Club Nacional (Uruguay id=828). Falls back to the global first-match
    if the scoped lookup misses (keeps the same signature-optional semantics
    as resolve_team_id in db_utils).
    """
    import logging
    _log = logging.getLogger("statz_functions")
    team_name = TEAM_NAME_FIXES.get(team_name, team_name)

    if comp_teams is not None and not comp_teams.empty:
        # If competition_id is provided, narrow comp_teams to that comp first.
        # Otherwise, treat comp_teams as already pre-filtered by the caller
        # (e.g. get_ratings pre-filters so it can avoid triggering the
        # fixture-filter side-effect in get_team_fixtures).
        #
        # Euro comp projections pass a LIST of 8 domestic league IDs (PL,
        # La Liga, etc.) for competition_id. Scalar `==` errored with
        # "Lengths must match" because pandas tried element-wise compare
        # between the left Series and right list. Normalising to a list +
        # .isin() handles both shapes — scalar wraps in a 1-element list,
        # list passes through.
        if competition_id is not None:
            # hasattr check avoids needing pandas imported at module level;
            # Series / ndarray / list / tuple all iterable-friendly for .isin()
            if isinstance(competition_id, (list, tuple)) or hasattr(competition_id, '__iter__') and not isinstance(competition_id, (str, int, float)):
                comp_id_list = competition_id
            else:
                comp_id_list = [competition_id]
            scoped_ids = comp_teams.loc[
                comp_teams['competition_id'].isin(comp_id_list), 'team_id'
            ].unique()
        else:
            scoped_ids = comp_teams['team_id'].unique() if 'team_id' in comp_teams.columns else []

        if len(scoped_ids) > 0:
            scoped = teams[(teams['id'].isin(scoped_ids)) & (teams['name'] == team_name)]['id']
            if not scoped.empty:
                return int(scoped.iloc[0])
        _log.warning(
            f"get_team_id({team_name!r}): no match within competition {competition_id} scope — falling back to global lookup"
        )

    matches = teams[teams['name'] == team_name]['id']
    if matches.empty:
        raise IndexError(f"get_team_id: no team named {team_name!r} in teams table")
    if len(matches) > 1:
        _log.warning(
            f"get_team_id({team_name!r}): ambiguous — {len(matches)} global matches ({list(matches)}), picking first (comp_id={competition_id})"
        )
    return int(matches.iloc[0])


def get_stat_id(stat_name, stats_types):
    stats = stats_types
    stat_id = stats[stats['name'] == stat_name]['id'].values[0]
    return stat_id


def fit_model(trainX, trainY):
    from sklearn.linear_model import PoissonRegressor
    model = PoissonRegressor(solver='newton-cholesky')
    model.fit(trainX, trainY)
    return model


def grid_search(trainX, trainY):
    import numpy as np
    from sklearn.linear_model import PoissonRegressor
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'alpha': np.arange(0, 1, 0.1),
        'max_iter': [100, 200, 500],
        'fit_intercept': [True, False]
    }
    pr = PoissonRegressor()
    gs = GridSearchCV(pr, param_grid, cv=5, scoring='neg_mean_squared_error')
    model = gs.fit(trainX, trainY)
    return model


def get_comp_teams(league_id, season_id, comp_teams, teams):
    comp_teams = comp_teams[(comp_teams['competition_id'] == league_id) &
                            (comp_teams['season_id'] == season_id)].reset_index(drop=True)
    team_names = []
    for index, row in comp_teams.iterrows():
        team_names.append(get_team(row['team_id'], teams))
    return team_names


def get_team_fixtures(team_name, fixtures, teams, comp_id=None, season_id=None, comp_teams=None):
    team_id = get_team_id(team_name, teams, comp_id, comp_teams)
    fixtures = fixtures[(fixtures['home_team_id'] == team_id) | (fixtures['away_team_id'] == team_id)].reset_index(
        drop=True)
    # Derive home/away team names from IDs rather than splitting fixtures['name']
    # on ' vs '. The split path broke on World Cup placeholder fixtures
    # whose name reads "Winner Match 73" (no separator) — ValueError:
    # "Columns must be same length as key". ID-mapped lookup also avoids
    # ambiguity for teams whose names legitimately contain " vs ".
    _id_to_name = teams.set_index('id')['name']
    fixtures['home_team'] = fixtures['home_team_id'].map(_id_to_name)
    fixtures['away_team'] = fixtures['away_team_id'].map(_id_to_name)
    fixtures['opponent'] = fixtures.apply(lambda x: x['away_team'] if x['home_team'] == team_name else x['home_team'],
                                          axis=1)
    if comp_id is not None:
        try:
            fixtures = fixtures[fixtures['competition_id'].isin(comp_id)].reset_index(drop=True)
        except:
            fixtures = fixtures[fixtures['competition_id'] == comp_id].reset_index(drop=True)
    if season_id is not None:
        try:
            fixtures = fixtures[fixtures['season_id'].isin(season_id)].reset_index(drop=True)
        except:
            fixtures = fixtures[fixtures['season_id'] == season_id].reset_index(drop=True)
    fixtures = fixtures[
        ['id', 'competition_id', 'round_id', 'season_id', 'kickoff_datetime', 'opponent', 'home_team', 'away_team',
         'home_team_id', 'away_team_id', 'home_team_goals', 'away_team_goals',
         'stats_imported']]  # UPDATED - added stats_imported
    return fixtures.sort_values(by='kickoff_datetime').reset_index(drop=True)


# Point-in-time cutoff for history features (George, 2026-08-02).
# `as_of=None` means "now", which is correct for the live path: it projects
# UPCOMING fixtures, so every past match is legitimately available. Backfilling
# training rows for fixtures ALREADY PLAYED needs the cutoff pinned to that
# fixture's kickoff instead — otherwise the history feature contains the very
# result the model is being trained to predict, and the model looks excellent
# while being worthless. Threading it here (rather than reimplementing the
# weighted-average maths in a backfill script) keeps training features and
# serving features computed by exactly the same code.
def _as_of_cutoff(as_of):
    import pandas as pd
    return pd.to_datetime('today') if as_of is None else pd.to_datetime(as_of)


def get_team_stats(stat, team, fixtures, team_stats, teams, stats_types, venue='Yes', comp_id=None, season_id=None,
                   games=None, comp_teams=None, as_of=None):
    team_stats.drop_duplicates(subset=['fixture_id', 'stats_type_id', 'team_id'], inplace=True)
    team_id = get_team_id(team, teams, comp_id, comp_teams)
    fixtures = get_team_fixtures(team, fixtures, teams, comp_id=comp_id,
                                 season_id=season_id, comp_teams=comp_teams)
    _lookup_stat = 'Fouls' if stat == 'Fouls Drawn' else stat
    _stat_id = get_stat_id(_lookup_stat, stats_types)
    if not team_stats.empty and team_stats[team_stats['stats_type_id'] == _stat_id].empty:
        _warn_stat_coverage_miss(_lookup_stat, "team")
    if stat == 'Fouls Drawn':
        team_stats = team_stats[team_stats['fixture_id'].isin((fixtures).id.unique()) &
                                (team_stats['stats_type_id'] == _stat_id) &
                                (team_stats['team_id'] != team_id)].reset_index(drop=True)
    else:
        team_stats = team_stats[team_stats['fixture_id'].isin((fixtures).id.unique()) &
                                (team_stats['stats_type_id'] == _stat_id) &
                                (team_stats['team_id'] == team_id)].reset_index(drop=True)
    team_stats = team_stats[['fixture_id', 'value', 'team_id']]
    team_stats = team_stats.merge(fixtures, left_on='fixture_id', right_on='id',
                                  how='right')  # UPDATED - changed to right join to include all fixtures
    team_stats = team_stats[team_stats['stats_imported'] == 1].reset_index(
        drop=True)  # NEW - only include fixtures where stats have been imported
    team_stats['value'].fillna(0, inplace=True)  # NEW - fill NaN values with 0
    if venue == 'Yes':
        venue = []
        for index, row in team_stats.iterrows():
            if row['home_team_id'] == team_id:  # UPDATED - use home_team_id and team_id variable
                venue.append('H')
            else:
                venue.append('A')
        team_stats['venue'] = venue
        team_stats = team_stats[['kickoff_datetime', 'season_id', 'opponent', 'value', 'venue']]
    else:
        team_stats = team_stats[['kickoff_datetime', 'season_id', 'opponent', 'value']]
    team_stats.rename(columns={'value': f'Team {stat}'}, inplace=True)
    team_stats = team_stats.sort_values(by='kickoff_datetime').reset_index(drop=True)
    # Cutoff BEFORE the tail-slice: filtering after would return fewer than
    # `games` rows for a point-in-time call.
    if as_of is not None:
        import pandas as _pd
        team_stats = team_stats[
            _pd.to_datetime(team_stats['kickoff_datetime']) < _as_of_cutoff(as_of)
        ].reset_index(drop=True)
    if games is not None:
        team_stats = team_stats.iloc[-games:]
    return team_stats.reset_index(drop=True)


def get_opp_stats(stat, team, fixtures, team_stats, teams, stats_types, venue='Yes', comp_id=None, season_id=None,
                  games=None, comp_teams=None, as_of=None):
    team_stats.drop_duplicates(subset=['fixture_id', 'stats_type_id', 'team_id'], inplace=True)
    team_id = get_team_id(team, teams, comp_id, comp_teams)
    fixtures = get_team_fixtures(team, fixtures, teams, comp_id=comp_id,
                                 season_id=season_id, comp_teams=comp_teams)
    _lookup_stat = 'Fouls' if stat == 'Fouls Drawn' else stat
    _stat_id = get_stat_id(_lookup_stat, stats_types)
    if not team_stats.empty and team_stats[team_stats['stats_type_id'] == _stat_id].empty:
        _warn_stat_coverage_miss(_lookup_stat, "team")
    if stat == 'Fouls Drawn':
        team_stats = team_stats[team_stats['fixture_id'].isin((fixtures).id.unique()) &
                                (team_stats['stats_type_id'] == _stat_id) &
                                (team_stats['team_id'] == team_id)].reset_index(drop=True)
    else:
        team_stats = team_stats[team_stats['fixture_id'].isin((fixtures).id.unique()) &
                                (team_stats['stats_type_id'] == _stat_id) &
                                (team_stats['team_id'] != team_id)].reset_index(drop=True)
    team_stats = team_stats[['fixture_id', 'value', 'team_id']]
    team_stats = team_stats.merge(fixtures, left_on='fixture_id', right_on='id',
                                  how='right')  # UPDATED - changed to right join to include all fixtures
    team_stats = team_stats[team_stats['stats_imported'] == 1].reset_index(
        drop=True)  # NEW - only include fixtures where stats have been imported
    team_stats['value'].fillna(0, inplace=True)  # NEW - fill NaN values with 0
    if venue == 'Yes':
        venue = []
        for index, row in team_stats.iterrows():
            if row['home_team_id'] == team_id:  # UPDATED - use home_team_id and team_id variable
                venue.append('H')  # UPDATED - H instead of A
            else:
                venue.append('A')  # UPDATED - A instead of H
        team_stats['venue'] = venue
        team_stats = team_stats[['kickoff_datetime', 'season_id', 'opponent', 'value', 'venue']]
    else:
        team_stats = team_stats[['kickoff_datetime', 'season_id', 'opponent', 'value']]
    team_stats.rename(columns={'value': f'Team {stat}'}, inplace=True)
    team_stats = team_stats.sort_values(by='kickoff_datetime').reset_index(drop=True)
    # Cutoff BEFORE the tail-slice: filtering after would return fewer than
    # `games` rows for a point-in-time call.
    if as_of is not None:
        import pandas as _pd
        team_stats = team_stats[
            _pd.to_datetime(team_stats['kickoff_datetime']) < _as_of_cutoff(as_of)
        ].reset_index(drop=True)
    if games is not None:
        team_stats = team_stats.iloc[-games:]
    return team_stats.reset_index(drop=True)


# UPDATED - New Parameter: previous_team_ratings
def get_ratings(league_id, previous_team_ratings, current_season_id, all_season_ids, comp_teams, teams_df, fixtures_df,
                team_stats, stats_types, weight, games, weightings,
                league_above_id=None, league_below_id=None):
    import pandas as pd
    # Pre-filter comp_teams to THIS league's rows, so downstream get_team_id
    # can disambiguate duplicate names (e.g. 'Nacional' Portugal vs Uruguay)
    # without us having to pass comp_id — which would trigger the fixture
    # filter in get_team_fixtures and drop the cross-comp previous-season data
    # we deliberately rely on for promoted/relegated teams.
    scoped_comp_teams = comp_teams[comp_teams['competition_id'] == league_id] if comp_teams is not None else None
    # Restrict previous_team_ratings to the projecting league + tier-adjacent
    # leagues only (current, above, below). Cross-tier inclusion lets us look
    # up Championship opponents' ratings when computing a promoted team's
    # PL rating (their previous-season fixtures are still in scope per
    # all_season_ids). Both tiers share the domestic mean=100 storage scale.
    #
    # Euro comps (EL/CL/EConfL) are NOT in this list — they're stored on a
    # different scale (raw xG values) and would explode the divide-by-Defense
    # step. Anderlecht's EL Defense=10.7 inflated Brugge's xG by ~9× on one
    # Belgian Pro League fixture before this filter shipped 2026-04-30.
    if previous_team_ratings is not None and 'competition_id' in previous_team_ratings.columns:
        allowed_comp_ids = [league_id]
        if league_above_id is not None:
            allowed_comp_ids.append(league_above_id)
        if league_below_id is not None:
            allowed_comp_ids.append(league_below_id)
        previous_team_ratings = previous_team_ratings[
            previous_team_ratings['competition_id'].isin(allowed_comp_ids)
        ]
    comp_teams = get_comp_teams(league_id, current_season_id, comp_teams, teams=teams_df)
    team_ratings = []
    for team in comp_teams:
        xG = get_team_stats('Expected Goals (xG)', team, fixtures_df, team_stats, teams_df, stats_types, games=games,
                            season_id=all_season_ids, comp_teams=scoped_comp_teams)
        xGA = get_opp_stats('Expected Goals (xG)', team, fixtures_df, team_stats, teams_df, stats_types, games=games,
                            season_id=all_season_ids, comp_teams=scoped_comp_teams)
        xGA = xGA.rename(columns={'Team Expected Goals (xG)': 'Opponent Expected Goals (xG)'})
        GF = get_team_stats('Goals', team, fixtures_df, team_stats, teams_df, stats_types, games=games,
                            season_id=all_season_ids, comp_teams=scoped_comp_teams)
        GA = get_opp_stats('Goals', team, fixtures_df, team_stats, teams_df, stats_types, games=games,
                           season_id=all_season_ids, comp_teams=scoped_comp_teams)
        GA = GA.rename(columns={'Team Goals': 'Opponent Goals'})
        matches = GF.merge(xG[['kickoff_datetime', 'Team Expected Goals (xG)']], on='kickoff_datetime', how='left')
        matches = matches.merge(GA[['kickoff_datetime', 'Opponent Goals']], on='kickoff_datetime', how='left')
        matches = matches.merge(xGA[['kickoff_datetime', 'Opponent Expected Goals (xG)']], on='kickoff_datetime',
                                how='left')
        matches['Team Expected Goals (xG)'].fillna(matches['Team Goals'], inplace=True)
        matches['Opponent Expected Goals (xG)'].fillna(matches['Opponent Goals'], inplace=True)
        matches['Adjusted Goals'] = matches['Team Goals'] * 0.3 + matches[
            'Team Expected Goals (xG)'] * 0.7  # UPDATED - changed weightings to 0.3 and 0.7
        matches['Adjusted Goals Against'] = matches['Opponent Goals'] * 0.3 + matches[
            'Opponent Expected Goals (xG)'] * 0.7  # UPDATED - changed weightings to 0.3 and 0.7
        matches = matches[['kickoff_datetime', 'season_id', 'opponent', 'Adjusted Goals', 'Adjusted Goals Against']]
        for i in range(len(matches)):  # NEW - adjust for opponent ratings
            kickoff_datetime = matches.iloc[i]['kickoff_datetime']  # NEW - get kickoff datetime
            opponent = matches.iloc[i]['opponent']  # NEW - get opponent name
            opponent_rating = previous_team_ratings[
                previous_team_ratings['Team'] == opponent]  # NEW - get opponent rating
            opponent_rating = opponent_rating[
                opponent_rating['Date'] < pd.to_datetime(kickoff_datetime).date()].sort_values(by='Date',
                                                                                               ascending=False).head(
                1)  # NEW - get latest rating before match date
            if not opponent_rating.empty:  # NEW - check if opponent rating exists
                if opponent_rating['Inverse'].values[0] == 'Yes':  # NEW - check if inverse adjustment is needed
                    matches.at[matches.index[i], 'Adjusted Goals Against'] /= (
                                opponent_rating['Attack'].values[0] / 100)  # NEW - adjust goals against
                    matches.at[matches.index[i], 'Adjusted Goals'] *= (
                                opponent_rating['Defense'].values[0] / 100)  # NEW - adjust goals
                else:  # NEW - normal adjustment
                    matches.at[matches.index[i], 'Adjusted Goals Against'] /= (
                                opponent_rating['Attack'].values[0] / 100)  # NEW - adjust goals against
                    matches.at[matches.index[i], 'Adjusted Goals'] /= (
                                opponent_rating['Defense'].values[0] / 100)  # NEW - adjust goals
        if weightings[2] and weightings[3]:
            matches.loc[matches['season_id'] == all_season_ids[3], 'Adjusted Goals'] *= weightings[2]
            matches.loc[matches['season_id'] == all_season_ids[3], 'Adjusted Goals Against'] /= weightings[3]
        if weightings[0] and weightings[1]:
            matches.loc[matches['season_id'] == all_season_ids[2], 'Adjusted Goals'] *= weightings[0]
            matches.loc[matches['season_id'] == all_season_ids[2], 'Adjusted Goals Against'] /= weightings[1]
        matches['Weeks Since Game'] = (pd.to_datetime(matches['kickoff_datetime'].max()) - pd.to_datetime(
            matches['kickoff_datetime'])).dt.days // 7
        matches['Game Weight'] = weight ** (matches['Weeks Since Game'] - 3)  # UPDATED - changed to -3
        matches.loc[
            matches['Weeks Since Game'] < 4, 'Game Weight'] = 1  # NEW - set weight to 1 for games within last 4 weeks
        matches['Weighted Goals'] = matches['Adjusted Goals'] * matches['Game Weight']
        matches['Weighted Goals Against'] = matches['Adjusted Goals Against'] * matches['Game Weight']
        attack_rating = matches['Weighted Goals'].sum() / matches['Game Weight'].sum()
        defense_rating = matches['Weighted Goals Against'].sum() / matches['Game Weight'].sum()
        team_ratings.append([team, attack_rating, defense_rating])
    team_ratings = pd.DataFrame(team_ratings, columns=['Team', 'Attack', 'Defense'])
    return team_ratings[['Team', 'Attack', 'Defense']]


async def get_market_value_with_cache(league_dashed, div, country_code):
    """Snapshot-first (2026-07-21): projection runs READ the weekly
    transfermarkt_market_value_snapshots refresh instead of scraping
    Transfermarkt on every run — "we don't need to scrape every time we
    project, it can be a weekly thing" (George). MVs drift on
    weeks-to-months timescales, and per-run scraping did nothing since the
    AWS WAF block (2026-07-14) except burn ~10s/league and keep the server
    IP flagged. Snapshots are refreshed out-of-band by
    scripts/scrape_market_values.py (residential IP) or the GitHub Action.

    A live scrape remains ONLY as the first-onboard fallback: a league
    with no snapshot at all (newly added comp) gets one attempt so it
    isn't stuck valueless until the next weekly refresh.
    """
    import logging
    _log = logging.getLogger("projection")
    from app.repository.transfermarkt_mv_repo import (
        insert_market_value_snapshots_async,
        read_latest_market_values_async,
    )
    cached = await read_latest_market_values_async(league_dashed)
    if not cached.empty:
        return cached

    _log.info(
        f"[transfermarkt_mv:{league_dashed}] No snapshot yet (new league?) — "
        f"attempting one live scrape."
    )
    df = get_market_value(league_dashed, div, country_code)  # raises on failure → MV block skips, as before
    try:
        await insert_market_value_snapshots_async(df, league_dashed)
    except Exception as cache_write_err:
        _log.warning(
            f"[transfermarkt_mv:{league_dashed}] Snapshot write failed (non-fatal): {cache_write_err}"
        )
    return df


def get_market_value(league_dashed, div, country_code):
    import time
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    url = f'https://www.transfermarkt.co.uk/{league_dashed.lower()}/startseite/wettbewerb/{country_code}{div}'
    # Full browser-shaped header set: since 2026-07-14 Transfermarkt's bot
    # protection answers bare-UA requests with HTTP 202 + a challenge page
    # (0 teams parsed, bs4 decode warnings). A UA alone stopped being enough;
    # the Accept/Accept-Language pair is what currently gets a real page.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Referer": "https://www.transfermarkt.co.uk/",
        "Upgrade-Insecure-Requests": "1",
    }
    # Retries cover transient gateway errors (2026-07-14: a one-off 504 on
    # the MLS page) AND the 202 challenge, which often clears on a re-request
    # once the session cookie from the first response is presented. The
    # timeout guards against a hung connection stalling the run — this
    # function is sync, so a stuck socket blocks the whole MV block.
    session = requests.Session()
    response = None
    for attempt in (1, 2, 3):
        try:
            response = session.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                break
        except requests.RequestException:
            if attempt == 3:
                raise
        if attempt < 3:
            time.sleep(3 * attempt)
    if response.status_code != 200:
        # Never parse a challenge/error body — that's where the bs4
        # REPLACEMENT CHARACTER noise came from. Raise the same shape of
        # error the empty-scrape path uses so the cache fallback kicks in.
        raise RuntimeError(
            f"get_market_value: Transfermarkt returned HTTP {response.status_code} for {league_dashed} after 3 attempts (url={url})"
        )
    soup = BeautifulSoup(response.content, 'html.parser')
    list = soup.select('td[class="hauptlink no-border-links"]')
    teams = []
    for y in range(len(list)):
        team = list[y].text.strip()
        teams.append(team)
    if not teams:
        # Empty scrape — Transfermarkt rate-limit, page change, or wrong URL.
        # Raise a descriptive error so the MV block's try/except logs WHICH
        # league failed instead of a cryptic "Length mismatch" from the
        # downstream df.columns = ['Team'] assignment on a 0-col DataFrame.
        raise RuntimeError(
            f"get_market_value: Transfermarkt scrape returned 0 teams for {league_dashed} (HTTP {response.status_code}, url={url})"
        )
    df = pd.DataFrame(teams)
    df.columns = ['Team']
    mvalue = []
    value = soup.select('td[class="rechts"]')
    value = value[1::2]
    value = value[1:]
    for x in range(len(value)):
        mvalue.append(value[x].text.strip())
    df['Market Value'] = mvalue
    df['Team'] = df['Team'].str.replace('AFC', '')
    df['Team'] = df['Team'].str.replace('FC', '')
    df['Team'] = df['Team'].str.replace('SC', '')
    df['Team'] = df['Team'].str.replace('CF', '')
    df['Team'] = df['Team'].str.replace('RCD', '')
    df['Team'] = df['Team'].str.replace('SS', '')
    df['Team'] = df['Team'].str.replace('AS', '')
    df['Team'] = df['Team'].str.replace('BC', '')
    df['Team'] = df['Team'].str.replace('US', '')
    df['Team'] = df['Team'].str.replace('AC', '')
    df['Team'] = df['Team'].str.strip()
    df['Market Value'] = df['Market Value'].str.replace('.', '')
    # [bnmk]: 'k' was absent from this pattern for years while the k→000 map
    # entry below waited for it — any club under €1m was silently dropped
    # (Rochdale AFC €975k, League Two 2026/27, was the first real casualty).
    df_temp = df['Market Value'].str.strip('€').str.extract(r'(\d+)([bnmk]+)')
    df_temp2 = df_temp[0] + df_temp[1].map({'bn': '0000000', 'm': '0000', 'k': '000'})
    df.drop('Market Value', axis=1, inplace=True)
    df['Market Value'] = df_temp2
    return df


def get_home_goal_avg(league_id, team_stats, fixtures, stats_types):
    import pandas as pd
    fixtures = fixtures[fixtures['competition_id'] == league_id]
    df = fixtures[['id', 'home_team_id', 'away_team_id', 'kickoff_datetime']].merge(
        team_stats[['fixture_id', 'team_id', 'stats_type_id', 'value']], left_on='id', right_on='fixture_id',
        how='inner')
    df.drop_duplicates(subset=['id', 'home_team_id', 'away_team_id', 'team_id', 'stats_type_id'], inplace=True)
    df['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(df['kickoff_datetime'])).dt.days // 7
    df['Weeks Since Kickoff'] = df['Weeks Since Kickoff'].astype(int)
    df['Weight'] = 0.9 ** (df['Weeks Since Kickoff'] - 5)  # UPDATED - changed to 0.9 and -5
    df.loc[df['Weeks Since Kickoff'] < 6, 'Weight'] = 1  # NEW - set weight to 1 for games within last 6 weeks
    df['Weighted Value'] = df['value'] * df['Weight']
    goals = df[df['stats_type_id'] == get_stat_id('Goals', stats_types)]
    home_goals = goals[goals['team_id'] == goals['home_team_id']]
    home_goals_average = home_goals['Weighted Value'].sum() / home_goals['Weight'].sum()
    xG = df[df['stats_type_id'] == get_stat_id('Expected Goals (xG)', stats_types)]
    home_xG = xG[xG['team_id'] == xG['home_team_id']]
    home_xG_average = home_xG['Weighted Value'].sum() / home_xG['Weight'].sum()
    adjusted_home_goal_average = home_goals_average * 0.3 + home_xG_average * 0.7
    return adjusted_home_goal_average


def get_away_goal_avg(league_id, team_stats, fixtures, stats_types):
    import pandas as pd
    fixtures = fixtures[fixtures['competition_id'] == league_id]
    df = fixtures[['id', 'home_team_id', 'away_team_id', 'kickoff_datetime']].merge(
        team_stats[['fixture_id', 'team_id', 'stats_type_id', 'value']], left_on='id', right_on='fixture_id',
        how='inner')
    df.drop_duplicates(subset=['id', 'home_team_id', 'away_team_id', 'team_id', 'stats_type_id'], inplace=True)
    df['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(df['kickoff_datetime'])).dt.days // 7
    df['Weeks Since Kickoff'] = df['Weeks Since Kickoff'].astype(int)
    df['Weight'] = 0.9 ** (df['Weeks Since Kickoff'] - 5)  # UPDATED - changed to 0.9 and -5
    df.loc[df['Weeks Since Kickoff'] < 6, 'Weight'] = 1  # NEW - set weight to 1 for games within last 6 weeks
    df['Weighted Value'] = df['value'] * df['Weight']
    goals = df[df['stats_type_id'] == get_stat_id('Goals', stats_types)]
    away_goals = goals[goals['team_id'] == goals['away_team_id']]
    away_goals_average = away_goals['Weighted Value'].sum() / away_goals['Weight'].sum()
    xG = df[df['stats_type_id'] == get_stat_id('Expected Goals (xG)', stats_types)]
    away_xG = xG[xG['team_id'] == xG['away_team_id']]
    away_xG_average = away_xG['Weighted Value'].sum() / away_xG['Weight'].sum()
    adjusted_away_goal_average = away_goals_average * 0.3 + away_xG_average * 0.7
    return adjusted_away_goal_average


def make_goal_prediction(attack_rating, defense_rating, average_goals):
    pred = average_goals * (attack_rating / 100) * (defense_rating / 100)
    return pred


def make_round_goal_prediction(fixtures, team_ratings, average_home_goals, average_away_goals):
    import pandas as pd
    import math
    import logging
    _log = logging.getLogger("projection")

    def _safe_rating(df, col):
        if len(df) == 0:
            return None
        v = df[col].values[0]
        return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v

    # For neutral-venue fixtures (e.g. CL/Europa/ECL finals at a third
    # ground), the home-advantage goals bias is wrong — neither team is
    # actually playing at home. Use the midpoint of league avg H + avg A
    # goals as a symmetric baseline for both teams. Per-fixture flag
    # arrives via the fixtures DataFrame's `neutral_venue` column when
    # populated (see migration 2026_05_27_165528 + ImportFixtureStoreJob).
    neutral_baseline = (average_home_goals + average_away_goals) / 2

    predictions = []
    for i in range(len(fixtures)):
        id = fixtures.iloc[i]['id']
        kickoff_datetime = fixtures.iloc[i]['kickoff_datetime']
        home_team = fixtures.iloc[i]['home_team']
        away_team = fixtures.iloc[i]['away_team']
        is_neutral = bool(fixtures.iloc[i].get('neutral_venue', False))
        _home = team_ratings[team_ratings['Team'] == home_team]
        _away = team_ratings[team_ratings['Team'] == away_team]
        home_attack = _safe_rating(_home, 'Attack')
        home_defense = _safe_rating(_home, 'Defense')
        away_attack = _safe_rating(_away, 'Attack')
        away_defense = _safe_rating(_away, 'Defense')
        if home_attack is None:
            _log.warning(f"Team missing/NaN in ratings: '{home_team}' — using mean (100)")
        if away_attack is None:
            _log.warning(f"Team missing/NaN in ratings: '{away_team}' — using mean (100)")
        home_attack_rating = home_attack if home_attack is not None else 100
        home_defense_rating = home_defense if home_defense is not None else 100
        away_attack_rating = away_attack if away_attack is not None else 100
        away_defense_rating = away_defense if away_defense is not None else 100
        home_baseline = neutral_baseline if is_neutral else average_home_goals
        away_baseline = neutral_baseline if is_neutral else average_away_goals
        home_goals = (make_goal_prediction(home_attack_rating, away_defense_rating, home_baseline))
        away_goals = (make_goal_prediction(away_attack_rating, home_defense_rating, away_baseline))
        # home_goal_boost, away_goal_boost = get_goal_boost(team_ratings, average_home_goals, average_away_goals)
        # home_goals = (home_goals * home_goal_boost).round(2)
        # away_goals = (away_goals * away_goal_boost).round(2)
        predictions.append([id, kickoff_datetime, home_team, home_goals, away_goals, away_team])
    return pd.DataFrame(predictions,
                        columns=['id', 'kickoff_datetime', 'Home Team', 'Home Goals', 'Away Goals', 'Away Team'])


def get_team(team_id, teams):
    team = teams[teams['id'] == team_id]
    return team['name'].values[0]


def get_round_id(fixtures, previous=False):
    import pandas as pd
    date = pd.to_datetime('today')
    fixtures.loc[:, 'kickoff_datetime'] = pd.to_datetime(fixtures['kickoff_datetime'])
    if previous == True:
        fixtures = fixtures[fixtures['kickoff_datetime'] < date].reset_index(drop=True)
        fixtures = fixtures.sort_values(by='kickoff_datetime', ascending=False)
    else:
        fixtures = fixtures[fixtures['kickoff_datetime'] > date].reset_index(drop=True)
        fixtures = fixtures.sort_values(by='kickoff_datetime', ascending=True)
    round_id = fixtures['round_id'].iloc[0]
    return round_id


def get_stage_id(fixtures, previous=False):
    import pandas as pd
    date = pd.to_datetime('today')
    fixtures.loc[:, 'kickoff_datetime'] = pd.to_datetime(fixtures['kickoff_datetime'])
    if previous == True:
        fixtures = fixtures[fixtures['kickoff_datetime'] < date].reset_index(drop=True)
        fixtures = fixtures.sort_values(by='kickoff_datetime', ascending=False)
    else:
        fixtures = fixtures[fixtures['kickoff_datetime'] > date].reset_index(drop=True)
        fixtures = fixtures.sort_values(by='kickoff_datetime', ascending=True)
    stage_id = fixtures['stage_id'].iloc[0]
    return stage_id


def get_fixtures(fixtures, teams, previous=False, odds=True, cup=False, leg=None, round_id=None):
    if cup == True:
        stage_id = get_stage_id(fixtures, previous)
        fixtures = fixtures[fixtures['stage_id'] == stage_id]
        if leg != None:
            fixtures = fixtures[fixtures['leg'] == f'{leg}/2']
    else:
        if round_id == None:
            round_id = get_round_id(fixtures, previous)
            fixtures = fixtures[fixtures['round_id'] == round_id]
        else:
            fixtures = fixtures[fixtures['round_id'] == round_id]
    fixtures = fixtures[['id', 'kickoff_datetime', 'name', 'home_team_id', 'away_team_id', 'bet365_home_odds_decimal',
                         'bet365_draw_odds_decimal', 'bet365_away_odds_decimal']]
    fixtures['home_team'] = fixtures['home_team_id'].apply(lambda x: get_team(x, teams))
    fixtures['away_team'] = fixtures['away_team_id'].apply(lambda x: get_team(x, teams))
    fixtures = fixtures[
        ['id', 'kickoff_datetime', 'home_team', 'away_team', 'bet365_home_odds_decimal', 'bet365_draw_odds_decimal',
         'bet365_away_odds_decimal']]
    fixtures.sort_values(by=['kickoff_datetime', 'home_team'], inplace=True)
    return fixtures.reset_index(drop=True)


def get_goal_boost(ratings, average_home_goals, average_away_goals):
    import numpy as np
    home_goals_list = []
    away_goals_list = []
    for i in range(len(ratings)):
        home_team = ratings.iloc[i]['Team']
        home_attack = ratings.iloc[i]['Attack']
        home_defense = ratings.iloc[i]['Defense']
        for i in range(len(ratings)):
            away_team = ratings.iloc[i]['Team']
            away_attack = ratings.iloc[i]['Attack']
            away_defense = ratings.iloc[i]['Defense']
            if home_team == away_team:
                continue
            home_goals = make_goal_prediction(home_attack, away_defense, average_home_goals)
            away_goals = make_goal_prediction(away_attack, home_defense, average_away_goals)
            home_goals_list.append(home_goals)
            away_goals_list.append(away_goals)
    home_goal_boost = average_home_goals / np.mean(home_goals_list)
    away_goal_boost = average_away_goals / np.mean(away_goals_list)
    return home_goal_boost, away_goal_boost


def get_draw_boost(ratings, average_home_goals, average_away_goals, draw_perc):
    import numpy as np
    from scipy.stats import poisson
    draw_probs = []
    for i in range(len(ratings)):
        home_team = ratings.iloc[i]['Team']
        home_attack = ratings.iloc[i]['Attack']
        home_defense = ratings.iloc[i]['Defense']
        for i in range(len(ratings)):
            away_team = ratings.iloc[i]['Team']
            away_attack = ratings.iloc[i]['Attack']
            away_defense = ratings.iloc[i]['Defense']
            if home_team == away_team:
                continue
            home_goals = make_goal_prediction(home_attack, away_defense, average_home_goals)
            away_goals = make_goal_prediction(away_attack, home_defense, average_away_goals)
            x = np.arange(0, 9)
            y = np.arange(0, 9)
            X, Y = np.meshgrid(x, y)
            Z = poisson.pmf(X, home_goals) * poisson.pmf(Y, away_goals)
            draw_prob = np.sum(np.diag(Z))
            draw_probs.append(draw_prob)

    projected_draw_prob = np.mean(draw_probs)
    draw_boost = draw_perc / projected_draw_prob
    return np.clip(draw_boost, 1, 1.1)  # NEW


def get_result_probs(home_goals: object, away_goals: object, boost: object) -> tuple[Any, Any, Any]:
    import numpy as np
    from scipy.stats import poisson
    x = np.arange(0, 9)
    y = np.arange(0, 9)
    X, Y = np.meshgrid(x, y)
    Z = poisson.pmf(X, home_goals) * poisson.pmf(Y, away_goals)
    home_win_prob = np.sum(np.triu(Z, k=1))
    draw_prob = np.sum(np.diag(Z))
    away_win_prob = np.sum(np.tril(Z, k=-1))
    draw_prob = draw_prob * boost
    remaining_prob = 1 - draw_prob
    original_home_probs = home_win_prob
    original_away_probs = away_win_prob
    home_win_prob = (original_home_probs / (original_home_probs + original_away_probs)) * remaining_prob
    away_win_prob = (original_away_probs / (original_home_probs + original_away_probs)) * remaining_prob
    return round(home_win_prob * 100, 2), round(draw_prob * 100, 2), round(away_win_prob * 100, 2)


def find_inputs_for_probs(home_start, away_start, target_home, target_draw, target_away, boost=1.1):
    from scipy.optimize import minimize
    import numpy as np
    from scipy.stats import poisson
    def objective(x):
        home_goals, away_goals = x
        x_vals = np.arange(0, 9)
        y_vals = np.arange(0, 9)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = poisson.pmf(X, home_goals) * poisson.pmf(Y, away_goals)
        home_win_prob = np.sum(np.triu(Z, k=1))
        draw_prob = np.sum(np.diag(Z)) * boost
        away_win_prob = np.sum(np.tril(Z, k=-1))
        remaining_prob = 1 - draw_prob
        home_win_prob = (home_win_prob / (home_win_prob + away_win_prob)) * remaining_prob
        away_win_prob = (away_win_prob / (home_win_prob + away_win_prob)) * remaining_prob
        return (home_win_prob - target_home / 100) ** 2 + (draw_prob - target_draw / 100) ** 2 + (
                    away_win_prob - target_away / 100) ** 2

    x0 = [home_start, away_start]
    bounds = [(0.1, 5), (0.1, 5)]
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    return result.x


def sim_season(score_preds, current_league_table, _warned=set()):
    import numpy as np
    import copy
    import logging
    league_table = copy.deepcopy(current_league_table)
    for index, row in score_preds.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        for team in [home_team, away_team]:
            if team not in league_table:
                if team not in _warned:
                    logging.getLogger("projection").warning(f"Team not in league table: '{team}' — adding with 0 points")
                    _warned.add(team)
                league_table[team] = {'Points': 0, 'Goals For': 0, 'Goals Against': 0, 'Goal Difference': 0}

        home_goals = np.random.poisson(row['Home Goals'])
        away_goals = np.random.poisson(row['Away Goals'])

        if home_goals > away_goals:
            league_table[home_team]['Points'] += 3
        elif home_goals < away_goals:
            league_table[away_team]['Points'] += 3
        else:
            league_table[home_team]['Points'] += 1
            league_table[away_team]['Points'] += 1

        league_table[home_team]['Goals For'] += home_goals
        league_table[away_team]['Goals For'] += away_goals
        league_table[home_team]['Goals Against'] += away_goals
        league_table[away_team]['Goals Against'] += home_goals
        league_table[home_team]['Goal Difference'] += (home_goals - away_goals)
        league_table[away_team]['Goal Difference'] += (away_goals - home_goals)

    sorted_teams = sorted(
        league_table.items(),
        key=lambda item: (
            item[1]['Points'],
            item[1]['Goal Difference'],
            item[1]['Goals For']
        ),
        reverse=True
    )
    for idx, (team, stats) in enumerate(sorted_teams, 1):
        stats['Position'] = idx
    league_table_sorted = dict(sorted_teams)
    return league_table_sorted


def sim_multiple_seasons(score_preds, current_league_table, num_sims=100):
    import pandas as pd
    all_tables = []
    for sim in range(1, num_sims + 1):
        simulated_league_table = sim_season(score_preds, current_league_table)  # returns dict: {team: stats_dict}
        for team, stats in simulated_league_table.items():
            stats_copy = stats.copy()
            stats_copy['Team'] = team
            stats_copy['Simulation'] = sim
            all_tables.append(stats_copy)

    all_tables_df = pd.DataFrame(all_tables)
    avg_table = all_tables_df.groupby('Team').agg({
        'Points': 'mean',
        'Goals For': 'mean',
        'Goals Against': 'mean',
        'Goal Difference': 'mean'
    }).reset_index()
    avg_table['Position'] = avg_table['Points'].rank(method='min', ascending=False).astype(int)
    avg_table = avg_table.sort_values(by='Position').reset_index(drop=True)
    # debug prints removed


    avg_table = avg_table[['Position', 'Team', 'Points', 'Goals For', 'Goals Against', 'Goal Difference']]

    return avg_table, all_tables_df


# get_avg_table_with_probs (the old hardcoded per-league `lines` win/top-N/
# relegation calculator) was retired 2026-05-19 — every positional market is
# now a read-time range-sum over league_position_probabilities (written by
# app/repository/league_position_repo.py). See docs/league-projections-redesign.md.


def get_avg_table_with_probs_and_point_limits(avg_table_with_probs, all_tables):
    max_points = all_tables.groupby('Team')['Points'].max().reset_index()
    min_points = all_tables.groupby('Team')['Points'].min().reset_index()
    min_points.rename(columns={'Points': 'Min Points'}, inplace=True)
    max_points.rename(columns={'Points': 'Max Points'}, inplace=True)
    avg_table_with_probs = avg_table_with_probs.merge(max_points, on='Team', how='left')
    avg_table_with_probs = avg_table_with_probs.merge(min_points, on='Team', how='left')
    return avg_table_with_probs


# def load_model(stat, file_path, league):  # UPDATED - New Parameter: league
#     import pickle
#     if stat == 'Goals':  # NEW
#         return None  # No model for Goals, return None
#     filename = file_path + '\\' + stat + '_model.sav'
#     ## NEW lines below - try loading league-specific model first
#     try:
#         model = pickle.load(open(filename, 'rb'))
#     except:
#         file_name = file_path + '\\' + 'All_Leagues_' + stat + '_model.sav'
#         model = pickle.load(open(file_name, 'rb'))
#     return model
#

# def load_model(stat, file_path, league):
#     if stat == 'Goals':
#         return None  # nema modela za Goals
#
#     # pokušaj učitavanja modela specifičnog za ligu
#     league_model_path = os.path.join(file_path, league, f"{league}_{stat}_model.sav")
#     if os.path.exists(league_model_path):
#         with open(league_model_path, 'rb') as f:
#             model = pickle.load(f)
#         return model
#
#     # ako ne postoji, pokušaj učitavanje modela "All Leagues"
#     all_leagues_path = os.path.join(file_path, "All Leagues", f"All_Leagues_{stat}_model.sav")
#     if os.path.exists(all_leagues_path):
#         with open(all_leagues_path, 'rb') as f:
#             model = pickle.load(f)
#         return model
#
#     # Ako ne postoji ni jedan model
#     print(f"Model za {stat} nije pronađen ni u {league} ni u All Leagues.")
#     return None
#

def load_model(stat, file_path):
    """Load the global team-stat model for `stat`.

    Every league now reads from the same `All Leagues` model — per-league
    pickles were retired 2026-05-21. Spotted on Superliga: the per-league
    `Superliga_Fouls_model.sav` was trained on 62 model_dataset rows and
    drifted to predicting 23 fouls/game vs the league avg of 11. Tiny per-
    league training sets are below the retraining min-row threshold, so
    those stale files never get refreshed; pooling all leagues into one
    global model (trained on ~5k top-5 rows) avoids the failure mode.

    The active `All Leagues/All_Leagues_{stat}_model.sav` file is kept
    current by `promote_model()` in `projection_model_repo` — it atomically
    overwrites the unversioned filename whenever a new All-Leagues model
    is promoted.
    """
    if stat == 'Goals':
        return None

    global_path = os.path.join(file_path, "All Leagues", f"All_Leagues_{stat}_model.sav")
    try:
        with open(global_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        _PROJ_LOGGER.warning(f"[load_model] no global model file for {stat}: {global_path}")
        return None

def load_all_models(stat_list, file_path):
    models = {}
    for stat in stat_list:
        models[stat] = load_model(stat, file_path)
    return models

def get_weighted_team_stats(stat, team, fixtures, team_stats, teams, stats_types, weight, venue='Yes', comp_id=None,
                            season_id=None, games=None, comp_teams=None, as_of=None):
    import pandas as pd
    date_from = _as_of_cutoff(as_of)
    team_stats = get_team_stats(stat, team, fixtures, team_stats, teams, stats_types, venue, comp_id, season_id, games,
                                comp_teams=comp_teams, as_of=as_of)
    team_stats = team_stats[pd.to_datetime(team_stats['kickoff_datetime']) < date_from].reset_index(drop=True)
    team_stats['Weeks Since Kickoff'] = (date_from - pd.to_datetime(team_stats['kickoff_datetime'])).dt.days // 7
    team_stats['Weight'] = weight ** (team_stats['Weeks Since Kickoff'])
    team_stats['Weighted ' + stat] = team_stats['Team ' + stat] * team_stats['Weight']
    return team_stats


def get_team_weighted_average(stat, team, fixtures, team_stats, teams, stats_types, weight, venue='Yes', ratings=None,
                              comp_id=None, league_weightings=None, season_id=None, games=None, comp_teams=None,
                              as_of=None):
    team_stats_df = get_weighted_team_stats(stat, team, fixtures, team_stats, teams, stats_types, weight, venue,
                                            comp_id, season_id, games, comp_teams=comp_teams, as_of=as_of)
    stat_list = ['Shots Total', 'Shots On Target', 'Passes', 'Successful Passes', 'Corners', 'Total Crosses']
    if league_weightings is not None:
        if stat in stat_list:
            league_weightings[2] = (1 - league_weightings[2]) * 0.25 + league_weightings[2]
            league_weightings[0] = (1 - league_weightings[0]) * 0.6 + league_weightings[0] if league_weightings[
                                                                                                  0] is not None else \
            league_weightings[0]
            if season_id[2] is not None:
                team_stats_df.loc[team_stats_df['season_id'] == season_id[2], 'Weighted ' + stat] *= league_weightings[
                    0]
            if season_id[3] is not None:
                team_stats_df.loc[team_stats_df['season_id'] == season_id[3], 'Weighted ' + stat] *= league_weightings[
                    2]
    if len(team_stats_df) < 5:
        league_fixtures = fixtures[fixtures['season_id'].isin(season_id)]
        league_stats = team_stats[team_stats['fixture_id'].isin(league_fixtures['id'])]
        league_stats = league_stats[league_stats['stats_type_id'] == get_stat_id(stat, stats_types)]
        average_stats = league_stats['value'].mean()
        if stat in stat_list:
            _r = ratings[ratings['Team'] == team]['Attack']
            attack_ratings = _r.values[0] if len(_r) > 0 else 100
            return average_stats * (attack_ratings / 100)
        else:
            return average_stats
    return team_stats_df['Weighted ' + stat].sum() / team_stats_df['Weight'].sum()


def get_weighted_opp_stats(stat, team, fixtures, team_stats, teams, stats_types, weight, venue='Yes', comp_id=None,
                           season_id=None, games=None, comp_teams=None, as_of=None):
    import pandas as pd
    date_from = _as_of_cutoff(as_of)
    team_stats = get_opp_stats(stat, team, fixtures, team_stats, teams, stats_types, venue, comp_id, season_id, games,
                               comp_teams=comp_teams, as_of=as_of)
    team_stats = team_stats[pd.to_datetime(team_stats['kickoff_datetime']) < date_from].reset_index(drop=True)
    team_stats['Weeks Since Kickoff'] = (date_from - pd.to_datetime(team_stats['kickoff_datetime'])).dt.days // 7
    team_stats['Weight'] = weight ** (team_stats['Weeks Since Kickoff'])
    team_stats['Weighted' + stat] = team_stats['Team ' + stat] * team_stats['Weight']
    return team_stats


def get_opp_weighted_average(stat, team, fixtures, team_stats, teams, stats_types, weight, venue='Yes', ratings=None,
                             comp_id=None, league_weightings=None, season_id=None, games=None, comp_teams=None,
                             as_of=None):
    team_stats_df = get_weighted_opp_stats(stat, team, fixtures, team_stats, teams, stats_types, weight, venue, comp_id,
                                           season_id, games, comp_teams=comp_teams, as_of=as_of)
    stat_list = ['Shots Total', 'Shots On Target', 'Passes', 'Successful Passes', 'Corners', 'Total Crosses']
    if league_weightings is not None:
        if stat in stat_list:
            league_weightings[3] = (1 - league_weightings[2]) * 0.25 + league_weightings[3]
            league_weightings[1] = (1 - league_weightings[0]) * 0.6 + league_weightings[1] if league_weightings[
                                                                                                  1] is not None else \
            league_weightings[1]
            if season_id[2] is not None:
                team_stats_df.loc[team_stats_df['season_id'] == season_id[2], 'Weighted' + stat] /= league_weightings[1]
            if season_id[3] is not None:
                team_stats_df.loc[team_stats_df['season_id'] == season_id[3], 'Weighted' + stat] /= league_weightings[3]
    if len(team_stats_df) < 5:
        league_fixtures = fixtures[fixtures['season_id'].isin(season_id)]
        league_stats = team_stats[team_stats['fixture_id'].isin(league_fixtures['id'])]
        league_stats = league_stats[league_stats['stats_type_id'] == get_stat_id(stat, stats_types)]
        average_stats = league_stats['value'].mean()
        if stat in stat_list:
            _r = ratings[ratings['Team'] == team]['Defense']
            defense_ratings = _r.values[0] if len(_r) > 0 else 100
            return average_stats / (defense_ratings / 100)
        else:
            return average_stats
    return team_stats_df['Weighted' + stat].sum() / team_stats_df['Weight'].sum()


def calculate_team_venue_effect(team, stat, fixtures, team_stats_df, teams, stats_types, venue, comp_id=None,
                                games=None, season_id=None, comp_teams=None, as_of=None):
    team_stats = get_team_stats(stat, team, fixtures, team_stats_df, teams, stats_types, 'Yes', comp_id=comp_id,
                                games=games, season_id=season_id, comp_teams=comp_teams, as_of=as_of)
    if len(team_stats) < 5:
        if venue == 'H':
            return 1.1
        else:
            return 0.9
    home = team_stats[team_stats['venue'] == 'H'][f'Team {stat}'].mean()
    away = team_stats[team_stats['venue'] == 'A'][f'Team {stat}'].mean()
    avg = team_stats[f'Team {stat}'].mean()
    if venue == 'H':
        return home / avg
    else:
        return away / avg


def calculate_opp_venue_effect(team, stat, fixtures, team_stats_df, teams, stats_types, venue, comp_id=None, games=None,
                               season_id=None, comp_teams=None, as_of=None):
    team_stats = get_opp_stats(stat, team, fixtures, team_stats_df, teams, stats_types, 'Yes', comp_id=comp_id,
                               games=games, season_id=season_id, comp_teams=comp_teams, as_of=as_of)
    if len(team_stats) < 5:
        if venue == 'H':
            return 0.9
        else:
            return 1.1
    home = team_stats[team_stats['venue'] == 'H'][f'Team {stat}'].mean()
    away = team_stats[team_stats['venue'] == 'A'][f'Team {stat}'].mean()
    avg = team_stats[f'Team {stat}'].mean()
    if venue == 'H':
        return home / avg
    else:
        return away / avg


def get_team_stat_prediction(team, opponent, fixtures, stat, team_stats, teams, stats_types, model, ratings=None,
                             venue=None, comp_id=None, league_weightings=None, season_id=None, games=None,
                             comp_teams=None, as_of=None):
    import warnings
    if venue == None:
        team_history = get_team_weighted_average(stat, team, fixtures, team_stats, teams, stats_types, 0.98,
                                                 ratings=ratings, comp_id=comp_id, league_weightings=league_weightings,
                                                 season_id=season_id, games=games, comp_teams=comp_teams, as_of=as_of)
        opponent_history = get_opp_weighted_average(stat, opponent, fixtures, team_stats, teams, stats_types, 0.98,
                                                    ratings=ratings, comp_id=comp_id,
                                                    league_weightings=league_weightings, season_id=season_id,
                                                    games=games, comp_teams=comp_teams, as_of=as_of)
    else:
        team_history = get_team_weighted_average(stat, team, fixtures, team_stats, teams, stats_types, 0.98,
                                                 ratings=ratings, comp_id=comp_id, league_weightings=league_weightings,
                                                 season_id=season_id, games=games, comp_teams=comp_teams, as_of=as_of) * calculate_team_venue_effect(team,
                                                                                                                 stat,
                                                                                                                 fixtures,
                                                                                                                 team_stats,
                                                                                                                 teams,
                                                                                                                 stats_types,
                                                                                                                 venue,
                                                                                                                 comp_id=comp_id,
                                                                                                                 games=games * 2,
                                                                                                                 season_id=season_id,
                                                                                                                 comp_teams=comp_teams, as_of=as_of)
        if venue == 'H':
            opponent_venue = 'A'
        else:
            opponent_venue = 'H'
        opponent_history = get_opp_weighted_average(stat, opponent, fixtures, team_stats, teams, stats_types, 0.98,
                                                    ratings=ratings, comp_id=comp_id,
                                                    league_weightings=league_weightings, season_id=season_id,
                                                    games=games, comp_teams=comp_teams, as_of=as_of) * calculate_opp_venue_effect(opponent, stat, fixtures,
                                                                                              team_stats, teams,
                                                                                              stats_types,
                                                                                              opponent_venue,
                                                                                              comp_id=comp_id,
                                                                                              games=games * 2,
                                                                                              season_id=season_id,
                                                                                              comp_teams=comp_teams, as_of=as_of)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        team_stat = model.predict([[team_history, opponent_history]])
    return (team_stat[0]).round(2), team_history, opponent_history  # UPDATED - return histories


# Weight on team_history in get_simple_team_stat_prediction's team-vs-opp
# blend. team_proj = ALPHA * team_history + (1 - ALPHA) * opp_history.
# Higher = trust team's intrinsic style more, less responsive to opponent.
#
# Why 0.7 and not 1.0 or 0.5: validated 2026-05-05 against actual PL hit
# rates this season. Anderson (MID, Forest) calibrated to 70% predicted
# vs 70.6% actual at α=0.7. Senesi (DEF, Bournemouth) — α=0.7 lifts him
# to ~43% vs 70.6% actual; the residual gap there is in the SHARE
# calculation (his per-game DefCon avg is 11.2 but shares × team_hist
# gives ~10.05), not the team-projection formula. See in-progress memory
# `team_down_cbit_in_progress.md` for the share-calc follow-up.
#
# Was multiplicative (`team_hist × opp_hist / league_avg`) before
# 2026-05-05. Multiplicative compounded both teams' below-average factors
# and projected BELOW both inputs — fine for offensive stats where
# attack/defence interaction really is multiplicative, structurally wrong
# for defensive stats where defensive activity is mostly determined by
# game state (which opp_history measures imperfectly anyway — high-
# possession teams' opp_history_CBI is LOWER because their opponents
# don't get as many CBI events, not because they're "tough on CBI").
SIMPLE_PROJECTION_ALPHA = 0.7


def get_simple_team_stat_prediction(team, opponent, fixtures, stat, team_stats, teams, stats_types,
                                    ratings=None, venue=None, comp_id=None, league_weightings=None,
                                    season_id=None, games=None, comp_teams=None):
    """Model-free sibling of get_team_stat_prediction.

    Same recency-weighted machinery (get_team_weighted_average +
    get_opp_weighted_average + venue effects) — combines the two with a
    weighted blend instead of a multiplicative ratio:

        team_stat = α × team_history + (1 - α) × opp_history

    where α = SIMPLE_PROJECTION_ALPHA (currently 0.7). Used for stats
    where no trained PoissonRegressor exists yet (Ball Recovery, FPL CBI).
    Returns (projected_value, team_history, opp_history) — same tuple
    shape as the model path so call sites are interchangeable.
    """
    import pandas as pd
    if venue is None:
        team_history = get_team_weighted_average(stat, team, fixtures, team_stats, teams, stats_types, 0.98,
                                                 ratings=ratings, comp_id=comp_id,
                                                 league_weightings=league_weightings,
                                                 season_id=season_id, games=games, comp_teams=comp_teams)
        opponent_history = get_opp_weighted_average(stat, opponent, fixtures, team_stats, teams, stats_types, 0.98,
                                                    ratings=ratings, comp_id=comp_id,
                                                    league_weightings=league_weightings,
                                                    season_id=season_id, games=games, comp_teams=comp_teams)
    else:
        team_history = get_team_weighted_average(stat, team, fixtures, team_stats, teams, stats_types, 0.98,
                                                 ratings=ratings, comp_id=comp_id,
                                                 league_weightings=league_weightings,
                                                 season_id=season_id, games=games, comp_teams=comp_teams) * \
            calculate_team_venue_effect(team, stat, fixtures, team_stats, teams, stats_types, venue,
                                        comp_id=comp_id, games=games * 2 if games else None,
                                        season_id=season_id, comp_teams=comp_teams)
        opponent_venue = 'A' if venue == 'H' else 'H'
        opponent_history = get_opp_weighted_average(stat, opponent, fixtures, team_stats, teams, stats_types, 0.98,
                                                    ratings=ratings, comp_id=comp_id,
                                                    league_weightings=league_weightings,
                                                    season_id=season_id, games=games, comp_teams=comp_teams) * \
            calculate_opp_venue_effect(opponent, stat, fixtures, team_stats, teams, stats_types,
                                       opponent_venue, comp_id=comp_id, games=games * 2 if games else None,
                                       season_id=season_id, comp_teams=comp_teams)

    t_nan = pd.isna(team_history)
    o_nan = pd.isna(opponent_history)
    if not t_nan and not o_nan:
        team_stat = SIMPLE_PROJECTION_ALPHA * team_history + (1 - SIMPLE_PROJECTION_ALPHA) * opponent_history
    elif not t_nan:
        team_stat = team_history
    elif not o_nan:
        team_stat = opponent_history
    else:
        team_stat = 0

    return round(float(team_stat), 2), team_history, opponent_history


def get_team_all_stats_prediction(team, opponent, fixtures, stat_list, team_stats, teams, stats_types, models,
                                  ratings=None, venue=None, comp_id=None, league_weightings=None, season_id=None,
                                  games=None, comp_teams=None):
    predictions = {}
    predictions['Team'] = team
    predictions['Opponent'] = opponent
    predictions['Venue'] = venue
    original_weightings = league_weightings.copy() if league_weightings is not None else None
    for stat in stat_list:
        model = models[stat]
        league_weightings = original_weightings.copy() if original_weightings is not None else None
        # UPDATED - store history in predictions
        predictions[stat], predictions['Team ' + stat + ' History'], predictions[
            'Opponent ' + stat + ' History Against'] = get_team_stat_prediction(team, opponent, fixtures, stat,
                                                                                team_stats, teams, stats_types, model,
                                                                                ratings=ratings, venue=venue,
                                                                                comp_id=comp_id,
                                                                                league_weightings=league_weightings,
                                                                                season_id=season_id, games=games,
                                                                                comp_teams=comp_teams)
    return predictions


def get_team_round_predictions(next_fix, stat_list, fixtures, team_stats, teams, stats_types, models, goals=False,
                               ratings=None, comp_id=None, league_weightings=None, season_id=None, games=None,
                               neutral_venue=False, comp_teams=None):
    import pandas as pd
    if goals == False:
        stat_list.remove('Goals')
    round_preds = []
    original_weightings = league_weightings.copy() if league_weightings is not None else None
    for index, row in next_fix.iterrows():
        id = row['id']
        kickoff_datetime = row['kickoff_datetime']
        league_weightings = original_weightings.copy() if original_weightings is not None else None
        # Per-row neutral check — the DataFrame's `neutral_venue` column
        # (added by ImportFixtureStoreJob via the `fixtures` table)
        # overrides the function-level kwarg so we can have mixed
        # neutral/home fixtures in one batch. Falls back to the kwarg
        # when the column is absent (older code paths or DataFrames
        # that pre-date the migration).
        row_neutral = bool(row.get('neutral_venue', neutral_venue))
        if row_neutral == True:
            home_team_preds = get_team_all_stats_prediction(row['home_team'], row['away_team'], fixtures, stat_list,
                                                            team_stats, teams, stats_types, models, ratings=ratings,
                                                            comp_id=comp_id, league_weightings=league_weightings,
                                                            season_id=season_id, games=games, comp_teams=comp_teams)
            away_team_preds = get_team_all_stats_prediction(row['away_team'], row['home_team'], fixtures, stat_list,
                                                            team_stats, teams, stats_types, models, ratings=ratings,
                                                            comp_id=comp_id, league_weightings=league_weightings,
                                                            season_id=season_id, games=games, comp_teams=comp_teams)
            # team_projections.venue is NOT NULL — venue=None propagates
            # through get_team_all_stats_prediction's Venue field and
            # blows up the insert. Label the rows with 'N' (Neutral) so
            # the stat predictions stay venue-agnostic but the row still
            # has a non-null Venue label for storage / downstream UI.
            home_team_preds['Venue'] = 'N'
            away_team_preds['Venue'] = 'N'
        else:
            home_team_preds = get_team_all_stats_prediction(row['home_team'], row['away_team'], fixtures, stat_list,
                                                            team_stats, teams, stats_types, models, ratings=ratings,
                                                            venue='H', comp_id=comp_id,
                                                            league_weightings=league_weightings, season_id=season_id,
                                                            games=games, comp_teams=comp_teams)
            away_team_preds = get_team_all_stats_prediction(row['away_team'], row['home_team'], fixtures, stat_list,
                                                            team_stats, teams, stats_types, models, ratings=ratings,
                                                            venue='A', comp_id=comp_id,
                                                            league_weightings=league_weightings, season_id=season_id,
                                                            games=games, comp_teams=comp_teams)
        home_team_preds['Fouls Drawn'] = away_team_preds['Fouls']
        away_team_preds['Fouls Drawn'] = home_team_preds['Fouls']
        if goals == True:
            home_team_preds['Assists'] = (home_team_preds['Goals'] * 0.82).round(2)
            away_team_preds['Assists'] = (away_team_preds['Goals'] * 0.82).round(2)
            home_team_preds['Saves'] = away_team_preds['Shots On Target'] - away_team_preds['Goals']
            away_team_preds['Saves'] = home_team_preds['Shots On Target'] - home_team_preds['Goals']
        home_team_preds['fixture_id'] = id
        away_team_preds['fixture_id'] = id
        home_team_preds['kickoff_datetime'] = kickoff_datetime
        away_team_preds['kickoff_datetime'] = kickoff_datetime
        round_preds.append(home_team_preds)
        round_preds.append(away_team_preds)
        df = pd.DataFrame(round_preds)
    if goals == True:
        # UPDATED - return history columns
        return df[
            ['fixture_id', 'kickoff_datetime', 'Team', 'Opponent', 'Venue', 'Goals', 'Assists'] + stat_list[1:] + [
                'Fouls Drawn', 'Saves'] + ['Team ' + stat + ' History' for stat in stat_list] + [
                'Opponent ' + stat + ' History Against' for stat in stat_list]]
    else:
        # UPDATED - return history columns
        return df[['fixture_id', 'kickoff_datetime', 'Team', 'Opponent', 'Venue'] + stat_list[0:] + ['Fouls Drawn'] + [
            'Team ' + stat + ' History' for stat in stat_list] + ['Opponent ' + stat + ' History Against' for stat in
                                                                  stat_list]]


def adjust_shots_projection(projected_goals, projected_shots, projected_shots_on_target, avg_shots_per_goal,
                            avg_shots_on_target_per_goal, weight=0.5):
    shots_based_on_goals = projected_goals * avg_shots_per_goal
    shots_on_target_based_on_goals = projected_goals * avg_shots_on_target_per_goal
    shots_diff = shots_based_on_goals - projected_shots
    shots_on_target_diff = shots_on_target_based_on_goals - projected_shots_on_target

    adjusted_shots = projected_shots + (shots_diff * weight)
    adjusted_shots_on_target = projected_shots_on_target + (shots_on_target_diff * weight)
    return round(adjusted_shots, 2), round(adjusted_shots_on_target, 2)


# UPDATED - New Parameter: season_id
def player_criteria(player, team, fixtures, player_stats, players, teams, season_id=None, competition_id=None, comp_teams=None,
                    in_confirmed_xi=False):
    import pandas as pd
    fixtures['kickoff_datetime'] = pd.to_datetime(fixtures['kickoff_datetime'])  # NEW - ensure datetime format
    # Pre-filter comp_teams for disambiguation only. Do NOT pass comp_id to
    # get_team_fixtures — that would activate its fixture filter and drop
    # cross-comp matches (same reasoning as get_ratings).
    scoped_comp_teams = comp_teams[comp_teams['competition_id'] == competition_id] if (competition_id is not None and comp_teams is not None and 'competition_id' in comp_teams.columns) else comp_teams
    team_fixtures = get_team_fixtures(team, fixtures, teams, season_id=season_id, comp_teams=scoped_comp_teams)
    todays_date = pd.to_datetime('today')
    team_fixtures['kickoff_datetime'] = pd.to_datetime(team_fixtures['kickoff_datetime'])
    team_fixtures = team_fixtures[team_fixtures['kickoff_datetime'] < todays_date]
    last5_fixtures = team_fixtures.tail(5)
    last5_fixtures = last5_fixtures[last5_fixtures['kickoff_datetime'] > todays_date - pd.DateOffset(weeks=20)]
    last5_fixture_ids = last5_fixtures['id'].values
    try:
        player_stats_df = player_stats[player_stats['player_id'] == get_player_id(player, players, team, teams, competition_id, comp_teams)]
        player_stats_df = player_stats_df[player_stats_df['stats_type_id'] == 119]
        player_stats_df = player_stats_df[player_stats_df['value'] > 45]  # NEW - filter out very low minutes
        player_stats_df = player_stats_df.merge(fixtures, left_on='fixture_id',
                                                right_on='id')  # NEW - merge to get kickoff_datetime
        player_stats_df = player_stats_df[['fixture_id', 'value', 'kickoff_datetime']]  # NEW - include kickoff_datetime
        player_stats_df = player_stats_df[
            player_stats_df['kickoff_datetime'] > todays_date - pd.DateOffset(weeks=40)]  # NEW - only last 40 weeks

    except:
        return False
    # if mins > 2500:
    #    return True
    # player_stats_df = player_stats_df[player_stats_df['fixture_id'].isin(last5_fixture_ids)]
    # player_stats_df = player_stats_df[player_stats_df['value'] > 45]
    # if len(player_stats_df) > 0:
    #   return True
    player_stats_df_last_5 = player_stats_df[
        player_stats_df['fixture_id'].isin(last5_fixture_ids)]  # NEW - filter last 5 fixtures
    player_stats_df_last_5 = player_stats_df_last_5[
        player_stats_df_last_5['value'] > 45]  # NEW - filter out very low minutes
    # Total-games gate (>5 historical appearances ≥45 min in last 40wk) is
    # non-negotiable — genuine no-data players still filter out.
    if len(player_stats_df) <= 5:
        return False
    # Confirmed-XI players bypass the team's-last-5 appearance gate: a
    # manager rotating starters out of the run-up to a cup final / play-off
    # is exactly the case where the player_criteria default silently dropped
    # them (e.g. Hakimi rested for PSG's last 5 Ligue 1 before the UCL final).
    # Lineup confirmation overrides that gate; total-games gate above still
    # protects against zero-history XI selections.
    if in_confirmed_xi:
        return True
    if len(player_stats_df_last_5) > 0:
        return True
    # New-joiner grace (George, 2026-07-23 — 26/27 launch). The old hard rule
    # ("must have appeared in THIS club's last 5") excluded every summer
    # signing until GW1: they can't have played for a club whose last 5 games
    # predate their arrival, even though their cross-club history (loaded by
    # design) projects them fine. Reframe the question from "did he play in
    # our last 5?" to "have we played 5 fixtures since he last played
    # ANYWHERE?" — old-club and international appearances count as activity.
    # Pre-season signings project from day one (club has played ~0 since
    # May); an unused joiner drops out automatically once his club plays 5
    # without him; long-inactive players fail implicitly (their club has
    # inevitably played 5+ since). Same threshold, same loaded data.
    last_appearance = player_stats_df['kickoff_datetime'].max()
    games_since = len(team_fixtures[team_fixtures['kickoff_datetime'] > last_appearance])
    return games_since < 5


def get_player_id(player_name, player_df, team, teams, competition_id=None, comp_teams=None):
    team_id = get_team_id(team, teams, competition_id, comp_teams)
    player_df = player_df[player_df['current_team_id'] == team_id]
    player_id = player_df[player_df['display_name'] == player_name]['id']
    return player_id.values[0]


# UPDATED - New Parameters: comps and include_international
def get_player_stats(stat_df, team_df, player_id, stat, stats_types, fixtures, comps, mins=50, games=None,
                     include_international=False):
    player_df = stat_df[stat_df['player_id'] == player_id]
    player_stats = player_df.merge(stats_types, left_on='stats_type_id', right_on='id')
    player_minutes = player_stats[player_stats['name'] == 'Minutes Played']
    player_minutes = player_minutes[['fixture_id', 'value', 'team_id']]
    player_minutes.rename(columns={'value': 'minutes'}, inplace=True)
    player_minutes['minutes'] = player_minutes['minutes'].astype(int)
    player_minutes = player_minutes[player_minutes['minutes'] > mins]
    player_minutes.reset_index(drop=True, inplace=True)  # NEW - reset index after filtering

    # Vectorized international filter: merge fixture → competition → sub_type,
    # then keep only domestic/domestic_cup/cup_international. Preserves the original
    # behavior that any unmapped fixture or competition is dropped (via the how='left'
    # + .isin() which returns False for NaN).
    if not include_international:
        _fix_to_comp = (
            fixtures[['id', 'competition_id']]
            .drop_duplicates(subset=['id'])
            .rename(columns={'id': 'fixture_id'})
        )
        _comp_to_subtype = (
            comps[['id', 'sub_type']]
            .drop_duplicates(subset=['id'])
            .rename(columns={'id': 'competition_id'})
        )
        _fixture_subtype = _fix_to_comp.merge(
            _comp_to_subtype, on='competition_id', how='left'
        )[['fixture_id', 'sub_type']]
        player_minutes = player_minutes.merge(_fixture_subtype, on='fixture_id', how='left')
        player_minutes = player_minutes[
            player_minutes['sub_type'].isin(['domestic', 'domestic_cup', 'cup_international'])
        ]
        player_minutes = player_minutes.drop(columns=['sub_type']).reset_index(drop=True)

    # Stat-coverage check: a per-player filter being empty is NOT a signal
    # of loader-filter omission (a CB with 0 Assists in their last 50
    # fixtures is normal). The right check is global: does the loaded
    # stat_df contain ANY rows for this stat_type_id across all players?
    # If not, the loader filter (PLAYER_STAT_NAMES in projection_stats.py)
    # is missing it and projections silently zero out for everyone.
    if not stat_df.empty:
        _st_match = stats_types[stats_types['name'] == stat]
        if not _st_match.empty:
            _target_sid = _st_match['id'].iloc[0]
            if stat_df[stat_df['stats_type_id'] == _target_sid].empty:
                _warn_stat_coverage_miss(stat, "player")
    player_stats = player_stats[player_stats['name'] == stat]
    player_stats.drop(columns=['team_id'], inplace=True)
    player_stats = player_minutes.merge(player_stats, left_on='fixture_id', right_on='fixture_id', how='left')
    player_stats['value'].fillna(0, inplace=True)
    player_stats = player_stats[['fixture_id', 'value', 'minutes', 'team_id']]
    player_stats = player_stats.merge(fixtures, left_on='fixture_id', right_on='id')
    player_stats = player_stats[['kickoff_datetime', 'fixture_id', 'team_id', 'name', 'value', 'minutes']].sort_values(
        by='kickoff_datetime')
    # NOTE: window slice (iloc[-games:]) deferred until AFTER the team-stat
    # merge below — so the window is always `games` fixtures with valid team
    # denominators, not `games` fixtures some of which the team-merge later
    # zero-fills. Sparse-coverage stats (FPL synthetics, intl xG) gain a
    # full-sized sample this way.
    player_stats.reset_index(drop=True, inplace=True)

    if stat == 'Expected Goals (xG)':
        # xG missing-data guard — drop fixtures whose xG=0 is missing data,
        # not a genuine zero, so it can't deflate the player's xG average.
        # Two independent drop conditions:
        #   1. per-player  — xG=0 but the player took shots>0 (a shot can't
        #      be worth 0.00 xG, so the xG row is missing).
        #   2. per-fixture — the fixture has NO player-xG coverage at all
        #      (no player has an xG row), i.e. xG simply wasn't tracked for
        #      it. `stat_df` is the all-players frame, so its xG fixture_ids
        #      are exactly the covered set.
        # Genuine zeros (xG=0, shots=0, fixture IS covered) are kept.
        player_stats['value'] = player_stats['value'].astype(float)
        _shots_id = get_stat_id('Shots Total', stats_types)
        _shots_lookup = (
            player_df[player_df['stats_type_id'] == _shots_id][['fixture_id', 'value']]
            .rename(columns={'value': '_shots'})
            .drop_duplicates(subset=['fixture_id'])
        )
        player_stats = player_stats.merge(_shots_lookup, on='fixture_id', how='left')
        _missing_player = (
            (player_stats['value'] == 0)
            & player_stats['_shots'].notna()
            & (player_stats['_shots'] > 0)
        )
        _xg_sid = get_stat_id('Expected Goals (xG)', stats_types)
        _xg_covered = set(stat_df.loc[stat_df['stats_type_id'] == _xg_sid, 'fixture_id'])
        _no_coverage = ~player_stats['fixture_id'].isin(_xg_covered)
        player_stats = (
            player_stats[~(_missing_player | _no_coverage)]
            .drop(columns=['_shots'])
            .reset_index(drop=True)
        )

    elif stat == 'Expected Assists (xA)':
        # xA is fractional like xG — the int cast truncated every per-match
        # value (0.2, 0.4, ...) to ZERO, so xA shares were ~0 for everyone
        # and the PL assists blend silently degenerated to assists-only
        # since the feature shipped (found 2026-07-30).
        player_stats['value'] = player_stats['value'].astype(float)
    else:
        player_stats['value'] = player_stats['value'].astype(int)
    player_stats.rename(columns={'name': 'Game', 'value': f'Player {stat}'}, inplace=True)

    # Vectorized team stat lookup: replaces the per-row loop that filtered team_df
    # by (team_id, fixture_id, stats_type_id) for every player row.
    # Three cases:
    #   - Fouls Drawn → look up the OPPONENT's Fouls value (team_id != player's team_id)
    #   - Accurate Passes → look up the team's "Successful Passes" stat
    #   - Everything else → look up the team's own value for this stat
    # Missing data preserves the original fall-through to 0 via fillna(0).
    _team_col = f'Team {stat}'
    if stat == 'Fouls Drawn':
        _fouls_id = get_stat_id('Fouls', stats_types)
        _fouls = team_df[team_df['stats_type_id'] == _fouls_id][['fixture_id', 'team_id', 'value']]
        # Self-merge on fixture_id to get (self, opponent) pairs per fixture,
        # then keep only opposite-team rows so each player gets the OPPONENT's fouls.
        _fouls_cross = _fouls.merge(_fouls, on='fixture_id', suffixes=('_self', '_opp'))
        _fouls_cross = _fouls_cross[_fouls_cross['team_id_self'] != _fouls_cross['team_id_opp']]
        _team_lookup = (
            _fouls_cross[['fixture_id', 'team_id_self', 'value_opp']]
            .rename(columns={'team_id_self': 'team_id', 'value_opp': _team_col})
            .drop_duplicates(subset=['fixture_id', 'team_id'])
        )
    else:
        _lookup_stat_name = 'Successful Passes' if stat == 'Accurate Passes' else stat
        _target_stat_id = get_stat_id(_lookup_stat_name, stats_types)
        _target_team_rows = team_df[team_df['stats_type_id'] == _target_stat_id]
        if _target_team_rows.empty and not team_df.empty:
            # Loaded team_df has rows for OTHER stats but not this one —
            # missing from TEAM_STAT_NAMES.
            _warn_stat_coverage_miss(_lookup_stat_name, "team")
        _team_lookup = (
            _target_team_rows[['fixture_id', 'team_id', 'value']]
            .rename(columns={'value': _team_col})
            .drop_duplicates(subset=['fixture_id', 'team_id'])
        )
    player_stats = player_stats.merge(_team_lookup, on=['fixture_id', 'team_id'], how='left')
    player_stats[_team_col] = player_stats[_team_col].fillna(0)
    # if stat != 'Goals' and stat != 'Assists':
    #    player_stats[f'Team {stat}'].replace({0:None}, inplace=True)
    #    player_stats.dropna(subset=[f'Team {stat}'], inplace=True)
    #    if stat == 'Expected Goals (xG)':
    #        player_stats[f'Team {stat}'] = player_stats[f'Team {stat}'].astype(float)
    #    else:
    #        player_stats[f'Team {stat}'] = player_stats[f'Team {stat}'].astype(int)
    #    player_stats['Game'] = player_stats['Game'].str.split(' v ').str[0]
    #    player_stats[f'{stat} Proportion'] = ((player_stats[f'Player {stat}'] / player_stats[f'Team {stat}'])).round(3)
    #    player_stats[f'{stat} Proportion'] = player_stats[f'{stat} Proportion'].apply(lambda x: 1 if x > 1 else x)
    #    player_stats[f'{stat} Proportion'].fillna(0, inplace=True)
    #    return player_stats.reset_index(drop=True)
    # else:
    #    player_stats[f'Team {stat}'] = player_stats[f'Team {stat}'].astype(int)
    #    player_stats['Game'] = player_stats['Game'].str.split(' v ').str[0]

    if stat in ('Expected Goals (xG)', 'Expected Assists (xA)'):  # NEW - moved from above; xA fractional too (2026-07-30)
        player_stats[f'Team {stat}'] = player_stats[f'Team {stat}'].astype(float)  # NEW
    else:  # NEW
        player_stats[f'Team {stat}'] = player_stats[f'Team {stat}'].astype(int)  # NEW
    player_stats['Game'] = player_stats['Game'].str.split(' v ').str[0]  # NEW

    # NEW (2026-06-05): Drop fixtures where team has no value for this stat
    # BEFORE the window slice. Previously this filter ran downstream in
    # get_player_weighted_average AFTER the window, producing variable
    # sample sizes (e.g. 12 valid out of last 30 for a player with cup
    # history on an FPL-only stat). Now we keep walking back the player's
    # history until we have `games` fixtures with valid team denominators
    # — fewer if the player simply doesn't have enough history.
    player_stats = player_stats[player_stats[f'Team {stat}'] > 0].reset_index(drop=True)

    # Window slice deferred from earlier in the function — applied now,
    # after team-merge + team-zero filter, so the window is stable in size
    # and consists only of fixtures with a valid share-calc denominator.
    if games is not None:
        player_stats = player_stats.iloc[-games:].reset_index(drop=True)

    return player_stats.reset_index(drop=True)


# UPDATED - New Parameters: team_id and comps
def get_weighted_player_stats(df, team_df, player_id, team_id, stat, stats_types, fixtures, comps, weight, mins=50,
                              games=None):
    import pandas as pd
    # UDATED - pass comps to get_player_stats function
    player_stats = get_player_stats(df, team_df, player_id, stat, stats_types, fixtures, comps, mins, games)
    player_stats = player_stats[pd.to_datetime(player_stats['kickoff_datetime']) < pd.to_datetime('today')].reset_index(
        drop=True)
    player_stats['Weeks Since Kickoff'] = (pd.to_datetime('today') - pd.to_datetime(
        player_stats['kickoff_datetime'])).dt.days // 7
    player_stats['Weight'] = weight ** (player_stats['Weeks Since Kickoff'] - 3)  # UPDATED - changed to -3
    player_stats.loc[
        player_stats['Weeks Since Kickoff'] < 4, 'Weight'] = 1  # UPDATED - set weight to 1 for last 4 weeks
    # Vectorized: halve the weight on games played for a different team
    player_stats.loc[player_stats['team_id'] != team_id, 'Weight'] *= 0.5
    # if stat != 'Goals' and stat != 'Assists':
    #    player_stats[f'Weighted {stat} Proportion'] = player_stats[f'{stat} Proportion'] * player_stats['Weight']
    # else:
    player_stats[f'Weighted Player {stat}'] = player_stats[f'Player {stat}'] * player_stats[
        'Weight']  # UPDATED - No indent
    player_stats[f'Weighted Team {stat}'] = player_stats[f'Team {stat}'] * player_stats['Weight']  # UPDATED - No indent
    return player_stats


# UPDATED - New Parameters: team_id and comps
def get_player_weighted_average(df, team_df, player_id, team_id, stat, stats_types, fixtures, comps, weight, mins=50,
                                games=None, per90_details=None):
    """per90_details: optional dict — when passed, this function ALSO fills it
    with the per-90 view of the SAME sample (xmins-methodology.md §11 Task 1):
        m_bar   — typical start length, Σ(Weight×minutes) ÷ Σ(Weight) over the
                  exact frame the share used (same xG guard / team-zero drops /
                  window), so parity is exact per stat
        share90 — returned share × 90 ÷ m_bar (parity: share90 × m_bar/90 ≡ share)
        n_games — surviving sample size
        series  — per-match per-90 share list, DISPLAY ONLY, never the estimate
    The returned share is BYTE-IDENTICAL with or without this argument — all
    per-90 work is additive reads on the already-built frame."""
    import logging as _logging
    _logger = _logging.getLogger("projection")
    # UDATED - pass team_id and comps to get_weighted_player_stats function
    player_stats = get_weighted_player_stats(df, team_df, player_id, team_id, stat, stats_types, fixtures, comps,
                                             weight, mins, games)

    def _fill_per90(final_share):
        # Additive reads on the frame the share used — cannot alter the share.
        if per90_details is None:
            return
        _w_sum = player_stats['Weight'].sum() if len(player_stats) else 0
        if _w_sum == 0:
            per90_details.update({'m_bar': None, 'share90': None, 'n_games': 0, 'series': []})
            return
        _m_bar = float((player_stats['Weight'] * player_stats['minutes']).sum() / _w_sum)
        _series = [
            # Team {stat} > 0 guaranteed for surviving rows (upstream drop);
            # minutes > 50 by the sample filter.
            {'fixture_id': int(_f), 'date': str(_k)[:10], 'mins': int(_m),
             'share90': round((float(_p) * 90.0 / int(_m)) / float(_t), 4)}
            for _f, _k, _m, _p, _t in zip(
                player_stats['fixture_id'], player_stats['kickoff_datetime'], player_stats['minutes'],
                player_stats[f'Player {stat}'], player_stats[f'Team {stat}'])
        ]
        per90_details.update({
            'm_bar': round(_m_bar, 2),
            'share90': float(final_share) * 90.0 / _m_bar if _m_bar > 0 else None,
            'n_games': int(len(player_stats)),
            'series': _series,
        })

    # NOTE (2026-06-05): The Team {stat} > 0 filter used to live here. It now
    # runs upstream in get_player_stats *before* the window slice, so the
    # window is `games` fixtures of valid team denominators instead of
    # `games` minus the zero-team-stat ones. By the time we reach this
    # function the data is already filtered — this comment is a tombstone.
    weighted_sum = player_stats[f'Weighted Player {stat}'].sum()
    if weighted_sum == 0:
        _fill_per90(0)
        return 0
    team_weighted_sum = player_stats[f'Weighted Team {stat}'].sum()
    # Denominator guard — without this, 0/0 produces NaN which cascades
    # through poisson.pmf into the DB as NULL projection_percent and
    # breaks the insert. Separately: log diagnostics so we can audit
    # WHICH (player, team, stat) triggers this — in theory a player
    # can't have stat > 0 while their team's aggregate is 0, so any
    # hit here points at a data/schema mismatch worth chasing.
    if team_weighted_sum == 0:
        raw_team_total = player_stats[f'Team {stat}'].sum()
        n_player_rows = int((player_stats[f'Player {stat}'] > 0).sum())
        # Aggregate per-run for a single summary line at end of
        # distribute_team_predictions_to_players (~50-100 hits per nightly
        # run pre-aggregation made the digest unreadable). Per-hit detail
        # demoted to DEBUG so it's still reachable when needed.
        _NAN_GUARD_HITS.append({
            'player_id': player_id,
            'team_id': team_id,
            'stat': stat,
            'weighted_player_sum': float(weighted_sum),
            'raw_team_total': float(raw_team_total) if raw_team_total is not None else 0,
            'n_player_rows': n_player_rows,
            'total_rows': len(player_stats),
        })
        try:
            sample = player_stats[['fixture_id', 'team_id', f'Player {stat}', f'Team {stat}']].head(5).to_dict('records')
            team_id_dtype = str(player_stats['team_id'].dtype)
            fixture_id_dtype = str(player_stats['fixture_id'].dtype)
        except Exception:
            sample = 'n/a'
            team_id_dtype = fixture_id_dtype = '?'
        _logger.debug(
            f"[NaN-guard] player_id={player_id} team_id={team_id} stat={stat!r} "
            f"has weighted_player_sum={weighted_sum:.3f} but team_weighted_sum=0 "
            f"(raw team total={raw_team_total}, player-recorded fixtures={n_player_rows}, "
            f"total rows={len(player_stats)}). "
            f"dtypes: team_id={team_id_dtype}, fixture_id={fixture_id_dtype}. "
            f"sample: {sample}. Returning 0."
        )
        _fill_per90(0)
        return 0
    weighted_average = weighted_sum / team_weighted_sum
    # else:
    #    weighted_sum = player_stats[f'Weighted {stat} Proportion'].sum()
    #    if weighted_sum == 0:
    #        return 0
    #    else:
    #        weighted_average = player_stats[f'Weighted {stat} Proportion'].sum() / player_stats['Weight'].sum()
    if len(player_stats) < 10 and weighted_average > 0.2:
        # Parity is against the RETURNED (damped) share — share90 carries the
        # damp too, so share90 × m_bar/90 ≡ the value every consumer sees.
        _fill_per90(weighted_average * 0.75)
        return weighted_average * 0.75
    _fill_per90(weighted_average)
    return weighted_average


# UPDATED - New Parameters: season_id and comps, Removed xG parameter
def distribute_team_predictions_to_players(player_stats, team_df, team_predictions, stats_types, fixtures, players,
                                           teams, comps, weight, season_id=None, competition_id=None, comp_teams=None,
                                           confirmed_lineups=None,
                                           odds_for_fixture_players=None, odds_blend_weight=0.3,
                                           per90_collector=None, only_player_ids=None):
    """
    confirmed_lineups: optional {(fixture_id, team_id): set(player_id)} — when
    a key exists for the (fixture, team) being projected, restrict the
    player iteration to those IDs only. Used by per-fixture re-projections
    triggered on confirmed-lineup arrival (Phase 1 of the lineup-aware
    rerun work). Bench / non-XI players just get dropped from this run's
    output — share renormalization is deferred to v2.

    odds_for_fixture_players: optional pre-loaded player-prop odds in the
    shape returned by load_player_odds:
        {fixture_id: {player_id: {stats_type_id: {book: ladder}}}}
    When set, the per-(player, fixture, stat) λ is blended toward the
    bookmaker-implied λ for the 3 v1 markets (Goals/Shots Total/SoT).
    Other stats pass through unchanged.

    odds_blend_weight: α applied to the bookie λ when blending. Caller
    passes its service-level weight (0.3 domestic, 0.5 euro_comp, 0.3 WC).

    per90_collector: optional list — when provided, every share computed here
    is ALSO captured in per-90 form (xmins-methodology.md §11 Task 1): one
    entry per (player, stat) with share_legacy, share90, m_bar, n_games and
    the display-only per-match series. Purely additive: shares and the
    returned frame are byte-identical whether or not the collector is passed.
    Only the FPL branch passes it (PL full runs); confirmed-lineup / props
    re-projection modes do not (George, 2026-07-29).
    """

    def _collect_per90(_pid, _stat_name, _share, _details):
        if per90_collector is not None and _details:
            per90_collector.append({
                'player_id': int(_pid), 'stat_name': _stat_name,
                'share_legacy': float(_share), **_details,
            })
    # Player-prop blend helpers hoisted out of the per-row hot loop —
    # PLAYER_BLEND_STAT_NAMES maps DataFrame stat column name to
    # stats_type_id for the 3 v1 markets (Goals/Shots Total/SoT).
    # v1.5+ stats added in odds_blend.py propagate here automatically.
    from app.services.odds_blend import blend_player_stat, PLAYER_BLEND_STAT_NAMES
    import numpy as np
    import pandas as pd
    # Reset per-run accumulators. Stat-coverage warnings dedup across all
    # get_player_stats / get_team_stats / get_opp_stats calls in this run.
    _NAN_GUARD_HITS.clear()
    _STAT_COVERAGE_WARNINGS.clear()
    team_predictions = team_predictions.drop(columns=['Corners'])
    stat_list = team_predictions.columns[5:].to_list()
    # debug print removed
    # stat_list.remove('Saves')
    if 'Saves' in stat_list:
        stat_list.remove('Saves')

    full_predicted_stats = []
    for team in team_predictions['Team'].unique():  # UPDATED - loop through each team
        specific_team_predictions = team_predictions[
            team_predictions['Team'] == team]  # NEW - filter predictions for the specific team
        team_id = get_team_id(team, teams, competition_id, comp_teams)
        # team_stat_values = row[1].values
        team_players = players[players['current_team_id'] == team_id]  # UPDATED - use team_id
        # only_player_ids (2026-07-30, George: "for premier league we should
        # map all players in FPL, that should be the gate"): second-pass mode
        # that projects EXACTLY these players and bypasses player_criteria —
        # used by the FPL branch to add FPL-mapped squad players the
        # appearance gate excludes (backup GKs etc.). xMins prices their
        # minutes honestly, so per-start shares are safe to compute.
        if only_player_ids is not None:
            team_players = team_players[team_players['id'].isin(only_player_ids)]

        # Integration ramp (George, 2026-07-23): this club's fixture ids, used
        # below to count each player's appearances FOR THIS CLUB. New joiners
        # project at a confidence haircut that lifts to full value by 5 club
        # games — mirroring the 5-game new-joiner grace window in
        # player_criteria (gate = whether you project, ramp = how confidently).
        _ramp_team_fix_ids = set(
            fixtures[(fixtures['home_team_id'] == team_id) | (fixtures['away_team_id'] == team_id)]['id'].astype(int)
        )

        # Lineup-aware restriction (Phase 1 of confirmed-lineup rerun).
        # Build a per-fixture XI map for THIS team. Each fixture independently
        # gates bench rows in the inner loop below — earlier single-fixture
        # design captured a team-wide set + restricted team_players upfront,
        # which silently dropped bench rows from OTHER fixtures (the league
        # nightly batch covers ~7 upcoming gameweeks). Now: fixtures without
        # a confirmed XI keep their full squad; fixtures with a confirmed XI
        # only emit rows for those 11 starters.
        #
        # _player_in_any_xi flows into player_criteria's in_confirmed_xi
        # kwarg — when a player is confirmed for ANY upcoming fixture they
        # bypass the team's-last-5 appearance gate (manager has signalled
        # match availability, rotation rest is no longer disqualifying).
        # Matches the Hakimi case: confirmed for UCL final overrides his
        # 5-fixture absence in PSG's Ligue 1 run-up.
        _confirmed_xi_per_fix = {}    # {fixture_id: set(player_id)}
        if confirmed_lineups:
            for _fid in specific_team_predictions['fixture_id'].unique():
                _lineup_ids = confirmed_lineups.get((int(_fid), int(team_id)))
                if _lineup_ids:
                    _confirmed_xi_per_fix[int(_fid)] = _lineup_ids
        _player_in_any_xi = set().union(*_confirmed_xi_per_fix.values()) if _confirmed_xi_per_fix else set()
        for name, id in team_players[['display_name', 'id']].values:
            # player_pred_stats = {}
            # UPDATED - New Parameter: season_id and we can now use team
            if only_player_ids is not None or player_criteria(name, team, fixtures, player_stats, players, teams, season_id, competition_id, comp_teams,
                               in_confirmed_xi=int(id) in _player_in_any_xi):
                # Integration ramp: min(1.0, 0.75 + 0.05 × apps-for-this-club).
                # 0 club apps → 0.75, full value from the 5th club game.
                # Old-club shares are the best prior for a new joiner but an
                # optimistic one at full weight (adaptation, rotation while
                # integrating). Established players hit the 1.0 cap by
                # construction. Confirmed-XI fixtures bypass below.
                _p_apps_df = player_stats[(player_stats['player_id'] == int(id))
                                          & (player_stats['stats_type_id'] == 119)]
                _p_apps_df = _p_apps_df[_p_apps_df['value'] > 45]
                _club_apps = int(_p_apps_df['fixture_id'].isin(_ramp_team_fix_ids).sum())
                _ramp = min(1.0, 0.75 + 0.05 * _club_apps)
                for stat in range(len(stat_list)):  # UPDATED - use stat instead of i
                    if stat_list[stat] == 'Goals':  # UPDATED - use stat_list[stat] instead of i
                        try:
                            # UPDATED - Pass team_id and comps
                            _d90 = {} if per90_collector is not None else None
                            stat_prop_goals = get_player_weighted_average(player_stats, team_df, id, team_id, 'Goals',
                                                                          stats_types, fixtures, comps, weight,
                                                                          games=50, per90_details=_d90)
                            _collect_per90(id, 'Goals', stat_prop_goals, _d90)
                            # if xG == True:
                            try:  # NEW - try-except block to handle cases where xG data may be insufficient
                                # UPDATED - Pass team_id and comps
                                _d90 = {} if per90_collector is not None else None
                                stat_prop_xG = get_player_weighted_average(player_stats, team_df, id, team_id,
                                                                           'Expected Goals (xG)', stats_types, fixtures,
                                                                           comps, weight, games=50, per90_details=_d90)
                                _collect_per90(id, 'Expected Goals (xG)', stat_prop_xG, _d90)
                                if stat_prop_xG == 0:  # NEW - if xG proportion is 0, use only goals proportion
                                    stat_prop = stat_prop_goals  # NEW
                                else:  # NEW - calculate average of goals and xG proportions
                                    stat_prop = (stat_prop_goals + stat_prop_xG) / 2
                                    # else:
                            except:  # NEW - if xG data is insufficient, use only goals proportion
                                stat_prop = stat_prop_goals
                            # if np.isnan(stat_prop) == False:
                            #    if stat_prop == 0:
                            #        player_pred_stats[stat_list[i]] = 0.00
                            #    else:
                            #        predicted_stat = stat_prop * team_stat_values[i+5]
                            #        player_pred_stats[stat_list[i]] = predicted_stat.round(2)
                        except:
                            pass
                    elif stat_list[stat] == 'Assists' and competition_id == 8:
                        # PL only: blend Assists with FPL xA, mirroring the Goals/xG
                        # blend above. xA is FPL-only — Sportmonks doesn't provide it,
                        # so the LeagueDataLoader._overlay_fpl_xg_xa hook injects xA
                        # rows in-memory before this runs. Other leagues fall through
                        # to the default Assists-only branch below.
                        try:
                            _d90 = {} if per90_collector is not None else None
                            stat_prop_assists = get_player_weighted_average(
                                player_stats, team_df, id, team_id, 'Assists',
                                stats_types, fixtures, comps, weight, games=50,
                                per90_details=_d90,
                            )
                            _collect_per90(id, 'Assists', stat_prop_assists, _d90)
                            try:
                                _d90 = {} if per90_collector is not None else None
                                stat_prop_xA = get_player_weighted_average(
                                    player_stats, team_df, id, team_id,
                                    'Expected Assists (xA)', stats_types, fixtures,
                                    comps, weight, games=50,
                                    per90_details=_d90,
                                )
                                _collect_per90(id, 'Expected Assists (xA)', stat_prop_xA, _d90)
                                if stat_prop_xA == 0:
                                    stat_prop = stat_prop_assists
                                else:
                                    stat_prop = (stat_prop_assists + stat_prop_xA) / 2
                            except:
                                stat_prop = stat_prop_assists
                        except:
                            pass
                    else:
                        try:
                            # UPDATED - Pass team_id and comps
                            _d90 = {} if per90_collector is not None else None
                            stat_prop = get_player_weighted_average(player_stats, team_df, id, team_id, stat_list[stat],
                                                                    stats_types, fixtures, comps, weight, games=50,
                                                                    per90_details=_d90)
                            _collect_per90(id, stat_list[stat], stat_prop, _d90)
                            # if np.isnan(stat_prop) == False:
                            #    if stat_prop == 0:
                            #        player_pred_stats[stat_list[i]] = 0.00
                            #    else:
                            #        predicted_stat = stat_prop * team_stat_values[i+5]
                            #        player_pred_stats[stat_list[i]] = predicted_stat.round(2)
                        except:
                            pass

                    ## NEW - Loop through specific team predictions to create entries for each fixture
                    for i in range(len(specific_team_predictions)):
                        _fid = int(specific_team_predictions['fixture_id'].iloc[i])
                        # Per-fixture XI gate: when THIS fixture has a confirmed
                        # XI and the player isn't in it, skip writing this row.
                        # Fixtures without a confirmed XI fall through (full
                        # squad iterated). Replaces the old team-wide
                        # team_players restriction that silently leaked into
                        # other fixtures in multi-fixture batches.
                        _xi_for_fix = _confirmed_xi_per_fix.get(_fid)
                        if _xi_for_fix is not None and int(id) not in _xi_for_fix:
                            continue
                        player_pred_stats = {}
                        player_pred_stats['player_id'] = id
                        player_pred_stats['Player'] = name
                        player_pred_stats['Team'] = team
                        player_pred_stats['Opponent'] = specific_team_predictions['Opponent'].iloc[i]
                        player_pred_stats['Venue'] = specific_team_predictions['Venue'].iloc[i]
                        player_pred_stats['fixture_id'] = specific_team_predictions['fixture_id'].iloc[i]
                        player_pred_stats['kickoff_datetime'] = specific_team_predictions['kickoff_datetime'].iloc[i]
                        player_pred_stats['stat_name'] = stat_list[stat]
                        _value = specific_team_predictions[stat_list[stat]].iloc[i] * stat_prop
                        # Integration ramp — skipped when the manager has
                        # spoken (player in this fixture's confirmed XI).
                        # Applied BEFORE the odds blend so the market can pull
                        # ramped values back up for players it believes in.
                        if not (_xi_for_fix is not None and int(id) in _xi_for_fix):
                            _value = _value * _ramp
                        _blend_st = PLAYER_BLEND_STAT_NAMES.get(stat_list[stat])
                        if odds_for_fixture_players and _blend_st is not None:
                            _ladders = (odds_for_fixture_players
                                        .get(_fid, {})
                                        .get(int(id), {})
                                        .get(_blend_st, {}))
                            _value = blend_player_stat(
                                float(_value), _ladders, _blend_st, odds_blend_weight,
                            )
                        player_pred_stats['value'] = _value
                        full_predicted_stats.append(player_pred_stats)

                        # if sum(player_pred_stats.values()) == 0:
                #   continue
            # else:
            #    continue
            # player_pred_stats['player_id'] = id
            # player_pred_stats['Player'] = name
            # player_pred_stats['Team'] = team_stat_values[2]
            # player_pred_stats['Opponent'] = team_stat_values[3]
            # if pd.isna(team_stat_values[2]):
            #    player_pred_stats['Venue'] = 'Neutral'
            # else:
            #    player_pred_stats['Venue'] = team_stat_values[4]
            # player_pred_stats['fixture_id'] = team_stat_values[0]
            # player_pred_stats['kickoff_datetime'] = team_stat_values[1]
            # full_predicted_stats.append(player_pred_stats)
    df = pd.DataFrame(full_predicted_stats)

    ## NEW - Pivot the dataframe to have stats as columns
    # debug print removed

    if df.empty or 'value' not in df.columns or 'stat_name' not in df.columns:
        cols = ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Team', 'Opponent', 'Venue'] + stat_list
        _log_nan_guard_summary(competition_id, comps, teams)
        return pd.DataFrame(columns=cols)

    df = df.pivot_table(
        index=['player_id', 'Player', 'Team', 'Opponent', 'Venue', 'fixture_id', 'kickoff_datetime'],
        columns='stat_name',
        values='value',
        aggfunc='first'
    ).reset_index()

    df.columns.name = None  # Remove the aggregation name
    df = df.round(2)  # Round all values to 2 decimal places
    existing_stats = [s for s in stat_list if s in df.columns]
    # All-zero column scan: any projected stat where 100% of player rows
    # are 0 is almost certainly a silent bug (loader-filter omission, share
    # collapse, model error, etc). Surface as a single WARNING so the
    # daily digest catches it. Genuinely all-zero is implausible at the
    # league level — no league has 0 Yellowcards, 0 Goals, etc. across
    # every projected player.
    if len(df) > 0:
        zero_cols = [s for s in existing_stats if df[s].sum() == 0]
        if zero_cols:
            import logging as _logging
            _logging.getLogger("projection").warning(
                f"[stat-coverage] All-zero projection columns: {zero_cols}. "
                f"Likely loader-filter omission (check projection_stats.py) or "
                f"share-calc bug. {len(df)} player×fixture rows scanned."
            )
    _log_nan_guard_summary(competition_id, comps, teams)
    return df[['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Team', 'Opponent', 'Venue'] + existing_stats]

def get_player_position(player, team, players, teams, competition_id=None, comp_teams=None):
    if player == 'Caoimhin Kelleher':
        return 'GK'
    team_id = get_team_id(team, teams, competition_id, comp_teams)
    player_row = players[(players['display_name'] == player) & (players['current_team_id'] == team_id)]
    if player_row.empty:
        return None
    position = player_row['position'].values[0]
    if position == 'goalkeeper':
        return 'GK'
    elif position == 'defender':
        return 'DEF'
    elif position == 'midfielder':
        return 'MID'
    elif position == 'attacker':
        return 'FWD'
    return position


def get_poisson_probs(projections, stats, numbers):
    """
    For each (player-fixture row, stat, line) compute P(X >= line) under Poisson(λ=projection).

    Returns one row per (player, fixture, market, prop) combination.

    Vectorised 2026-05-08: previous implementation grew the result DataFrame one
    row at a time via `new_df.loc[len(new_df)] = row`, which is O(N²) in pandas
    (each assignment copies the entire frame). At ~9k rows for PL it ran in
    ~8.4s; adding stats made it scale poorly. Now ~0.5-1s for the same load.
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import poisson

    base_cols = ['fixture_id', 'kickoff_datetime', 'player_id', 'Player',
                 'Position', 'Team', 'Opponent', 'Venue']
    if len(projections) == 0:
        return pd.DataFrame(columns=base_cols + ['Market', 'Prop', 'Projection %'])

    base = projections[base_cols].reset_index(drop=True)
    blocks = []

    for stat in stats:
        # Coerce projection values to float; missing/NaN → 0 so poisson.sf returns 0.
        proj_values = pd.to_numeric(projections[stat], errors='coerce').fillna(0).values
        for number in numbers:
            # poisson.sf(k-1, lam) = P(X >= k) — vectorised across all players.
            probs = poisson.sf(number - 1, proj_values) * 100
            # Clamp to [0.01, 99.99] to match the previous implementation.
            probs = np.clip(probs, 0.01, 99.99).round(2)

            block = base.copy()
            block['Market'] = stat
            block['Prop'] = f"{number}+"
            block['Projection %'] = probs
            blocks.append(block)

    return pd.concat(blocks, ignore_index=True)


## THE FOLLOWING FUNCTIONS ARE FOR THE PREMIER LEAGUE SCRIPT

def get_extra_stats(player, position, team, teams, players, player_stats, fixtures, stats_types, weight=0.98, mins=50,
                    games=50, competition_id=None, comp_teams=None):
    import pandas as pd
    player_id = get_player_id(player, players, team, teams, competition_id, comp_teams)
    player_stats = player_stats[player_stats['player_id'] == player_id]
    player_df = player_stats[player_stats['stats_type_id'] == get_stat_id('Minutes Played', stats_types)]
    player_df = player_df[player_df['value'] >= mins]
    player_df.rename(columns={'value': 'Player Minutes Played'}, inplace=True)
    if player_df.empty:
        return None
    player_df = player_df.merge(fixtures, left_on='fixture_id', right_on='id')
    player_df = player_df[['fixture_id', 'kickoff_datetime', 'player_id', 'Player Minutes Played']]
    for stat in ['Clearances', 'Blocked Shots', 'Interceptions', 'Tackles', 'Ball Recovery', 'Tackles Won']:
        player_stat = player_stats[player_stats['stats_type_id'] == get_stat_id(stat, stats_types)]
        player_stat = player_stat[['fixture_id', 'value']]
        player_stat.rename(columns={'value': f'Player {stat}'}, inplace=True)
        player_stat[f'Player {stat}'] = player_stat[f'Player {stat}'].astype(int)
        player_df = player_df.merge(player_stat, on='fixture_id', how='left')
    player_df.fillna(0).reset_index(drop=True)
    if position == 'DEF':
        player_df['Player Def Con'] = player_df[
            ['Player Clearances', 'Player Blocked Shots', 'Player Interceptions', 'Player Tackles']].sum(axis=1)
    else:
        player_df['Player Def Con'] = player_df[
            ['Player Clearances', 'Player Blocked Shots', 'Player Interceptions', 'Player Tackles',
             'Player Ball Recovery']].sum(axis=1)
    player_df['Player Def Con'] = player_df['Player Def Con'].astype(int)
    player_df = player_df.sort_values(by='kickoff_datetime', ascending=False)
    player_df = player_df.head(games)
    if position == 'DEF':
        player_df['Hit?'] = player_df['Player Def Con'] >= 10
    else:
        player_df['Hit?'] = player_df['Player Def Con'] >= 12
    player_df['90 Hit?'] = player_df['Player Minutes Played'] >= 90
    player_df['Weeks Since Kickoff'] = (pd.to_datetime('today') - pd.to_datetime(
        player_df['kickoff_datetime'])).dt.days // 7
    player_df['Weight'] = weight ** (player_df['Weeks Since Kickoff'] - 5)
    player_df.loc[player_df['Weeks Since Kickoff'] < 6, 'Weight'] = 1
    player_df['Weighted Hit Rate'] = player_df['Hit?'] * player_df['Weight']
    player_df['Weighted Def Con'] = player_df['Player Def Con'] * player_df['Weight']
    player_df['Weighted Clearances'] = player_df['Player Clearances'] * player_df['Weight']
    player_df['Weighted Blocked Shots'] = player_df['Player Blocked Shots'] * player_df['Weight']
    player_df['Weighted Ball Recovery'] = player_df['Player Ball Recovery'] * player_df['Weight']
    player_df['Weighted Tackles Won'] = player_df['Player Tackles Won'] * player_df['Weight']
    player_df['Weighted 90 Hit Rate'] = player_df['90 Hit?'] * player_df['Weight']
    if position == 'GK':
        weighted_hit_rate = 0
    else:
        weighted_hit_rate = player_df['Weighted Hit Rate'].sum() / player_df['Weight'].sum()
    weighted_def_con = player_df['Weighted Def Con'].sum() / player_df['Weight'].sum()
    weighted_clearances = player_df['Weighted Clearances'].sum() / player_df['Weight'].sum()
    weighted_blocked_shots = player_df['Weighted Blocked Shots'].sum() / player_df['Weight'].sum()
    weighted_ball_recovery = player_df['Weighted Ball Recovery'].sum() / player_df['Weight'].sum()
    weighted_tackles_won = player_df['Weighted Tackles Won'].sum() / player_df['Weight'].sum()
    weighted_90_hit_rate = player_df['Weighted 90 Hit Rate'].sum() / player_df['Weight'].sum()
    return weighted_hit_rate, weighted_def_con, weighted_clearances, weighted_blocked_shots, weighted_ball_recovery, weighted_tackles_won, weighted_90_hit_rate


def get_fpl_points(pl_projections, score_preds, fpl_points_dict_gk, fpl_points_dict_def, fpl_points_dict_mid,
                   fpl_points_dict_fwd):
    import pandas as pd
    import numpy as np
    from scipy.stats import poisson
    fpl_points_df = {'fixture_id': [], 'kickoff_datetime': [], 'player_id': [], 'Player': [], 'Position': [],
                     'Team': [], 'Opponent': [], 'Venue': [], 'PTS': []}
    fpl_points_df['fixture_id'] = pl_projections['fixture_id'].tolist()
    fpl_points_df['kickoff_datetime'] = pl_projections['kickoff_datetime'].tolist()
    fpl_points_df['player_id'] = pl_projections['player_id'].tolist()
    fpl_points_df['Player'] = pl_projections['Player'].tolist()
    fpl_points_df['Position'] = pl_projections['FPL Position'].tolist()
    fpl_points_df['Team'] = pl_projections['Team'].tolist()
    fpl_points_df['Opponent'] = pl_projections['Opponent'].tolist()
    fpl_points_df['Venue'] = pl_projections['Venue'].tolist()
    # xMinutes: when the caller stamped xmin_p_play/xmin_p60 (FPL_XMINUTES on),
    # appearance points and the genuinely 60-minute-gated component (clean
    # sheet) stop assuming a 90-minute start. Absent columns -> everything
    # defaults to a full 90 and the output is identical to pre-xMinutes.
    _has_xmin = 'xmin_p_play' in pl_projections.columns and 'xmin_p60' in pl_projections.columns
    _has_bands = _has_xmin and 'xmin_bands' in pl_projections.columns
    for i in range(len(pl_projections)):
        fixture_id = pl_projections['fixture_id'][i]
        fix_score_pred = score_preds[score_preds['id'] == fixture_id]
        position = pl_projections['FPL Position'][i]
        if _has_xmin:
            _p_play = float(pl_projections['xmin_p_play'][i])
            _p60 = float(pl_projections['xmin_p60'][i])
        else:
            _p_play = 1.0
            _p60 = 1.0
        # Share of the match he is on the pitch for. Goals conceded and penalty
        # saves accrue per event while he plays and carry NO 60' threshold
        # (that rule is clean-sheet-only), so they scale by exposure, not by
        # p60. George, 2026-08-02.
        _exposure = min(1.0, float(pl_projections['xmin_bands'][i]) / 90.0) if _has_bands else 1.0
        if position == 'GK':
            fpl_points_dict = fpl_points_dict_gk
        elif position == 'DEF':
            fpl_points_dict = fpl_points_dict_def
        elif position == 'MID':
            fpl_points_dict = fpl_points_dict_mid
        elif position == 'FWD':
            fpl_points_dict = fpl_points_dict_fwd
        else:
            fpl_points_df['PTS'].append(0)
            continue
        team = pl_projections['Team'][i]
        goal_points = pl_projections['Goals'][i] * fpl_points_dict['Goals']
        assists = pl_projections['Assists'][i] * fpl_points_dict['Assists']
        yellow_cards = pl_projections['Yellow Cards'][i] * fpl_points_dict['Yellow Card']
        saves = pl_projections['Saves'][i]
        saves_points = poisson.pmf(3, saves) + poisson.pmf(4, saves) + poisson.pmf(5, saves) + (
                    (poisson.pmf(6, saves) + poisson.pmf(7, saves) + poisson.pmf(8, saves)) * 2) + (
                                   poisson.pmf(9, saves) + poisson.pmf(10, saves) + poisson.pmf(11,
                                                                                                saves)) * 3 if saves > 0 else 0
        if 'Goals Conceded' in fpl_points_dict:
            goals_conceded = fix_score_pred[fix_score_pred['Home Team'] == team]['Away Goals'].values[0] if team in \
                                                                                                            fix_score_pred[
                                                                                                                'Home Team'].values else \
            fix_score_pred[fix_score_pred['Away Team'] == team]['Home Goals'].values[0]
            # He can only concede while he is on the pitch, so the EXPOSURE
            # scales the rate that goes into the Poisson — not the answer that
            # comes out. The dock is floor(GC/2), which is non-linear, so
            # scaling the result over-charges (~7% at lam 1.4, exposure 0.85).
            _gc_lambda = goals_conceded * _exposure
            goal_conceded_points = (poisson.pmf(2, _gc_lambda) + poisson.pmf(3, _gc_lambda) + (
                        (poisson.pmf(4, _gc_lambda) + poisson.pmf(5, _gc_lambda)) * 2) + (
                                                poisson.pmf(6, _gc_lambda) + poisson.pmf(7, _gc_lambda)) * 3) * \
                                   fpl_points_dict['Goals Conceded'] if _gc_lambda > 0 else 0
        else:
            goal_conceded_points = 0
        if 'Clean Sheet' in fpl_points_dict:
            clean_sheet_perc = fix_score_pred[fix_score_pred['Home Team'] == team]['Home Clean Sheet %'].values[
                0] if team in fix_score_pred['Home Team'].values else \
            fix_score_pred[fix_score_pred['Away Team'] == team]['Away Clean Sheet %'].values[0]
            # FPL rule: clean-sheet points require 60+ minutes played.
            clean_sheet_points = (float(clean_sheet_perc.replace('%', '')) / 100) * fpl_points_dict['Clean Sheet'] * _p60
        else:
            clean_sheet_points = 0
        if 'Penalties Saved' in fpl_points_dict:
            # A penalty can only be faced while he is on the pitch — no 60'
            # gate here either, so exposure not p60. Linear in goals_conceded,
            # so scaling the result is exact. George, 2026-08-02.
            pen_save_points = (0.1 * goals_conceded) * 0.16 * fpl_points_dict['Penalties Saved'] * _exposure
        else:
            pen_save_points = 0
        cbit_points = pl_projections['CBIT Hit Rate'][i] * 2
        # xMinutes: 2 appearance pts need 60+ minutes; 1 pt for any shorter
        # appearance. Collapses to the old flat +2 when p_play = p60 = 1.
        appearance_points = 2 * _p60 + 1 * max(0.0, _p_play - _p60)
        fpl_points = goal_points + assists + yellow_cards + saves_points + clean_sheet_points + goal_conceded_points + cbit_points + pen_save_points + appearance_points
        fpl_points_df['PTS'].append(fpl_points)
    fpl_points_df = pd.DataFrame(fpl_points_df)
    fpl_points_df.sort_values(by='PTS', ascending=False, inplace=True)
    fpl_points_df['PTS'] = fpl_points_df['PTS'].round(2)
    fpl_points_df.reset_index(drop=True, inplace=True)
    return fpl_points_df


def bonus_points_score(projections, score_preds, fpl_bonus_dict_gk, fpl_bonus_dict_def, fpl_bonus_dict_mid,
                       fpl_bonus_dict_fwd):
    import pandas as pd
    fpl_bonus_df = {'fixture_id': [], 'kickoff_datetime': [], 'player_id': [], 'Player': [], 'Team': [], 'Opponent': [],
                    'Venue': [], 'Goal Bonus': [], 'Assist Bonus': [], 'Save Bonus': [], 'CBIT Bonus': [],
                    'Goal Conceded Bonus': [], 'Clean Sheet Bonus': [], 'Pass Completion Bonus': [], 'Total': []}
    fpl_bonus_df['fixture_id'] = projections['fixture_id'].tolist()
    fpl_bonus_df['kickoff_datetime'] = projections['kickoff_datetime'].tolist()
    fpl_bonus_df['player_id'] = projections['player_id'].tolist()
    fpl_bonus_df['Player'] = projections['Player'].tolist()
    fpl_bonus_df['Team'] = projections['Team'].tolist()
    fpl_bonus_df['Opponent'] = projections['Opponent'].tolist()
    fpl_bonus_df['Venue'] = projections['Venue'].tolist()
    # xMinutes: the count-shaped columns arrive pre-scaled by exposure on the
    # FPL-local frame. Two team-derived terms need explicit weighting here and
    # they take DIFFERENT weights: clean sheet is a genuine 60' threshold (p60),
    # goals conceded is per-event-while-on-pitch (exposure = xMins/90).
    _has_xmin = 'xmin_p60' in projections.columns
    _has_bands = 'xmin_bands' in projections.columns
    for i in range(len(projections)):
        fixture_id = projections['fixture_id'][i]
        fix_score_pred = score_preds[score_preds['id'] == fixture_id]
        position = projections['FPL Position'][i]
        _p60 = float(projections['xmin_p60'][i]) if _has_xmin else 1.0
        _exposure = min(1.0, float(projections['xmin_bands'][i]) / 90.0) if _has_bands else 1.0
        if position == 'GK':
            fpl_bonus_dict = fpl_bonus_dict_gk
        elif position == 'DEF':
            fpl_bonus_dict = fpl_bonus_dict_def
        elif position == 'MID':
            fpl_bonus_dict = fpl_bonus_dict_mid
        elif position == 'FWD':
            fpl_bonus_dict = fpl_bonus_dict_fwd
        else:
            for _k in ['Goal Bonus', 'Assist Bonus', 'Save Bonus', 'CBIT Bonus',
                       'Goal Conceded Bonus', 'Clean Sheet Bonus', 'Pass Completion Bonus', 'Total']:
                fpl_bonus_df[_k].append(0)
            continue
        team = projections['Team'][i]
        save_bonus = projections['Saves'][i] * fpl_bonus_dict['Saves'] if position == 'GK' else 0
        key_passes = projections['Key Passes'][i] * fpl_bonus_dict['Key Passes']
        big_chances_created = projections['Key Passes'][i] * fpl_bonus_dict['Big Chances Created'] * 0.2
        chances_created_bonus = key_passes + big_chances_created
        cbi = projections['Clearances Average'][i] + projections['Blocked Shots Average'][i] + \
              projections['Interceptions'][i]
        cbi_points = cbi * fpl_bonus_dict['Clearances, Blocks & Interceptions']
        recoveries = projections['Ball Recovery Average'][i] * fpl_bonus_dict['Recoveries']
        cbit_bonus = cbi_points + recoveries
        shots_on_target = projections['Shots On Target'][i] * fpl_bonus_dict['Shots On Target']
        shots_off_target = projections['Shots Total'][i] - projections['Shots On Target'][i]
        shots_off_target = shots_off_target * fpl_bonus_dict['Shots Off Target'] if shots_off_target > 0 else 0
        shots_bonus = shots_on_target + shots_off_target
        goal_bonus = (projections['Goals'][i] * fpl_bonus_dict['Goals']) + shots_bonus
        assist_bonus = projections['Assists'][i] * fpl_bonus_dict['Assists'] + chances_created_bonus
        if 'Goals Conceded' in fpl_bonus_dict:
            goals_conceded = fix_score_pred[fix_score_pred['Home Team'] == team]['Away Goals'].values[0] if team in \
                                                                                                            fix_score_pred[
                                                                                                                'Home Team'].values else \
            fix_score_pred[fix_score_pred['Away Team'] == team]['Home Goals'].values[0]
            # -4 BPS per goal conceded WHILE ON THE PITCH — no 60' gate, so
            # exposure not p60. Linear in goals, so scaling the answer is exact
            # here (unlike the points path's floor(GC/2)). George, 2026-08-02.
            goal_conceded_points = (goals_conceded * fpl_bonus_dict['Goals Conceded'] if goals_conceded > 0 else 0) * _exposure
        else:
            goal_conceded_points = 0
        if 'Clean Sheet' in fpl_bonus_dict:
            clean_sheet_perc = fix_score_pred[fix_score_pred['Home Team'] == team]['Home Clean Sheet %'].values[
                0] if team in fix_score_pred['Home Team'].values else \
            fix_score_pred[fix_score_pred['Away Team'] == team]['Away Clean Sheet %'].values[0]
            clean_sheet_points = (float(clean_sheet_perc.replace('%', '')) / 100) * fpl_bonus_dict['Clean Sheet'] * _p60
        else:
            clean_sheet_points = 0
        passes = projections['Passes'][i] if projections['Passes'][i] > 0 else 0
        accurate_passes = projections['Accurate Passes'][i] if projections['Accurate Passes'][i] > 0 else 0
        pass_completion = accurate_passes / passes if passes > 0 else 0
        if 20 <= passes < 25 and pass_completion >= 0.7:
            pass_completion_points = 0.25
        elif 25 <= passes < 30 and 0.7 < pass_completion < 0.75:
            pass_completion_points = 0.75
        elif 30 <= passes < 40 and 0.7 < pass_completion < 0.75:
            pass_completion_points = 1
        elif 30 <= passes < 40 and 0.75 <= pass_completion < 0.8:
            pass_completion_points = 1.5
        elif 30 <= passes < 40 and pass_completion >= 0.8:
            pass_completion_points = 2
        elif 40 <= passes < 50 and 0.7 < pass_completion < 0.75:
            pass_completion_points = 1.5
        elif 40 <= passes < 50 and 0.75 <= pass_completion < 0.8:
            pass_completion_points = 2
        elif 40 <= passes < 50 and pass_completion >= 0.8:
            pass_completion_points = 2.5
        elif 50 <= passes < 60 and 0.7 < pass_completion < 0.8:
            pass_completion_points = 2
        elif 50 <= passes < 60 and 0.8 <= pass_completion < 0.85:
            pass_completion_points = 3
        elif 50 <= passes < 60 and pass_completion >= 0.85:
            pass_completion_points = 3.5
        elif 60 <= passes < 70 and 0.7 < pass_completion < 0.8:
            pass_completion_points = 2.5
        elif 60 <= passes < 70 and 0.8 <= pass_completion < 0.85:
            pass_completion_points = 3.5
        elif 60 <= passes < 70 and pass_completion >= 0.85:
            pass_completion_points = 4
        elif 70 <= passes < 80 and 0.7 <= pass_completion < 0.8:
            pass_completion_points = 3
        elif 70 <= passes < 80 and pass_completion >= 0.8:
            pass_completion_points = 4
        elif 80 <= passes < 90:
            pass_completion_points = 4.75
        elif passes > 100:
            pass_completion_points = 5
        else:
            pass_completion_points = 0
        fpl_bonus = goal_bonus + assist_bonus + cbit_bonus + goal_conceded_points + clean_sheet_points + pass_completion_points + save_bonus
        fpl_bonus_df['Goal Bonus'].append(goal_bonus)
        fpl_bonus_df['Assist Bonus'].append(assist_bonus)
        fpl_bonus_df['Save Bonus'].append(save_bonus)
        fpl_bonus_df['CBIT Bonus'].append(cbit_bonus)
        fpl_bonus_df['Goal Conceded Bonus'].append(goal_conceded_points)
        fpl_bonus_df['Clean Sheet Bonus'].append(clean_sheet_points)
        fpl_bonus_df['Pass Completion Bonus'].append(pass_completion_points)
        fpl_bonus_df['Total'].append(fpl_bonus)
    fpl_bonus_df = pd.DataFrame(fpl_bonus_df)
    fpl_bonus_df.sort_values(by='Total', ascending=False, inplace=True)
    fpl_bonus_df = fpl_bonus_df.round(2)
    fpl_bonus_df['Total'] = fpl_bonus_df['Total'].clip(lower=0)
    fpl_bonus_df.reset_index(drop=True, inplace=True)
    return fpl_bonus_df


def get_bonus_points(bps_df, score_preds, expo_factor=0.1):
    import pandas as pd
    import numpy as np
    df = pd.DataFrame()
    for i in range(len(score_preds)):
        fixture_id = score_preds['id'].iloc[i]
        fixture_bps = bps_df[bps_df['fixture_id'] == fixture_id]
        fixture_bps = fixture_bps.copy()
        fixture_bps['Total Scaled'] = np.exp(expo_factor * fixture_bps['Total'])
        fixture_bps = fixture_bps.copy()
        fixture_bps['Bonus Points'] = (fixture_bps['Total Scaled'] / fixture_bps['Total Scaled'].sum()) * (
                    len(fixture_bps[fixture_bps['Total'] >= 7.5]) * 0.5)
        df = pd.concat([df, fixture_bps[['Player', 'Team', 'Opponent', 'Bonus Points']]], ignore_index=True)
    return df


def get_bonus_points_by_fixture(bps_df, score_preds, expo_factor=0.1):
    """Same arithmetic as get_bonus_points but keyed by (fixture_id,
    player_id). The Player/Team/Opponent merge downstream of the original
    cross-joins the two legs of any matchup repeated inside the projection
    window — per-row values were right (upsert last-wins) but the bonus
    could land on the wrong leg. Callers merge on ['fixture_id',
    'player_id'] (2026-07-30)."""
    import pandas as pd
    import numpy as np
    parts = []
    for i in range(len(score_preds)):
        fixture_id = score_preds['id'].iloc[i]
        fixture_bps = bps_df[bps_df['fixture_id'] == fixture_id].copy()
        if fixture_bps.empty:
            continue
        fixture_bps['Total Scaled'] = np.exp(expo_factor * fixture_bps['Total'])
        fixture_bps['Bonus Points'] = (fixture_bps['Total Scaled'] / fixture_bps['Total Scaled'].sum()) * (
                    len(fixture_bps[fixture_bps['Total'] >= 7.5]) * 0.5)
        parts.append(fixture_bps[['fixture_id', 'player_id', 'Bonus Points']])
    if not parts:
        return pd.DataFrame(columns=['fixture_id', 'player_id', 'Bonus Points'])
    return pd.concat(parts, ignore_index=True)


def get_dream11_points(pl_projections, score_preds, dream11_points_dict_gk, dream11_points_dict_def,
                       dream11_points_dict_mid, dream11_points_dict_fwd):
    import pandas as pd
    dream11_points_df = {'fixture_id': [], 'kickoff_datetime': [], 'player_id': [], 'Player': [], 'Position': [],
                         'Team': [], 'Opponent': [], 'Venue': [], 'PTS': []}
    dream11_points_df['fixture_id'] = pl_projections['fixture_id'].tolist()
    dream11_points_df['kickoff_datetime'] = pl_projections['kickoff_datetime'].tolist()
    dream11_points_df['player_id'] = pl_projections['player_id'].tolist()
    dream11_points_df['Player'] = pl_projections['Player'].tolist()
    dream11_points_df['Position'] = pl_projections['Dream11 Position'].tolist()
    dream11_points_df['Team'] = pl_projections['Team'].tolist()
    dream11_points_df['Opponent'] = pl_projections['Opponent'].tolist()
    dream11_points_df['Venue'] = pl_projections['Venue'].tolist()
    for i in range(len(pl_projections)):
        fixture_id = pl_projections['fixture_id'][i]
        fix_score_pred = score_preds[score_preds['id'] == fixture_id]
        position = pl_projections['Dream11 Position'][i]
        if position == 'GK':
            dream11_points_dict = dream11_points_dict_gk
        elif position == 'DEF':
            dream11_points_dict = dream11_points_dict_def
        elif position == 'MID':
            dream11_points_dict = dream11_points_dict_mid
        elif position == 'FWD':
            dream11_points_dict = dream11_points_dict_fwd
        else:
            dream11_points_df['PTS'].append(0)
            continue
        team = pl_projections['Team'][i]
        goal_points = pl_projections['Goals'][i] * dream11_points_dict['Goals']
        assists = pl_projections['Assists'][i] * dream11_points_dict['Assists']
        shots_on_target = pl_projections['Shots On Target'][i] * dream11_points_dict['Shots On Target']
        tackles_won = pl_projections['Tackles Won Average'][i] * dream11_points_dict['Tackles Won']
        key_passes = pl_projections['Key Passes'][i] * dream11_points_dict['Key Passes']
        accurate_passes = pl_projections['Accurate Passes'][i] * dream11_points_dict['Successful Passes']
        interceptions = pl_projections['Interceptions'][i] * dream11_points_dict['Interceptions']
        yellow_cards = pl_projections['Yellow Cards'][i] * dream11_points_dict['Yellow Card']
        saves = pl_projections['Saves'][i] * dream11_points_dict['Saves'] if position == 'GK' else 0
        goals_conceded = fix_score_pred[fix_score_pred['Home Team'] == team]['Away Goals'].values[0] if team in \
                                                                                                        fix_score_pred[
                                                                                                            'Home Team'].values else \
        fix_score_pred[fix_score_pred['Away Team'] == team]['Home Goals'].values[0]
        goal_conceded_points = goals_conceded * dream11_points_dict[
            'Goals Conceded'] if position == 'GK' and goals_conceded > 0 else 0
        clean_sheet_perc = fix_score_pred[fix_score_pred['Home Team'] == team]['Home Clean Sheet %'].values[
            0] if team in fix_score_pred['Home Team'].values else \
        fix_score_pred[fix_score_pred['Away Team'] == team]['Away Clean Sheet %'].values[0]
        clean_sheet_points = (float(clean_sheet_perc.replace('%', '')) / 100) * dream11_points_dict[
            'Clean Sheet'] if 'Clean Sheet' in dream11_points_dict else 0
        pen_save_points = (0.1 * goals_conceded) * 0.16 * dream11_points_dict[
            'Penalties Saved'] if 'Penalties Saved' in dream11_points_dict else 0
        dream11_points = goal_points + assists + shots_on_target + tackles_won + key_passes + accurate_passes + yellow_cards + interceptions + saves + goal_conceded_points + clean_sheet_points + pen_save_points + 4
        dream11_points_df['PTS'].append(dream11_points)
    dream11_points_df = pd.DataFrame(dream11_points_df)
    dream11_points_df.sort_values(by='PTS', ascending=False, inplace=True)
    dream11_points_df['PTS'] = dream11_points_df['PTS'].round(2)
    dream11_points_df.reset_index(drop=True, inplace=True)
    return dream11_points_df


def get_draftkings_points(pl_projections, score_preds, draftkings_points_dict_gk, draftkings_points_dict_def,
                          draftkings_points_dict_mid, draftkings_points_dict_fwd):
    import pandas as pd
    import numpy as np
    from scipy.stats import poisson
    draftkings_points_df = {'fixture_id': [], 'kickoff_datetime': [], 'player_id': [], 'Player': [], 'Position': [],
                            'Team': [], 'Opponent': [], 'Venue': [], 'PTS': []}
    draftkings_points_df['fixture_id'] = pl_projections['fixture_id'].tolist()
    draftkings_points_df['kickoff_datetime'] = pl_projections['kickoff_datetime'].tolist()
    draftkings_points_df['player_id'] = pl_projections['player_id'].tolist()
    draftkings_points_df['Player'] = pl_projections['Player'].tolist()
    draftkings_points_df['Position'] = pl_projections['Draftkings Position'].tolist()
    draftkings_points_df['Team'] = pl_projections['Team'].tolist()
    draftkings_points_df['Opponent'] = pl_projections['Opponent'].tolist()
    draftkings_points_df['Venue'] = pl_projections['Venue'].tolist()
    for i in range(len(pl_projections)):
        fixture_id = pl_projections['fixture_id'][i]
        fix_score_pred = score_preds[score_preds['id'] == fixture_id]
        position = pl_projections['Draftkings Position'][i]
        if position == 'GK':
            draftkings_points_dict = draftkings_points_dict_gk
        elif position == 'DEF':
            draftkings_points_dict = draftkings_points_dict_def
        elif position == 'MID':
            draftkings_points_dict = draftkings_points_dict_mid
        elif position == 'FWD':
            draftkings_points_dict = draftkings_points_dict_fwd
        else:
            draftkings_points_df['PTS'].append(0)
            continue
        team = pl_projections['Team'][i]
        goal_points = pl_projections['Goals'][i] * draftkings_points_dict['Goals']
        assists = pl_projections['Assists'][i] * draftkings_points_dict['Assists']
        shots_on_target = pl_projections['Shots On Target'][i] * draftkings_points_dict['Shots On Target']
        shots_total = pl_projections['Shots Total'][i] * draftkings_points_dict['Shots Total']
        crosses_total = pl_projections['Total Crosses'][i] * draftkings_points_dict['Total Crosses']
        tackles_won = pl_projections['Tackles Won Average'][i] * draftkings_points_dict['Tackles Won']
        key_passes = pl_projections['Key Passes'][i] * draftkings_points_dict['Key Passes']
        accurate_passes = pl_projections['Accurate Passes'][i] * draftkings_points_dict['Successful Passes']
        fouls_drawn = pl_projections['Fouls Drawn'][i] * draftkings_points_dict['Fouls Drawn']
        fouls_committed = pl_projections['Fouls'][i] * draftkings_points_dict['Fouls Committed']
        interceptions = pl_projections['Interceptions'][i] * draftkings_points_dict[
            'Interceptions'] if position != 'GK' else 0
        yellow_cards = pl_projections['Yellow Cards'][i] * draftkings_points_dict['Yellow Card']
        saves = pl_projections['Saves'][i] * draftkings_points_dict['Saves'] if position == 'GK' else 0
        goals_conceded = fix_score_pred[fix_score_pred['Home Team'] == team]['Away Goals'].values[0] if team in \
                                                                                                        fix_score_pred[
                                                                                                            'Home Team'].values else \
        fix_score_pred[fix_score_pred['Away Team'] == team]['Home Goals'].values[0]
        goal_conceded_points = goals_conceded * draftkings_points_dict[
            'Goals Conceded'] if position == 'GK' and goals_conceded > 0 else 0
        clean_sheet_perc = fix_score_pred[fix_score_pred['Home Team'] == team]['Home Clean Sheet %'].values[
            0] if team in fix_score_pred['Home Team'].values else \
        fix_score_pred[fix_score_pred['Away Team'] == team]['Away Clean Sheet %'].values[0]
        clean_sheet_points = (float(clean_sheet_perc.replace('%', '')) / 100) * draftkings_points_dict[
            'Clean Sheet'] if 'Clean Sheet' in draftkings_points_dict else 0
        pen_save_points = (0.1 * goals_conceded) * 0.16 * draftkings_points_dict[
            'Penalties Saved'] if 'Penalties Saved' in draftkings_points_dict else 0
        if 'Win' in draftkings_points_dict:
            win_perc = fix_score_pred[fix_score_pred['Home Team'] == team]['Home Win %'].values[0] if team in \
                                                                                                      fix_score_pred[
                                                                                                          'Home Team'].values else \
            fix_score_pred[fix_score_pred['Away Team'] == team]['Away Win %'].values[0]
            win_points = (float(win_perc.replace('%', '')) / 100) * draftkings_points_dict['Win']
        else:
            win_points = 0
        draftkings_points = goal_points + assists + shots_on_target + shots_total + crosses_total + tackles_won + key_passes + accurate_passes + fouls_drawn + fouls_committed + yellow_cards + interceptions + saves + goal_conceded_points + clean_sheet_points + pen_save_points + win_points
        draftkings_points_df['PTS'].append(draftkings_points)
    draftkings_points_df = pd.DataFrame(draftkings_points_df)
    draftkings_points_df.sort_values(by='PTS', ascending=False, inplace=True)
    draftkings_points_df['PTS'] = draftkings_points_df['PTS'].round(2)
    draftkings_points_df.reset_index(drop=True, inplace=True)
    return draftkings_points_df


def get_fanteam_points(pl_projections, score_preds, fanteam_points_dict_gk, fanteam_points_dict_def,
                       fanteam_points_dict_mid, fanteam_points_dict_fwd):
    import pandas as pd
    from scipy.stats import poisson
    fanteam_points_df = {'fixture_id': [], 'kickoff_datetime': [], 'player_id': [], 'Player': [], 'Position': [],
                         'Team': [], 'Opponent': [], 'Venue': [], 'FanTeam Points': []}
    fanteam_points_df['fixture_id'] = pl_projections['fixture_id'].tolist()
    fanteam_points_df['kickoff_datetime'] = pl_projections['kickoff_datetime'].tolist()
    fanteam_points_df['player_id'] = pl_projections['player_id'].tolist()
    fanteam_points_df['Player'] = pl_projections['Player'].tolist()
    fanteam_points_df['Position'] = pl_projections['FanTeam Position'].tolist()
    fanteam_points_df['Team'] = pl_projections['Team'].tolist()
    fanteam_points_df['Opponent'] = pl_projections['Opponent'].tolist()
    fanteam_points_df['Venue'] = pl_projections['Venue'].tolist()
    for i in range(len(pl_projections)):
        fixture_id = pl_projections['fixture_id'][i]
        fix_score_pred = score_preds[score_preds['id'] == fixture_id]
        position = pl_projections['FanTeam Position'][i]
        if position == 'GK':
            fanteam_points_dict = fanteam_points_dict_gk
        elif position == 'DEF':
            fanteam_points_dict = fanteam_points_dict_def
        elif position == 'MID':
            fanteam_points_dict = fanteam_points_dict_mid
        elif position == 'FWD':
            fanteam_points_dict = fanteam_points_dict_fwd
        else:
            fanteam_points_df['FanTeam Points'].append(0)
            continue
        team = pl_projections['Team'][i]
        goal_points = pl_projections['Goals'][i] * fanteam_points_dict['Goals']
        assists = pl_projections['Assists'][i] * fanteam_points_dict['Assists']
        shots_on_target = pl_projections['Shots On Target'][i] * fanteam_points_dict['Shots On Target']
        yellow_cards = pl_projections['Yellow Cards'][i] * fanteam_points_dict['Yellow Card']
        saves = pl_projections['Saves'][i] * fanteam_points_dict['Saves'] if position == 'GK' else 0
        full_match_perc = pl_projections['Full Match Hit Rate'][i]
        full_match_points = full_match_perc * fanteam_points_dict[
            'Full Match'] if 'Full Match' in fanteam_points_dict else 0
        if 'Goals Conceded' in fanteam_points_dict:
            # Get goals conceded from fix_score_pred
            goals_conceded = fix_score_pred[fix_score_pred['Home Team'] == team]['Away Goals'].values[0] if team in \
                                                                                                            fix_score_pred[
                                                                                                                'Home Team'].values else \
            fix_score_pred[fix_score_pred['Away Team'] == team]['Home Goals'].values[0]
            goal_conceded_points = (poisson.pmf(2, goals_conceded) + poisson.pmf(3, goals_conceded) + (
                        (poisson.pmf(4, goals_conceded) + poisson.pmf(5, goals_conceded)) * 2) + (
                                                poisson.pmf(6, goals_conceded) + poisson.pmf(7, goals_conceded)) * 3) * \
                                   fanteam_points_dict['Goals Conceded'] if goals_conceded > 0 else 0
        else:
            goal_conceded_points = 0
        if 'Clean Sheet' in fanteam_points_dict:
            # Get clean sheet percentage from fix_score_pred
            clean_sheet_perc = fix_score_pred[fix_score_pred['Home Team'] == team]['Home Clean Sheet %'].values[
                0] if team in fix_score_pred['Home Team'].values else \
            fix_score_pred[fix_score_pred['Away Team'] == team]['Away Clean Sheet %'].values[0]
            clean_sheet_points = (float(clean_sheet_perc.replace('%', '')) / 100) * fanteam_points_dict['Clean Sheet']
        else:
            clean_sheet_points = 0
        if 'Penalties Saved' in fanteam_points_dict:
            pen_save_points = (0.1 * goals_conceded) * 0.16 * fanteam_points_dict['Penalties Saved']
        else:
            pen_save_points = 0
        if 'Win' in fanteam_points_dict:
            win_perc = fix_score_pred[fix_score_pred['Home Team'] == team]['Home Win %'].values[0] if team in \
                                                                                                      fix_score_pred[
                                                                                                          'Home Team'].values else \
            fix_score_pred[fix_score_pred['Away Team'] == team]['Away Win %'].values[0]
            win_points = (float(win_perc.replace('%', '')) / 100) * fanteam_points_dict['Win']
        else:
            win_points = 0
        if 'Lose' in fanteam_points_dict:
            lose_perc = fix_score_pred[fix_score_pred['Home Team'] == team]['Away Win %'].values[0] if team in \
                                                                                                       fix_score_pred[
                                                                                                           'Home Team'].values else \
            fix_score_pred[fix_score_pred['Away Team'] == team]['Home Win %'].values[0]
            lose_points = (float(lose_perc.replace('%', '')) / 100) * fanteam_points_dict['Lose']
        fanteam_points = goal_points + assists + shots_on_target + yellow_cards + saves + clean_sheet_points + goal_conceded_points + pen_save_points + win_points + lose_points + full_match_points + 2
        fanteam_points_df['FanTeam Points'].append(fanteam_points)
    fanteam_points_df = pd.DataFrame(fanteam_points_df)
    fanteam_points_df.sort_values(by='FanTeam Points', ascending=False, inplace=True)
    fanteam_points_df['FanTeam Points'] = fanteam_points_df['FanTeam Points'].round(2)
    fanteam_points_df.reset_index(drop=True, inplace=True)
    return fanteam_points_df


def get_opta_points(pl_projections, score_preds, opta_points_dict):
    import pandas as pd
    opta_points_df = {'fixture_id': [], 'kickoff_datetime': [], 'player_id': [], 'Player': [], 'Position': [],
                      'Team': [], 'Opponent': [], 'Venue': [], 'PTS': [], 'Floor PTS': []}
    opta_points_df['fixture_id'] = pl_projections['fixture_id'].tolist()
    opta_points_df['kickoff_datetime'] = pl_projections['kickoff_datetime'].tolist()
    opta_points_df['player_id'] = pl_projections['player_id'].tolist()
    opta_points_df['Player'] = pl_projections['Player'].tolist()
    opta_points_df['Position'] = pl_projections['Position'].tolist()
    opta_points_df['Team'] = pl_projections['Team'].tolist()
    opta_points_df['Opponent'] = pl_projections['Opponent'].tolist()
    opta_points_df['Venue'] = pl_projections['Venue'].tolist()

    for i in range(len(pl_projections)):
        fixture_id = pl_projections['fixture_id'][i]
        fix_score_pred = score_preds[score_preds['id'] == fixture_id]
        team = pl_projections['Team'][i]
        position = pl_projections['Position'][i]
        goals = pl_projections['Goals'][i] * opta_points_dict['Goals']
        assists = pl_projections['Assists'][i] * opta_points_dict['Assists']
        shots_off = (pl_projections['Shots Total'][i] - pl_projections['Shots On Target'][i]) * opta_points_dict[
            'Shots Off']
        shots_on_target = pl_projections['Shots On Target'][i] * opta_points_dict['Shots On Target']
        passes = pl_projections['Passes'][i] * opta_points_dict['Passes']
        interceptions = pl_projections['Interceptions'][i] * opta_points_dict['Interceptions']
        tackles = pl_projections['Tackles'][i] * opta_points_dict['Tackles']
        crosses = pl_projections['Total Crosses'][i] * opta_points_dict['Total Crosses']
        yellow_cards = pl_projections['Yellow Cards'][i] * opta_points_dict['Yellow Cards']
        fouls = pl_projections['Fouls'][i] * opta_points_dict['Fouls']
        fouls_drawn = pl_projections['Fouls Drawn'][i] * opta_points_dict['Fouls Drawn']
        saves = pl_projections['Saves'][i] * opta_points_dict['Saves']
        offsides = pl_projections['Offsides'][i] * opta_points_dict['Offsides']
        blocked_shots = pl_projections['Blocked Shots Average'][i] * opta_points_dict['Blocked Shots']
        goals_conceded = fix_score_pred[fix_score_pred['Home Team'] == team]['Away Goals'].values[0] if team in \
                                                                                                        fix_score_pred[
                                                                                                            'Home Team'].values else \
        fix_score_pred[fix_score_pred['Away Team'] == team]['Home Goals'].values[0]
        goals_conceded_points = goals_conceded * opta_points_dict['Goals Conceded']
        pen_save_points = (0.1 * goals_conceded) * 0.16 * opta_points_dict['Penalties Saved'] if position == 'GK' else 0
        if position == 'GK':
            goals_conceded_points = goals_conceded_points * 6
        points = goals + assists + shots_off + shots_on_target + passes + interceptions + tackles + crosses + yellow_cards + fouls + fouls_drawn + saves + offsides + blocked_shots + goals_conceded_points + pen_save_points
        floor_points = shots_off + shots_on_target + passes + interceptions + tackles + crosses + yellow_cards + fouls + fouls_drawn + saves + offsides + blocked_shots + goals_conceded_points + pen_save_points
        opta_points_df['PTS'].append(points)
        opta_points_df['Floor PTS'].append(floor_points)
    opta_points = pd.DataFrame(opta_points_df)
    opta_points.sort_values(by='PTS', ascending=False, inplace=True)
    opta_points = opta_points.round(2)
    opta_points.reset_index(drop=True, inplace=True)
    return opta_points