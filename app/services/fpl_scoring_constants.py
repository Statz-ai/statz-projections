"""FPL scoring dictionaries — verbatim copies of the literals in
projection_service.py / projection_all_teams_service.py (~L1959 / ~L1481).

Used by the instant-recalc path (fpl_recalc_service) so it scores with
exactly the run's rules WITHOUT touching the live scoring blocks. If the
scoring rules change in the services, change them here too — a drift makes
recalc'd points disagree with the next full run (the daily digest's
all-zero/consistency checks won't catch a constant drift, so treat these
as one unit in review)."""

FPL_POINTS_GK = {'Goals': 10, 'Assists': 3, 'Clean Sheet': 4, 'Saves': 1, 'Penalties Saved': 5, 'Goals Conceded': -1, 'Yellow Card': -1}
FPL_POINTS_DEF = {'Goals': 6, 'Assists': 3, 'Clean Sheet': 4, 'Goals Conceded': -1, 'Yellow Card': -1}
FPL_POINTS_MID = {'Goals': 5, 'Assists': 3, 'Clean Sheet': 1, 'Yellow Card': -1}
FPL_POINTS_FWD = {'Goals': 4, 'Assists': 3, 'Yellow Card': -1}

FPL_BONUS_GK = {'Goals': 12, 'Winning Goal': 3, 'Assists': 9, 'Clean Sheet': 12, 'Saves': 2.85, 'Penalties Saved': 7, 'Key Passes': 1, 'Big Chances Created': 3, 'Successful Dribbles': 1, 'Clearance Offline': 9, 'Big Chances Missed': -3, 'Clearances, Blocks & Interceptions': 0.333, 'Recoveries': 0.33, 'Tackles Won': 2, 'Fouls Drawn': 1, 'Shots On Target': 2, 'Shots Off Target': -1, 'Offsides': -1, 'Fouls': -1, '70-79% Passes Completed': 2, '80-89% Passes Completed': 4, '90%+ Passes Completed': 6, 'Goals Conceded': -4, 'Yellow Card': -3}
FPL_BONUS_DEF = {'Goals': 12, 'Winning Goal': 3, 'Assists': 9, 'Clean Sheet': 12, 'Clearances, Blocks & Interceptions': 0.333, 'Recoveries': 0.33, 'Tackles Won': 2, 'Fouls Drawn': 1, 'Shots On Target': 2, 'Shots Off Target': -1, 'Offsides': -1, 'Fouls': -1, '70-79% Passes Completed': 2, '80-89% Passes Completed': 4, '90%+ Passes Completed': 6, 'Key Passes': 1, 'Big Chances Created': 3, 'Successful Dribbles': 1, 'Clearance Offline': 9, 'Big Chances Missed': -3, 'Goals Conceded': -4, 'Yellow Card': -3}
FPL_BONUS_MID = {'Goals': 18, 'Winning Goal': 3, 'Assists': 9, 'Clearances, Blocks & Interceptions': 0.333, 'Recoveries': 0.33, 'Tackles Won': 2, 'Fouls Drawn': 1, 'Shots On Target': 2, 'Shots Off Target': -1, 'Offsides': -1, 'Fouls': -1, '70-79% Passes Completed': 2, '80-89% Passes Completed': 4, '90%+ Passes Completed': 6, 'Key Passes': 1, 'Big Chances Created': 3, 'Successful Dribbles': 1, 'Clearance Offline': 9, 'Big Chances Missed': -3, 'Yellow Card': -3}
FPL_BONUS_FWD = {'Goals': 24, 'Winning Goal': 3, 'Assists': 9, 'Key Passes': 1, 'Big Chances Created': 3, 'Successful Dribbles': 1, 'Clearance Offline': 9, 'Big Chances Missed': -3, 'Clearances, Blocks & Interceptions': 0.333, 'Recoveries': 0.33, 'Tackles Won': 2, 'Fouls Drawn': 1, 'Shots On Target': 2, 'Shots Off Target': -1, 'Offsides': -1, 'Fouls': -1, '70-79% Passes Completed': 2, '80-89% Passes Completed': 4, '90%+ Passes Completed': 6, 'Yellow Card': -3}
