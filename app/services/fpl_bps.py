"""FPL Bonus Points System — scoring a REALISED stat line.

Spec: statz repo `docs/fpl-bps-rebuild-spec.md` (George, 2026-08-02), which
records the action-by-action decisions and the published 26/27 BPS table this
implements.

The contract that matters: everything here takes **realised integers**, not
expectations. `floor(cbi / 3)`, the 30-attempt pass-completion gate and the
minutes bands are step functions, and evaluating a step function at a mean is
the error that produced the old hardcoded pass-completion ladder. The caller
(the fixture simulator) draws a minutes band, draws the actions conditional on
those minutes, and passes the drawn numbers in here.

NOT wired into the pipeline yet — `bonus_points_score` in statz_functions.py
is still the live path.
"""

# Positions, as FPL's element_type
GK, DEF, MID, FWD = 1, 2, 3, 4

# --- published 26/27 BPS table ---------------------------------------------
APPEARANCE_SHORT = 3          # played 1-60 minutes
APPEARANCE_LONG = 6           # played over 60 minutes
GOAL = {GK: 12, DEF: 12, MID: 18, FWD: 24}   # non-penalty
GOAL_FROM_PENALTY = 12        # flat, all positions — PARKED (see spec)
ASSIST = 9
CLEAN_SHEET = 12              # GK/DEF only
GOAL_CONCEDED = -4            # GK/DEF only, per goal
SAVE_IN_BOX = 3
SAVE_OTHER = 2
SAVE_BIG_CHANCE = 1           # no stat — folded into the saves coefficient
PENALTY_SAVE = 7              # DROPPED as noise (~0.14 BPS)
CBI_PER = 3                   # 1 BPS per 3
RECOVERIES_PER = 3            # 1 BPS per 3
KEY_PASS = 1
BIG_CHANCE_CREATED = 3
SUCCESSFUL_TACKLE = 2
SUCCESSFUL_DRIBBLE = 1
SUCCESSFUL_CROSS = 1          # EXCLUDED — we hold total crosses, not successful
WINNING_GOAL = 3
GOAL_LINE_CLEARANCE = 9       # DROPPED as noise
FOUL_WON = 1
SHOT_ON_TARGET = 2
SHOT_OFF_TARGET = -1
PASS_COMPLETION_MIN_ATTEMPTS = 30
PASS_COMPLETION = ((0.90, 6), (0.80, 4), (0.70, 2))   # highest band first
PENALTY_CONCEDED = -3         # DROPPED as noise
PENALTY_MISSED = -6           # PARKED with the penalties work
YELLOW_CARD = -3
RED_CARD = -9                 # DROPPED as noise
OWN_GOAL = -6                 # DROPPED as noise
BIG_CHANCE_MISSED = -3
ERROR_LEADING_TO_GOAL = -3    # DROPPED as noise
ERROR_LEADING_TO_SHOT = -1    # DROPPED as noise
FOUL_CONCEDED = -1
OFFSIDE = -1


def appearance_bps(minutes):
    """3 BPS for 1-60 minutes, 6 for over 60. There is NO extra band for the
    full 90 — the top band is 60+, so p90 never enters this term."""
    if minutes <= 0:
        return 0
    return APPEARANCE_LONG if minutes > 60 else APPEARANCE_SHORT


def pass_completion_bps(passes, accurate_passes):
    """2 / 4 / 6 at 70-79% / 80-89% / 90%+, on 30+ attempted passes.

    Replaces the 11-branch hardcoded ladder in bonus_points_score, which
    started at 20 attempts, capped at 5 where the real rule caps at 6, and
    scored ZERO for anyone landing between 90 and 100 passes.
    """
    if passes < PASS_COMPLETION_MIN_ATTEMPTS or passes <= 0:
        return 0
    pct = accurate_passes / passes
    for threshold, value in PASS_COMPLETION:
        if pct >= threshold:
            return value
    return 0


def score(stats, position, *, include_dropped=False):
    """BPS for one realised player-match.

    `stats` keys (all counts, all optional and defaulting to 0):
        minutes, goals, assists, clean_sheet, goals_conceded,
        saves, saves_in_box, key_passes, big_chances_created,
        big_chances_missed, cbi, recoveries, tackles_won,
        dribbles, shots, shots_on_target, passes, accurate_passes,
        fouls, fouls_drawn, offsides, yellow_cards, winning_goal

    `include_dropped` additionally scores the actions the spec drops as noise
    (penalty saves, red cards, own goals, penalties missed) — used by the
    validation harness to size what dropping them costs, never in production.
    Requires: pens_saved, red_cards, own_goals, pens_missed.
    """
    g = stats.get
    minutes = g('minutes', 0)
    if minutes <= 0:
        return 0

    total = appearance_bps(minutes)

    # Attacking
    total += g('goals', 0) * GOAL.get(position, 0)
    total += g('assists', 0) * ASSIST
    total += g('key_passes', 0) * KEY_PASS
    total += g('big_chances_created', 0) * BIG_CHANCE_CREATED
    total += g('big_chances_missed', 0) * BIG_CHANCE_MISSED
    total += g('shots_on_target', 0) * SHOT_ON_TARGET
    total += max(0, g('shots', 0) - g('shots_on_target', 0)) * SHOT_OFF_TARGET
    total += g('dribbles', 0) * SUCCESSFUL_DRIBBLE
    total += g('winning_goal', 0) * WINNING_GOAL

    # Defending
    total += (g('cbi', 0) // CBI_PER)
    total += (g('recoveries', 0) // RECOVERIES_PER)
    total += g('tackles_won', 0) * SUCCESSFUL_TACKLE
    if position in (GK, DEF):
        total += g('clean_sheet', 0) * CLEAN_SHEET
        total += g('goals_conceded', 0) * GOAL_CONCEDED

    # Goalkeeping. Saves split in-box (3) vs other (2); big-chance saves have
    # no stat and are absorbed by the projection-side coefficient.
    if position == GK:
        in_box = g('saves_in_box', 0)
        total += in_box * SAVE_IN_BOX
        total += max(0, g('saves', 0) - in_box) * SAVE_OTHER

    # Passing
    total += pass_completion_bps(g('passes', 0), g('accurate_passes', 0))

    # Discipline
    total += g('yellow_cards', 0) * YELLOW_CARD
    total += g('fouls', 0) * FOUL_CONCEDED
    total += g('fouls_drawn', 0) * FOUL_WON
    total += g('offsides', 0) * OFFSIDE

    if include_dropped:
        total += g('pens_saved', 0) * PENALTY_SAVE
        total += g('red_cards', 0) * RED_CARD
        total += g('own_goals', 0) * OWN_GOAL
        total += g('pens_missed', 0) * PENALTY_MISSED

    return total
