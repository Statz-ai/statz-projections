"""Monte Carlo bonus: simulate the fixture, rank BPS, award 3/2/1.

Spec: statz `docs/fpl-bps-rebuild-spec.md`.

Replaces `get_bonus_points`, which shares out a pool sized
`0.5 x count(expected BPS >= 7.5)` via `exp(0.1 x BPS)`. Measured on prod that
pool averages **3.44 per fixture against FPL's 6**, and it hands a full unit of
softmax weight to a player with zero expected BPS — 855 rows with xMins = 0
currently receive bonus.

The reason a softmax over *expected* BPS cannot be patched into correctness:
bonus is a RANK statistic. Halving a player's stat line models "half as good
for 90 minutes", not "50% chance of a normal game" — for a rank contest those
are completely different. So we sample instead.

Per sample, per fixture:
  1. draw each team's goals conceded ONCE and share it across that side —
     clean sheets are perfectly correlated within a team and worth 12 BPS to
     every GK/DEF
  2. draw each player's minutes band from p_play / p60 / p90
  3. draw his actions conditional on those minutes
  4. score with fpl_bps.score on the drawn integers — every threshold
     (floor(cbi/3), the 30-attempt pass gate, appearance bands) is then
     evaluated on a realised value rather than a mean, which is what the old
     hardcoded pass-completion ladder existed to fudge
  5. rank, award 3/2/1 with FPL's tie rules

The pool problem disappears by construction: every sample awards 6 (slightly
more on ties, which is correct — a tie for first pays 3+3+1).
"""
import logging

import numpy as np

from app.services import fpl_bps

logger = logging.getLogger("projection")

DEFAULT_SAMPLES = 5000

# Minutes used for each band. Matches bands_to_xmins (xminutes.py) so the
# simulator's average minutes reconcile with the xMins we publish.
BAND_MINUTES = {"none": 0.0, "short": 30.0, "long": 75.0, "full": 90.0}

# Attempted-count stats are overdispersed relative to Poisson. Measured on
# Bruno's last 80 matches, pass counts run variance/mean = 4.65, and 4.15 even
# within the 80+ minute band — so minutes explain only a little of it. Poisson
# would understate the spread badly for the high-volume stats, which is where
# the 30-attempt pass-completion gate lives.
# Only the high-volume pass counts are overdispersed enough to model. The
# defensive actions are LOW counts (a defender makes ~9 CBI, ~3 recoveries, ~2
# tackles won a match), so Poisson is close enough and the absolute error is
# small: at a mean of 9, measured dispersion 1.27 widens the spread from 3.0 to
# 3.4, which is ~0.1 BPS once floored by the 1-per-3 divisor.
#
# cbi and recoveries previously sat at 2.0 — a guess with nothing behind it,
# against a MEASURED 1.27 for CBIT. That drew them ~25% wider than reality,
# which matters because bonus is a RANK: over-wide tails let a mediocre
# defender roll a freak line and take a bonus place off a better one.
#
# Poisson also matches the DefCon threshold, which runs _TD_DISPERSION = 1.0 by
# the same call — one spread model for the same actions. George, 2026-08-05.
DISPERSION = {
    "passes": 4.15,
    "accurate_passes": 4.15,
}
DEFAULT_DISPERSION = 1.0  # Poisson

# Stats drawn per player from a per-90 rate. Key is the fpl_bps.score kwarg.
COUNT_STATS = (
    "goals", "assists", "key_passes", "big_chances_created", "big_chances_missed",
    "shots", "shots_on_target", "cbi", "recoveries", "tackles_won", "dribbles",
    "fouls", "fouls_drawn", "offsides", "yellow_cards", "saves", "passes",
)


def _draw_counts(rng, lam, n, dispersion=DEFAULT_DISPERSION):
    """n draws from Poisson(lam), or a negative binomial with the same mean
    when overdispersed. lam may be a vector (one entry per sample)."""
    lam = np.asarray(lam, dtype=float)
    lam = np.where(np.isfinite(lam) & (lam > 0), lam, 0.0)
    if dispersion <= 1.0001:
        return rng.poisson(lam)
    # var = mean * dispersion  ->  r = mean/(d-1), p = r/(r+mean)
    r = np.maximum(lam / (dispersion - 1.0), 1e-9)
    p = r / (r + np.maximum(lam, 1e-12))
    out = rng.negative_binomial(r, p)
    return np.where(lam > 0, out, 0)


def _draw_bands(rng, p_play, p60, p90, n):
    """Minutes band per sample. Returns (minutes, played_mask, over60_mask).

    Bands are nested: p90 <= p60 <= p_play, enforced here because a dialled
    value can violate it.
    """
    p_play = float(np.clip(p_play, 0.0, 1.0))
    p60 = float(np.clip(min(p60, p_play), 0.0, 1.0))
    p90 = float(np.clip(min(p90, p60), 0.0, 1.0))
    u = rng.random(n)
    minutes = np.zeros(n)
    minutes = np.where(u < p90, BAND_MINUTES["full"], minutes)
    minutes = np.where((u >= p90) & (u < p60), BAND_MINUTES["long"], minutes)
    minutes = np.where((u >= p60) & (u < p_play), BAND_MINUTES["short"], minutes)
    return minutes, minutes > 0, minutes > 60


def _award_bonus(bps_matrix):
    """3/2/1 per sample with FPL's tie rules, vectorised over samples.

    COMPETITION RANKING (1-2-2-4 style): rank = 1 + how many players scored
    strictly higher; 3 points for rank 1, 2 for rank 2, 1 for rank 3.

    That single rule reproduces all three published tie cases with no special
    casing, which an earlier "award the 1st/2nd/3rd distinct score" version did
    not — it paid 40/40/30/10 as 3/3/2/1 where FPL pays 3/3/1/0.

        40 30 20 10  -> ranks 1 2 3 4  -> 3 2 1 0
        40 40 30 10  -> ranks 1 1 3 4  -> 3 3 1 0   (tie 1st)
        40 30 30 10  -> ranks 1 2 2 4  -> 3 2 2 0   (tie 2nd)
        40 30 20 20  -> ranks 1 2 3 3  -> 3 2 1 1   (tie 3rd)
    """
    # strictly-greater count per player, per sample
    greater = (bps_matrix[:, None, :] > bps_matrix[:, :, None]).sum(axis=2)
    rank = greater + 1
    return np.where(rank == 1, 3.0, np.where(rank == 2, 2.0, np.where(rank == 3, 1.0, 0.0)))


def simulate_fixture(players, team_goals_against, n_samples=DEFAULT_SAMPLES, seed=0):
    """Expected bonus per player for one fixture.

    players: list of dicts —
        player_id, position (fpl_bps.GK/DEF/MID/FWD), team,
        p_play / p60 / p90, and per-90 rates keyed as in COUNT_STATS.
    team_goals_against: {team: expected goals conceded over 90}
    seed: derived from fixture_id by the caller. Deterministic on purpose —
        an unseeded simulator would jitter every player's bonus on every run
        for no reason, which users would read as the model changing its mind.

    Returns {player_id: expected_bonus}.
    """
    if not players:
        return {}
    rng = np.random.default_rng(seed)
    n = int(n_samples)

    # Team goals conceded: ONE draw per team per sample, shared by that side.
    # Independent per-player draws would let one defender keep a clean sheet
    # while his centre-back partner concedes, which is impossible and would
    # flatten exactly the correlated hauls that win bonus.
    team_conceded = {
        team: rng.poisson(max(float(ga), 0.0), n)
        for team, ga in team_goals_against.items()
    }

    bps = np.zeros((n, len(players)))
    for idx, p in enumerate(players):
        minutes, played, over60 = _draw_bands(
            rng, p.get("p_play", 1.0), p.get("p60", 1.0), p.get("p90", 0.5), n
        )
        scale = minutes / 90.0
        drawn = {}
        for stat in COUNT_STATS:
            rate = float(p.get(stat, 0.0) or 0.0)
            if rate <= 0:
                drawn[stat] = np.zeros(n, dtype=int)
                continue
            drawn[stat] = _draw_counts(
                rng, rate * scale, n, DISPERSION.get(stat, DEFAULT_DISPERSION)
            )
        # Accurate passes are a binomial on the drawn attempts, not their own
        # independent draw — otherwise completion% could exceed 100.
        comp = float(np.clip(p.get("pass_completion", 0.8), 0.0, 1.0))
        drawn["accurate_passes"] = rng.binomial(drawn["passes"], comp)

        conceded_full = team_conceded.get(p.get("team"), np.zeros(n, dtype=int))
        # Only concedes while on the pitch — the same exposure rule as the
        # points path, applied here to a drawn value.
        conceded = rng.binomial(conceded_full, np.clip(scale, 0.0, 1.0))
        clean_sheet = ((conceded == 0) & over60).astype(int)

        for s in range(n):
            if not played[s]:
                continue
            bps[s, idx] = fpl_bps.score(
                {
                    "minutes": float(minutes[s]),
                    "clean_sheet": int(clean_sheet[s]),
                    "goals_conceded": int(conceded[s]),
                    **{k: int(v[s]) for k, v in drawn.items()},
                },
                p.get("position"),
            )

    bonus = _award_bonus(bps)
    return {
        p["player_id"]: float(bonus[:, i].mean())
        for i, p in enumerate(players)
    }


# ---------------------------------------------------------------------------
#  Adapter: FPL scoring frame -> simulator inputs
# ---------------------------------------------------------------------------

# frame column -> simulator stat key. Values are read from the "{col} per90"
# companion (xminutes.PER90_SUFFIX), NOT the column itself: the frame column is
# already at expected minutes, and the simulator needs the rate BEFORE the
# minutes term so it can scale by each drawn band.
FRAME_TO_SIM = {
    "Goals": "goals",
    "Assists": "assists",
    "Key Passes": "key_passes",
    "Big Chances Created": "big_chances_created",
    "Big Chances Missed": "big_chances_missed",
    "Shots Total": "shots",
    "Shots On Target": "shots_on_target",
    # Team-down PROJECTIONS, not the "* Average" career means these used to
    # read. The Averages are identical in every fixture, so a defender's bonus
    # never moved with the opponent even though his DefCon did.
    "Ball Recovery": "recoveries",
    "Tackles Won": "tackles_won",
    "Successful Dribbles": "dribbles",
    "Fouls": "fouls",
    "Fouls Drawn": "fouls_drawn",
    "Offsides": "offsides",
    "Yellow Cards": "yellow_cards",
    "Saves": "saves",
    "Passes": "passes",
}

# CBI has no single frame column — Sportmonks tracks the three components
# separately, and bonus_points_score sums them the same way.
# Projected team-down CBI, replacing the sum of three career-average columns.
# Falls back to the old parts when the FPL-only combined column is absent.
CBI_PROJECTED = "Clearances Blocks Interceptions (FPL)"
CBI_PARTS = ("Clearances Average", "Blocked Shots Average", "Interceptions")

POSITION_CODES = {"GK": fpl_bps.GK, "DEF": fpl_bps.DEF, "MID": fpl_bps.MID, "FWD": fpl_bps.FWD}

PER90 = " per90"


_MISSING_PER90_WARNED = set()


def _rate(row, col):
    """Per-90 rate for a frame column.

    Reads the "{col} per90" companion. There is deliberately NO fallback to the
    un-suffixed column: that value is lambda at EXPECTED minutes, so using it as
    a per-90 rate understates every player — badly for subs — and does so
    silently, which is worse than returning nothing. Missing -> 0.0 plus a
    one-off warning naming the column.
    """
    key = col + PER90
    if key in row and row[key] is not None:
        try:
            v = float(row[key])
            if v == v:  # not NaN
                return max(v, 0.0)
        except (TypeError, ValueError):
            pass
    elif col not in _MISSING_PER90_WARNED:
        _MISSING_PER90_WARNED.add(col)
        logger.warning(
            "[bonus-sim] no '%s' column — that stat contributes 0 to BPS. "
            "Check it is stamped by apply_per90_scaling and present in BUNDLE_COLS.",
            key,
        )
    return 0.0


def _goals_against(score_preds):
    """{fixture_id: {team: expected goals conceded}} — a team concedes what the
    OTHER side is projected to score."""
    out = {}
    for r in score_preds.to_dict("records"):
        fid = r.get("id")
        if fid is None:
            continue
        home, away = r.get("Home Team"), r.get("Away Team")
        try:
            hg, ag = float(r.get("Home Goals") or 0), float(r.get("Away Goals") or 0)
        except (TypeError, ValueError):
            continue
        out[int(fid)] = {home: ag, away: hg}
    return out


def simulate_bonus_for_frame(frame, score_preds, n_samples=DEFAULT_SAMPLES):
    """Expected bonus per (fixture_id, player_id) for a whole FPL frame.

    Drop-in for bonus_points_score + get_bonus_points_by_fixture. Returns a
    DataFrame [fixture_id, player_id, 'Bonus Points'].

    Seeded per fixture_id, so the same inputs always give the same bonus —
    an unseeded simulator would jitter every player's number on every run.
    """
    import pandas as pd

    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=["fixture_id", "player_id", "Bonus Points"])

    ga_by_fix = _goals_against(score_preds)
    rows = []
    for fid, grp in frame.groupby("fixture_id"):
        try:
            fid_int = int(fid)
        except (TypeError, ValueError):
            continue
        players = []
        for row in grp.to_dict("records"):
            pos = POSITION_CODES.get(row.get("FPL Position"))
            if pos is None:
                continue  # unmapped position can't be scored
            p = {
                "player_id": row.get("player_id"),
                "position": pos,
                "team": row.get("Team"),
                "p_play": float(row.get("xmin_p_play") or 0.0),
                "p60": float(row.get("xmin_p60") or 0.0),
                "p90": float(row.get("xmin_p90") or 0.0),
            }
            for col, key in FRAME_TO_SIM.items():
                p[key] = _rate(row, col)
            _cbi_proj = _rate(row, CBI_PROJECTED) if (CBI_PROJECTED + PER90) in row else 0.0
            p["cbi"] = _cbi_proj if _cbi_proj > 0 else sum(_rate(row, c) for c in CBI_PARTS)
            passes = p.get("passes", 0.0)
            acc = _rate(row, "Accurate Passes")
            p["pass_completion"] = min(acc / passes, 1.0) if passes > 0 else 0.8
            players.append(p)

        if not players:
            continue
        bonus = simulate_fixture(
            players, ga_by_fix.get(fid_int, {}), n_samples=n_samples, seed=fid_int,
        )
        for pid, val in bonus.items():
            rows.append({"fixture_id": fid_int, "player_id": pid, "Bonus Points": val})

    df = pd.DataFrame(rows, columns=["fixture_id", "player_id", "Bonus Points"])
    if len(df):
        logger.info(
            "[bonus-sim] %d fixtures, %d player rows, mean pool %.2f/fixture",
            df["fixture_id"].nunique(), len(df),
            df.groupby("fixture_id")["Bonus Points"].sum().mean(),
        )
    return df
