"""Expected-minutes (xMinutes) model for FPL fantasy points.

Why this exists (George, 2026-07-24): FPL points assumed every projected
player starts and plays ~90 — the per-start rates upstream drop every game
under ~50 minutes (get_player_stats), get_fpl_points awarded a flat +2
appearance and full clean-sheet weight. A rotation forward whose recent
record is mostly cameos (Osula: 7 starts + 7 subs of 2-25 min across his
last 30) projected like a nailed starter.

Scope: FPL fantasy scoring ONLY. player_projections / prop pipelines /
other fantasy scorers (Opta, FanTeam, Dream11) are untouched — betting
props are deliberately priced "if he plays".

Model — per player, over the club's last XMIN_WINDOW fixtures (crossing the
season boundary; same domestic/cup competition scope as the rate sample):

  p_play    P(features at all)      — 0-minute club fixtures count against
  p60       P(plays 60+)            — FPL threshold for 2 appearance pts + CS
  xmin      expected minutes per club fixture (incl. DNPs)
  xmin_start_sample
            weighted avg minutes in his >XMIN_START_MIN-minute games — the
            sample the existing attacking rates were computed from, so
            exposure = xmin / xmin_start_sample is the correct rate scaler
            (rates are per-qualifying-game, NOT per-90).

Cross-club history fills the window for movers (weight x XMIN_CROSS_CLUB_WEIGHT,
mirroring get_weighted_player_stats). Sparse histories blend toward neutral
per-position priors. All constants are env-tunable — pick by judgment,
tune live.
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger("projection")

# Master switch. FPL_XMINUTES=0 -> callers skip stamping entirely and the
# scoring functions see no xmin columns, reproducing pre-xMinutes output
# byte-for-byte. Instant rollback = env change + container restart.
XMIN_ENABLED = os.getenv("FPL_XMINUTES", "1") not in ("0", "false", "False")

# Weekly recency decay — same shape as the existing rate weighting
# (get_weighted_player_stats: weight ** (weeks - 3), floor 1 inside 4 weeks).
XMIN_DECAY = float(os.getenv("XMIN_DECAY", "0.96"))
# Club fixtures in the window ("last 30 appearances" per George's brief).
XMIN_WINDOW = int(os.getenv("XMIN_WINDOW", "30"))
# Weight multiplier on games for a previous club (mirrors the x0.5 in
# get_weighted_player_stats line ~1649).
XMIN_CROSS_CLUB_WEIGHT = float(os.getenv("XMIN_CROSS_CLUB_WEIGHT", "0.5"))
# Below this many appearances in the window, blend toward neutral priors
# (alpha = n_apps / XMIN_PRIOR_APPS, capped at 1).
XMIN_PRIOR_APPS = int(os.getenv("XMIN_PRIOR_APPS", "5"))
# The per-start rate sample keeps games with minutes > 50 (get_player_stats
# mins=50) — exposure must be measured against that same population.
XMIN_START_MIN = int(os.getenv("XMIN_START_MIN", "50"))
# Fallback average start length when a player has no >50-min games at all.
XMIN_DEFAULT_START_MINUTES = float(os.getenv("XMIN_DEFAULT_START_MINUTES", "82"))
# Sub cameo length used when deriving a prior xmin from the priors below.
XMIN_PRIOR_SUB_MINUTES = float(os.getenv("XMIN_PRIOR_SUB_MINUTES", "25"))

# (p_play, p60) neutral priors by FPL position for players with little or no
# usable history. Deliberately conservative — an unknown quantity should not
# out-project a known nailed starter.
XMIN_NEUTRAL_PRIORS = {
    "GK": (0.55, 0.55),
    "DEF": (0.60, 0.50),
    "MID": (0.60, 0.40),
    "FWD": (0.60, 0.40),
}
XMIN_DEFAULT_PRIOR = (0.60, 0.45)

MINUTES_STAT_ID = 119  # stats_types 'Minutes Played'

# Competition scope matching get_player_stats' international filter — the
# minutes window must cover the same population of games as the rate sample.
_ALLOWED_SUB_TYPES = ("domestic", "domestic_cup", "cup_international")

# Count-shaped stat columns it is valid to scale linearly by exposure on the
# FPL-local frame. Ratios (pass completion) stay correct because numerator
# and denominator scale together; hit-rate columns are recomputed from the
# scaled inputs by the caller, never scaled directly.
XMIN_SCALED_STAT_COLS = [
    "Goals", "Assists", "Yellow Cards", "Saves",
    "Key Passes", "Accurate Passes", "Passes",
    "Shots Total", "Shots On Target", "Total Crosses",
    "Interceptions", "Tackles", "Offsides", "Fouls", "Fouls Drawn",
    "Ball Recovery", "Clearances Blocks Interceptions (FPL)",
    "Clearances Average", "Blocked Shots Average", "Ball Recovery Average",
    "Tackles Won Average", "CBIT Average",
]


def eligible_past_fixtures(fixtures, comps=None, today=None):
    """Past fixtures restricted to the sub_types the rate sample uses.

    Returns [id, home_team_id, away_team_id, kickoff_datetime] with
    kickoff_datetime as datetime64, sorted ascending.
    """
    today = pd.to_datetime("today") if today is None else pd.to_datetime(today)
    fx = fixtures[["id", "home_team_id", "away_team_id", "kickoff_datetime"]].copy()
    if "competition_id" in fixtures.columns and comps is not None and "sub_type" in comps.columns:
        _sub = (
            fixtures[["id", "competition_id"]]
            .drop_duplicates(subset=["id"])
            .merge(
                comps[["id", "sub_type"]].drop_duplicates(subset=["id"]).rename(
                    columns={"id": "competition_id"}
                ),
                on="competition_id",
                how="left",
            )
        )
        _ok_ids = _sub[_sub["sub_type"].isin(_ALLOWED_SUB_TYPES)]["id"]
        fx = fx[fx["id"].isin(_ok_ids)]
    fx["kickoff_datetime"] = pd.to_datetime(fx["kickoff_datetime"])
    fx = fx[fx["kickoff_datetime"] < today]
    return fx.sort_values("kickoff_datetime").reset_index(drop=True)


def build_minutes_frame(player_stats, past_fixtures):
    """All players' per-appearance minutes joined to kickoff.

    player_stats: the pipeline's loaded per-player stat rows
    (player_id, fixture_id, team_id, stats_type_id, value).
    past_fixtures: output of eligible_past_fixtures — implicitly applies
    the same competition-scope filter to the appearance rows.

    Returns [player_id, fixture_id, team_id, minutes, kickoff_datetime].
    """
    mins = player_stats[player_stats["stats_type_id"] == MINUTES_STAT_ID][
        ["player_id", "fixture_id", "team_id", "value"]
    ].copy()
    mins = mins.rename(columns={"value": "minutes"})
    mins["minutes"] = pd.to_numeric(mins["minutes"], errors="coerce").fillna(0).astype(int)
    mins = mins.merge(
        past_fixtures[["id", "kickoff_datetime"]],
        left_on="fixture_id",
        right_on="id",
        how="inner",
    ).drop(columns=["id"])
    mins = mins.drop_duplicates(subset=["player_id", "fixture_id"])
    return mins


def _weights(kickoffs, today):
    weeks = ((today - pd.to_datetime(kickoffs)).dt.days // 7).astype(float)
    w = XMIN_DECAY ** (weeks - 3)
    w = np.where(weeks < 4, 1.0, w)
    return w


def _priors_for(position):
    p_play, p60 = XMIN_NEUTRAL_PRIORS.get(position, XMIN_DEFAULT_PRIOR)
    prior_xmin = p60 * XMIN_DEFAULT_START_MINUTES + max(0.0, p_play - p60) * XMIN_PRIOR_SUB_MINUTES
    return p_play, p60, prior_xmin


def get_expected_minutes(player_id, team_id, minutes_frame, past_fixtures,
                         position=None, today=None):
    """Expected-minutes profile for one player at his current club.

    minutes_frame: build_minutes_frame output (all players).
    past_fixtures: eligible_past_fixtures output.

    Returns dict:
      p_play, p60, xmin, xmin_start_sample, exposure, p60_if_start,
      n_window (rows in window), n_apps (appearances in window).
    Never raises on thin data — falls back to positional priors.
    """
    today = pd.to_datetime("today") if today is None else pd.to_datetime(today)
    prior_p_play, prior_p60, prior_xmin = _priors_for(position)

    pm = minutes_frame[minutes_frame["player_id"] == player_id]
    rows = []  # (kickoff, minutes, weight_mult)

    club_apps = pm[pm["team_id"] == team_id]
    if not club_apps.empty:
        # Current club: every club fixture since his first club appearance is
        # a trial — played 0 minutes counts against him. This is the core fix:
        # DNPs and cameos were previously invisible.
        first_club_kick = pd.to_datetime(club_apps["kickoff_datetime"]).min()
        club_fx = past_fixtures[
            (past_fixtures["home_team_id"] == team_id)
            | (past_fixtures["away_team_id"] == team_id)
        ]
        club_fx = club_fx[club_fx["kickoff_datetime"] >= first_club_kick]
        club_window = club_fx.merge(
            club_apps[["fixture_id", "minutes"]],
            left_on="id",
            right_on="fixture_id",
            how="left",
        )
        club_window["minutes"] = club_window["minutes"].fillna(0).astype(int)
        for kick, m in club_window[["kickoff_datetime", "minutes"]].itertuples(index=False):
            rows.append((kick, m, 1.0))

    # Previous clubs, most recent stint first, until the window is full.
    other = pm[pm["team_id"] != team_id]
    if not other.empty and len(rows) < XMIN_WINDOW:
        stints = (
            other.assign(kick=pd.to_datetime(other["kickoff_datetime"]))
            .groupby("team_id")["kick"]
            .agg(["min", "max"])
            .sort_values("max", ascending=False)
        )
        pad = pd.Timedelta(days=3)
        for old_team_id, stint in stints.iterrows():
            if len(rows) >= XMIN_WINDOW:
                break
            old_fx = past_fixtures[
                (past_fixtures["home_team_id"] == old_team_id)
                | (past_fixtures["away_team_id"] == old_team_id)
            ]
            old_fx = old_fx[
                (old_fx["kickoff_datetime"] >= stint["min"] - pad)
                & (old_fx["kickoff_datetime"] <= stint["max"] + pad)
            ]
            old_apps = other[other["team_id"] == old_team_id][["fixture_id", "minutes"]]
            if not old_fx.empty:
                stint_window = old_fx.merge(
                    old_apps, left_on="id", right_on="fixture_id", how="left"
                )
                stint_window["minutes"] = stint_window["minutes"].fillna(0).astype(int)
                pairs = stint_window[["kickoff_datetime", "minutes"]]
            else:
                # Old club's fixtures not loaded (foreign league) —
                # appearance rows only; optimistic on p_play, hence the
                # cross-club discount + priors blend below.
                pairs = other[other["team_id"] == old_team_id][
                    ["kickoff_datetime", "minutes"]
                ].assign(kickoff_datetime=lambda d: pd.to_datetime(d["kickoff_datetime"]))
            for kick, m in pairs.itertuples(index=False):
                rows.append((kick, m, XMIN_CROSS_CLUB_WEIGHT))

    if not rows:
        exposure = min(1.0, prior_xmin / XMIN_DEFAULT_START_MINUTES)
        return {
            "p_play": prior_p_play, "p60": prior_p60, "xmin": prior_xmin,
            "xmin_start_sample": XMIN_DEFAULT_START_MINUTES,
            "exposure": exposure, "p60_if_start": 0.9,
            "n_window": 0, "n_apps": 0,
        }

    window = pd.DataFrame(rows, columns=["kickoff_datetime", "minutes", "mult"])
    window = window.sort_values("kickoff_datetime", ascending=False).head(XMIN_WINDOW)
    window["w"] = _weights(window["kickoff_datetime"], today) * window["mult"]
    w_sum = window["w"].sum()
    if w_sum <= 0:
        window["w"] = window["mult"]
        w_sum = window["w"].sum()

    played = window["minutes"] > 0
    sixty = window["minutes"] >= 60
    p_play_hat = float((window["w"] * played).sum() / w_sum)
    p60_hat = float((window["w"] * sixty).sum() / w_sum)
    xmin_hat = float((window["w"] * window["minutes"]).sum() / w_sum)

    starts = window[window["minutes"] > XMIN_START_MIN]
    if not starts.empty:
        s_w = starts["w"].sum()
        xmin_start_sample = float((starts["w"] * starts["minutes"]).sum() / s_w)
        p60_if_start = float((starts["w"] * (starts["minutes"] >= 60)).sum() / s_w)
    else:
        xmin_start_sample = XMIN_DEFAULT_START_MINUTES
        p60_if_start = 0.9

    n_apps = int(played.sum())
    alpha = min(1.0, n_apps / float(XMIN_PRIOR_APPS)) if XMIN_PRIOR_APPS > 0 else 1.0
    p_play = alpha * p_play_hat + (1 - alpha) * prior_p_play
    p60 = alpha * p60_hat + (1 - alpha) * prior_p60
    xmin = alpha * xmin_hat + (1 - alpha) * prior_xmin
    p60 = min(p60, p_play)

    exposure = min(1.0, xmin / xmin_start_sample) if xmin_start_sample > 0 else 1.0
    return {
        "p_play": round(p_play, 4), "p60": round(p60, 4),
        "xmin": round(xmin, 1),
        "xmin_start_sample": round(xmin_start_sample, 1),
        "exposure": round(exposure, 4), "p60_if_start": round(p60_if_start, 4),
        "n_window": int(len(window)), "n_apps": n_apps,
    }


def starter_override(profile):
    """Confirmed-XI override: he IS starting this fixture. Certainty replaces
    the base rates; how long he stays on is still his own start history."""
    p60 = profile.get("p60_if_start", 0.9)
    return {
        **profile,
        "p_play": 1.0,
        "p60": round(p60, 4),
        "xmin": profile.get("xmin_start_sample", XMIN_DEFAULT_START_MINUTES),
        "exposure": 1.0,
    }


def stamp_xmin_columns(frame, profiles, confirmed_xi=None):
    """Stamp xmin_p_play / xmin_p60 / xmin_exposure / xmin_expected onto a
    per-(player, fixture) projections frame.

    profiles: {player_id: get_expected_minutes dict}
    confirmed_xi: optional {(fixture_id, team_id): set(player_id)} from
    load_confirmed_lineups — matching rows get the starter_override.
    """
    def _get(pid, key, default):
        prof = profiles.get(pid)
        return prof.get(key, default) if prof else default

    frame["xmin_p_play"] = frame["player_id"].map(lambda p: _get(p, "p_play", 1.0))
    frame["xmin_p60"] = frame["player_id"].map(lambda p: _get(p, "p60", 1.0))
    frame["xmin_exposure"] = frame["player_id"].map(lambda p: _get(p, "exposure", 1.0))
    frame["xmin_expected"] = frame["player_id"].map(
        lambda p: _get(p, "xmin", XMIN_DEFAULT_START_MINUTES)
    )

    if confirmed_xi:
        xi_players = {
            (fid, pid)
            for (fid, _tid), pids in confirmed_xi.items()
            for pid in pids
        }
        if xi_players:
            mask = [
                (int(f), int(p)) in xi_players if pd.notna(f) and pd.notna(p) else False
                for f, p in zip(frame["fixture_id"], frame["player_id"])
            ]
            mask = pd.Series(mask, index=frame.index)
            if mask.any():
                for idx in frame.index[mask]:
                    prof = profiles.get(int(frame.at[idx, "player_id"]))
                    if not prof:
                        continue
                    over = starter_override(prof)
                    frame.at[idx, "xmin_p_play"] = over["p_play"]
                    frame.at[idx, "xmin_p60"] = over["p60"]
                    frame.at[idx, "xmin_exposure"] = over["exposure"]
                    frame.at[idx, "xmin_expected"] = over["xmin"]
                logger.info(
                    "xMinutes: starter override applied to %d confirmed-XI rows",
                    int(mask.sum()),
                )
    return frame


def apply_exposure_scaling(frame):
    """Scale the count-shaped stat columns by xmin_exposure in place.
    Call ONLY on an FPL-local copy — never on the frame that feeds
    player_projections / Opta / FanTeam / Dream11."""
    if "xmin_exposure" not in frame.columns:
        return frame
    for col in XMIN_SCALED_STAT_COLS:
        if col in frame.columns:
            frame[col] = frame[col] * frame["xmin_exposure"]
    return frame
