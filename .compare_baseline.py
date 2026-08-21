"""Diff current fixture projections against the 2026-08-19 pre-change baseline.

    python3 .compare_baseline.py            # summary by competition
    python3 .compare_baseline.py "Premier League"

Baseline was taken 2026-08-19 ~21:40 UTC, before the overnight sweep picked up
the goal-average rework (228b621) and the standings-based ratio (9d63375).

CAVEAT: La Liga's baseline rows are ALREADY post-change — that league was
re-run at 21:15 to verify the deploy. Its diff will show ~nothing. Every other
competition is a genuine before/after.
"""
import csv, subprocess, sys, io

REMOTE = ("ssh -o ConnectTimeout=30 projections@176.74.18.125 -p2223 "
          "'cd ~/site/statz-projection && C=$(docker compose ps -q statz-projection) && "
          "docker cp /tmp/snap.py $C:/tmp/ >/dev/null && "
          "docker compose exec -T -w /app -e PYTHONPATH=/app statz-projection python /tmp/snap.py 2>/dev/null'")

def load(path):
    return {r["fixture_id"]: r for r in csv.DictReader(open(path))
            if r.get("fixture_id")}

def devig(r):
    try:
        o = [float(r["b365_h"]), float(r["b365_d"]), float(r["b365_a"])]
    except (ValueError, KeyError, TypeError):
        return None
    if min(o) <= 1:
        return None
    s = sum(1 / x for x in o)
    return [100 / x / s for x in o]

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    base = load(".baseline_2026_08_19.csv")
    print("re-snapshotting current projections…", file=sys.stderr)
    out = subprocess.run(REMOTE, shell=True, capture_output=True, text=True).stdout
    out = "\n".join(l for l in out.splitlines() if not l.startswith("Pool is initialized"))
    now = {r["fixture_id"]: r for r in csv.DictReader(io.StringIO(out)) if r.get("fixture_id")}

    rows = []
    for fid, n in now.items():
        b = base.get(fid)
        if not b or (only and n["competition"] != only):
            continue
        bk = devig(n)
        rows.append((n["competition"], n["home"], n["away"],
                     float(b["our_home"]), float(n["our_home"]),
                     float(b["our_draw"]), float(n["our_draw"]),
                     bk,
                     float(b["our_hg"]), float(n["our_hg"]),
                     float(b["our_ag"]), float(n["our_ag"])))
    if not rows:
        print("no overlapping fixtures — has the overnight run happened yet?")
        return

    if only:
        print("\n  SCORELINES (expected goals per side)\n")
        print("  %-20s %-20s %17s %17s %13s" % ("home", "away", "home goals b->a", "away goals b->a", "total b->a"))
        print("  " + "-" * 94)
        for c, h, a, hb, hn, db_, dn, bk, ghb, ghn, gab, gan in rows:
            print("  %-20s %-20s %7.2f -> %-7.2f %7.2f -> %-7.2f %5.2f -> %-5.2f"
                  % (h[:20], a[:20], ghb, ghn, gab, gan, ghb + gab, ghn + gan))

        print("\n  PROBABILITIES\n")
        print("  %-20s %-20s %13s %13s %11s" % ("home", "away", "home before/after", "draw before/after", "vs book"))
        print("  " + "-" * 86)
        for c, h, a, hb, hn, db_, dn, bk, *_ in rows:
            v = "%+6.1f -> %+6.1f" % (hb - bk[0], hn - bk[0]) if bk else "          -"
            print("  %-20s %-20s %6.1f -> %5.1f %6.1f -> %5.1f %11s" % (h[:20], a[:20], hb, hn, db_, dn, v))

    print("\n  %-24s %4s %10s %10s %10s %10s %11s"
          % ("competition", "fix", "home gls", "away gls", "home %", "draw %", "|err| b->a"))
    print("  " + "-" * 86)
    for comp in sorted({r[0] for r in rows}):
        g = [r for r in rows if r[0] == comp]
        n = len(g)
        gh = sum(r[9] - r[8] for r in g) / n     # home goals moved
        ga = sum(r[11] - r[10] for r in g) / n   # away goals moved
        hm = sum(r[4] - r[3] for r in g) / n
        dm = sum(r[6] - r[5] for r in g) / n
        pv = [r for r in g if r[7]]
        err = ("%.1f -> %.1f" % (sum(abs(r[3] - r[7][0]) for r in pv) / len(pv),
                                 sum(abs(r[4] - r[7][0]) for r in pv) / len(pv))) if pv else "-"
        print("  %-24s %4d %+10.3f %+10.3f %+10.2f %+10.2f %11s"
              % (comp[:24], n, gh, ga, hm, dm, err))
    print("\n  goals columns are the change in expected goals per side;")
    print("  home %/draw % are percentage-point moves; |err| is mean absolute")
    print("  home-probability error against bet365, before -> after.")

if __name__ == "__main__":
    main()
