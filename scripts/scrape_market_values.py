"""Off-server Transfermarkt market-value scraper.

Runs as a GitHub Action (.github/workflows/transfermarkt-mv.yml) because
Transfermarkt's AWS WAF challenges the projection server's datacenter IP
(HTTP 202) since ~2026-07-14. GitHub runner IPs pass; results are pushed
into statz's transfermarkt_market_value_snapshots table via the internal
API, where the projection pipeline already reads them as its fallback.

Politeness rules (we do NOT want this egress flagged too):
- weekly schedule, one pass, no parallelism
- 10-20s jittered pause between leagues, shuffled order
- abort the WHOLE run on the first challenge/non-200 (no hammering)
- one retry only, on 5xx, after 60s

Parsing mirrors app/services/statz_functions.py::get_market_value exactly
(prefix strips, € suffix expansion) so stored rows are byte-identical to
what the server-side scraper would have written.

Env: STATZ_BASE_URL (default https://statz.ai), STATZ_INTERNAL_API_TOKEN.
"""

import os
import random
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = os.environ.get("STATZ_BASE_URL", "https://statz.ai").rstrip("/")
TOKEN = os.environ["STATZ_INTERNAL_API_TOKEN"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
}

# Same order-sensitive prefix strips as get_market_value.
PREFIX_STRIPS = ["AFC", "FC", "SC", "CF", "RCD", "SS", "AS", "BC", "US", "AC"]
SUFFIX_MAP = {"bn": "0000000", "m": "0000", "k": "000"}


def api(method, path, **kwargs):
    resp = requests.request(
        method, f"{BASE_URL}/api/internal{path}",
        headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"},
        timeout=30, **kwargs,
    )
    resp.raise_for_status()
    return resp.json()


def parse_league(html):
    """Return [(team_name, market_value_digits)] — mirrors get_market_value."""
    soup = BeautifulSoup(html, "html.parser")
    teams = [td.text.strip() for td in soup.select('td[class="hauptlink no-border-links"]')]
    values = soup.select('td[class="rechts"]')[1::2][1:]
    values = [td.text.strip() for td in values]

    rows = []
    for team, raw in zip(teams, values):
        for prefix in PREFIX_STRIPS:
            team = team.replace(prefix, "")
        team = team.strip()
        raw = raw.replace(".", "").strip("€")
        # [bnmk]: 'k' was missing from the original pattern for years — the
        # k→000 map entry existed but could never match, silently dropping
        # any club under €1m (first real casualty: Rochdale AFC, €975k,
        # League Two 2026/27 — flagged by the missing-MV badge).
        m = re.search(r"(\d+)([bnmk]+)", raw)
        if not m or m.group(2) not in SUFFIX_MAP:
            continue  # genuinely unparseable

        rows.append((team, m.group(1) + SUFFIX_MAP[m.group(2)]))
    return rows


def scrape(session, league):
    url = f"https://www.transfermarkt.co.uk/{league['league_dashed'].lower()}/startseite/wettbewerb/{league['code']}{league['div']}"
    for attempt in (1, 2):
        # A dropped connection or a read timeout is a transient network fault,
        # not a block — treat it like a 5xx rather than letting it surface as
        # a traceback. Matters more now the scrape runs from a laptop on wifi
        # than it did on a datacentre link.
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException as err:
            if attempt == 1:
                print(f"{league['league_dashed']}: {type(err).__name__} — retrying in 60s")
                time.sleep(60)
                continue
            print(f"ABORT: {type(err).__name__} on {url} — network unreachable, stopping run.")
            sys.exit(1)
        if resp.status_code == 200:
            return parse_league(resp.text)
        if 500 <= resp.status_code < 600 and attempt == 1:
            time.sleep(60)
            continue
        # 202/403/etc = challenge or block. Abort the whole run — retrying
        # or continuing across leagues is how an IP gets flagged.
        #
        # Always exit non-zero. A 202 used to exit 0 as a polite back-off,
        # which meant a blocked run showed green: the Action reported success
        # daily from 21 Jul to 7 Aug 2026 while landing no data at all. A run
        # that scraped nothing is a failed run, whatever the reason.
        reason = "challenged (WAF)" if resp.status_code == 202 else "blocked/unavailable"
        print(f"ABORT: HTTP {resp.status_code} on {url} — {reason}, stopping run.")
        sys.exit(1)
    return []


def main():
    leagues = api("GET", "/transfermarkt-mv/leagues")["leagues"]
    random.shuffle(leagues)
    print(f"{len(leagues)} leagues to scrape")

    session = requests.Session()
    ok = 0
    for i, league in enumerate(leagues):
        if i:
            time.sleep(random.uniform(10, 20))
        rows = scrape(session, league)
        if not rows:
            print(f"{league['league_dashed']}: 0 rows parsed (page layout change?) — skipping")
            continue
        result = api("POST", "/transfermarkt-mv", json={
            "league_dashed": league["league_dashed"],
            "rows": [{"team_name": t, "market_value": v} for t, v in rows],
        })
        ok += 1
        print(f"{league['league_dashed']}: upserted {result['upserted']}")

    print(f"Done — {ok}/{len(leagues)} leagues refreshed")
    if not ok:
        # Reached the end without a single league landing — every page parsed
        # to zero rows. Same rule as the abort above: nothing scraped is a
        # failure, not a quiet success.
        print("FAIL: no leagues refreshed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
