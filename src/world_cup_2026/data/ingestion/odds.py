"""
Fetch pre-match 1X2 bookmaker odds from Betexplorer for major international tournaments.

Scrapes match-level average odds (home / draw / away) for:
  - FIFA World Cup 2006–2022
  - UEFA Euro 2004–2024
  - Copa América 2016–2024

Each tournament's index page is fetched first to discover stage IDs (which expose
group-stage qualifiers in addition to the knockout-round results page). All unique
match rows are deduplicated by Betexplorer match ID and written to raw/bookmaker_odds.csv.

Run with:
    uv run python -m world_cup_2026 data fetch --source odds
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

DEFAULT_RAW_DIR = Path("raw")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# (label, betexplorer base URL)
TARGET_TOURNAMENTS = [
    ("FIFA World Cup 2022", "https://www.betexplorer.com/football/world/world-cup-2022/"),
    ("FIFA World Cup 2018", "https://www.betexplorer.com/football/world/world-cup-2018/"),
    ("FIFA World Cup 2014", "https://www.betexplorer.com/football/world/world-cup-2014/"),
    ("FIFA World Cup 2010", "https://www.betexplorer.com/football/world/world-cup-2010/"),
    ("FIFA World Cup 2006", "https://www.betexplorer.com/football/world/world-cup-2006/"),
    ("UEFA Euro 2024",      "https://www.betexplorer.com/football/europe/euro-2024/"),
    ("UEFA Euro 2020",      "https://www.betexplorer.com/football/europe/euro-2020/"),
    ("UEFA Euro 2016",      "https://www.betexplorer.com/football/europe/euro-2016/"),
    ("UEFA Euro 2012",      "https://www.betexplorer.com/football/europe/euro-2012/"),
    ("UEFA Euro 2008",      "https://www.betexplorer.com/football/europe/euro-2008/"),
    ("UEFA Euro 2004",      "https://www.betexplorer.com/football/europe/euro-2004/"),
    ("Copa América 2024",   "https://www.betexplorer.com/football/south-america/copa-america-2024/"),
    ("Copa América 2021",   "https://www.betexplorer.com/football/south-america/copa-america-2021/"),
    ("Copa América 2019",   "https://www.betexplorer.com/football/south-america/copa-america-2019/"),
    ("Copa América 2016",   "https://www.betexplorer.com/football/south-america/copa-america-2016/"),
]

_MIN_MATCHES_PER_STAGE = 3   # skip stages that are clearly empty/noise
_REQUEST_DELAY = 2.5          # seconds between requests (be polite)
_TIMEOUT = 15


def _get(url: str) -> str:
    time.sleep(_REQUEST_DELAY)
    r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.text


def _get_stage_ids(html: str) -> list[str]:
    """Extract all unique ?stage=XXX IDs from tournament index page."""
    return list(dict.fromkeys(re.findall(r"\?stage=([\w]+)", html)))


def _parse_matches(html: str, competition: str) -> list[dict]:
    """
    Parse match rows from a Betexplorer results page.

    Each row looks like:
      <tr>
        <td><a data-test="ID" href="..." class="in-match"><span>Home</span> - <span>Away</span></a></td>
        <td>2:1</td>
        <td class="table-main__odds" data-odd="2.10"></td>
        <td class="table-main__odds colored"><span><span><span data-odd="3.20"></span>…</td>
        <td class="table-main__odds" data-odd="3.80"></td>
        <td class="h-text-right h-text-no-wrap">18.12.2022</td>
      </tr>
    """
    rows = []
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL):
        match_id_m = re.search(r'data-test="(\d+)"', row_html)
        date_m = re.search(r"(\d{2}\.\d{2}\.\d{4})", row_html)
        if not match_id_m or not date_m:
            continue

        # Teams from within the in-match anchor
        teams_html_m = re.search(r'class="in-match">(.*?)</a>', row_html, re.DOTALL)
        if not teams_html_m:
            continue
        teams = re.findall(r"<(?:span|strong)>([^<]+)</(?:span|strong)>", teams_html_m.group(1))
        if len(teams) < 2:
            continue

        # Odds: all data-odd values in row — order is home / draw / away
        odds = [float(x) for x in re.findall(r'data-odd="([\d.]+)"', row_html)]
        if len(odds) < 3:
            continue

        date_str = date_m.group(1)  # DD.MM.YYYY
        try:
            match_date = f"{date_str[6:]}-{date_str[3:5]}-{date_str[:2]}"  # YYYY-MM-DD
        except Exception:
            continue

        rows.append({
            "match_id_betexplorer": match_id_m.group(1),
            "match_date": match_date,
            "home_team": teams[0].strip(),
            "away_team": teams[1].strip(),
            "h_odd": odds[0],
            "d_odd": odds[1],
            "a_odd": odds[2],
            "competition": competition,
        })
    return rows


def _fetch_tournament(label: str, base_url: str) -> list[dict]:
    """Fetch all stages for one tournament; return deduplicated list of match dicts."""
    print(f"  {label}...")

    try:
        index_html = _get(base_url)
    except Exception as e:
        print(f"    ✗ index fetch failed: {e}")
        return []

    stage_ids = _get_stage_ids(index_html)

    # Always include the main results page (knockout rounds)
    pages_to_fetch: list[str] = [base_url + "results/"]
    for sid in stage_ids:
        pages_to_fetch.append(base_url + f"results/?stage={sid}")

    seen_ids: set[str] = set()
    all_rows: list[dict] = []

    for url in pages_to_fetch:
        try:
            html = _get(url)
        except Exception:
            continue

        matches = _parse_matches(html, label)
        for m in matches:
            if m["match_id_betexplorer"] not in seen_ids:
                seen_ids.add(m["match_id_betexplorer"])
                all_rows.append(m)

    print(f"    {len(all_rows)} unique matches")
    return all_rows


def fetch_bookmaker_odds(raw_dir: str | Path = DEFAULT_RAW_DIR) -> Path:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    out_path = raw_path / "bookmaker_odds.csv"

    # Load existing data to avoid re-fetching already-covered competitions
    existing: pd.DataFrame | None = None
    existing_competitions: set[str] = set()
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing_competitions = set(existing["competition"].unique())
        print(f"Existing data: {len(existing)} rows across {len(existing_competitions)} competitions")

    all_rows: list[dict] = []
    for label, base_url in TARGET_TOURNAMENTS:
        if label in existing_competitions:
            print(f"  {label}... (skipped, already fetched)")
            continue
        all_rows.extend(_fetch_tournament(label, base_url))

    if not all_rows:
        print("Nothing new to fetch.")
        return out_path

    new_df = pd.DataFrame(all_rows)
    if existing is not None:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset="match_id_betexplorer", inplace=True)
    else:
        combined = new_df

    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} total matches → {out_path}  (+{len(new_df)} new)")
    return out_path
