from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_RAW_DIR = Path("raw/fifa")
FIFA_RANKING_API_URL = "https://inside.fifa.com/api/ranking-overview?locale=en&dateId="
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

def fetch_fifa_ranking(date_id: str, raw_dir: str | Path = DEFAULT_RAW_DIR) -> Path:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    target_path = raw_path / f"{date_id}.json"
    if target_path.exists():
        return target_path
    
    url = f"{FIFA_RANKING_API_URL}{date_id}"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Referer": "https://inside.fifa.com/fifa-world-ranking/men"})
    
    with urlopen(request) as response:
        data = json.loads(response.read().decode("utf-8"))
        
    target_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return target_path
