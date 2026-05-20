from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

# A small dictionary of common international match cities and their (lat, lon, altitude_m)
CITY_GEO_DATA = {
    "Kuala Lumpur": (3.1390, 101.6869, 66),
    "Bangkok": (13.7563, 100.5018, 2),
    "Doha": (25.2854, 51.5310, 13),
    "London": (51.5074, -0.1278, 11),
    "Budapest": (47.4979, 19.0402, 102),
    "Kuwait City": (29.3759, 47.9774, 5),
    "Montevideo": (-34.9011, -56.1645, 43),
    "Vienna": (48.2082, 16.3738, 190),
    "Oslo": (59.9139, 10.7522, 23),
    "Copenhagen": (55.6761, 12.5683, 5),
    "Glasgow": (55.8642, -4.2518, 26),
    "Buenos Aires": (-34.6037, -58.3816, 25),
    "Dublin": (53.3498, -6.2603, 20),
    "Santiago": (-33.4489, -70.6693, 570),
    "Belfast": (54.5973, -5.9301, 3),
    "Nairobi": (-1.2921, 36.8219, 1795),
    "Jakarta": (-6.2088, 106.8456, 8),
    "Seoul": (37.5665, 126.9780, 38),
    "Lima": (-12.0464, -77.0428, 154),
    "Cairo": (30.0444, 31.2357, 23),
    "Mexico City": (19.4326, -99.1332, 2240),
    "La Paz": (-16.4897, -68.1193, 3640),
    "Quito": (-0.1807, -78.4678, 2850),
    "Johannesburg": (-26.2041, 28.0473, 1753),
    "Bogotá": (4.7110, -74.0721, 2640),
    "Addis Ababa": (9.0192, 38.7469, 2355),
}

COUNTRY_HOME_GEO = {
    "England": (51.5074, -0.1278),
    "Scotland": (55.9533, -3.1883),
    "Brazil": (-15.7975, -47.8919),
    "Argentina": (-34.6037, -58.3816),
    "France": (48.8566, 2.3522),
    "Germany": (52.5200, 13.4050),
    "Mexico": (19.4326, -99.1332),
    "United States": (38.9072, -77.0369),
    "Japan": (35.6762, 139.6503),
    "South Korea": (37.5665, 126.9780),
    "Thailand": (13.7563, 100.5018),
    "Malaysia": (3.1390, 101.6869),
    "Uruguay": (-34.9011, -56.1645),
    "Chile": (-33.4489, -70.6693),
    "Bolivia": (-16.4897, -68.1193),
    "Ecuador": (-0.1807, -78.4678),
    "Colombia": (4.7110, -74.0721),
}

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def build_geo_features(matches_path: str | Path, output_path: str | Path) -> Path:
    df = pd.read_csv(matches_path)
    df["match_lat"] = df["city"].map(lambda x: CITY_GEO_DATA.get(x, (0, 0, 0))[0])
    df["match_lon"] = df["city"].map(lambda x: CITY_GEO_DATA.get(x, (0, 0, 0))[1])
    df["match_altitude"] = df["city"].map(lambda x: CITY_GEO_DATA.get(x, (0, 0, 0))[2])
    
    def get_travel(row, team_col):
        home_coords = COUNTRY_HOME_GEO.get(row[team_col])
        if not home_coords or (row["match_lat"] == 0 and row["match_lon"] == 0):
            return 0.0
        return haversine_distance(home_coords[0], home_coords[1], row["match_lat"], row["match_lon"])

    df["home_travel_km"] = df.apply(lambda r: get_travel(r, "home_team"), axis=1)
    df["away_travel_km"] = df.apply(lambda r: get_travel(r, "away_team"), axis=1)
    df["travel_diff_km"] = df["home_travel_km"] - df["away_travel_km"]
    
    output_path = Path(output_path)
    df.to_csv(output_path, index=False)
    return output_path
