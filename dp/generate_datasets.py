from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import random

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
random.seed(42)

users = [
    {"user_id": "U001", "profile": "teen", "timezone": "UTC+0"},
    {"user_id": "U002", "profile": "family", "timezone": "UTC+1"},
    {"user_id": "U003", "profile": "student", "timezone": "UTC+0"},
    {"user_id": "U004", "profile": "gamer", "timezone": "UTC+2"},
    {"user_id": "U005", "profile": "researcher", "timezone": "UTC-1"},
    {"user_id": "U006", "profile": "traveler", "timezone": "UTC+3"},
]

movies = [
    {
        "movie_id": "M001",
        "title": "Skyward Adventure",
        "genre": "Adventure",
        "duration_min": 118,
        "rating": 4.6,
        "enjoyment_score": 92,
        "age_rating": "12+",
    },
    {
        "movie_id": "M002",
        "title": "Mystic River",
        "genre": "Mystery",
        "duration_min": 105,
        "rating": 4.1,
        "enjoyment_score": 81,
        "age_rating": "16+",
    },
    {
        "movie_id": "M003",
        "title": "Galactic Beats",
        "genre": "Sci-Fi",
        "duration_min": 96,
        "rating": 4.4,
        "enjoyment_score": 88,
        "age_rating": "12+",
    },
    {
        "movie_id": "M004",
        "title": "Culinary Quest",
        "genre": "Documentary",
        "duration_min": 83,
        "rating": 4.8,
        "enjoyment_score": 95,
        "age_rating": "6+",
    },
    {
        "movie_id": "M005",
        "title": "Pixel Racers",
        "genre": "Action",
        "duration_min": 132,
        "rating": 3.9,
        "enjoyment_score": 73,
        "age_rating": "16+",
    },
    {
        "movie_id": "M006",
        "title": "Ocean Whispers",
        "genre": "Drama",
        "duration_min": 114,
        "rating": 4.5,
        "enjoyment_score": 90,
        "age_rating": "12+",
    },
    {
        "movie_id": "M007",
        "title": "Quantum Puzzles",
        "genre": "Sci-Fi",
        "duration_min": 101,
        "rating": 4.2,
        "enjoyment_score": 83,
        "age_rating": "12+",
    },
    {
        "movie_id": "M008",
        "title": "Rhythm of Rain",
        "genre": "Romance",
        "duration_min": 124,
        "rating": 4.0,
        "enjoyment_score": 78,
        "age_rating": "12+",
    },
    {
        "movie_id": "M009",
        "title": "Hidden Trails",
        "genre": "Thriller",
        "duration_min": 109,
        "rating": 4.3,
        "enjoyment_score": 85,
        "age_rating": "16+",
    },
    {
        "movie_id": "M010",
        "title": "Starlight Stories",
        "genre": "Fantasy",
        "duration_min": 95,
        "rating": 4.7,
        "enjoyment_score": 94,
        "age_rating": "6+",
    },
    {
        "movie_id": "M011",
        "title": "Code Rush",
        "genre": "Documentary",
        "duration_min": 88,
        "rating": 4.5,
        "enjoyment_score": 89,
        "age_rating": "12+",
    },
    {
        "movie_id": "M012",
        "title": "Forest Echo",
        "genre": "Drama",
        "duration_min": 122,
        "rating": 4.1,
        "enjoyment_score": 80,
        "age_rating": "12+",
    },
]

user_df = pd.DataFrame(users)
user_df.to_csv(DATA_DIR / "workshop_users.csv", index=False)

movie_df = pd.DataFrame(movies)
movie_df.to_csv(DATA_DIR / "workshop_movies.csv", index=False)

user_favorites = {
    "U001": {"M001", "M004", "M010"},
    "U002": {"M002", "M005", "M011"},
    "U003": {"M003", "M004", "M007"},
    "U004": {"M001", "M005", "M009"},
    "U005": {"M006", "M008", "M010"},
    "U006": {"M002", "M003", "M012"},
}

devices = ["mobile", "tablet", "smart_tv", "laptop"]

start_date = datetime(2025, 3, 1)
end_date = datetime(2025, 3, 15)
delta_days = (end_date - start_date).days + 1

session_records: list[dict[str, object]] = []
session_id = 1

for offset in range(delta_days):
    day = start_date + timedelta(days=offset)
    for user in users:

        sessions_per_day = int(rng.integers(2, 6))
        possible_starts = rng.integers(6 * 60, 23 * 60, size=sessions_per_day)
        possible_starts.sort()

        for base_start in possible_starts:
            movie = movies[rng.integers(0, len(movies))]
            planned_duration = movie["duration_min"]
            raw_watch = rng.normal(loc=planned_duration * 0.82, scale=12)
            watched_minutes = int(max(15, min(planned_duration + 20, raw_watch)))

            start_dt = day + timedelta(minutes=int(base_start))
            end_dt = start_dt + timedelta(minutes=watched_minutes)

            rewatch_count = 1
            
            if rng.random() < 0.35 and watched_minutes > 35:
                extra_fragments = 2 if rng.random() < 0.35 else 1
                rewatch_count += extra_fragments

            for part in range(rewatch_count):
                if part == 0:
                    local_start = start_dt
                else:
                    overlap_shift = rng.integers(5, min(25, watched_minutes - 10))
                    local_start = start_dt + timedelta(minutes=int(overlap_shift))
                    watched_minutes = int(
                        max(10, min(planned_duration + 25, watched_minutes - overlap_shift + rng.integers(5, 25)))
                    )
                    end_dt = local_start + timedelta(minutes=watched_minutes)

                session_records.append(
                    {
                        "session_id": f"S{session_id:04d}",
                        "user_id": user["user_id"],
                        "movie_id": movie["movie_id"],
                        "profile": user["profile"],
                        "device": random.choice(devices),
                        "favorite_film": movie["movie_id"] in user_favorites[user["user_id"]],
                        "planned_duration_min": planned_duration,
                        "watched_minutes": watched_minutes,
                        "start_time": local_start.strftime("%Y-%m-%d %H:%M"),
                        "end_time": end_dt.strftime("%Y-%m-%d %H:%M"),
                        "timezone": user["timezone"],
                        "weekday": local_start.strftime("%A"),
                        "is_rewatch_fragment": part > 0,
                    }
                )
                session_id += 1

edge_cases = [
    {
        "session_id": "S9991",
        "user_id": "U003",
        "movie_id": "M004",
        "profile": "student",
        "device": "smart_tv",
        "favorite_film": True,
        "planned_duration_min": 83,
        "watched_minutes": 30,
        "start_time": "2025-03-05 09:00",
        "end_time": "2025-03-05 09:30",
        "timezone": "UTC+0",
        "weekday": "Wednesday",
        "is_rewatch_fragment": False,
    },
    {
        "session_id": "S9992",
        "user_id": "U003",
        "movie_id": "M004",
        "profile": "student",
        "device": "smart_tv",
        "favorite_film": True,
        "planned_duration_min": 83,
        "watched_minutes": 50,
        "start_time": "2025-03-05 09:25",
        "end_time": "2025-03-05 10:15",
        "timezone": "UTC+0",
        "weekday": "Wednesday",
        "is_rewatch_fragment": True,
    },
    {
        "session_id": "S9993",
        "user_id": "U005",
        "movie_id": "M008",
        "profile": "researcher",
        "device": "laptop",
        "favorite_film": True,
        "planned_duration_min": 124,
        "watched_minutes": 70,
        "start_time": "2025-03-08 23:40",
        "end_time": "2025-03-09 00:50",
        "timezone": "UTC-1",
        "weekday": "Saturday",
        "is_rewatch_fragment": False,
    },
]

session_df = pd.DataFrame(session_records + edge_cases)

session_df["start_time_dt"] = pd.to_datetime(session_df["start_time"])
session_df.sort_values(["start_time_dt", "user_id", "movie_id"], inplace=True)
session_df.drop(columns=["start_time_dt"], inplace=True)

session_df.to_csv(DATA_DIR / "workshop_sessions.csv", index=False)

print(f"Generated {len(session_df)} watch sessions across {len(users)} users.")
print(f"Saved workshop_users.csv with {len(user_df)} entries.")
print(f"Saved workshop_movies.csv with {len(movie_df)} entries.")
