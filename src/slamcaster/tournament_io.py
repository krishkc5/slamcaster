from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RESULTS_REQUIRED_COLS = [
    "tournament",
    "year",
    "surface",
    "round",
    "match_num",
    "winner_name",
    "loser_name",
    "winner_id_optional",
    "loser_id_optional",
    "winner_seed",
    "loser_seed",
    "score",
    "best_of",
    "match_date_optional",
]


@dataclass(frozen=True)
class TournamentMatch:
    tournament: str
    year: int
    surface: str
    round: str
    match_num: int
    winner_name: str
    loser_name: str
    winner_id: int | None
    loser_id: int | None
    winner_seed: int | None
    loser_seed: int | None
    score: str | None
    best_of: int | None
    match_date: pd.Timestamp | None


def read_results_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in RESULTS_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Results file missing columns: {missing}")
    return df


def read_draw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "slot" not in df.columns or "player_name" not in df.columns:
        raise ValueError("Draw CSV must include columns: slot, player_name")
    if "seed" not in df.columns:
        df["seed"] = pd.NA
    if "player_id_optional" not in df.columns:
        df["player_id_optional"] = pd.NA
    return df


def is_bye_name(name: str) -> bool:
    return str(name).strip().upper() == "BYE"

