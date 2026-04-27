 from __future__ import annotations
 
 from dataclasses import dataclass
 from pathlib import Path
 from typing import Iterable
 
 import pandas as pd
 
 from .config import normalize_surface
 from .utils import ensure_dir
 
 
 MATCH_COLS_KEEP = [
     "tourney_id",
     "tourney_name",
     "tourney_level",
     "tourney_date",
     "surface",
     "draw_size",
     "round",
     "best_of",
     "match_num",
     "winner_id",
     "winner_name",
     "winner_rank",
     "winner_rank_points",
     "loser_id",
     "loser_name",
     "loser_rank",
     "loser_rank_points",
 ]
 
 
 @dataclass(frozen=True)
 class RawData:
     matches: pd.DataFrame
     rankings: pd.DataFrame
     players: pd.DataFrame | None
 
 
 def _read_csv(path: Path) -> pd.DataFrame:
     if not path.exists():
         raise FileNotFoundError(f"Missing file: {path}")
     return pd.read_csv(path, low_memory=False)
 
 
 def load_matches(raw_dir: Path, years: Iterable[int]) -> pd.DataFrame:
     """Load yearly match CSVs from data/raw into a single DataFrame."""
     dfs: list[pd.DataFrame] = []
     for y in years:
         p = raw_dir / f"atp_matches_{y}.csv"
         if not p.exists():
             continue
         df = _read_csv(p)
         # Keep a stable subset across years.
         keep = [c for c in MATCH_COLS_KEEP if c in df.columns]
         df = df[keep].copy()
         dfs.append(df)
 
     if not dfs:
         raise FileNotFoundError(
             f"No match files found in {raw_dir}. "
             "Run scripts/download_data.py first."
         )
     out = pd.concat(dfs, ignore_index=True)
     out["tourney_date"] = pd.to_datetime(out["tourney_date"], format="%Y%m%d", errors="coerce")
     out["surface"] = out["surface"].map(normalize_surface)
     out["tourney_level"] = out["tourney_level"].fillna("U")
     out["round"] = out["round"].fillna("U")
     return out
 
 
 def load_rankings(raw_dir: Path) -> pd.DataFrame:
     """Load rankings decade files into a single DataFrame."""
     parts = []
     for name in ("atp_rankings_00s.csv", "atp_rankings_10s.csv", "atp_rankings_20s.csv"):
         p = raw_dir / name
         if p.exists():
             parts.append(_read_csv(p))
     if not parts:
         raise FileNotFoundError(
             f"No ranking files found in {raw_dir}. Run scripts/download_data.py first."
         )
     r = pd.concat(parts, ignore_index=True)
     # Expected columns: ranking_date, rank, player, points
     if "ranking_date" in r.columns:
         r["ranking_date"] = pd.to_datetime(r["ranking_date"], format="%Y%m%d", errors="coerce")
     r = r.rename(columns={"player": "player_id"})
     r = r.dropna(subset=["ranking_date", "player_id"])
     r["player_id"] = r["player_id"].astype("int64", errors="ignore")
     return r[["ranking_date", "player_id", "rank", "points"]].copy()
 
 
 def load_players(raw_dir: Path) -> pd.DataFrame | None:
     p = raw_dir / "atp_players.csv"
     if not p.exists():
         return None
     df = _read_csv(p)
     # Columns vary; we keep common fields.
     cols = [c for c in ["player_id", "name_first", "name_last", "dob"] if c in df.columns]
     if not cols:
         return None
     out = df[cols].copy()
     if "dob" in out.columns:
         out["dob"] = pd.to_datetime(out["dob"], format="%Y%m%d", errors="coerce")
     return out
 
 
 def build_name_to_id_map(matches: pd.DataFrame) -> dict[str, int]:
     """Best-effort mapping from player_name to a recent player_id."""
     mapping: dict[str, int] = {}
     for side in ("winner", "loser"):
         id_col = f"{side}_id"
         name_col = f"{side}_name"
         if id_col not in matches.columns or name_col not in matches.columns:
             continue
         tmp = matches[[name_col, id_col, "tourney_date"]].dropna(subset=[name_col, id_col])
         tmp = tmp.sort_values("tourney_date")
         # last observed id per name
         for name, grp in tmp.groupby(name_col, sort=False):
             last = grp.iloc[-1]
             try:
                 mapping[str(name)] = int(last[id_col])
             except Exception:
                 continue
     return mapping
 
 
 def ensure_data_dirs(*dirs: Path) -> None:
     for d in dirs:
         ensure_dir(d)
