 from __future__ import annotations
 
 from dataclasses import dataclass
 from datetime import timedelta
 from typing import Any
 
 import numpy as np
 import pandas as pd
 
 from .elo import EloSnapshot, EloTracker
 from .utils import stable_int_hash
 
 
 @dataclass(frozen=True)
 class FeatureConfig:
     recent_n_1: int = 5
     recent_n_2: int = 10
     last_days_1: int = 14
     last_days_2: int = 30
 
 
 def _safe_div(num: float, den: float) -> float:
     if den <= 0:
         return 0.0
     return float(num) / float(den)
 
 
 def _encode_best_of(best_of: Any) -> int:
     try:
         return 1 if int(best_of) == 5 else 0
     except Exception:
         return 0
 
 
 def _recent_win_rate(history: list[int], n: int) -> float:
     if not history:
         return 0.0
     h = history[-n:]
     return float(sum(h)) / float(len(h))
 
 
 class RollingForm:
     """Rolling form stats keyed by player_id, leakage-safe by chronological iteration."""
 
     def __init__(self) -> None:
         self.overall_results: dict[int, list[int]] = {}
         self.surface_results: dict[tuple[int, str], list[int]] = {}
         self.match_dates: dict[int, list[pd.Timestamp]] = {}
 
     def snapshot(
         self, pid: int, surface: str, asof: pd.Timestamp, cfg: FeatureConfig
     ) -> dict[str, float]:
         overall = self.overall_results.get(pid, [])
         surf = self.surface_results.get((pid, surface), [])
         dates = self.match_dates.get(pid, [])
 
         w5 = _recent_win_rate(overall, cfg.recent_n_1)
         w10 = _recent_win_rate(overall, cfg.recent_n_2)
         ws = _recent_win_rate(surf, cfg.recent_n_2)
 
         # counts in last 14/30 days relative to asof (pre-match only).
         c14 = 0
         c30 = 0
         if dates:
             cutoff30 = asof - pd.Timedelta(days=cfg.last_days_2)
             cutoff14 = asof - pd.Timedelta(days=cfg.last_days_1)
             # dates are appended in chronological order
             for d in reversed(dates):
                 if d >= cutoff30:
                     c30 += 1
                 else:
                     break
                 if d >= cutoff14:
                     c14 += 1
         return {
             "recent_win_rate_5": w5,
             "recent_win_rate_10": w10,
             "recent_surface_win_rate": ws,
             "matches_last_14d": float(c14),
             "matches_last_30d": float(c30),
         }
 
     def update(self, winner: int, loser: int, surface: str, match_date: pd.Timestamp) -> None:
         self.overall_results.setdefault(winner, []).append(1)
         self.overall_results.setdefault(loser, []).append(0)
         self.surface_results.setdefault((winner, surface), []).append(1)
         self.surface_results.setdefault((loser, surface), []).append(0)
         self.match_dates.setdefault(winner, []).append(match_date)
         self.match_dates.setdefault(loser, []).append(match_date)
 
 
 def choose_symmetric_assignment(match_key: Any, seed: int = 0) -> int:
     """Return 0/1 deterministically for symmetric row assignment."""
     return stable_int_hash(match_key, seed) % 2
 
 
 def build_feature_row(
     *,
     p1_id: int,
     p2_id: int,
     surface: str,
     tourney_level: str,
     round_name: str,
     best_of: Any,
     p1_rank: Any,
     p2_rank: Any,
     p1_points: Any,
     p2_points: Any,
     p1_age_years: float | None,
     p2_age_years: float | None,
     elo_snap: EloSnapshot,
     form: RollingForm,
     match_date: pd.Timestamp,
     cfg: FeatureConfig,
 ) -> dict[str, Any]:
     """Compute leakage-safe, difference-based features for (p1 vs p2)."""
     f1 = form.snapshot(p1_id, surface, match_date, cfg)
     f2 = form.snapshot(p2_id, surface, match_date, cfg)
 
     def _to_float(x: Any) -> float:
         try:
             if x is None or (isinstance(x, float) and np.isnan(x)):
                 return float("nan")
             return float(x)
         except Exception:
             return float("nan")
 
     r1 = _to_float(p1_rank)
     r2 = _to_float(p2_rank)
     pts1 = _to_float(p1_points)
     pts2 = _to_float(p2_points)
 
     age_diff = float("nan")
     if p1_age_years is not None and p2_age_years is not None:
         age_diff = float(p1_age_years - p2_age_years)
 
     return {
         "p1_id": int(p1_id),
         "p2_id": int(p2_id),
         "surface": surface,
         "tourney_level": str(tourney_level),
         "round": str(round_name),
         "best_of_5": _encode_best_of(best_of),
         "match_date": match_date,
         # Differences (p1 - p2)
         "elo_diff": float(elo_snap.overall_p1 - elo_snap.overall_p2),
         "surface_elo_diff": float(elo_snap.surface_p1 - elo_snap.surface_p2),
         "rank_diff": r1 - r2,
         "rank_points_diff": pts1 - pts2,
         "recent_win_rate_5_diff": float(f1["recent_win_rate_5"] - f2["recent_win_rate_5"]),
         "recent_win_rate_10_diff": float(f1["recent_win_rate_10"] - f2["recent_win_rate_10"]),
         "recent_surface_win_rate_diff": float(
             f1["recent_surface_win_rate"] - f2["recent_surface_win_rate"]
         ),
         "matches_last_14d_diff": float(f1["matches_last_14d"] - f2["matches_last_14d"]),
         "matches_last_30d_diff": float(f1["matches_last_30d"] - f2["matches_last_30d"]),
         "age_diff": age_diff,
     }


 def add_rank_asof(
     matches: pd.DataFrame,
     rankings: pd.DataFrame,
     *,
     side: str,
 ) -> pd.DataFrame:
     """Add leakage-safe ranking snapshots for `winner` or `loser` using merge_asof.
 
     Rankings: columns [ranking_date, player_id, rank, points]
     Matches: requires columns [tourney_date, {side}_id]
     """
     df = matches.copy()
     pid_col = f"{side}_id"
     if pid_col not in df.columns:
         return df
 
     left = df[["tourney_date", pid_col]].copy()
     left = left.rename(columns={pid_col: "player_id"}).sort_values("tourney_date")
 
     right = rankings.sort_values("ranking_date").copy()
     out = pd.merge_asof(
         left,
         right,
         left_on="tourney_date",
         right_on="ranking_date",
         by="player_id",
         direction="backward",
         allow_exact_matches=True,
     )
     df[f"{side}_rank_asof"] = out["rank"].to_numpy()
     df[f"{side}_points_asof"] = out["points"].to_numpy()
     return df


 def build_match_feature_dataset(
     *,
     matches: pd.DataFrame,
     rankings: pd.DataFrame,
     players: pd.DataFrame | None = None,
     feature_cfg: FeatureConfig | None = None,
     elo_cfg=None,
 ) -> pd.DataFrame:
     """Build symmetric, leakage-safe match dataset.
 
     Leakage prevention:
     - matches are processed in chronological order
     - Elo and rolling-form states are snapshotted pre-match and updated post-match
     - rankings are merged as-of the match date (latest <= match date)
     """
     feature_cfg = feature_cfg or FeatureConfig()
     matches = matches.dropna(subset=["tourney_date", "winner_id", "loser_id"]).copy()
     matches = matches.sort_values(["tourney_date", "tourney_id", "match_num"], na_position="last")
 
     # Ranking snapshots (as-of match date)
     m2 = add_rank_asof(matches, rankings, side="winner")
     m2 = add_rank_asof(m2, rankings, side="loser")
 
     dob_map: dict[int, pd.Timestamp] = {}
     if players is not None and {"player_id", "dob"}.issubset(players.columns):
         tmp = players.dropna(subset=["player_id", "dob"]).copy()
         for _, r in tmp.iterrows():
             try:
                 dob_map[int(r["player_id"])] = pd.Timestamp(r["dob"])
             except Exception:
                 continue
 
     elo = EloTracker(cfg=elo_cfg)
     form = RollingForm()
 
     rows: list[dict[str, Any]] = []
     for _, r in m2.iterrows():
         dt = pd.Timestamp(r["tourney_date"])
         surface = str(r.get("surface", "Unknown"))
         w = int(r["winner_id"])
         l = int(r["loser_id"])
         match_key = (r.get("tourney_id"), int(r.get("match_num", 0)))
 
         swap = choose_symmetric_assignment(match_key, seed=0)
         if swap == 0:
             p1, p2 = w, l
             y = 1
             p1_rank = r.get("winner_rank_asof")
             p2_rank = r.get("loser_rank_asof")
             p1_pts = r.get("winner_points_asof")
             p2_pts = r.get("loser_points_asof")
         else:
             p1, p2 = l, w
             y = 0
             p1_rank = r.get("loser_rank_asof")
             p2_rank = r.get("winner_rank_asof")
             p1_pts = r.get("loser_points_asof")
             p2_pts = r.get("winner_points_asof")
 
         def _age(pid: int) -> float | None:
             dob = dob_map.get(pid)
             if dob is None or pd.isna(dob):
                 return None
             return float((dt - dob).days / 365.25)
 
         snap = elo.pre_match_snapshot(p1, p2, surface)
         feat = build_feature_row(
             p1_id=p1,
             p2_id=p2,
             surface=surface,
             tourney_level=str(r.get("tourney_level", "U")),
             round_name=str(r.get("round", "U")),
             best_of=r.get("best_of"),
             p1_rank=p1_rank,
             p2_rank=p2_rank,
             p1_points=p1_pts,
             p2_points=p2_pts,
             p1_age_years=_age(p1),
             p2_age_years=_age(p2),
             elo_snap=snap,
             form=form,
             match_date=dt,
             cfg=feature_cfg,
         )
         feat["target_p1_win"] = int(y)
         rows.append(feat)
 
         # Update state after match using true winner/loser.
         elo.update(winner=w, loser=l, surface=surface)
         form.update(winner=w, loser=l, surface=surface, match_date=dt)
 
     out = pd.DataFrame(rows)
     return out
