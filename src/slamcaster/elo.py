 from __future__ import annotations
 
 from dataclasses import dataclass
 from typing import DefaultDict
 
 from .config import EloConfig, normalize_surface
 
 
 @dataclass
 class EloSnapshot:
     overall_p1: float
     overall_p2: float
     surface_p1: float
     surface_p2: float
 
 
 def expected_score(r_a: float, r_b: float) -> float:
     """Return expected probability A beats B under Elo."""
     return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))
 
 
 class EloTracker:
     """Tracks overall and surface-specific Elo ratings with leakage-safe updates.
 
     Usage pattern:
     - Call pre_match_snapshot() BEFORE updating for a match (features).
     - Then call update() AFTER the match result (state).
     """
 
     def __init__(self, cfg: EloConfig | None = None) -> None:
         self.cfg = cfg or EloConfig()
         self.overall: dict[int, float] = {}
         # surface -> player_id -> rating
         self.surface: dict[str, dict[int, float]] = {}
 
     def _get_overall(self, pid: int) -> float:
         return self.overall.get(pid, self.cfg.start_rating)
 
     def _get_surface(self, surface: str, pid: int) -> float:
         surface = normalize_surface(surface)
         if surface not in self.surface:
             self.surface[surface] = {}
         return self.surface[surface].get(pid, self.cfg.start_rating)
 
     def pre_match_snapshot(self, p1: int, p2: int, surface: str) -> EloSnapshot:
         s = normalize_surface(surface)
         return EloSnapshot(
             overall_p1=self._get_overall(p1),
             overall_p2=self._get_overall(p2),
             surface_p1=self._get_surface(s, p1),
             surface_p2=self._get_surface(s, p2),
         )
 
     def update(self, winner: int, loser: int, surface: str) -> None:
         """Update ratings after a match."""
         s = normalize_surface(surface)
         # Overall
         r_w = self._get_overall(winner)
         r_l = self._get_overall(loser)
         e_w = expected_score(r_w, r_l)
         e_l = 1.0 - e_w
         k = self.cfg.k_factor
         self.overall[winner] = r_w + k * (1.0 - e_w)
         self.overall[loser] = r_l + k * (0.0 - e_l)
 
         # Surface
         rs_w = self._get_surface(s, winner)
         rs_l = self._get_surface(s, loser)
         es_w = expected_score(rs_w, rs_l)
         es_l = 1.0 - es_w
         if s not in self.surface:
             self.surface[s] = {}
         self.surface[s][winner] = rs_w + k * (1.0 - es_w)
         self.surface[s][loser] = rs_l + k * (0.0 - es_l)
