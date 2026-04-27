 from __future__ import annotations
 
 from dataclasses import dataclass
 
 
 @dataclass(frozen=True)
 class SackmannRepo:
     owner: str = "JeffSackmann"
     repo: str = "tennis_atp"
     branch: str = "master"
 
     @property
     def raw_base(self) -> str:
         return f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/{self.branch}"
 
 
 def atp_matches_filename(year: int) -> str:
     return f"atp_matches_{year}.csv"
 
 
 def atp_rankings_filenames() -> list[str]:
     # Weekly rankings are stored in decade files.
     return ["atp_rankings_00s.csv", "atp_rankings_10s.csv", "atp_rankings_20s.csv"]
 
 
 def atp_players_filename() -> str:
     return "atp_players.csv"
