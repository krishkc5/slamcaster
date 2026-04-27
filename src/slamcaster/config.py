 from __future__ import annotations
 
 from dataclasses import dataclass
 from pathlib import Path
 
 
 @dataclass(frozen=True)
 class Paths:
     root: Path
     data: Path
     raw: Path
     interim: Path
     processed: Path
     draws: Path
     models: Path
     outputs: Path
     reports: Path
     predictions: Path
     plots: Path
     tables: Path
 
 
 def get_paths(root: Path | None = None) -> Paths:
     """Return project paths rooted at repo root (default: auto-detect)."""
     if root is None:
         # repo root is two levels up from this file: src/slamcaster/config.py
         root = Path(__file__).resolve().parents[2]
 
     data = root / "data"
     outputs = root / "outputs"
     return Paths(
         root=root,
         data=data,
         raw=data / "raw",
         interim=data / "interim",
         processed=data / "processed",
         draws=data / "draws",
         models=root / "models",
         outputs=outputs,
         reports=outputs / "reports",
         predictions=outputs / "predictions",
         plots=outputs / "plots",
         tables=outputs / "tables",
     )
 
 
 @dataclass(frozen=True)
 class EloConfig:
     k_factor: float = 32.0
     start_rating: float = 1500.0
 
 
 SURFACE_ALIASES: dict[str, str] = {
     "Hard": "Hard",
     "Clay": "Clay",
     "Grass": "Grass",
     "Carpet": "Carpet",
     "H": "Hard",
     "C": "Clay",
     "G": "Grass",
 }
 
 
 def normalize_surface(surface: str | None) -> str:
     if not surface:
         return "Unknown"
     s = str(surface).strip()
     return SURFACE_ALIASES.get(s, s)
