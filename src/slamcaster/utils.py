 from __future__ import annotations
 
 import hashlib
 import json
 from dataclasses import asdict, is_dataclass
 from datetime import date, datetime
 from pathlib import Path
 from typing import Any
 
 import pandas as pd
 
 
 def ensure_dir(path: Path) -> None:
     path.mkdir(parents=True, exist_ok=True)
 
 
 def parse_date(d: Any) -> date | None:
     """Parse YYYYMMDD or YYYY-MM-DD or datetime/date."""
     if d is None or (isinstance(d, float) and pd.isna(d)):
         return None
     if isinstance(d, date) and not isinstance(d, datetime):
         return d
     if isinstance(d, datetime):
         return d.date()
     s = str(d).strip()
     if not s:
         return None
     if len(s) == 8 and s.isdigit():
         return datetime.strptime(s, "%Y%m%d").date()
     return datetime.strptime(s, "%Y-%m-%d").date()
 
 
 def stable_int_hash(*parts: Any) -> int:
     """Deterministic integer hash across runs/machines."""
     payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
     h = hashlib.sha256(payload).hexdigest()
     return int(h[:16], 16)
 
 
 def to_jsonable(obj: Any) -> Any:
     if is_dataclass(obj):
         return {k: to_jsonable(v) for k, v in asdict(obj).items()}
     if isinstance(obj, Path):
         return str(obj)
     if isinstance(obj, (datetime, date)):
         return obj.isoformat()
     if isinstance(obj, dict):
         return {str(k): to_jsonable(v) for k, v in obj.items()}
     if isinstance(obj, (list, tuple)):
         return [to_jsonable(x) for x in obj]
     return obj
