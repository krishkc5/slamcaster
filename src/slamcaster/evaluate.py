 from __future__ import annotations
 
 from dataclasses import dataclass
 from pathlib import Path
 
 import pandas as pd
 
 from .utils import ensure_dir
 
 
 @dataclass(frozen=True)
 class EvalSummary:
     champion: str | None
     champion_title_prob: float | None
 
 
 def evaluate_tournament_predictions(
     predictions_csv: Path,
     actual_csv: Path,
 ) -> EvalSummary:
     pred = pd.read_csv(predictions_csv)
     act = pd.read_csv(actual_csv)
 
     champ = None
     if "champion" in act.columns:
         champ = str(act["champion"].dropna().iloc[0]) if act["champion"].dropna().shape[0] else None
     elif {"player_name", "result"}.issubset(act.columns):
         # expects one row with result == "W"
         w = act.loc[act["result"].astype(str) == "W", "player_name"]
         champ = str(w.iloc[0]) if len(w) else None
 
     title_prob = None
     if champ is not None:
         m = pred.loc[pred["player_name"].astype(str) == champ]
         if len(m):
             title_prob = float(m["p_reach_W"].iloc[0])
 
     return EvalSummary(champion=champ, champion_title_prob=title_prob)
 
 
 def write_eval_report_md(out_path: Path, summary: EvalSummary) -> None:
     ensure_dir(out_path.parent)
     lines = ["# Tournament evaluation\n"]
     lines.append("## Champion\n")
     if summary.champion is None:
         lines.append("- **champion**: (not found in actual file)")
     else:
         lines.append(f"- **champion**: {summary.champion}")
         if summary.champion_title_prob is not None:
             lines.append(f"- **predicted title probability**: {summary.champion_title_prob:.4f}")
     lines.append("")
     out_path.write_text("\n".join(lines), encoding="utf-8")
