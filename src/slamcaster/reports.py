 from __future__ import annotations
 
 from dataclasses import dataclass
 from datetime import date
 from pathlib import Path
 from typing import Any
 
 import pandas as pd
 
 from .utils import ensure_dir
 
 
 @dataclass(frozen=True)
 class TournamentReportInputs:
     tournament: str
     year: int
     surface: str
     start_date: date
     model_path: str
     sims: int
 
 
 def write_prediction_report_md(
     *,
     out_path: Path,
     inputs: TournamentReportInputs,
     probs: pd.DataFrame,
     notes: list[str] | None = None,
 ) -> None:
     ensure_dir(out_path.parent)
     top = probs.sort_values("p_reach_W", ascending=False).head(20)
     lines: list[str] = []
     lines.append(f"# {inputs.tournament} {inputs.year} prediction\n")
     lines.append("## Metadata\n")
     lines.append(f"- **tournament**: {inputs.tournament}")
     lines.append(f"- **year**: {inputs.year}")
     lines.append(f"- **surface**: {inputs.surface}")
     lines.append(f"- **start_date**: {inputs.start_date.isoformat()}")
     lines.append(f"- **model**: `{inputs.model_path}`")
     lines.append(f"- **sims**: {inputs.sims}")
     lines.append("")
     lines.append("## Top title favorites\n")
     lines.append("| player | title_prob | final_prob | sf_prob | qf_prob |")
     lines.append("|---|---:|---:|---:|---:|")
     for _, r in top.iterrows():
         lines.append(
             f"| {r['player_name']} | {r.get('p_reach_W', 0.0):.4f} | {r.get('p_reach_F', 0.0):.4f} | {r.get('p_reach_SF', 0.0):.4f} | {r.get('p_reach_QF', 0.0):.4f} |"
         )
     lines.append("")
     if notes:
         lines.append("## Notes\n")
         for n in notes:
             lines.append(f"- {n}")
         lines.append("")
     out_path.write_text("\n".join(lines), encoding="utf-8")
