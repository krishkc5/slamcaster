 from __future__ import annotations
 
 import argparse
 from pathlib import Path
 
 import pandas as pd
 
 from slamcaster.config import EloConfig, get_paths
 from slamcaster.features import FeatureConfig, build_match_feature_dataset
 from slamcaster.load_data import load_matches, load_players, load_rankings
 from slamcaster.utils import ensure_dir
 
 
 def main() -> None:
     ap = argparse.ArgumentParser()
     ap.add_argument("--start-year", type=int, required=True)
     ap.add_argument("--end-year", type=int, required=True)
     ap.add_argument("--output", type=str, required=True)
     ap.add_argument("--k-factor", type=float, default=32.0)
     args = ap.parse_args()
 
     paths = get_paths()
     years = list(range(args.start_year, args.end_year + 1))
     matches = load_matches(paths.raw, years)
     rankings = load_rankings(paths.raw)
     players = load_players(paths.raw)
 
     df = build_match_feature_dataset(
         matches=matches,
         rankings=rankings,
         players=players,
         feature_cfg=FeatureConfig(),
         elo_cfg=EloConfig(k_factor=float(args.k_factor)),
     )
 
     out_path = Path(args.output)
     ensure_dir(out_path.parent)
     # Store match_date as ISO for portability.
     df["match_date"] = pd.to_datetime(df["match_date"]).dt.date.astype(str)
     df.to_csv(out_path, index=False)
     print(f"Wrote {len(df):,} rows to {out_path}")
 
 
 if __name__ == "__main__":
     main()
