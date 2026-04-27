 from __future__ import annotations
 
 import argparse
 from pathlib import Path
 
 import numpy as np
 import pandas as pd
 
 from slamcaster.calibrate import calibrate_model
 from slamcaster.config import get_paths
 from slamcaster.model import (
     ModelArtifacts,
     evaluate_prob_model,
     make_logistic_regression,
     make_random_forest,
     make_xgb_or_fallback,
     save_artifacts,
 )
 from slamcaster.utils import ensure_dir
 
 
 TARGET = "target_p1_win"
 CAT_COLS = ["surface", "tourney_level", "round"]
 NUM_COLS = [
     "elo_diff",
     "surface_elo_diff",
     "rank_diff",
     "rank_points_diff",
     "recent_win_rate_5_diff",
     "recent_win_rate_10_diff",
     "recent_surface_win_rate_diff",
     "matches_last_14d_diff",
     "matches_last_30d_diff",
     "age_diff",
     "best_of_5",
 ]
 
 
 def _time_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
     years = pd.to_datetime(df["match_date"]).dt.year
     uniq = sorted(int(y) for y in years.dropna().unique())
     if len(uniq) < 6:
         raise ValueError("Not enough years of data for time-aware splits. Download more years.")
     test_years = set(uniq[-2:])
     val_years = set(uniq[-4:-2])
     train_years = set(uniq[:-4])
     train = df[years.isin(train_years)].copy()
     val = df[years.isin(val_years)].copy()
     test = df[years.isin(test_years)].copy()
     return train, val, test
 
 
 def main() -> None:
     ap = argparse.ArgumentParser()
     ap.add_argument("--input", required=True)
     ap.add_argument("--model-output", required=True)
     ap.add_argument("--calibrate", action="store_true")
     args = ap.parse_args()
 
     paths = get_paths()
     df = pd.read_csv(args.input)
     if TARGET not in df.columns:
         raise ValueError(f"Missing target column: {TARGET}")
 
     train, val, test = _time_splits(df)
     X_train = train[NUM_COLS + CAT_COLS]
     y_train = train[TARGET].astype(int).to_numpy()
     X_val = val[NUM_COLS + CAT_COLS]
     y_val = val[TARGET].astype(int).to_numpy()
     X_test = test[NUM_COLS + CAT_COLS]
     y_test = test[TARGET].astype(int).to_numpy()
 
     models = {
         "logreg": make_logistic_regression(CAT_COLS, NUM_COLS),
         "rf": make_random_forest(CAT_COLS, NUM_COLS),
         "xgb_or_hgb": make_xgb_or_fallback(CAT_COLS, NUM_COLS),
     }
 
     metrics_rows = []
     best_name = None
     best_val_ll = float("inf")
     best_model = None
 
     for name, m in models.items():
         m.fit(X_train, y_train)
         p_val = m.predict_proba(X_val)[:, 1]
         val_metrics = evaluate_prob_model(y_val, p_val)
         p_test = m.predict_proba(X_test)[:, 1]
         test_metrics = evaluate_prob_model(y_test, p_test)
 
         row = {
             "model": name,
             "val_accuracy": val_metrics["accuracy"],
             "val_log_loss": val_metrics["log_loss"],
             "val_brier": val_metrics["brier"],
             "test_accuracy": test_metrics["accuracy"],
             "test_log_loss": test_metrics["log_loss"],
             "test_brier": test_metrics["brier"],
         }
         metrics_rows.append(row)
 
         if val_metrics["log_loss"] < best_val_ll:
             best_val_ll = val_metrics["log_loss"]
             best_name = name
             best_model = m
 
     metrics = pd.DataFrame(metrics_rows).sort_values("val_log_loss")
     ensure_dir(paths.tables)
     metrics_path = paths.tables / "model_metrics.csv"
     metrics.to_csv(metrics_path, index=False)
     print(f"Wrote metrics to {metrics_path}")
 
     assert best_model is not None and best_name is not None
 
     final_model = best_model
     if args.calibrate:
         final_model = calibrate_model(best_model, X_val, y_val, method="isotonic")
 
     artifacts = ModelArtifacts(
         model=final_model,
         feature_columns=NUM_COLS + CAT_COLS,
         target_column=TARGET,
         metadata={
             "best_model": best_name,
             "splits": {"train_years": "older", "val_years": "mid", "test_years": "latest_2"},
             "calibrated": bool(args.calibrate),
         },
     )
     out_path = Path(args.model_output)
     ensure_dir(out_path.parent)
     save_artifacts(str(out_path), artifacts)
     print(f"Saved model artifacts to {out_path}")
 
 
 if __name__ == "__main__":
     main()
