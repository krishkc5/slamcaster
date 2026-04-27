 from __future__ import annotations
 
 from dataclasses import dataclass
 from typing import Any
 
 import numpy as np
 import pandas as pd
 from joblib import dump, load
 from sklearn.compose import ColumnTransformer
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.impute import SimpleImputer
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
 from sklearn.pipeline import Pipeline
 from sklearn.preprocessing import OneHotEncoder, StandardScaler
 
 
 @dataclass(frozen=True)
 class ModelArtifacts:
     model: Any
     feature_columns: list[str]
     target_column: str
     metadata: dict[str, Any]
 
 
 def _build_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
     numeric = Pipeline(
         steps=[
             ("imputer", SimpleImputer(strategy="median")),
             ("scaler", StandardScaler(with_mean=False)),
         ]
     )
     categorical = Pipeline(
         steps=[
             ("imputer", SimpleImputer(strategy="most_frequent")),
             ("onehot", OneHotEncoder(handle_unknown="ignore")),
         ]
     )
     return ColumnTransformer(
         transformers=[
             ("num", numeric, num_cols),
             ("cat", categorical, cat_cols),
         ],
         remainder="drop",
     )
 
 
 def make_logistic_regression(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
     pre = _build_preprocessor(cat_cols, num_cols)
     clf = LogisticRegression(max_iter=2000, n_jobs=None)
     return Pipeline([("pre", pre), ("clf", clf)])
 
 
 def make_random_forest(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
     pre = _build_preprocessor(cat_cols, num_cols)
     clf = RandomForestClassifier(
         n_estimators=400,
         random_state=42,
         n_jobs=-1,
         min_samples_leaf=2,
     )
     return Pipeline([("pre", pre), ("clf", clf)])
 
 
 def make_xgb_or_fallback(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
     pre = _build_preprocessor(cat_cols, num_cols)
     try:
         from xgboost import XGBClassifier  # type: ignore
 
         clf = XGBClassifier(
             n_estimators=600,
             learning_rate=0.05,
             max_depth=4,
             subsample=0.9,
             colsample_bytree=0.9,
             reg_lambda=1.0,
             random_state=42,
             n_jobs=-1,
             eval_metric="logloss",
         )
         return Pipeline([("pre", pre), ("clf", clf)])
     except Exception:
         from sklearn.ensemble import HistGradientBoostingClassifier
 
         clf = HistGradientBoostingClassifier(random_state=42, max_depth=6)
         return Pipeline([("pre", pre), ("clf", clf)])
 
 
 def evaluate_prob_model(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
     p = np.clip(p, 1e-6, 1 - 1e-6)
     y_hat = (p >= 0.5).astype(int)
     return {
         "accuracy": float(accuracy_score(y_true, y_hat)),
         "log_loss": float(log_loss(y_true, p)),
         "brier": float(brier_score_loss(y_true, p)),
     }
 
 
 def save_artifacts(path: str, artifacts: ModelArtifacts) -> None:
     dump(artifacts, path)
 
 
 def load_artifacts(path: str) -> ModelArtifacts:
     return load(path)
