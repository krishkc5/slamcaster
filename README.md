# slamcaster

ML-based tennis tournament outcome predictor for ATP men’s tennis.

## MVP pipeline (working)

### 1) Create a virtualenv and install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional (if you want XGBoost):

```bash
pip install -e ".[xgb]"
```

### 2) Download raw data (Jeff Sackmann tennis_atp)

```bash
python scripts/download_data.py --start-year 2000 --end-year 2026
```

Files are saved to `data/raw/`. The downloader is idempotent (won’t re-download existing files unless `--overwrite`).

### 3) Build leakage-safe match feature dataset

```bash
python scripts/build_dataset.py --start-year 2000 --end-year 2026 --output data/processed/matches_features.csv
```

Leakage prevention (critical):
- Elo + recent form are computed **chronologically**. For each match, features are snapshotted **before** the match and state is updated **after**.
- Rankings are merged “as-of” the match date (latest ranking date \(\le\) match date).

### 4) Train models and save the best

```bash
python scripts/train_model.py --input data/processed/matches_features.csv --model-output models/match_model.joblib
```

Metrics are written to `outputs/tables/model_metrics.csv`.

### 5) Predict a tournament from a draw CSV and simulate

This repo includes small sample draws (8 players) at:
- `data/draws/us_open_2025_mens.csv`
- `data/draws/australian_open_2026_mens.csv`

Run:

```bash
python scripts/predict_tournament.py \
  --draw data/draws/us_open_2025_mens.csv \
  --tournament "US Open" \
  --year 2025 \
  --surface Hard \
  --start-date 2025-08-24 \
  --model models/match_model.joblib \
  --sims 10000 \
  --output outputs/predictions/us_open_2025_predictions.csv
```

### 6) Evaluate predictions vs actual (MVP)

```bash
python scripts/evaluate_tournament.py \
  --predictions outputs/predictions/us_open_2025_predictions.csv \
  --actual data/draws/us_open_2025_actual_results.csv \
  --output outputs/reports/us_open_2025_eval.md
```

### 7) Toy sanity-check batch (sample draws)

```bash
python scripts/run_backtests.py
```

## Milestone 2: Real tournament backtesting (honest)

Key rule: **do not invent bracket slot order** from match logs.

Workflow:
- Use Sackmann match data for **model training**.
- Use a **real ordered draw CSV** for simulation.
- Use a **match-results CSV (one row per match)** for evaluation.

### Build a results file from Sackmann (draw will fail on purpose)

This writes a `*_actual_results.csv` in the required match-row schema, but will refuse to fabricate `slot` order:

```bash
python scripts/build_tournament_files.py \
  --tournament "US Open" \
  --year 2025 \
  --surface Hard \
  --output-draw data/draws/us_open_2025_mens.csv \
  --output-results data/draws/us_open_2025_actual_results.csv
```

### Validate a real draw CSV you provide

```bash
python scripts/validate_draw.py --draw data/draws/us_open_2025_mens.csv --check-known-names
```

### Run a real backtest once you have real draw + real results

```bash
python scripts/predict_tournament.py \
  --draw data/draws/us_open_2025_mens.csv \
  --tournament "US Open" \
  --year 2025 \
  --surface Hard \
  --start-date 2025-08-24 \
  --model models/match_model.joblib \
  --sims 10000 \
  --output outputs/predictions/us_open_2025_predictions.csv

python scripts/evaluate_tournament.py \
  --predictions outputs/predictions/us_open_2025_predictions.csv \
  --actual data/draws/us_open_2025_actual_results.csv \
  --output outputs/reports/us_open_2025_eval.md
```

## Draw CSV format (MVP)

Required columns:
- `slot` (1-indexed bracket slot)
- `player_name`

Optional:
- `seed`
- `player_id_optional` (recommended for robustness)

The simulator assumes ordered slots: round one is (1 vs 2), (3 vs 4), etc.

## Results CSV format (required for real evaluation)

For completed tournaments, use **one row per match** in `data/draws/<tourney>_<year>_actual_results.csv`:

Columns:
- `tournament`, `year`, `surface`
- `round`, `match_num`
- `winner_name`, `loser_name`
- `winner_id_optional`, `loser_id_optional`
- `winner_seed`, `loser_seed`
- `score`, `best_of`, `match_date_optional`