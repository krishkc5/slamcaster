from __future__ import annotations

import subprocess
from pathlib import Path

from slamcaster.config import get_paths


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    paths = get_paths()
    model_path = paths.models / "match_model.joblib"

    # Toy sanity checks (small draws). Replace with real ordered draws + match-results CSVs for real backtests.
    cases = [
        {
            "tournament": "US Open",
            "year": 2025,
            "surface": "Hard",
            "start_date": "2025-08-24",
            "draw": paths.draws / "us_open_2025_mens.csv",
            "pred": paths.predictions / "us_open_2025_predictions.csv",
            "actual": paths.draws / "us_open_2025_actual_results.csv",
            "report": paths.reports / "us_open_2025_eval.md",
        },
        {
            "tournament": "Australian Open",
            "year": 2026,
            "surface": "Hard",
            "start_date": "2026-01-12",
            "draw": paths.draws / "australian_open_2026_mens.csv",
            "pred": paths.predictions / "australian_open_2026_predictions.csv",
            "actual": paths.draws / "australian_open_2026_actual_results.csv",
            "report": paths.reports / "australian_open_2026_eval.md",
        },
        {
            "tournament": "Monte Carlo Masters",
            "year": 2026,
            "surface": "Clay",
            "start_date": "2026-04-06",
            "draw": paths.draws / "monte_carlo_2026_mens.csv",
            "pred": paths.predictions / "monte_carlo_2026_predictions.csv",
            "actual": paths.draws / "monte_carlo_2026_actual_results.csv",
            "report": paths.reports / "monte_carlo_2026_eval.md",
        },
    ]

    for c in cases:
        if not Path(c["draw"]).exists():
            print(f"SKIP (missing draw): {c['draw']}")
            continue
        if not Path(c["actual"]).exists():
            print(f"SKIP (missing actual results): {c['actual']}")
            continue

        _run(
            [
                "python",
                "scripts/predict_tournament.py",
                "--draw",
                str(c["draw"]),
                "--tournament",
                c["tournament"],
                "--year",
                str(c["year"]),
                "--surface",
                c["surface"],
                "--start-date",
                c["start_date"],
                "--model",
                str(model_path),
                "--sims",
                "5000",
                "--output",
                str(c["pred"]),
            ]
        )
        _run(
            [
                "python",
                "scripts/evaluate_tournament.py",
                "--predictions",
                str(c["pred"]),
                "--actual",
                str(c["actual"]),
                "--output",
                str(c["report"]),
            ]
        )


if __name__ == "__main__":
    main()
