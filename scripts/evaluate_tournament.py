from __future__ import annotations

import argparse
from pathlib import Path

from slamcaster.evaluate import evaluate_tournament_predictions, write_eval_report_md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--actual", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    summary = evaluate_tournament_predictions(Path(args.predictions), Path(args.actual))
    write_eval_report_md(Path(args.output), summary)
    print(f"Wrote evaluation report to {args.output}")


if __name__ == "__main__":
    main()
