from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from slamcaster.config import get_paths, normalize_surface
from slamcaster.load_data import load_matches
from slamcaster.utils import ensure_dir


def _filter_tournament(matches: pd.DataFrame, tournament: str, year: int, surface: str) -> pd.DataFrame:
    surface = normalize_surface(surface)
    m = matches.copy()
    m = m.dropna(subset=["tourney_date"])
    m["year"] = pd.to_datetime(m["tourney_date"]).dt.year
    m = m[(m["year"] == year) & (m["surface"].astype(str) == surface)]

    # Prefer exact match; fallback to case-insensitive contains.
    exact = m[m["tourney_name"].astype(str) == tournament]
    if len(exact):
        return exact
    return m[m["tourney_name"].astype(str).str.lower().str.contains(tournament.lower())]


def _write_results_csv(m: pd.DataFrame, out_path: Path, tournament: str, year: int, surface: str) -> None:
    out = pd.DataFrame(
        {
            "tournament": tournament,
            "year": year,
            "surface": surface,
            "round": m["round"].astype(str),
            "match_num": m.get("match_num", pd.Series(range(1, len(m) + 1))).astype(int, errors="ignore"),
            "winner_name": m["winner_name"].astype(str),
            "loser_name": m["loser_name"].astype(str),
            "winner_id_optional": m["winner_id"],
            "loser_id_optional": m["loser_id"],
            "winner_seed": pd.NA,
            "loser_seed": pd.NA,
            "score": pd.NA,
            "best_of": m.get("best_of", pd.NA),
            "match_date_optional": pd.to_datetime(m["tourney_date"]).dt.date.astype(str),
        }
    )
    ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tournament", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--surface", required=True)
    ap.add_argument("--output-draw", required=True)
    ap.add_argument("--output-results", required=True)
    args = ap.parse_args()

    paths = get_paths()
    matches = load_matches(paths.raw, [args.year])
    surface = normalize_surface(args.surface)
    tm = _filter_tournament(matches, args.tournament, args.year, surface)
    if len(tm) == 0:
        raise SystemExit(
            f"No matches found for tournament={args.tournament!r} year={args.year} surface={surface!r}. "
            "Check naming vs Sackmann tourney_name."
        )

    results_path = Path(args.output_results)
    _write_results_csv(tm, results_path, args.tournament, args.year, surface)
    print(f"Wrote results to {results_path}")

    # Draw reconstruction is NOT safe from Sackmann match logs in general.
    # We fail clearly rather than inventing bracket order.
    raise SystemExit(
        "Cannot safely reconstruct ordered draw slot file from Sackmann match data alone.\n"
        "Provide a real ordered draw CSV manually (128 rows for Grand Slams; 64-slot with BYEs for Masters) and then rerun prediction/evaluation.\n"
        f"Expected output draw path (you requested): {args.output_draw}"
    )


if __name__ == "__main__":
    main()

