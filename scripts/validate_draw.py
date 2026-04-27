from __future__ import annotations

import argparse
import difflib
from pathlib import Path

import pandas as pd

from slamcaster.bracket import validate_power_of_two
from slamcaster.config import get_paths
from slamcaster.load_data import load_matches
from slamcaster.tournament_io import is_bye_name, read_draw_csv


def _known_player_names(paths_years: range) -> set[str]:
    paths = get_paths()
    m = load_matches(paths.raw, paths_years)
    names: set[str] = set()
    for col in ("winner_name", "loser_name"):
        if col in m.columns:
            names |= set(m[col].dropna().astype(str).unique().tolist())
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", required=True)
    ap.add_argument("--check-known-names", action="store_true")
    ap.add_argument("--names-start-year", type=int, default=2000)
    ap.add_argument("--names-end-year", type=int, default=2026)
    args = ap.parse_args()

    df = read_draw_csv(Path(args.draw))
    df["slot"] = df["slot"].astype(int)
    n = len(df)

    errors: list[str] = []
    warnings: list[str] = []

    try:
        validate_power_of_two(n)
    except ValueError as e:
        errors.append(str(e))

    slots = df["slot"].tolist()
    if len(set(slots)) != len(slots):
        errors.append("Duplicate slot numbers found.")
    if set(slots) != set(range(1, n + 1)):
        missing = sorted(set(range(1, n + 1)) - set(slots))
        extra = sorted(set(slots) - set(range(1, n + 1)))
        errors.append(f"Slots must be 1..{n}. missing={missing[:10]} extra={extra[:10]}")

    players = df["player_name"].astype(str).tolist()
    real_players = [p for p in players if not is_bye_name(p)]
    if len(set(real_players)) != len(real_players):
        dupes = pd.Series(real_players).value_counts()
        dupes = dupes[dupes > 1]
        errors.append(f"Duplicate player_name entries (excluding BYE): {dupes.to_dict()}")

    # BYE checks: no BYE vs BYE pairing in R1
    ordered = df.sort_values("slot")
    for i in range(1, n, 2):
        a = str(ordered.iloc[i - 1]["player_name"])
        b = str(ordered.iloc[i]["player_name"])
        if is_bye_name(a) and is_bye_name(b):
            errors.append(f"Invalid BYE vs BYE at slots {i} and {i+1}.")

    bye_count = sum(1 for p in players if is_bye_name(p))
    if bye_count > n // 2:
        errors.append(f"Too many BYEs: {bye_count} for draw size {n}.")

    if args.check_known_names:
        known = _known_player_names(range(args.names_start_year, args.names_end_year + 1))
        unknown = [p for p in real_players if p not in known]
        if unknown:
            warnings.append(f"{len(unknown)} names not found exactly in historical data.")
            for p in unknown[:25]:
                close = difflib.get_close_matches(p, known, n=3, cutoff=0.85)
                if close:
                    warnings.append(f"  - {p} (did you mean: {close}?)")
                else:
                    warnings.append(f"  - {p} (no close match)")

    if errors:
        msg = "DRAW VALIDATION FAILED\n" + "\n".join(f"- {e}" for e in errors)
        if warnings:
            msg += "\n\nWARNINGS\n" + "\n".join(f"- {w}" for w in warnings)
        raise SystemExit(msg)

    print(f"OK: draw {args.draw} size={n} byes={bye_count}")
    if warnings:
        print("\nWARNINGS")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()

