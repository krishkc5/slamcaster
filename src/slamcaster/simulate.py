from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .bracket import DrawEntry, first_round_pairs, is_bye, validate_power_of_two


ROUND_NAMES_128 = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "W"]
ROUND_NAMES_64 = ["R64", "R32", "R16", "QF", "SF", "F", "W"]


@dataclass(frozen=True)
class SimulationResult:
    advancement: pd.DataFrame
    metadata: dict[str, Any]


def _round_name_for_size(size: int) -> str:
    if size == 128:
        return "R128"
    if size == 64:
        return "R64"
    if size == 32:
        return "R32"
    if size == 16:
        return "R16"
    if size == 8:
        return "QF"
    if size == 4:
        return "SF"
    if size == 2:
        return "F"
    if size == 1:
        return "W"
    return f"R{size}"


def simulate_tournament(
    *,
    entries: list[DrawEntry],
    predict_proba: Any,
    sims: int = 10_000,
    rng_seed: int = 42,
) -> SimulationResult:
    """Monte Carlo tournament simulation for power-of-two draws.

    BYE rows are supported (auto-advance).
    """
    validate_power_of_two(len(entries))
    entries = sorted(entries, key=lambda e: e.slot)
    players = sorted({e.player_name for e in entries if not is_bye(e)})

    adv_counts: dict[tuple[str, str], int] = {}  # (player, round_name) -> count reached that round
    opp_elo_sum: dict[str, float] = {p: 0.0 for p in players}
    opp_elo_n: dict[str, int] = {p: 0 for p in players}

    rng = np.random.default_rng(rng_seed)

    start_round = _round_name_for_size(len(entries))
    for p in players:
        adv_counts[(p, start_round)] = sims

    for _ in range(sims):
        alive = entries[:]
        while len(alive) > 1:
            winners: list[DrawEntry] = []
            for a, b in first_round_pairs(alive):
                if is_bye(a) and is_bye(b):
                    continue
                if is_bye(a) and not is_bye(b):
                    w = b
                    winners.append(w)
                    next_round = _round_name_for_size(len(alive) // 2)
                    adv_counts[(w.player_name, next_round)] = adv_counts.get(
                        (w.player_name, next_round), 0
                    ) + 1
                    continue
                if is_bye(b) and not is_bye(a):
                    w = a
                    winners.append(w)
                    next_round = _round_name_for_size(len(alive) // 2)
                    adv_counts[(w.player_name, next_round)] = adv_counts.get(
                        (w.player_name, next_round), 0
                    ) + 1
                    continue

                p = float(predict_proba(a, b))
                win_a = rng.random() < p
                w = a if win_a else b
                winners.append(w)

                next_round = _round_name_for_size(len(alive) // 2)
                adv_counts[(w.player_name, next_round)] = adv_counts.get(
                    (w.player_name, next_round), 0
                ) + 1

                if hasattr(a, "elo") and hasattr(b, "elo"):
                    opp = b.elo if win_a else a.elo
                    opp_elo_sum[w.player_name] += float(opp)
                    opp_elo_n[w.player_name] += 1

            alive = winners

    round_order = ROUND_NAMES_128 if len(entries) == 128 else ROUND_NAMES_64
    rounds = sorted(
        {r for (_, r) in adv_counts.keys()},
        key=lambda x: round_order.index(x) if x in round_order else 999,
    )

    rows = []
    for p in players:
        row = {"player_name": p}
        for r in rounds:
            row[f"p_reach_{r}"] = adv_counts.get((p, r), 0) / sims
        exp = 0.0
        for i, r in enumerate(rounds):
            exp += i * row.get(f"p_reach_{r}", 0.0)
        row["expected_round_index"] = exp
        row["avg_opponent_elo"] = (
            (opp_elo_sum[p] / opp_elo_n[p]) if opp_elo_n[p] > 0 else float("nan")
        )
        rows.append(row)

    adv = pd.DataFrame(rows).sort_values("p_reach_W", ascending=False, na_position="last")
    return SimulationResult(advancement=adv, metadata={"sims": sims, "rng_seed": rng_seed, "draw_size": len(entries)})
