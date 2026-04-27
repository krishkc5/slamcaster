 from __future__ import annotations
 
from dataclasses import dataclass
 from pathlib import Path
 
 import pandas as pd
 
 from .utils import ensure_dir
from .tournament_io import read_results_csv
 
 
 @dataclass(frozen=True)
 class EvalSummary:
     champion: str | None
     champion_title_prob: float | None
    predicted_favorite: str | None
    favorite_won: bool | None
    top10: pd.DataFrame
    actual_round_index: pd.DataFrame
    brier_by_round: pd.DataFrame
    performance_delta: pd.DataFrame
 
 
def _round_index_mapping(draw_size: int) -> dict[str, int]:
    if draw_size == 128:
        return {"R128": 0, "R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "F": 6, "W": 7}
    if draw_size == 64:
        return {"R64": 0, "R32": 1, "R16": 2, "QF": 3, "SF": 4, "F": 5, "W": 6}
    raise ValueError(f"Unsupported draw size for evaluation: {draw_size}")


def _infer_draw_size_from_predictions(pred: pd.DataFrame) -> int:
    cols = set(pred.columns)
    if "p_reach_R128" in cols:
        return 128
    if "p_reach_R64" in cols:
        return 64
    raise ValueError("Could not infer draw size from predictions columns.")


def _normalize_round(r: str) -> str:
    s = str(r).strip().upper()
    # Common aliases
    if s in {"QF", "QUARTERFINAL", "QUARTERFINALS"}:
        return "QF"
    if s in {"SF", "SEMIFINAL", "SEMIFINALS"}:
        return "SF"
    if s in {"F", "FINAL"}:
        return "F"
    return s


def _actual_round_indices_from_results(results: pd.DataFrame, draw_size: int) -> pd.DataFrame:
    idx = _round_index_mapping(draw_size)

    # Track maximum reached index for each player.
    reached: dict[str, int] = {}
    for _, m in results.iterrows():
        rnd = _normalize_round(m["round"])
        if rnd not in idx:
            continue
        w = str(m["winner_name"])
        l = str(m["loser_name"])
        r_i = idx[rnd]
        # loser "reached" this round index
        reached[l] = max(reached.get(l, 0), r_i)
        # winner reached at least next round
        reached[w] = max(reached.get(w, 0), min(r_i + 1, idx["W"]))

    out = pd.DataFrame({"player_name": list(reached.keys()), "actual_round_index": list(reached.values())})
    return out


def _brier_by_round(pred: pd.DataFrame, actual_idx: pd.DataFrame, draw_size: int) -> pd.DataFrame:
    idx = _round_index_mapping(draw_size)
    merged = pred.merge(actual_idx, on="player_name", how="left")
    merged["actual_round_index"] = merged["actual_round_index"].fillna(0).astype(int)

    rows = []
    for r, r_i in idx.items():
        col = f"p_reach_{r}"
        if col not in merged.columns:
            continue
        y = (merged["actual_round_index"] >= r_i).astype(float)
        p = merged[col].clip(1e-9, 1 - 1e-9).astype(float)
        brier = float(((p - y) ** 2).mean())
        rows.append({"round": r, "brier": brier})
    return pd.DataFrame(rows)

 
def _performance_table(pred: pd.DataFrame, actual_idx: pd.DataFrame) -> pd.DataFrame:
    merged = pred.merge(actual_idx, on="player_name", how="left")
    merged["actual_round_index"] = merged["actual_round_index"].fillna(0).astype(int)
    if "expected_round_index" not in merged.columns:
        merged["expected_round_index"] = 0.0
    merged["delta_actual_minus_expected"] = merged["actual_round_index"] - merged["expected_round_index"].astype(float)
    return merged[["player_name", "expected_round_index", "actual_round_index", "delta_actual_minus_expected"]].sort_values(
        "delta_actual_minus_expected"
    )


def evaluate_tournament_predictions(
    predictions_csv: Path,
    actual_results_csv: Path,
) -> EvalSummary:
    pred = pd.read_csv(predictions_csv)

    # Support legacy "champion-only" file OR the new match-results schema.
    act_raw = pd.read_csv(actual_results_csv)
    if "champion" in act_raw.columns and len(act_raw.columns) == 1:
        champ = str(act_raw["champion"].dropna().iloc[0]) if act_raw["champion"].dropna().shape[0] else None
        title_prob = None
        if champ is not None and "p_reach_W" in pred.columns:
            m = pred.loc[pred["player_name"].astype(str) == champ]
            if len(m):
                title_prob = float(m["p_reach_W"].iloc[0])
        favorite = None
        favorite_won = None
        if "p_reach_W" in pred.columns and len(pred):
            favorite = str(pred.sort_values("p_reach_W", ascending=False).iloc[0]["player_name"])
            favorite_won = (champ == favorite) if champ is not None else None
        top10 = pred.sort_values("p_reach_W", ascending=False).head(10).copy() if "p_reach_W" in pred.columns else pred.head(10).copy()
        empty = pd.DataFrame()
        return EvalSummary(
            champion=champ,
            champion_title_prob=title_prob,
            predicted_favorite=favorite,
            favorite_won=favorite_won,
            top10=top10,
            actual_round_index=empty,
            brier_by_round=empty,
            performance_delta=empty,
        )

    results = read_results_csv(actual_results_csv)
    draw_size = _infer_draw_size_from_predictions(pred)
    actual_idx = _actual_round_indices_from_results(results, draw_size)

    champ = None
    if len(results):
        finals = results.loc[results["round"].astype(str).str.upper().isin(["F", "FINAL"])]
        if len(finals):
            champ = str(finals.iloc[-1]["winner_name"])
        else:
            # Fallback: player with max actual_round_index
            if len(actual_idx):
                champ = str(actual_idx.sort_values("actual_round_index", ascending=False).iloc[0]["player_name"])

    title_prob = None
    if champ is not None and "p_reach_W" in pred.columns:
        m = pred.loc[pred["player_name"].astype(str) == champ]
        if len(m):
            title_prob = float(m["p_reach_W"].iloc[0])

    favorite = None
    favorite_won = None
    if "p_reach_W" in pred.columns and len(pred):
        favorite = str(pred.sort_values("p_reach_W", ascending=False).iloc[0]["player_name"])
        favorite_won = (champ == favorite) if champ is not None else None

    top10 = pred.sort_values("p_reach_W", ascending=False).head(10).copy()
    brier = _brier_by_round(pred, actual_idx, draw_size)
    perf = _performance_table(pred, actual_idx)

    return EvalSummary(
        champion=champ,
        champion_title_prob=title_prob,
        predicted_favorite=favorite,
        favorite_won=favorite_won,
        top10=top10,
        actual_round_index=actual_idx,
        brier_by_round=brier,
        performance_delta=perf,
    )
 
 
def write_eval_report_md(out_path: Path, summary: EvalSummary) -> None:
     ensure_dir(out_path.parent)
     lines = ["# Tournament evaluation\n"]
    lines.append("## Headline results\n")
    lines.append(f"- **actual champion**: {summary.champion if summary.champion is not None else '(not found)'}")
    if summary.champion_title_prob is not None:
        lines.append(f"- **champion predicted title probability**: {summary.champion_title_prob:.4f}")
    if summary.predicted_favorite is not None:
        lines.append(f"- **predicted favorite**: {summary.predicted_favorite}")
        if summary.favorite_won is not None:
            lines.append(f"- **favorite won**: {summary.favorite_won}")
    lines.append("")

    if summary.top10 is not None and len(summary.top10):
        lines.append("## Top 10 title probabilities\n")
        lines.append("| player | title_prob | final_prob | sf_prob | qf_prob |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in summary.top10.iterrows():
            lines.append(
                f"| {r['player_name']} | {float(r.get('p_reach_W', 0.0)):.4f} | {float(r.get('p_reach_F', 0.0)):.4f} | {float(r.get('p_reach_SF', 0.0)):.4f} | {float(r.get('p_reach_QF', 0.0)):.4f} |"
            )
        lines.append("")

    if summary.brier_by_round is not None and len(summary.brier_by_round):
        lines.append("## Round-by-round Brier scores (reach probabilities)\n")
        lines.append("| round | brier |")
        lines.append("|---|---:|")
        for _, r in summary.brier_by_round.iterrows():
            lines.append(f"| {r['round']} | {float(r['brier']):.6f} |")
        lines.append("")

    if summary.performance_delta is not None and len(summary.performance_delta):
        under = summary.performance_delta.head(10)
        over = summary.performance_delta.sort_values("delta_actual_minus_expected", ascending=False).head(10)
        lines.append("## Biggest underperformers (actual - expected)\n")
        lines.append("| player | expected_round | actual_round | delta |")
        lines.append("|---|---:|---:|---:|")
        for _, r in under.iterrows():
            lines.append(
                f"| {r['player_name']} | {float(r['expected_round_index']):.2f} | {int(r['actual_round_index'])} | {float(r['delta_actual_minus_expected']):.2f} |"
            )
        lines.append("")
        lines.append("## Biggest overperformers (actual - expected)\n")
        lines.append("| player | expected_round | actual_round | delta |")
        lines.append("|---|---:|---:|---:|")
        for _, r in over.iterrows():
            lines.append(
                f"| {r['player_name']} | {float(r['expected_round_index']):.2f} | {int(r['actual_round_index'])} | {float(r['delta_actual_minus_expected']):.2f} |"
            )
        lines.append("")

     out_path.write_text("\n".join(lines), encoding="utf-8")
