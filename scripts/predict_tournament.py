from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from slamcaster.bracket import DrawEntry
from slamcaster.config import EloConfig, get_paths, normalize_surface
from slamcaster.elo import EloTracker
from slamcaster.features import FeatureConfig, RollingForm, build_feature_row
from slamcaster.load_data import build_name_to_id_map, load_matches, load_players, load_rankings
from slamcaster.model import load_artifacts
from slamcaster.reports import TournamentReportInputs, write_prediction_report_md
from slamcaster.simulate import simulate_tournament
from slamcaster.tournament_io import is_bye_name
from slamcaster.utils import ensure_dir, parse_date


@dataclass
class PlayerContext:
    elo: EloTracker
    form: RollingForm
    rankings: pd.DataFrame
    dob_map: dict[int, pd.Timestamp]


def _build_dob_map(players: pd.DataFrame | None) -> dict[int, pd.Timestamp]:
    m: dict[int, pd.Timestamp] = {}
    if players is None:
        return m
    if {"player_id", "dob"}.issubset(players.columns):
        tmp = players.dropna(subset=["player_id", "dob"]).copy()
        for _, r in tmp.iterrows():
            try:
                m[int(r["player_id"])] = pd.Timestamp(r["dob"])
            except Exception:
                continue
    return m


def build_player_context_asof(
    *,
    matches: pd.DataFrame,
    rankings: pd.DataFrame,
    players: pd.DataFrame | None,
    asof: pd.Timestamp,
    elo_cfg: EloConfig | None = None,
) -> PlayerContext:
    """Build Elo + rolling-form state using matches strictly before `asof`."""
    m = matches.dropna(subset=["tourney_date", "winner_id", "loser_id"]).copy()
    m = m[m["tourney_date"] < asof].sort_values(["tourney_date", "tourney_id", "match_num"])

    elo = EloTracker(cfg=elo_cfg or EloConfig())
    form = RollingForm()
    for _, r in m.iterrows():
        dt = pd.Timestamp(r["tourney_date"])
        surface = str(r.get("surface", "Unknown"))
        w = int(r["winner_id"])
        l = int(r["loser_id"])
        elo.update(winner=w, loser=l, surface=surface)
        form.update(winner=w, loser=l, surface=surface, match_date=dt)

    return PlayerContext(elo=elo, form=form, rankings=rankings, dob_map=_build_dob_map(players))


def _rank_points_asof(rankings: pd.DataFrame, pid: int, asof: pd.Timestamp) -> tuple[float, float]:
    r = rankings[rankings["player_id"] == pid]
    r = r[r["ranking_date"] <= asof].sort_values("ranking_date")
    if len(r) == 0:
        return float("nan"), float("nan")
    last = r.iloc[-1]
    try:
        return float(last["rank"]), float(last["points"])
    except Exception:
        return float("nan"), float("nan")


def _age_years(dob_map: dict[int, pd.Timestamp], pid: int, asof: pd.Timestamp) -> float | None:
    dob = dob_map.get(pid)
    if dob is None or pd.isna(dob):
        return None
    return float((asof - dob).days / 365.25)


def read_draw(draw_path: Path, name_to_id: dict[str, int]) -> list[DrawEntry]:
    df = pd.read_csv(draw_path)
    if "slot" not in df.columns or "player_name" not in df.columns:
        raise ValueError("Draw CSV must include columns: slot, player_name (seed optional)")
    entries: list[DrawEntry] = []
    for _, r in df.iterrows():
        slot = int(r["slot"])
        player_name = str(r["player_name"])
        if is_bye_name(player_name):
            entries.append(DrawEntry(slot=slot, player_name="BYE", seed=None, player_id=None))
            continue
        seed = None
        if "seed" in df.columns and not pd.isna(r.get("seed")):
            try:
                seed = int(r["seed"])
            except Exception:
                seed = None
        pid = None
        if "player_id_optional" in df.columns and not pd.isna(r.get("player_id_optional")):
            try:
                pid = int(r["player_id_optional"])
            except Exception:
                pid = None
        if pid is None:
            pid = name_to_id.get(player_name)
        entries.append(DrawEntry(slot=slot, player_name=player_name, seed=seed, player_id=pid))
    return sorted(entries, key=lambda e: e.slot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", required=True)
    ap.add_argument("--tournament", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--surface", required=True)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sims", type=int, default=10_000)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    surface = normalize_surface(args.surface)
    start_dt = parse_date(args.start_date)
    if start_dt is None:
        raise ValueError("--start-date must be YYYY-MM-DD")
    asof = pd.Timestamp(start_dt)

    paths = get_paths()
    artifacts = load_artifacts(args.model)
    model = artifacts.model

    # Build state strictly before tournament start date.
    matches = load_matches(paths.raw, range(2000, args.year + 1))
    rankings = load_rankings(paths.raw)
    players = load_players(paths.raw)

    name_to_id = build_name_to_id_map(matches)
    entries = read_draw(Path(args.draw), name_to_id)

    ctx = build_player_context_asof(
        matches=matches,
        rankings=rankings,
        players=players,
        asof=asof,
        elo_cfg=EloConfig(),
    )

    feat_cfg = FeatureConfig()

    def predict_proba(a: DrawEntry, b: DrawEntry) -> float:
        if str(a.player_name).strip().upper() == "BYE" or str(b.player_name).strip().upper() == "BYE":
            # simulate.py will short-circuit BYEs, but keep predictable behavior here too.
            return 1.0 if str(b.player_name).strip().upper() == "BYE" else 0.0
        if a.player_id is None or b.player_id is None:
            # Unknown players: treat as coinflip.
            return 0.5

        pid_a = int(a.player_id)
        pid_b = int(b.player_id)
        snap = ctx.elo.pre_match_snapshot(pid_a, pid_b, surface)
        rank_a, pts_a = _rank_points_asof(ctx.rankings, pid_a, asof)
        rank_b, pts_b = _rank_points_asof(ctx.rankings, pid_b, asof)

        row = build_feature_row(
            p1_id=pid_a,
            p2_id=pid_b,
            surface=surface,
            tourney_level="G",  # TODO: use real tourney level from metadata
            round_name="R128",  # TODO: set based on draw size/round
            best_of=5
            if args.tournament.lower() in {"us open", "australian open", "roland garros", "wimbledon"}
            else 3,
            p1_rank=rank_a,
            p2_rank=rank_b,
            p1_points=pts_a,
            p2_points=pts_b,
            p1_age_years=_age_years(ctx.dob_map, pid_a, asof),
            p2_age_years=_age_years(ctx.dob_map, pid_b, asof),
            elo_snap=snap,
            form=ctx.form,
            match_date=asof,
            cfg=feat_cfg,
        )
        X = pd.DataFrame([row])[artifacts.feature_columns]
        return float(model.predict_proba(X)[:, 1][0])

    sim = simulate_tournament(entries=entries, predict_proba=predict_proba, sims=int(args.sims))

    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    sim.advancement.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")

    report_path = (
        Path(args.report)
        if args.report
        else (paths.reports / f"{args.tournament.lower().replace(' ', '_')}_{args.year}_prediction.md")
    )
    write_prediction_report_md(
        out_path=report_path,
        inputs=TournamentReportInputs(
            tournament=args.tournament,
            year=int(args.year),
            surface=surface,
            start_date=start_dt,
            model_path=args.model,
            sims=int(args.sims),
        ),
        probs=sim.advancement,
        notes=[
            "All features are computed using matches and rankings strictly before the tournament start date.",
            "This simulator uses ordered bracket slots (slot1 vs slot2, ...).",
        ],
    )
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
