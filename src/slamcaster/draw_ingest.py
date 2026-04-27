from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests


_COUNTRY_RE = re.compile(r"\b[A-Z]{3}\b")
_MARKER_RE = re.compile(r"\((W|WC|Q|LL|PR)\)")
_SEED_RE = re.compile(r"\[(\d{1,2})\]")


def download_file(url: str, output_path: Path) -> Path:
    """Download a file from URL to output_path (overwrites if exists)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download {url} (status={r.status_code})")
    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 128):
            if chunk:
                f.write(chunk)
    return output_path


def normalize_draw_player_name(raw: str) -> str:
    """Normalize player name from draw text into 'First Last'."""
    s = str(raw).strip()
    s = _MARKER_RE.sub("", s)
    s = _COUNTRY_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\[\d{1,2}\]", "", s).strip()
    s = s.replace(" ,", ",").strip()

    # LAST, First -> First Last
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        last = parts[0].title()
        first = parts[1].title() if len(parts) > 1 else ""
        s = f"{first} {last}".strip()
    else:
        # Title-case but keep internal capitalization reasonably
        s = " ".join(w.capitalize() if w.isupper() else w for w in s.split())
    return s.strip()


def extract_seed(raw: str) -> int | None:
    m = _SEED_RE.search(str(raw))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_generic_draw_text(text: str, draw_size: int) -> pd.DataFrame:
    """Parse bracket slot lines from extracted PDF or manual text.

    Expected patterns like:
      '1. SINNER, Jannik ITA [1]'
      '2 KOPRIVA, Vit CZE'
      '3. POPYRIN, Alexei AUS'
    """
    rows: list[dict[str, object]] = []
    slot_seen: set[int] = set()

    for line in str(text).splitlines():
        ln = line.strip()
        if not ln:
            continue
        # slot at start: 1. or 1
        m = re.match(r"^(\d{1,3})\s*[\.\)]?\s+(.*)$", ln)
        if not m:
            continue
        slot = int(m.group(1))
        if slot < 1 or slot > draw_size:
            continue
        if slot in slot_seen:
            continue
        rest = m.group(2).strip()
        if not rest:
            continue

        seed = extract_seed(rest)
        name = normalize_draw_player_name(rest)
        if not name:
            continue

        rows.append({"slot": slot, "player_name": name, "seed": seed, "player_id_optional": pd.NA})
        slot_seen.add(slot)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("slot")
    return df


def parse_us_open_pdf_draw(pdf_path: Path) -> pd.DataFrame:
    """US Open draw PDFs are usually parseable via plain text extraction."""
    from pypdf import PdfReader  # lazy import so the rest of the package works without it

    reader = PdfReader(str(pdf_path))
    pages_text = []
    for p in reader.pages:
        t = p.extract_text() or ""
        pages_text.append(t)
    text = "\n".join(pages_text)
    # draw_size is not known from pdf itself in this function; caller should validate.
    # We parse with a permissive upper bound; caller slices/validates.
    return parse_generic_draw_text(text, draw_size=128)


def write_draw_csv(draw_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["slot", "player_name", "seed", "player_id_optional"]
    out = draw_df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[cols].copy()
    out["slot"] = out["slot"].astype(int)
    out.to_csv(output_path, index=False)


def run_confidence_checks(draw_df: pd.DataFrame, draw_size: int) -> tuple[bool, list[str]]:
    """Return (ok, warnings). If ok=False, output should be treated as partial."""
    warnings: list[str] = []
    ok = True

    if len(draw_df) != draw_size:
        ok = False
        warnings.append(f"Row count {len(draw_df)} != draw_size {draw_size}.")

    if "slot" not in draw_df.columns:
        return False, ["Missing slot column."]

    slots = sorted(int(x) for x in draw_df["slot"].tolist() if pd.notna(x))
    if slots != list(range(1, draw_size + 1)):
        ok = False
        missing = sorted(set(range(1, draw_size + 1)) - set(slots))
        extra = sorted(set(slots) - set(range(1, draw_size + 1)))
        warnings.append(f"Slots are not exactly 1..{draw_size}. missing={missing[:10]} extra={extra[:10]}")

    if "player_name" in draw_df.columns:
        names = [str(x).strip() for x in draw_df["player_name"].tolist()]
        if any(n == "" for n in names):
            ok = False
            warnings.append("Some player_name values are empty.")
        non_bye = [n for n in names if n.upper() != "BYE"]
        if len(set(non_bye)) != len(non_bye):
            ok = False
            warnings.append("Duplicate non-BYE player_name detected.")

    # Seed coverage heuristic (warning, not a hard fail for partial PDFs)
    seeds = []
    if "seed" in draw_df.columns:
        seeds = [int(s) for s in draw_df["seed"].tolist() if pd.notna(s)]
    expected_min_seeds = 32 if draw_size == 128 else 16 if draw_size == 64 else 0
    if expected_min_seeds and len(set(seeds)) < expected_min_seeds:
        warnings.append(
            f"Only found {len(set(seeds))} seeds; expected ~{expected_min_seeds} for draw_size={draw_size}."
        )

    return ok, warnings

