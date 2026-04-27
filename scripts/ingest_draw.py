from __future__ import annotations

import argparse
from pathlib import Path

from slamcaster.config import get_paths
from slamcaster.draw_ingest import (
    download_file,
    parse_generic_draw_text,
    parse_us_open_pdf_draw,
    run_confidence_checks,
    write_draw_csv,
)


def _extract_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    texts = []
    for p in reader.pages:
        texts.append(p.extract_text() or "")
    return "\n".join(texts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-url", default=None)
    ap.add_argument("--source-file", default=None)
    ap.add_argument("--tournament", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--draw-size", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--parser", required=True, choices=["usopen_pdf", "generic_pdf_text", "manual_text"])
    args = ap.parse_args()

    if not args.source_url and not args.source_file:
        raise SystemExit("Provide either --source-url or --source-file")
    if args.source_url and args.source_file:
        raise SystemExit("Provide only one of --source-url or --source-file")

    paths = get_paths()
    source_path: Path
    if args.source_url:
        src_dir = paths.raw / "draw_sources" / f"{args.year}"
        src_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(args.source_url).name or "draw_source.pdf"
        source_path = download_file(args.source_url, src_dir / filename)
        print(f"Downloaded source to {source_path}")
    else:
        source_path = Path(args.source_file).expanduser().resolve()
        if not source_path.exists():
            raise SystemExit(f"Missing --source-file: {source_path}")

    draw_size = int(args.draw_size)
    out_path = Path(args.output)

    if args.parser == "usopen_pdf":
        df = parse_us_open_pdf_draw(source_path)
        df = df[df["slot"].between(1, draw_size)].copy()
    elif args.parser == "generic_pdf_text":
        text = _extract_pdf_text(source_path)
        df = parse_generic_draw_text(text, draw_size=draw_size)
    else:
        text = source_path.read_text(encoding="utf-8")
        df = parse_generic_draw_text(text, draw_size=draw_size)

    ok, warnings = run_confidence_checks(df, draw_size)
    if not ok:
        partial = out_path.with_suffix(out_path.suffix + ".partial.csv")
        write_draw_csv(df, partial)
        print("PARSING CONFIDENCE LOW — wrote partial draw CSV.")
        for w in warnings:
            print(f"- {w}")
        print(f"Partial output: {partial}")
        raise SystemExit(
            "Please manually correct the draw order/slots based on the official draw before using it."
        )

    write_draw_csv(df, out_path)
    for w in warnings:
        print(f"WARNING: {w}")
    print(f"Wrote draw CSV to {out_path}")
    print("Please manually compare the generated CSV against the official draw before using it for final results.")


if __name__ == "__main__":
    main()

