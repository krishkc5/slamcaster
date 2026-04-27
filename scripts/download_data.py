 from __future__ import annotations
 
 import argparse
 from pathlib import Path
 
 import requests
 from tqdm import tqdm
 
 from slamcaster.config import get_paths
 from slamcaster.data_sources import SackmannRepo, atp_matches_filename, atp_players_filename, atp_rankings_filenames
 from slamcaster.load_data import ensure_data_dirs
 
 
 def _download(url: str, dest: Path, *, overwrite: bool = False, timeout_s: int = 60) -> None:
     if dest.exists() and dest.stat().st_size > 0 and not overwrite:
         return
     dest.parent.mkdir(parents=True, exist_ok=True)
     r = requests.get(url, stream=True, timeout=timeout_s)
     if r.status_code != 200:
         raise RuntimeError(f"Failed to download {url} (status={r.status_code})")
     total = int(r.headers.get("content-length", 0))
     with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
         for chunk in r.iter_content(chunk_size=1024 * 128):
             if not chunk:
                 continue
             f.write(chunk)
             pbar.update(len(chunk))
 
 
 def main() -> None:
     ap = argparse.ArgumentParser()
     ap.add_argument("--start-year", type=int, required=True)
     ap.add_argument("--end-year", type=int, required=True)
     ap.add_argument("--overwrite", action="store_true")
     args = ap.parse_args()
 
     paths = get_paths()
     ensure_data_dirs(paths.raw)
 
     repo = SackmannRepo()
     # Rankings + players
     for fn in atp_rankings_filenames() + [atp_players_filename()]:
         url = f"{repo.raw_base}/{fn}"
         _download(url, paths.raw / fn, overwrite=args.overwrite)
 
     # Matches by year
     for y in range(args.start_year, args.end_year + 1):
         fn = atp_matches_filename(y)
         url = f"{repo.raw_base}/{fn}"
         try:
             _download(url, paths.raw / fn, overwrite=args.overwrite)
         except RuntimeError:
             # Some years may not exist yet; skip gracefully.
             continue
 
 
 if __name__ == "__main__":
     main()
