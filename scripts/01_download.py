#!/usr/bin/env python3
"""
Pobieranie plikow ENCODE z tabeli inwentaryzacji, z kontrola md5 i wznawianiem.

Kazdy plik trafia do  <outdir>/<test>/<eksperyment>/<akcesja>.<rozszerzenie>.
Pliki juz pobrane z poprawnym md5 sa pomijane, wiec skrypt mozna bezpiecznie uruchamiac ponownie.

Przyklady:
  # najpierw zobacz, co i ile zostanie pobrane
  python scripts/01_download.py --files data/meta/files_to_download_phase0.tsv --set kd_gene_quant --dry-run
  # kwantyfikacje genowe knockdownow i kontroli
  python scripts/01_download.py --files data/meta/files_to_download_phase0.tsv --set kd_gene_quant --threads 8
  # wybrane pliki Bru, przez S3 (szybciej, wymaga awscli)
  python scripts/01_download.py --files data/meta/files_to_download_phase0.tsv --set bru_nonbam \
      --output-type "signal of unique reads" --method s3
"""
import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.encodeproject.org"


def md5sum(path, chunk=8 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dest_path(outdir, row):
    assay = re.sub(r"[^A-Za-z0-9]+", "_", str(row["assay"])).strip("_") or "unknown"
    name = Path(str(row["url"])).name or f"{row['file_accession']}.{row['file_format']}"
    return Path(outdir) / assay / str(row["experiment"]) / name


def make_session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=32))
    s.headers.update({"User-Agent": "rbp-classes-download/0.1"})
    return s


def fetch_https(session, url, part):
    if not url.startswith("http"):
        url = BASE + url
    with session.get(url, stream=True, timeout=600, allow_redirects=True) as r:
        r.raise_for_status()
        with open(part, "wb") as fh:
            for block in r.iter_content(chunk_size=8 << 20):
                fh.write(block)


def fetch_s3(s3_uri, part):
    cmd = ["aws", "s3", "cp", "--no-sign-request", "--only-show-errors", s3_uri, str(part)]
    subprocess.run(cmd, check=True)


def download_one(session, row, outdir, method, attempts=3):
    dest = dest_path(outdir, row)
    expected = str(row.get("md5", "")).strip()
    if dest.exists() and (not expected or md5sum(dest) == expected):
        return row["file_accession"], str(dest), "skipped", ""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    err = ""
    for i in range(1, attempts + 1):
        try:
            if method == "s3" and str(row.get("s3_uri", "")).startswith("s3://"):
                fetch_s3(row["s3_uri"], part)
            else:
                fetch_https(session, str(row["url"]), part)
            if expected and md5sum(part) != expected:
                raise IOError("md5 mismatch")
            part.replace(dest)
            return row["file_accession"], str(dest), "ok", ""
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            part.unlink(missing_ok=True)
            time.sleep(5 * i)
    return row["file_accession"], str(dest), "failed", err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", required=True, help="TSV z inwentaryzacji (np. files_to_download_phase0.tsv)")
    ap.add_argument("--outdir", default="data/raw/encode")
    ap.add_argument("--set", nargs="*", default=[], help="filtr kolumny 'set'")
    ap.add_argument("--output-type", nargs="*", default=[], help="filtr kolumny 'output_type'")
    ap.add_argument("--assay", nargs="*", default=[], help="filtr kolumny 'assay'")
    ap.add_argument("--format", nargs="*", default=[], help="filtr kolumny 'file_format'")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--method", choices=["https", "s3"], default="https")
    ap.add_argument("--max-gb", type=float, default=50, help="limit bezpieczenstwa; wiecej tylko z --force")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.files, sep="\t", dtype=str).fillna("")
    for col, vals in [("set", args.set), ("output_type", args.output_type),
                      ("assay", args.assay), ("file_format", args.format)]:
        if vals:
            if col not in df.columns:
                sys.exit(f"[blad] brak kolumny '{col}' w {args.files}")
            df = df[df[col].isin(vals)]
    df = df.drop_duplicates("file_accession")
    total_gb = pd.to_numeric(df["size"], errors="coerce").fillna(0).sum() / 1e9

    print(f"Plikow do pobrania: {len(df)}, lacznie {total_gb:.2f} GB")
    print(df.groupby(["assay", "output_type"]).size().rename("n").to_string())
    if args.dry_run:
        return
    if total_gb > args.max_gb and not args.force:
        sys.exit(f"[stop] {total_gb:.1f} GB przekracza --max-gb {args.max_gb}; dodaj --force, jesli to zamierzone")
    if args.method == "s3" and shutil.which("aws") is None:
        sys.exit("[blad] --method s3 wymaga awscli (jest w srodowisku rbp-py)")

    session = make_session()
    results = []
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futs = [pool.submit(download_one, session, r, args.outdir, args.method) for _, r in df.iterrows()]
        for n, f in enumerate(as_completed(futs), 1):
            acc, path, status, err = f.result()
            results.append({"file_accession": acc, "path": path, "status": status, "error": err})
            if status == "failed" or n % 25 == 0 or n == len(futs):
                print(f"[{n}/{len(futs)}] {acc} {status} {err}", flush=True)

    res = pd.DataFrame(results)
    manifest = Path(args.outdir) / "download_manifest.tsv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        old = pd.read_csv(manifest, sep="\t", dtype=str)
        res = pd.concat([old[~old["file_accession"].isin(res["file_accession"])], res])
    res.to_csv(manifest, sep="\t", index=False)
    counts = res["status"].value_counts().to_dict()
    print(f"Gotowe: {counts}. Manifest: {manifest}")
    if counts.get("failed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
