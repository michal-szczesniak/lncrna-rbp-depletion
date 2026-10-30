#!/usr/bin/env bash
# Pobiera GTF GENCODE w wersji uzytej przez ENCODE w kwantyfikacjach knockdownow.
# Numer wersji odczytaj z data/meta/summary.md ("Wersje anotacji", np. V29 -> 29).
# Uzycie: bash scripts/02_get_gencode.sh 29
set -euo pipefail
V=${1:?podaj numer wersji GENCODE, np. 29}
OUT=data/ref
mkdir -p "$OUT"
BASE="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${V}"
for f in "gencode.v${V}.annotation.gtf.gz" "gencode.v${V}.long_noncoding_RNAs.gtf.gz"; do
  wget -c -P "$OUT" "$BASE/$f"
done
if wget -q -O "$OUT/MD5SUMS_v${V}" "$BASE/MD5SUMS"; then
  (cd "$OUT" && grep -E "gencode\.v${V}\.(annotation|long_noncoding_RNAs)\.gtf\.gz" "MD5SUMS_v${V}" | md5sum -c -)
else
  echo "[uwaga] brak pliku MD5SUMS dla release_${V}; pomijam weryfikacje"
fi
