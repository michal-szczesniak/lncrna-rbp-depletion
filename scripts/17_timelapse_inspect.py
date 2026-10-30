"""Faza 1, krok 6B: co zawieraja przetworzone pliki TimeLapse-seq (format, geny, klasy)."""
import glob
import os
import re
import pandas as pd

D = "data/raw/timelapse/processed_data_files"
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
cls["gene"] = cls["gene_id"].str.split(".").str[0]
by_id = cls.drop_duplicates("gene").set_index("gene")["klasa"]
by_name = cls.drop_duplicates("gene_name").set_index("gene_name")["klasa"]

files = sorted(glob.glob(f"{D}/*.tsv"))
kinds = {}
for f in files:
    kinds.setdefault(re.sub(r"_rep\d", "_repN", os.path.basename(f)), []).append(f)
print(f"1. Plikow: {len(files)}; typow (bez numeru repliki): {len(kinds)}")
for k in sorted(kinds):
    print(f"   {k}  ({len(kinds[k])} repliki)")

print("\n2. Format i geny w przykladowych plikach (po jednym z kazdego rodzaju)")
seen = set()
for k in sorted(kinds):
    suffix = re.sub(r"^K562_(dox_)?[A-Za-z0-9]+_(chr|nuc|cyto|tot)_repN", "", k) or k
    group = ("dox" if "_dox_" in k else "plain", suffix)
    if group in seen:
        continue
    seen.add(group)
    p = kinds[k][0]
    d = pd.read_csv(p, sep="\t", nrows=300000)
    print(f"\n== {os.path.basename(p)}: {len(d)} wierszy (max 300 tys.), kolumny: {list(d.columns)}")
    print(d.head(3).to_string(max_colwidth=40))
    idcol = next((c for c in d.columns if d[c].astype(str).str.startswith("ENSG").mean() > 0.5), None)
    if idcol:
        k_ = d[idcol].astype(str).str.extract(r"(ENSG\d+)", expand=False).map(by_id)
        src = f"kolumna {idcol} (ENSG)"
    else:
        namecol = max(d.columns, key=lambda c: d[c].astype(str).isin(by_name.index).mean())
        k_ = d[namecol].astype(str).map(by_name)
        src = f"kolumna {namecol} (symbole)"
    print(f"   klasy genow wg {src}:", k_.value_counts(dropna=False).to_dict())
