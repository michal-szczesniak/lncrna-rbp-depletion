"""
98 (20.10): laboratoria ENCODE, ktore wytworzyly uzywane dane (do podziekowan w manuskrypcie).
Szuka kolumn z 'lab' w plikach tabelarycznych w data/meta i data/raw/encode (bez podkatalogow i bez plikow > 300 MB)
i wypisuje liczebnosci laboratoriow, w miare mozliwosci w podziale na typ testu i linie komorkowa.
Tylko czyta; nic nie zapisuje poza logiem.
"""
import re
from pathlib import Path

import pandas as pd

found = False
for d in ["data/meta", "data/raw/encode", "data/raw", "data"]:
    p = Path(d)
    if not p.is_dir():
        continue
    for f in sorted(p.iterdir()):
        if not f.is_file() or f.stat().st_size > 300e6 or not re.search(r"\.(tsv|csv|txt)(\.gz)?$", f.name):
            continue
        sep = "," if ".csv" in f.name else "\t"
        try:
            head = pd.read_csv(f, sep=sep, nrows=5, dtype=str)
        except Exception as e:  # noqa: BLE001
            print(f"   {f}: nie czytam ({e.__class__.__name__})")
            continue
        labc = [c for c in head.columns if re.search(r"(^|[_ ])lab($|[_ ])|^Lab$|laborator", c, re.I)]
        if not labc:
            continue
        found = True
        t = pd.read_csv(f, sep=sep, dtype=str, usecols=lambda c: True)
        print(f"\n== {f} ({len(t)} wierszy), kolumny z 'lab': {labc}")
        grp = [c for c in t.columns if re.search(r"assay|Assay|biosample|Biosample term name|cell", c)][:2]
        for c in labc:
            if grp:
                print(t.groupby(grp + [c], dropna=False).size().to_string())
            else:
                print(t[c].value_counts(dropna=False).to_string())
if not found:
    print("Brak kolumny z laboratorium w data/meta, data/raw/encode, data/raw, data.")
    print("Lista plikow tabelarycznych w data/meta i data/raw/encode (nazwa, kolumny):")
    for d in ["data/meta", "data/raw/encode"]:
        for f in sorted(Path(d).glob("*.tsv*")):
            try:
                print(f"   {f}: {list(pd.read_csv(f, sep=chr(9), nrows=1).columns)[:25]}")
            except Exception:  # noqa: BLE001
                pass
print("\nGotowe.")
