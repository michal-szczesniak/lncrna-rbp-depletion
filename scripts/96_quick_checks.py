"""
96_quick_checks.py v2 (20.10) - liczba RBP (N002/N012/N166), ktorej wersja z 19.10 nie rozstrzygnela: liczyla kolumny
macierzy lfc_bc (815 eksperymentow po korekcie na wspolna kontrole) zamiast macierzy lfc (860 eksperymentow poly(A)).
Teraz: kolumny obu macierzy, z podzialem na linie, i porownanie z summary_all.tsv (876 eksperymentow, w tym total RNA).
Nic nie zmienia.
"""
import sys
from pathlib import Path

import pandas as pd

if not Path("results/phase2").is_dir():
    sys.exit("Uruchom z katalogu projektu rbp_classes.")
pd.set_option("display.width", 200)
M = Path("results/phase2/matrix")
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
print(f"summary_all.tsv: {len(summ)} eksperymentow, {summ['target'].nunique()} RBP; kolumny: {list(summ.columns)}")

sets = {}
for kind in ["lfc", "lfc_bc"]:
    print(f"\n== Macierz {kind}")
    kds = []
    for cell in ["K562", "HepG2"]:
        f = M / f"{kind}_{cell}.tsv.gz"
        if not f.exists():
            print(f"   brak {f}")
            continue
        cols = pd.read_csv(f, sep="\t", index_col=0, nrows=1).columns.tolist()
        kds += cols
        t = summ.reindex(cols)["target"]
        print(f"   {cell}: {len(cols)} eksperymentow, {t.nunique()} RBP, brak w summary_all: {t.isna().sum()}")
    t = summ.reindex(kds)["target"]
    sets[kind] = set(t.dropna())
    print(f"   razem: {len(kds)} eksperymentow, {t.nunique()} RBP")

allr = set(summ["target"])
print(f"\nRBP tylko w summary_all (poza macierza lfc): {sorted(allr - sets.get('lfc', set()))}")
print(f"RBP w lfc, ale nie w lfc_bc: {sorted(sets.get('lfc', set()) - sets.get('lfc_bc', set()))}")
for c in [c for c in summ.columns if c.lower() in ("assay", "rna_type", "library", "biosample", "method", "cell")]:
    print(f"\nsummary_all wg {c}:")
    print(summ.groupby(c)["target"].agg(["size", "nunique"]).to_string())
print("\nGotowe.")
