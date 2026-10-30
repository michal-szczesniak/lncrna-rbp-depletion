"""
95_number_checks.py (19.10) - liczby z manuskryptu, ktorych skan 90_number_provenance.py nie przypisal jednoznacznie do
pliku wynikowego. Wypisuje wartosci z plikow projektu z identyfikatorami liczb (N... z rejestru liczb, lista z 19.10).
Nic nie zmienia. Uruchamiac PO skryptach rycin z rundy 2 (czyta m.in. fig3/b_spliceosom_testy.tsv).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if not Path("results/phase2").is_dir():
    sys.exit("Uruchom z katalogu projektu rbp_classes.")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
FP = Path("results/figures_pub")
NMDF = ["UPF1", "UPF2", "UPF3B", "SMG1", "SMG5", "SMG6", "SMG7", "SMG8", "SMG9"]


def sec(t):
    print(f"\n== {t}", flush=True)


def safe(f):
    try:
        f()
    except Exception as e:  # noqa: BLE001
        print(f"   (nie powiodlo sie: {e.__class__.__name__}: {e})")


summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t")


def c1():
    sec("N001/N002/N166/N167/N169: liczba eksperymentow i RBP")
    print(f"   summary_all.tsv: {len(summ)} eksperymentow, {summ['target'].nunique()} unikalnych RBP; "
          f"wg linii: {summ.groupby('cell').size().to_dict()}; wg metody: {summ.groupby('method').size().to_dict()}")
    for f in sorted(Path("data/meta").glob("*.tsv")):
        try:
            d = pd.read_csv(f, sep="\t", dtype=str)
        except Exception:  # noqa: BLE001
            continue
        tcol = next((c for c in d.columns if c.lower() in ("target", "rbp", "target_label", "target_gene")), None)
        if tcol is None:
            continue
        line = f"   {f}: {len(d)} wierszy, unikalnych {tcol}: {d[tcol].nunique()}"
        for c in d.columns:
            v = d[c].astype(str)
            if v.str.contains("total", case=False).any() and v.str.contains("poly", case=False).any():
                for val, g in d.groupby(c):
                    line += f"\n      {c} = {val}: {len(g)} wierszy, {g[tcol].nunique()} RBP"
                line += f"\n      razem (bez kontroli, jesli sa oznaczone): {d[tcol].nunique()} RBP"
        print(line)


def c2():
    sec("N022/N023: mediana |class effect| (glowna analiza)")
    res = pd.read_csv("results/phase2/classes/class_effects_all.tsv.gz", sep="\t")
    m = res[(res["klasa"] == "lncRNA (razem)") & (res["macierz"] == "skorygowana") & (res["dopasowanie"] == "B")]
    a = m["roznica"].abs().median()
    print(f"   {len(m)} eksperymentow; mediana |ce| = {a:.4f} log2 = {100 * (2 ** a - 1):.1f}%")
    sec("N091-N096: percentyle class effect wyciszen czynnikow NMD (B = glowna, C = z GC) i mediany linii")
    gc = pd.read_csv("results/phase4/followups/gc_class_effects.tsv.gz", sep="\t")
    for name, d in [("B", m[["linia", "kd_experiment", "target", "roznica"]]),
                    ("C", gc.loc[gc["wariant"] == "C", ["linia", "kd_experiment", "target", "roznica"]])]:
        d = d.copy()
        d["centyl"] = d.groupby("linia")["roznica"].rank(pct=True) * 100
        med = d.groupby("linia")["roznica"].median()
        print(f"   {name}: mediany linii {med.round(4).to_dict()}")
        print(d[d["target"].isin(NMDF)].sort_values(["linia", "target"]).round(3).to_string(index=False))


def c3():
    sec("N066: Ryc. 2A (a_dopasowania.tsv)")
    print(pd.read_csv(FP / "fig2/a_dopasowania.tsv", sep="\t").round(4).to_string(index=False))


def c4():
    sec("N067-N071: eCLIP (Ryc. 2C, 2D)")
    ecl = pd.read_csv("results/phase3/eclip/eclip_directness.tsv", sep="\t")
    both = set(ecl.loc[ecl["linia"] == "K562", "RBP"]) & set(ecl.loc[ecl["linia"] == "HepG2", "RBP"])
    print(f"   RBP z eCLIP i wyciszeniem w obu liniach: {len(both)}")
    for cell in ["K562", "HepG2"]:
        for g in ["mRNA", "lncRNA"]:
            x = ecl.loc[ecl["linia"] == cell, f"abs_{g}"].dropna()
            print(f"   {cell} {g}: n = {len(x)}, mediana roznicy |log2FC| zwiazane - niezwiazane = {x.median():+.4f}")
    ta = pd.read_csv("results/phase4/followups/eclip_top_responders.tsv", sep="\t")
    for (cell, g), s in ta.groupby(["linia", "geny"]):
        sig = (s["q"] < 0.05) & (s["OR_MH"] > 1)
        print(f"   top 5%: {cell} {g}: wzbogacenie istotne dla {int(sig.sum())} z {len(s)} RBP ({100 * sig.mean():.1f}%)")


def c5():
    sec("N084/N085: Ryc. 3B, przedzialy ekspresji (b_spliceosom_testy.tsv)")
    t = pd.read_csv(FP / "fig3/b_spliceosom_testy.tsv", sep="\t")
    print(t.round(4).to_string(index=False))
    for cell, g in t.groupby("linia"):
        k = ((g["q"] < 0.05) & (g["mediana_roznicy_mRNA_minus_lncRNA"] > 0)).sum()
        print(f"   {cell}: mRNA istotnie wyzej niz lncRNA w {k} z {len(g)} przedzialow")


def c6():
    sec("N118/N119: KCNQ1OT1 (profil z Ryc. 4E) i dlugosc genu")
    print(pd.read_csv(FP / "fig4/d_KCNQ1OT1_profil.tsv", sep="\t").round(3).to_string(index=False))
    cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
    print(cls[cls["gene_name"] == "KCNQ1OT1"].to_string(index=False))
    sec("N120/N121: powtarzalnosc wspolodpowiedzi sasiadow (e_powtarzalnosc.tsv)")
    print(pd.read_csv(FP / "fig4/e_powtarzalnosc.tsv", sep="\t").round(4).to_string(index=False))


def c7():
    sec("N132-N135, N156: NORAD po wyciszeniu SRSF1, RCC2, EIF3D (log2FC surowe i po korekcie; centyl wsrod wyciszen)")
    cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
    gid = cls.loc[cls["gene_name"].isin(["NORAD", "LINC00657"]), "gene_id"]
    M = Path("results/phase2/matrix")
    for cell in ["K562", "HepG2"]:
        for kind in ["lfc", "lfc_bc"]:
            f = M / f"{kind}_{cell}.tsv.gz"
            if not f.exists():
                print(f"   brak {f}"); continue
            L = pd.read_csv(f, sep="\t", index_col=0)
            g = [x for x in gid if x in L.index]
            if not g:
                print(f"   {cell} {kind}: NORAD nieobecny w macierzy"); continue
            v = L.loc[g[0]]
            pct = v.rank(pct=True) * 100
            t = summ.set_index("kd_experiment").reindex(v.index)["target"]
            sel = t[t.isin(["SRSF1", "RCC2", "EIF3D"])].index
            print(f"   {cell} {kind}: " + "; ".join(f"{t[k]} {v[k]:+.2f} (centyl {pct[k]:.1f})" for k in sel))


for f in [c1, c2, c3, c4, c5, c6, c7]:
    safe(f)
print("\nGotowe.")
