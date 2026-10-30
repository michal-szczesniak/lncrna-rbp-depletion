"""
97_checks.py (20.10) - sprawdzenia do rejestru liczb i tabeli S5. Nic nie zmienia, tylko wypisuje.
  1. N107: mediana class effect lncRNA dywergentnych po wyciszeniu czynnikow nadzoru jadrowego w HepG2 (w tekscie 0,38;
     w A6_position_per_rbp.tsv 0,375) - z 5 miejscami po przecinku, na poziomie RBP i eksperymentow; tez NAT, w poblizu,
     miedzygenowe (0,24; 0,12; 0,17)
  2. definicje wariantow z Ryc. 1E do tabeli S5: 'ekspresja >= 50', 'bez splicingu i NMD', 'B_mRNA_bez_RI'
     (fragmenty kodu skryptow, ktore je licza) i miara podobienstwa RBP-RBP z Ryc. S4 (51_rbp_networks.py)
  3. czy w danych ENCODE na klastrze sa pliki wynikow po korekcie ComBat / z wirtualna kontrola (rekord D05)
"""
import re
import sys
from pathlib import Path

import pandas as pd

if not Path("results/phase2").is_dir():
    sys.exit("Uruchom z katalogu projektu rbp_classes.")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.precision", 5)
SURV = {"MTREX", "EXOSC1", "EXOSC2", "EXOSC3", "EXOSC4", "EXOSC5", "EXOSC6", "EXOSC7", "EXOSC8", "EXOSC9", "EXOSC10",
        "DIS3", "ZCCHC8", "RBM7", "ZFC3H1", "PABPN1"}
CATS = ["dywergentne", "antysensowe", "w poblizu", "miedzygenowe"]


def sec(t):
    print(f"\n== {t}", flush=True)


sec("1. N107-N110: mediany class effect kategorii po wyciszeniu nadzoru jadrowego (HepG2 i K562)")
ce = pd.read_csv("results/phase4/network/category_effects_per_kd.tsv", sep="\t")
for cell in ["HepG2", "K562"]:
    g = ce[(ce["linia"] == cell) & (ce["nadzor"])]
    per_rbp = g.groupby("target")[CATS].mean()
    print(f"   {cell}: {len(g)} wyciszen, {len(per_rbp)} RBP ({', '.join(sorted(per_rbp.index))})")
    print("     mediana po wyciszeniach: " + "; ".join(f"{c} {g[c].median():.5f}" for c in CATS))
    print("     mediana po RBP:          " + "; ".join(f"{c} {per_rbp[c].median():.5f}" for c in CATS))
f = Path("results/audit/A6_position_per_rbp.tsv")
if f.exists():
    a6 = pd.read_csv(f, sep="\t")
    print(f"   {f}: kolumny {list(a6.columns)}")
    print(a6.head(30).to_string(index=False))

sec("2. Definicje wariantow (fragmenty kodu)")
PATTERNS = [("ekspresja >= 50", 6), ("bez splicingu i NMD", 12), ("B_mRNA_bez_RI", 8), ("retained_intron", 4)]
for script in sorted(Path("scripts").glob("*.py")):
    try:
        lines = script.read_text(errors="replace").splitlines()
    except Exception:  # noqa: BLE001
        continue
    for pat, ctx in PATTERNS:
        for i, ln in enumerate(lines):
            if pat in ln:
                print(f"--- {script}:{i + 1} ({pat})")
                print("\n".join(f"{j + 1:5d}  {lines[j]}" for j in range(max(0, i - ctx), min(len(lines), i + ctx + 1))))
                break
s51 = Path("scripts/51_rbp_networks.py")
if s51.exists():
    print(f"--- {s51}: linie z corr / spearman / pearson / cosine")
    for i, ln in enumerate(s51.read_text(errors="replace").splitlines()):
        if re.search(r"corr|spearman|pearson|cosine", ln, re.I):
            print(f"{i + 1:5d}  {ln}")

sec("3. Pliki ENCODE: typy wynikow (szukam batch / ComBat / virtual)")
man = Path("data/raw/encode/download_manifest.tsv")
if man.exists():
    m = pd.read_csv(man, sep="\t", dtype=str)
    print(f"   {man}: {len(m)} wierszy; kolumny: {list(m.columns)}")
    for c in m.columns:
        if m[c].nunique() < 60:
            vc = m[c].value_counts()
            print(f"   {c}: " + "; ".join(f"{k} ({v})" for k, v in vc.items()))
    hit = m[m.apply(lambda r: r.astype(str).str.contains("batch|combat|virtual", case=False).any(), axis=1)]
    print(f"   wiersze z batch/combat/virtual: {len(hit)}")
    print(hit.head(10).to_string(index=False))
for p_ in ["data/meta", "data/raw/encode"]:
    for f_ in sorted(Path(p_).glob("*")):
        if re.search(r"batch|combat|virtual", f_.name, re.I):
            print(f"   plik: {f_}")
print("\nGotowe.")
