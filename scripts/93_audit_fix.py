"""
93_audit_fix.py - dokonczenie A5 i A6 z audytu (19.10). Niczego nie nadpisuje.
A5  RBP z wyraznym efektem (|ce| >= 0.1, q < 0.05 w obu liniach, ten sam znak): wariant glowny (B), z dopasowaniem GC (C),
    oba; oczekiwana liczba losowa (permutacja etykiet RBP w HepG2, 1000x). q liczone metoda BH z p; sprawdzam, ktory
    sposob liczenia q i laczenia eksperymentow tego samego RBP odtwarza 35 RBP z rbp_hits.tsv i 17 z tableS1_rbp.tsv.
A6  class effect kategorii polozenia (results/phase4/network/category_effects_per_kd.tsv) dla czynnikow nadzoru, NELFE, PPP1R10.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if not Path("results/phase2/matrix").is_dir():
    sys.exit("Uruchom z katalogu projektu rbp_classes.")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
rng = np.random.default_rng(3)
CELLS = ["K562", "HepG2"]
OUT = Path("results/audit"); OUT.mkdir(parents=True, exist_ok=True)


def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = np.isfinite(p)
    pv = p[ok]; o = np.argsort(pv); m = len(pv)
    qq = np.minimum.accumulate((pv[o] * m / (np.arange(m) + 1))[::-1])[::-1]
    t = np.empty(m); t[o] = np.minimum(qq, 1); q[ok] = t
    return q


res = pd.read_csv("results/phase2/classes/class_effects_all.tsv.gz", sep="\t")
B = res[(res["klasa"] == "lncRNA (razem)") & (res["macierz"] == "skorygowana") & (res["dopasowanie"] == "B")].copy()
gc = pd.read_csv("results/phase4/followups/gc_class_effects.tsv.gz", sep="\t")
Cv = gc[gc["wariant"] == "C"].copy()
hits = set(pd.read_csv("results/phase2/classes/rbp_hits.tsv", sep="\t", index_col=0).index)
s1 = pd.read_csv("results/figures_pub/tableS1_rbp.tsv", sep="\t", index_col=0)
robust = set(s1.index[s1["robust_to_GC"].astype(str) == "True"])
print(f"Odniesienie: {len(hits)} RBP (rbp_hits), {len(robust)} odpornych na GC (tableS1)")


def add_q(d, mode):
    d = d.copy()
    if mode == "wszystkie":
        d["q"] = bh(d["p"])
    else:
        d["q"] = np.nan
        for cell in CELLS:
            m = d["linia"] == cell
            d.loc[m, "q"] = bh(d.loc[m, "p"])
    return d


def table(d, qrule):
    g = d.groupby(["target", "linia"]).agg(ce=("roznica", "mean"), qmax=("q", "max"), qmin=("q", "min")).unstack("linia").dropna()
    return g[[("ce", "K562"), ("ce", "HepG2"), (qrule, "K562"), (qrule, "HepG2")]].set_axis(["ceK", "ceH", "qK", "qH"], axis=1)


def clear(g):
    return (g["ceK"].abs() >= 0.1) & (g["ceH"].abs() >= 0.1) & (g["qK"] < 0.05) & (g["qH"] < 0.05) & (np.sign(g["ceK"]) == np.sign(g["ceH"]))


best = None
for mode in ["wszystkie", "w linii"]:
    for qrule in ["qmax", "qmin"]:
        gB = table(add_q(B, mode), qrule); gC = table(add_q(Cv, mode), qrule)
        okB = clear(gB); okC = clear(gC).reindex(gB.index).fillna(False)
        sB, sBC = set(okB[okB].index), set((okB & okC)[okB & okC].index)
        score = len(sB ^ hits) + len(sBC ^ robust)
        print(f"  q {mode}, {qrule}: B {len(sB)} (zgodnych z 35: {len(sB & hits)}), B i C {len(sBC)} (zgodnych z 17: {len(sBC & robust)})")
        if best is None or score < best[0]:
            best = (score, mode, qrule, gB, gC)
_, mode, qrule, gB, gC = best
print(f"Dalej: q {mode}, {qrule}")
common = gB.index.intersection(gC.index)
gB, gC = gB.loc[common], gC.loc[common]
okB, okC = clear(gB), clear(gC)
obs = {"B": int(okB.sum()), "C": int(okC.sum()), "B i C": int((okB & okC).sum())}
null = {k: [] for k in obs}
for _ in range(1000):
    p = rng.permutation(len(common))
    b2, c2 = gB.copy(), gC.copy()
    for g2, g in [(b2, gB), (c2, gC)]:
        g2["ceH"] = g["ceH"].to_numpy()[p]; g2["qH"] = g["qH"].to_numpy()[p]
    o1, o2 = clear(b2), clear(c2)
    null["B"].append(int(o1.sum())); null["C"].append(int(o2.sum())); null["B i C"].append(int((o1 & o2).sum()))
print("\n== A5. RBP z wyraznym efektem: obserwowane i losowo (1000 permutacji)")
for k in obs:
    n = np.array(null[k])
    print(f"   {k}: obserwowane {obs[k]}, losowo srednio {n.mean():.1f} (95% {np.percentile(n, 2.5):.0f}-{np.percentile(n, 97.5):.0f}), "
          f"P(losowo >= obserwowane) = {(np.sum(n >= obs[k]) + 1) / 1001:.3f}")
print(f"   rho miedzy liniami: B {spearmanr(gB['ceK'], gB['ceH']).statistic:.3f}, C {spearmanr(gC['ceK'], gC['ceH']).statistic:.3f} (n = {len(common)})")
pd.DataFrame(null).to_csv(OUT / "A5_null_counts.tsv", sep="\t", index=False)
dirC = gC[okC].assign(kierunek=np.where(gC.loc[okC, "ceK"] > 0, "lncRNA wyzej", "lncRNA nizej"))
print(f"   wariant C: {len(dirC)} RBP, kierunek: {dirC['kierunek'].value_counts().to_dict()}")
print("   " + ", ".join(sorted(dirC.index)))

print("\n== A6. Kategorie polozenia (category_effects_per_kd.tsv)")
ce = pd.read_csv("results/phase4/network/category_effects_per_kd.tsv", sep="\t")
print(f"   kolumny: {list(ce.columns)}")
CATS = ["dywergentne", "antysensowe", "w poblizu", "miedzygenowe"]
ce["nadmiar"] = ce["dywergentne"] - ce["miedzygenowe"]
t = ce.groupby(["target", "linia"])[CATS + ["nadmiar"]].mean()
t["ranga_nadmiaru"] = t.groupby(level="linia")["nadmiar"].rank(ascending=False)
t["n"] = t.groupby(level="linia")["nadmiar"].transform("count")
for c in CATS:
    t[f"centyl_{c}"] = t.groupby(level="linia")[c].rank(pct=True).mul(100)
sel = ["MTREX", "NELFE", "PPP1R10", "EXOSC2", "EXOSC5", "EXOSC9", "EXOSC10", "XPO1", "SSB", "KPNB1"]
print(t[t.index.get_level_values(0).isin(sel)].round(3).to_string())
t.to_csv(OUT / "A6_position_per_rbp.tsv", sep="\t")
if "nadzor" in ce.columns:
    print("\n   czynniki nadzoru (mediana po wyciszeniach) i pozostale RBP:")
    print(ce.groupby(["linia", "nadzor"])[CATS].median().round(3).to_string())
print("\nGotowe.")
