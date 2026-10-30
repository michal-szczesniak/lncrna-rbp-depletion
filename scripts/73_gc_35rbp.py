"""
Stabilnosc listy 35 RBP po uwzglednieniu GC (wyniki skryptu 68: warianty B, C = B + GC, B_gc = B po usunieciu trendu GC).
Regula wyboru jak w skrypcie 40 (funkcja summarize): w kazdej linii korekta BH po wszystkich eksperymentach;
dla RBP srednia roznica i najwieksze q z jego eksperymentow; RBP "wyrazny", jesli |efekt| >= 0,1 i q < 0,05 w obu liniach,
z tym samym znakiem. Wariant B powinien odtworzyc liste 35 RBP (sprawdzian zgodnosci z glowna analiza).
Wyjscie: results/phase4/followups/gc_35rbp.tsv
"""
from pathlib import Path

import numpy as np
import pandas as pd

F = Path("results/phase4/followups")
EFFECT, Q = 0.1, 0.05
pd.set_option("display.width", 200)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


res = pd.read_csv(F / "gc_class_effects.tsv.gz", sep="\t")
res["q"] = res.groupby(["wariant", "linia"])["p"].transform(bh)
prof = Path("results/phase4/rbp_profiles/rbp_profiles.tsv")
ref35 = set(pd.read_csv(prof, sep="\t", index_col=0).index) if prof.exists() else set()


def clear(r):
    w = r.groupby(["target", "linia"]).agg(roznica=("roznica", "mean"), q=("q", "max")).unstack("linia").dropna()
    ok = lambda l: (w[("roznica", l)].abs() >= EFFECT) & (w[("q", l)] < Q)
    same = np.sign(w[("roznica", "K562")]) == np.sign(w[("roznica", "HepG2")])
    return w, set(w.index[ok("K562") & ok("HepG2") & same])


W, H = {}, {}
for v in ["B", "C", "B_gc"]:
    W[v], H[v] = clear(res[res["wariant"] == v])

print("1. Liczba RBP z wyraznym efektem w obu liniach")
for v in ["B", "C", "B_gc"]:
    extra = f"; zgodnosc z lista 35 z glownej analizy: {len(H[v] & ref35)} z {len(ref35)}" if ref35 else ""
    print(f"   {v:5s}: {len(H[v])} RBP; wspolne z B: {len(H[v] & H['B'])}{extra}")

rows = []
for rbp in sorted(H["B"] | H["C"] | H["B_gc"]):
    r = {"RBP": rbp, "w_B": rbp in H["B"], "w_C": rbp in H["C"], "w_B_gc": rbp in H["B_gc"], "na_liscie_35": rbp in ref35}
    for v in ["B", "C"]:
        for l in ["K562", "HepG2"]:
            r[f"{v}_{l}"] = W[v].loc[rbp, ("roznica", l)] if rbp in W[v].index else np.nan
    rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(F / "gc_35rbp.tsv", sep="\t", index=False)

print("\n2. RBP, ktore zmieniaja status po dodaniu GC (C wzgledem B)")
diff = t[t["w_B"] != t["w_C"]]
if diff.empty:
    print("   brak - lista identyczna")
else:
    print(diff[["RBP", "w_B", "w_C", "w_B_gc", "B_K562", "B_HepG2", "C_K562", "C_HepG2"]].round(3).to_string(index=False))
print(f"\n3. Kierunek: RBP podnoszace / obnizajace lncRNA (B: {sum(t.loc[t.w_B, 'B_K562'] > 0)} / {sum(t.loc[t.w_B, 'B_K562'] < 0)}; "
      f"C: {sum(t.loc[t.w_C, 'C_K562'] > 0)} / {sum(t.loc[t.w_C, 'C_K562'] < 0)})")
print(f"\nZapisano: {F / 'gc_35rbp.tsv'}")
