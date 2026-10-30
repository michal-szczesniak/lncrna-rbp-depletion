"""
Test motywu: czy RBP od terminacji transkrypcji, konca 3' i wczesnego losu RNA swoiscie zmieniaja lncRNA
dywergentne i antysensowe?

Grupa zdefiniowana Z GORY (anotacja ENCODE "3' end processing" + znane kompleksy), nie z naszych wynikow.
Miara dla kazdego knockdownu (z kroku 5.1; efekty kategorii wzgledem mRNA o podobnej ekspresji i dlugosci):
  nadmiar = srednia(efekt dywergentnych, efekt antysensowych) - efekt miedzygenowych,
  srednia z knockdownow tego samego RBP, potem srednia z dwoch linii (lub jedna linia, jesli tylko tam jest RBP).
Test glowny (jeden, ustalony z gory): mediana |nadmiaru| w grupie vs losowe grupy tej samej wielkosci (10 000 permutacji).
Test na odlozonej grupie: bez 8 RBP, ktore podsunely hipoteze. Testy pomocnicze: kazda linia osobno (Mann-Whitney),
podgrupy (kierunek efektu), korekta BH dla testow pomocniczych.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

OUT = Path("results/phase4/network")
ALIASES = {"SKIV2L2": "MTREX"}
N_PERM = 10_000
rng = np.random.default_rng(11)
pd.set_option("display.width", 220)

SUB = {
    "koniec 3' / poliadenylacja": {"CPSF1", "CPSF2", "CPSF3", "CPSF4", "CPSF6", "CPSF7", "FIP1L1", "WDR33", "SYMPK", "CSTF1",
                                   "CSTF2", "CSTF2T", "CSTF3", "NUDT21", "PCF11", "CLP1", "PAPOLA", "PABPN1", "RBBP6"},
    "pauza i terminacja": {"NELFA", "NELFB", "NELFCD", "NELFE", "SCAF4", "SCAF8", "PPP1R10", "XRN2", "SETX", "WDR82",
                           "INTS1", "INTS3", "INTS4", "INTS5", "INTS6", "INTS7", "INTS8", "INTS9", "INTS10", "INTS11", "INTS12", "ZC3H4"},
    "czapeczka i wczesny los RNA": {"NCBP1", "NCBP2", "NCBP3", "SRRT", "PHAX", "ZC3H18"},
    "nadzor jadrowy": {"MTREX", "EXOSC1", "EXOSC2", "EXOSC3", "EXOSC4", "EXOSC5", "EXOSC6", "EXOSC7", "EXOSC8", "EXOSC9",
                       "EXOSC10", "DIS3", "ZCCHC8", "RBM7", "ZFC3H1"},
}
HYPOTHESIS_SOURCES = {"CPSF6", "FIP1L1", "SCAF8", "NCBP1", "PPP1R10", "NELFE", "MTREX", "EXOSC2"}


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


# ---------------------------------------------------------------- grupa z anotacji ENCODE + kompleksy
raw = pd.read_excel("data/ref/rbp_annotation/supp_data1.xlsx", header=None)
sub = [str(s).strip() for s in raw.iloc[1].tolist()]
ann = raw.iloc[2:].copy()
ann.columns = ["rbp", "gene"] + sub[2:]
ann["rbp"] = ann["rbp"].astype(str).replace(ALIASES)
enc3 = set(ann.loc[pd.to_numeric(ann.get("3' end processing"), errors="coerce") == 1, "rbp"]) if "3' end processing" in ann.columns else set()
SUB["koniec 3' / poliadenylacja"] |= enc3
GROUP = set().union(*SUB.values())

ce = pd.read_csv(OUT / "category_effects_per_kd.tsv", sep="\t")
ce["target"] = ce["target"].replace(ALIASES)
ce["nadmiar"] = ce[["dywergentne", "antysensowe"]].mean(axis=1) - ce["miedzygenowe"]
per_line = ce.groupby(["target", "linia"])[["nadmiar", "dywergentne", "antysensowe", "miedzygenowe"]].mean()
comb = per_line["nadmiar"].unstack("linia").mean(axis=1)                 # srednia z dostepnych linii
present = sorted(GROUP & set(comb.index))
print(f"0. Grupa a priori: {len(GROUP)} RBP (w tym {len(enc3)} z kategorii ENCODE \"3' end processing\"); "
      f"z danymi: {len(present)} z {comb.shape[0]} RBP")
for name, s in SUB.items():
    have = sorted(s & set(comb.index))
    print(f"   {name}: {', '.join(have) if have else '-'}")


def perm_test(values, members):
    a = values.reindex(list(members)).dropna().abs()
    obs = a.median()
    pool = values.dropna().abs().values
    null = np.array([np.median(rng.choice(pool, len(a), replace=False)) for _ in range(N_PERM)])
    return len(a), obs, np.median(pool), (np.sum(null >= obs) + 1) / (N_PERM + 1)


print("\n1. TEST GLOWNY: mediana |nadmiaru| (dywergentne i antysensowe minus miedzygenowe), srednia z linii")
n, obs, allmed, p = perm_test(comb, GROUP)
print(f"   grupa: n = {n}, mediana |nadmiaru| = {obs:.3f}; wszystkie RBP: {allmed:.3f}; p permutacyjne = {p:.1e}")
held = GROUP - HYPOTHESIS_SOURCES
n2, obs2, _, p2 = perm_test(comb, held)
print(f"\n2. Grupa odlozona (bez {', '.join(sorted(HYPOTHESIS_SOURCES))}): n = {n2}, mediana |nadmiaru| = {obs2:.3f}; p permutacyjne = {p2:.1e}")

print("\n3. Testy pomocnicze: kazda linia osobno (Mann-Whitney, |nadmiar| w grupie vs pozostale RBP; korekta BH)")
rows = []
for cell in ["K562", "HepG2"]:
    v = per_line["nadmiar"].xs(cell, level="linia")
    for label, members in [("cala grupa", GROUP), ("grupa odlozona", held)]:
        a, b = v.reindex(list(members)).dropna().abs(), v[~v.index.isin(members)].abs()
        rows.append({"linia": cell, "zestaw": label, "n": len(a), "mediana |nadmiar| grupa": a.median(),
                     "mediana pozostale": b.median(), "p": mannwhitneyu(a, b, alternative="greater").pvalue})
t3 = pd.DataFrame(rows)
t3["q"] = bh(t3["p"].values)
print(t3.round({c: 3 for c in t3.columns if c not in ("p", "q")}).to_string(index=False, formatters={"p": "{:.1e}".format, "q": "{:.1e}".format}))

print("\n4. Podgrupy: kierunek i wielkosc efektu (mediany RBP; nadmiar ze znakiem; efekt dywergentnych wzgledem mRNA)")
rows = []
for name, s in SUB.items():
    for cell in ["K562", "HepG2"]:
        v = per_line.xs(cell, level="linia").reindex(sorted(s)).dropna(how="all")
        if v.empty:
            continue
        rows.append({"podgrupa": name, "linia": cell, "RBP": len(v), "nadmiar (ze znakiem)": v["nadmiar"].median(),
                     "|nadmiar|": v["nadmiar"].abs().median(), "dywergentne": v["dywergentne"].median(),
                     "antysensowe": v["antysensowe"].median(), "miedzygenowe": v["miedzygenowe"].median()})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n5. RBP o najwiekszym |nadmiarze| (srednia z linii; G = w grupie a priori, H = zrodlo hipotezy)")
top = comb.abs().sort_values(ascending=False).head(20)
for r in top.index:
    tag = ("G" if r in GROUP else " ") + ("H" if r in HYPOTHESIS_SOURCES else " ")
    vals = ", ".join(f"{c} {per_line.loc[(r, c), 'nadmiar']:+.2f}" for c in ["K562", "HepG2"] if (r, c) in per_line.index)
    print(f"   {tag} {r:<10} {comb[r]:+.2f}  ({vals})")
comb.rename("nadmiar_srednia").to_frame().assign(w_grupie=lambda d: d.index.isin(GROUP)).to_csv(OUT / "termination_theme_per_rbp.tsv", sep="\t")
print(f"\nZapisano: {OUT}/termination_theme_per_rbp.tsv")
