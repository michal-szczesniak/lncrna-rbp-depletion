"""
Faza 2, krok 2.6B: spojnosc funkcjonalna efektow na lncRNA.

Pytanie: czy RBP z tej samej kategorii funkcjonalnej (anotacja ENCODE, Van Nostrand i wsp. 2020, Supplementary Data 1)
przesuwaja lncRNA w tym samym kierunku? Test nie zalezy od progow istotnosci i jest odporny na efekty
niespecyficzne pojedynczych odczynnikow (kazde bialko wyciszano innym shRNA/gRNA).

Efekt RBP = lncRNA (razem) minus dopasowane mRNA (macierz skorygowana, dopasowanie B), srednia z dostepnych linii.
1. Dla kazdej kategorii funkcji i lokalizacji: mediana efektu czlonkow vs pozostalych RBP (Mann-Whitney, BH),
   spojnosc znaku, mediana osobno dla CRISPR i shRNA oraz dla genow niezbednych i pozostalych.
2. Profil klas (lincRNA, antysens, ... ) dla kategorii funkcji.
3. Geny gospodarze snoRNA: czy w danej kategorii zachowuja sie inaczej niz lncRNA razem
   (dla kazdego RBP roznica: gospodarze snoRNA minus lncRNA razem; test znakow rangowanych).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

C = Path("results/phase2/classes")
ANN = "data/ref/rbp_annotation/supp_data1.xlsx"
ALIASES = {"MTREX": ["SKIV2L2"], "ATP5A1": ["ATP5F1A"], "ATP5C1": ["ATP5F1C"], "GNB2L1": ["RACK1"]}
LNC = "lncRNA (razem)"
PROFILE = [LNC, "lincRNA", "antysens", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA", "pseudogen transkr."]
MIN_MEMBERS = 5
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 40)


def bh(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    q = p[ok]
    o = np.argsort(q)
    r = q[o] * len(q) / np.arange(1, len(q) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    tmp = np.empty_like(r)
    tmp[o] = np.minimum(r, 1)
    out[ok] = tmp
    return out


# ---------------------------------------------------------------- anotacja (dwa wiersze naglowka)
raw = pd.read_excel(ANN, header=None)
top, sub = raw.iloc[0].tolist(), raw.iloc[1].tolist()
group, cur = [], None
for t in top:
    cur = t if isinstance(t, str) else cur
    group.append(cur)
names = ["rbp", "gene"] + [str(s).strip() for s in sub[2:]]
ann = raw.iloc[2:].copy()
ann.columns = names
ann["gene"] = ann["gene"].astype(str).str.split(".").str[0]
FUNC = [n for n, g in zip(names, group) if g == "Functions" and n != "Essential Genes"]
LOC = [n for n, g in zip(names, group) if g == "Localization"]
for c in FUNC + LOC + ["Essential Genes"]:
    ann[c] = pd.to_numeric(ann[c], errors="coerce")
print(f"0. Anotacja: {len(ann)} RBP; kategorie funkcji: {len(FUNC)}, lokalizacji: {len(LOC)}")

# ---------------------------------------------------------------- efekty RBP
res = pd.read_csv(C / "class_effects_all.tsv.gz", sep="\t")
main = res[(res["macierz"] == "skorygowana") & (res["dopasowanie"] == "B")]
eff = main.groupby(["target", "klasa"])["roznica"].mean().unstack("klasa")       # srednia z linii
eff_line = main[main["klasa"] == LNC].groupby(["target", "linia"])["roznica"].mean().unstack("linia")
meth = main.groupby("target")["metoda"].agg(lambda s: "/".join(sorted(set(s))))

genes = pd.read_csv("data/ref/genes_v29.tsv", sep="\t")
genes["gene"] = genes["gene_id"].str.split(".").str[0]
name2gene = genes.drop_duplicates("gene_name").set_index("gene_name")["gene"]
ann_by_gene = ann.drop_duplicates("gene").set_index("gene")
ann_by_name = ann.drop_duplicates("rbp").set_index("rbp")
rows = {}
for t in eff.index:
    hit = None
    for n in [t] + ALIASES.get(t, []):
        g = name2gene.get(n)
        if g is not None and g in ann_by_gene.index:
            hit = ann_by_gene.loc[g]
            break
        if n in ann_by_name.index:
            hit = ann_by_name.loc[n]
            break
    if hit is not None:
        rows[t] = hit
A = pd.DataFrame(rows).T
print(f"   RBP z efektem: {len(eff)}; z anotacja ENCODE: {len(A)} (pozostale to glownie RBP dodane w ENCODE4)")

d = eff[[LNC]].rename(columns={LNC: "efekt"}).join(eff_line).join(meth.rename("metoda"))
d = d.join(A[FUNC + LOC + ["Essential Genes"]], how="inner")
d["niezbedny"] = d["Essential Genes"] == 1

# ---------------------------------------------------------------- 1. kategorie
rows = []
for cat in FUNC + LOC:
    m = d[cat] == 1
    nm = d[cat] == 0
    if m.sum() < MIN_MEMBERS or nm.sum() < MIN_MEMBERS:
        continue
    x, y = d.loc[m, "efekt"].dropna(), d.loc[nm, "efekt"].dropna()
    med = x.median()
    sub_m = d[m]
    rows.append({"kategoria": cat, "typ": "funkcja" if cat in FUNC else "lokalizacja", "n": len(x),
                 "mediana": med, "mediana_pozostalych": y.median(), "p": mannwhitneyu(x, y).pvalue,
                 "zgodny_znak_%": 100 * (np.sign(x) == np.sign(med)).mean(),
                 "K562": sub_m["K562"].median() if "K562" in sub_m else np.nan,
                 "HepG2": sub_m["HepG2"].median() if "HepG2" in sub_m else np.nan,
                 "CRISPR": sub_m.loc[sub_m["metoda"].str.contains("CRISPR"), "efekt"].median(),
                 "n_CRISPR": int(sub_m["metoda"].str.contains("CRISPR").sum()),
                 "shRNA": sub_m.loc[sub_m["metoda"].str.contains("shRNA"), "efekt"].median(),
                 "n_shRNA": int(sub_m["metoda"].str.contains("shRNA").sum()),
                 "niezbedne": sub_m.loc[sub_m["niezbedny"], "efekt"].median(),
                 "pozostale": sub_m.loc[~sub_m["niezbedny"], "efekt"].median()})
cat = pd.DataFrame(rows)
cat["q"] = bh(cat["p"].values)
cat = cat.sort_values("p")
cat.to_csv(C / "functional_categories.tsv", sep="\t", index=False)
print("\n1. Kategorie RBP a efekt na lncRNA (razem); mediana > 0 = po wyciszeniu lncRNA rosna wzgledem mRNA")
print("   kolumny K562/HepG2/CRISPR/shRNA/niezbedne/pozostale: mediany efektu czlonkow kategorii w danym podzbiorze")
print(cat.round({c: 3 for c in cat.columns if c not in ("p", "q")}).to_string(
    index=False, formatters={"p": "{:.1e}".format, "q": "{:.1e}".format}))

# ---------------------------------------------------------------- 2. profil klas
prof = []
full = eff.join(A[FUNC], how="inner")
for c in FUNC:
    m = full[c] == 1
    if m.sum() >= MIN_MEMBERS:
        prof.append(full.loc[m, [k for k in PROFILE if k in full.columns]].median().rename(f"{c} (n={int(m.sum())})"))
prof = pd.DataFrame(prof)
prof.to_csv(C / "functional_class_profile.tsv", sep="\t")
print("\n2. Profil klas: mediana efektu czlonkow kategorii funkcji, osobno dla kazdej klasy")
print(prof.round(3).to_string())

# ---------------------------------------------------------------- 3. geny gospodarze snoRNA
rows = []
for c in FUNC + ["wszystkie RBP"]:
    m = full[c] == 1 if c in full else pd.Series(True, index=full.index)
    x = (full.loc[m, "gospodarz snoRNA"] - full.loc[m, LNC]).dropna()
    if len(x) >= MIN_MEMBERS:
        rows.append({"kategoria": c, "n": len(x), "mediana_snoRNA_minus_lncRNA": x.median(),
                     "proc_dodatnich": 100 * (x > 0).mean(), "p": wilcoxon(x).pvalue})
sno = pd.DataFrame(rows)
sno["q"] = bh(sno["p"].values)
sno = sno.sort_values("p")
sno.to_csv(C / "functional_snohost.tsv", sep="\t", index=False)
print("\n3. Geny gospodarze snoRNA minus lncRNA (razem), w obrebie kategorii (dodatnia = gospodarze snoRNA rosna bardziej)")
print(sno.round({c: 3 for c in sno.columns if c not in ("p", "q")}).to_string(
    index=False, formatters={"p": "{:.1e}".format, "q": "{:.1e}".format}))
print(f"\nZapisano: {C}/functional_categories.tsv, {C}/functional_class_profile.tsv, {C}/functional_snohost.tsv")
