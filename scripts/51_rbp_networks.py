"""
Czesc 5.2: sieci RBP-RBP zbudowane osobno z odpowiedzi lncRNA i mRNA.

0. Ranking RBP wedlug swoistego nadmiaru lncRNA dywergentnych (dywergentne minus miedzygenowe; z kroku 5.1).
1. Dla kazdej linii: profil odpowiedzi kazdego RBP (sredni log2FC po korekcie na kontrole, srednia z jego knockdownow)
   na dwoch zbiorach genow o tej samej liczebnosci i ekspresji: lncRNA oraz dopasowane mRNA (najblizsza ekspresja,
   bez powtorzen). Podobienstwo dwoch RBP = korelacja Spearmana ich profili. Daje to dwie sieci RBP-RBP.
2. Czy sieci sa podobne? Korelacja podobienstw par RBP miedzy siecia lncRNA i mRNA (test Mantela, 1000 permutacji).
3. Czy tworza te same moduly? Grupowanie hierarchiczne (srednie wiazanie, 1 - korelacja), podzial na K modulow,
   zgodnosc podzialow (skorygowany indeks Randa, ARI; 0 = przypadek, 1 = identyczne).
4. Czy siec jest powtarzalna miedzy liniami? Dla RBP w obu liniach: korelacja podobienstw par K562 vs HepG2,
   osobno dla sieci lncRNA i mRNA.
5. Pary RBP podobne tylko przez lncRNA: roznica (lncRNA - mRNA) w obu liniach w tym samym kierunku.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import adjusted_rand_score

M, OUT = Path("results/phase2/matrix"), Path("results/phase4/network")
OUT.mkdir(parents=True, exist_ok=True)
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA", "sense-intronic"}
SURV = {"MTREX", "EXOSC2", "EXOSC5", "EXOSC9", "EXOSC10", "PABPN1", "ZCCHC8", "RBM7", "DIS3"}
MIN_EXPR, MIN_TESTED, K_MOD, N_PERM = 20, 0.9, 12, 1000
rng = np.random.default_rng(3)
pd.set_option("display.width", 220)

# ---------------------------------------------------------------- 0. ranking dywergentne minus miedzygenowe
ce = pd.read_csv(OUT / "category_effects_per_kd.tsv", sep="\t")
if {"dywergentne", "miedzygenowe"} <= set(ce.columns):
    ce["nadmiar"] = ce["dywergentne"] - ce["miedzygenowe"]
    w = ce.groupby(["target", "linia"])["nadmiar"].mean().unstack("linia").dropna()
    w["srednia"] = w.mean(axis=1)
    print("0. RBP o najwiekszym swoistym nadmiarze lncRNA dywergentnych (dywergentne minus miedzygenowe; * = nadzor jadrowy)")
    for rbp, r in w.sort_values("srednia", ascending=False).head(15).iterrows():
        print(f"   {rbp + ('*' if rbp in SURV else ''):<10} K562 {r['K562']:+.2f}, HepG2 {r['HepG2']:+.2f}")

# ---------------------------------------------------------------- 1. profile RBP na lncRNA i dopasowanych mRNA
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")["klasa"]


def rbp_corr(P):
    R = np.apply_along_axis(rankdata, 0, P.values)
    C = np.corrcoef(R.T)
    return pd.DataFrame(C, index=P.columns, columns=P.columns)


def upper(C, names):
    sub = C.loc[names, names].values
    return sub[np.triu_indices(len(names), 1)]


nets = {}
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"].reindex(L.index)
    ok = (L.notna().mean(axis=1) >= MIN_TESTED) & (expr >= MIN_EXPR)
    kl = cls.reindex(L.index)
    lnc = L.index[ok & kl.isin(LNC)]
    pool = pd.Series(np.log10(expr[ok & (kl == "mRNA")] + 1)).sort_values()
    used, mm = set(), []
    for g in lnc:                                                            # dopasowanie bez powtorzen: najblizsza ekspresja
        e = np.log10(expr[g] + 1)
        i = int(np.searchsorted(pool.values, e))
        for j in sorted(range(max(0, i - 50), min(len(pool), i + 50)), key=lambda j: abs(pool.values[j] - e)):
            if pool.index[j] not in used:
                used.add(pool.index[j]); mm.append(pool.index[j]); break
    targets = summ.reindex(L.columns)["target"]
    prof = {}
    for name, genes_ in [("lncRNA", lnc), ("mRNA", pd.Index(mm))]:
        P = L.loc[genes_].T.groupby(targets.values).mean().T.fillna(0.0)      # geny x RBP
        prof[name] = rbp_corr(P)
    nets[cell] = prof
    print(f"\n1. {cell}: {len(lnc)} lncRNA i {len(mm)} dopasowanych mRNA (mediana ekspresji {expr[lnc].median():.0f} vs {expr[mm].median():.0f}); "
          f"{prof['lncRNA'].shape[0]} RBP; srednie |podobienstwo| par: lncRNA {np.abs(upper(prof['lncRNA'], prof['lncRNA'].index)).mean():.3f}, "
          f"mRNA {np.abs(upper(prof['mRNA'], prof['mRNA'].index)).mean():.3f}")

# ---------------------------------------------------------------- 2-3. siec lncRNA vs siec mRNA
print("\n2-3. Siec lncRNA vs siec mRNA w tej samej linii")
for cell, prof in nets.items():
    names = prof["lncRNA"].index
    a, b = upper(prof["lncRNA"], names), upper(prof["mRNA"], names)
    r = spearmanr(a, b)[0]
    ra = rankdata(a)
    Cm = prof["mRNA"].loc[names, names].values
    iu_ = np.triu_indices(len(names), 1)
    perm = []
    for _ in range(N_PERM):                                              # test Mantela: permutacja etykiet RBP w sieci mRNA
        pn = rng.permutation(len(names))
        perm.append(np.corrcoef(ra, rankdata(Cm[np.ix_(pn, pn)][iu_]))[0, 1])
    p = (np.sum(np.array(perm) >= r) + 1) / (N_PERM + 1)
    mods = {}
    for k in ["lncRNA", "mRNA"]:
        D = 1 - prof[k].loc[names, names].values
        np.fill_diagonal(D, 0)
        mods[k] = fcluster(linkage(squareform(np.clip(D, 0, 2), checks=False), "average"), K_MOD, "maxclust")
    ari = adjusted_rand_score(mods["lncRNA"], mods["mRNA"])
    print(f"   {cell}: zgodnosc podobienstw par (Mantel) rho = {r:.2f}, p = {p:.1e}; zgodnosc modulow (ARI, K = {K_MOD}) = {ari:.2f}")
    pd.DataFrame({"RBP": names, "modul_lncRNA": mods["lncRNA"], "modul_mRNA": mods["mRNA"]}).to_csv(OUT / f"modules_{cell}.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- 4. powtarzalnosc miedzy liniami
both = nets["K562"]["lncRNA"].index.intersection(nets["HepG2"]["lncRNA"].index)
print(f"\n4. Powtarzalnosc sieci miedzy liniami ({len(both)} RBP w obu liniach): korelacja podobienstw par K562 vs HepG2")
for k in ["lncRNA", "mRNA"]:
    r = spearmanr(upper(nets["K562"][k], both), upper(nets["HepG2"][k], both))[0]
    print(f"   siec {k}: rho = {r:.2f}")

# ---------------------------------------------------------------- 5. pary podobne tylko przez lncRNA
d = {c: upper(nets[c]["lncRNA"], both) - upper(nets[c]["mRNA"], both) for c in ["K562", "HepG2"]}
iu = np.triu_indices(len(both), 1)
pairs = pd.DataFrame({"RBP1": both[iu[0]], "RBP2": both[iu[1]], "roznica_K562": d["K562"], "roznica_HepG2": d["HepG2"],
                      "lnc_K562": upper(nets["K562"]["lncRNA"], both), "lnc_HepG2": upper(nets["HepG2"]["lncRNA"], both),
                      "mRNA_K562": upper(nets["K562"]["mRNA"], both), "mRNA_HepG2": upper(nets["HepG2"]["mRNA"], both)})
pairs["min_roznica"] = np.where(np.sign(pairs["roznica_K562"]) == np.sign(pairs["roznica_HepG2"]),
                                pairs[["roznica_K562", "roznica_HepG2"]].abs().min(axis=1) * np.sign(pairs["roznica_K562"]), 0)
pairs.to_csv(OUT / "rbp_pairs_lnc_vs_mrna.tsv.gz", sep="\t", index=False)
r_d = spearmanr(d["K562"], d["HepG2"])[0]
print(f"\n5. Roznica podobienstw (lncRNA minus mRNA) powtarza sie miedzy liniami: rho = {r_d:.2f}")
cols = ["RBP1", "RBP2", "lnc_K562", "mRNA_K562", "lnc_HepG2", "mRNA_HepG2"]
print("   Pary RBP podobne przez lncRNA bardziej niz przez mRNA, w obu liniach (top 15):")
print(pairs.nlargest(15, "min_roznica")[cols].round(2).to_string(index=False))
print("   Pary RBP podobne przez mRNA bardziej niz przez lncRNA, w obu liniach (top 10):")
print(pairs.nsmallest(10, "min_roznica")[cols].round(2).to_string(index=False))
print(f"\nZapisano: {OUT}/modules_<linia>.tsv, {OUT}/rbp_pairs_lnc_vs_mrna.tsv.gz")
