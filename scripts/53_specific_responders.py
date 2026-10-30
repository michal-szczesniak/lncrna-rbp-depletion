"""
Swoiste odpowiedzi: geny, ktore silnie reaguja na usuniecie jednego RBP, a na pozostale znacznie slabiej.

Dla kazdej linii (macierz log2FC po korekcie na kontrole; dodatkowo maskowane geny w promieniu 50 kb od genu
usuwanego RBP, zeby wykluczyc efekty cis):
  z = (log2FC - mediana genu we wszystkich knockdownach) / (1.4826 * MAD), srednia z knockdownow tego samego RBP;
  najsilniejszy regulator genu = RBP z najwiekszym |z| (tylko RBP usuniete w obu liniach);
  dominacja = |z| najsilniejszego / |z| drugiego.
Replikacja: ten sam najsilniejszy regulator w obu liniach i ten sam kierunek (przypadek: ok. 1 / liczba RBP).
Porownanie lncRNA z mRNA o tej samej ekspresji; lista lncRNA o swoistej, powtarzalnej odpowiedzi; RBP najczesciej
bedace jedynym regulatorem.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

M, OUT = Path("results/phase2/matrix"), Path("results/phase4/specific")
OUT.mkdir(parents=True, exist_ok=True)
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA", "sense-intronic"}
MIN_EXPR, MIN_TESTED, MAD_FLOOR, MIN_Z = 20, 0.9, 0.05, 5
pd.set_option("display.width", 230)

cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
cis = pd.read_csv("data/meta/phase2_cis_genes.tsv", sep="\t")
cis50 = cis[cis["w50kb"]].groupby("kd_experiment")["gene_id"].apply(set)
poscat = pd.read_csv("results/phase4/network/lncRNA_position_category.tsv", sep="\t", index_col=0).iloc[:, 0] \
    if Path("results/phase4/network/lncRNA_position_category.tsv").exists() else pd.Series(dtype=str)
rbps_both = set(summ.groupby("target")["cell"].nunique().loc[lambda s: s == 2].index)

Zr, LF, EX = {}, {}, {}
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    for kd in L.columns.intersection(cis50.index):                       # maska 50 kb wokol genu RBP
        L.loc[L.index.isin(cis50[kd]), kd] = np.nan
    e = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"].reindex(L.index)
    L = L[(L.notna().mean(axis=1) >= MIN_TESTED) & (e >= MIN_EXPR)]
    med = L.median(axis=1)
    mad = (1.4826 * L.sub(med, axis=0).abs().median(axis=1)).clip(lower=MAD_FLOOR)
    Z = L.sub(med, axis=0).div(mad, axis=0)
    t = summ.reindex(L.columns)["target"].values
    keep = np.isin(t, list(rbps_both))
    Zr[cell] = Z.loc[:, keep].T.groupby(t[keep]).mean().T                 # geny x RBP
    LF[cell] = L.loc[:, keep].T.groupby(t[keep]).mean().T
    EX[cell] = e.reindex(L.index)


def top2(Z):
    A = Z.abs().fillna(0).values
    o = np.argsort(-A, axis=1)
    i1, i2 = o[:, 0], o[:, 1]
    r = np.arange(len(A))
    return pd.DataFrame({"rbp1": Z.columns[i1], "z1": Z.values[r, i1], "rbp2": Z.columns[i2], "absz2": A[r, i2]}, index=Z.index)


T = {c: top2(Zr[c]) for c in Zr}
common = T["K562"].index.intersection(T["HepG2"].index)
d = T["K562"].loc[common].join(T["HepG2"].loc[common], lsuffix="_K562", rsuffix="_HepG2")
d["dom_K562"] = d["z1_K562"].abs() / d["absz2_K562"]
d["dom_HepG2"] = d["z1_HepG2"].abs() / d["absz2_HepG2"]
d["ten_sam"] = (d["rbp1_K562"] == d["rbp1_HepG2"]) & (np.sign(d["z1_K562"]) == np.sign(d["z1_HepG2"]))
d["klasa"] = cls.reindex(d.index)["klasa"].values

# ---------------------------------------------------------------- porownanie klas (mRNA dopasowane ekspresja)
em = (np.log10(EX["K562"].reindex(common) + 1) + np.log10(EX["HepG2"].reindex(common) + 1)) / 2
lnc = common[pd.Series(d["klasa"], index=common).isin(LNC)]
pool = em[common[d["klasa"] == "mRNA"]].sort_values()
used, mm = set(), []
for g in lnc:
    i = int(np.searchsorted(pool.values, em[g]))
    for j in sorted(range(max(0, i - 60), min(len(pool), i + 60)), key=lambda j: abs(pool.values[j] - em[g])):
        if pool.index[j] not in used:
            used.add(pool.index[j]); mm.append(pool.index[j]); break
n_rbp = Zr["K562"].shape[1]
print(f"1. Ten sam najsilniejszy regulator w obu liniach (ten sam kierunek); {n_rbp} RBP usunietych w obu liniach, "
      f"przypadek ok. {100 / n_rbp:.2f}%")
rows = []
for name, g in [("lncRNA", lnc), ("mRNA o tej samej ekspresji", pd.Index(mm))]:
    x = d.loc[g]
    strong = x[(x["z1_K562"].abs() >= MIN_Z) & (x["z1_HepG2"].abs() >= MIN_Z)]
    rows.append({"geny": name, "n": len(x), "% ten sam regulator": 100 * x["ten_sam"].mean(),
                 f"wsrod genow z |z| >= {MIN_Z} w obu (n)": len(strong), "% ten sam regulator (silne)": 100 * strong["ten_sam"].mean(),
                 "mediana dominacji K562": x["dom_K562"].median(), "mediana dominacji HepG2": x["dom_HepG2"].median()})
t1 = pd.DataFrame(rows)
print(t1.round(2).to_string(index=False))
a, b = d.loc[lnc, "ten_sam"], d.loc[mm, "ten_sam"]
print(f"   lncRNA vs mRNA: iloraz szans {fisher_exact([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])[0]:.2f}, "
      f"p = {fisher_exact([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])[1]:.1e} (test Fishera)")

# ---------------------------------------------------------------- lista lncRNA
rep = d.loc[lnc][d.loc[lnc, "ten_sam"]].copy()
rep["dom_min"] = rep[["dom_K562", "dom_HepG2"]].min(axis=1)
rep["z_min"] = rep[["z1_K562", "z1_HepG2"]].abs().min(axis=1)
rep = rep[rep["z_min"] >= MIN_Z]
rep["lncRNA"] = cls.reindex(rep.index)["gene_name"].values
rep["polozenie"] = poscat.reindex(rep.index).fillna("-").values
rep["log2FC_K562"] = [LF["K562"].loc[g, r] for g, r in zip(rep.index, rep["rbp1_K562"])]
rep["log2FC_HepG2"] = [LF["HepG2"].loc[g, r] for g, r in zip(rep.index, rep["rbp1_HepG2"])]
rep = rep.sort_values(["dom_min", "z_min"], ascending=False)
rep.to_csv(OUT / "specific_lncRNA.tsv", sep="\t")
print(f"\n2. lncRNA o swoistej, powtarzalnej odpowiedzi: {len(rep)} (ten sam regulator w obu liniach, |z| >= {MIN_Z} w obu)")
print("   dominacja = ile razy silniej niz na drugi RBP; top 30 wedlug mniejszej dominacji z dwoch linii")
cols = ["lncRNA", "klasa", "polozenie", "rbp1_K562", "log2FC_K562", "log2FC_HepG2", "z1_K562", "z1_HepG2", "dom_K562", "dom_HepG2",
        "rbp2_K562", "rbp2_HepG2"]
print(rep[cols].head(30).rename(columns={"rbp1_K562": "RBP", "rbp2_K562": "drugi_K562", "rbp2_HepG2": "drugi_HepG2"})
      .round(2).to_string(index=False))

# ---------------------------------------------------------------- RBP jako jedyne regulatory
cnt_l = rep["rbp1_K562"].value_counts()
repm = d.loc[pd.Index(mm)][d.loc[pd.Index(mm), "ten_sam"]]
repm = repm[repm[["z1_K562", "z1_HepG2"]].abs().min(axis=1) >= MIN_Z]
cnt_m = repm["rbp1_K562"].value_counts()
hub = pd.DataFrame({"lncRNA": cnt_l, "mRNA o tej samej ekspresji": cnt_m}).fillna(0).astype(int).sort_values("lncRNA", ascending=False)
print("\n3. RBP, ktore najczesciej sa jedynym (powtarzalnym) regulatorem genu (top 15 wedlug lncRNA)")
print(hub.head(15).to_string())
hub.to_csv(OUT / "specific_hub_rbps.tsv", sep="\t")
print(f"\nZapisano: {OUT}/specific_lncRNA.tsv, {OUT}/specific_hub_rbps.tsv")
