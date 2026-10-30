"""
Czesc 5.3.

A. Diagnostyka: czy profil odpowiedzi tego samego RBP na poziomie pojedynczych genow powtarza sie miedzy liniami?
   Geny wyrazone w obu liniach; lncRNA oraz tyle samo mRNA o tej samej ekspresji. Dla kazdego RBP usunietego w obu
   liniach: korelacja Spearmana jego profilu w K562 i HepG2 (r_self) oraz pozycja "samego siebie" wsrod wszystkich RBP
   z drugiej linii (1 = najbardziej podobny jest ten sam RBP). Dwa warianty: log2FC po korekcie i log2FC / blad standardowy.
B. Wspolodpowiedz lncRNA z sasiadami: korelacja Spearmana zmian dwoch genow w ~400 knockdownach danej linii.
   Pary: lncRNA-mRNA dywergentne (glowa do glowy, TSS do 2 kb), lncRNA-mRNA antysensowe (nakladajace sie),
   lncRNA-mRNA w poblizu (do 10 kb); odniesienie: mRNA-mRNA dywergentne i w poblizu. Tlo: ten sam lncRNA z losowym
   mRNA z innego chromosomu o podobnej ekspresji co prawdziwy partner. Powtarzalnosc: korelacja wartosci par miedzy liniami.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, spearmanr

M, OUT = Path("results/phase2/matrix"), Path("results/phase4/network")
OUT.mkdir(parents=True, exist_ok=True)
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA", "sense-intronic"}
MIN_EXPR, MIN_TESTED, STRONG, BIN = 20, 0.9, 2000, 100_000
rng = np.random.default_rng(5)
pd.set_option("display.width", 220)

cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
genes = pd.read_csv("data/ref/genes_v29.tsv", sep="\t")
genes = genes[~genes["gene_id"].str.endswith("_PAR_Y")].copy()
genes["klasa"] = cls.reindex(genes["gene_id"])["klasa"].values
genes["tss"] = np.where(genes["strand"] == "+", genes["start"], genes["end"])
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
D = {}
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    S = pd.read_csv(M / f"se_{cell}.tsv.gz", sep="\t", index_col=0).reindex(index=L.index, columns=L.columns)
    e = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"].reindex(L.index)
    ok = (L.notna().mean(axis=1) >= MIN_TESTED) & (e >= MIN_EXPR)
    D[cell] = {"L": L[ok], "Z": (L[ok] / S[ok]).clip(-10, 10), "e": e[ok]}
kl = cls["klasa"]

# ================================================================ A. diagnostyka profili
common = D["K562"]["L"].index.intersection(D["HepG2"]["L"].index)
lnc = common[kl.reindex(common).isin(LNC)]
mpool = common[kl.reindex(common) == "mRNA"]
emean = (np.log10(D["K562"]["e"] + 1) + np.log10(D["HepG2"]["e"] + 1)).reindex(common) / 2
pool = emean[mpool].sort_values()
used, mm = set(), []
for g in lnc:
    i = int(np.searchsorted(pool.values, emean[g]))
    for j in sorted(range(max(0, i - 60), min(len(pool), i + 60)), key=lambda j: abs(pool.values[j] - emean[g])):
        if pool.index[j] not in used:
            used.add(pool.index[j]); mm.append(pool.index[j]); break
mm = pd.Index(mm)
strong = set(summ[summ["n_padj05"] >= STRONG].groupby("target")["cell"].nunique().loc[lambda s: s == 2].index) if "n_padj05" in summ else set()


def rbp_profiles(cell, kind, gset):
    X = D[cell][kind].loc[gset]
    t = summ.reindex(X.columns)["target"].values
    return X.T.groupby(t).mean().T                                # geny x RBP


def rankcorr_matrix(A, B):
    ra = np.apply_along_axis(rankdata, 0, A.fillna(A.median()).values)
    rb = np.apply_along_axis(rankdata, 0, B.fillna(B.median()).values)
    ra = (ra - ra.mean(0)) / ra.std(0)
    rb = (rb - rb.mean(0)) / rb.std(0)
    return pd.DataFrame(ra.T @ rb / ra.shape[0], index=A.columns, columns=B.columns)


print(f"A. Czy profil tego samego RBP powtarza sie miedzy liniami? ({len(lnc)} lncRNA i {len(mm)} mRNA o tej samej ekspresji, "
      f"mediana sredniej log10 ekspresji {emean[lnc].median():.2f} vs {emean[mm].median():.2f})")
rowsA = []
for kind, kname in [("L", "log2FC"), ("Z", "log2FC / SE")]:
    for gname, gset in [("lncRNA", lnc), ("mRNA", mm)]:
        PK, PH = rbp_profiles("K562", kind, gset), rbp_profiles("HepG2", kind, gset)
        both = PK.columns.intersection(PH.columns)
        C = rankcorr_matrix(PK[both], PH[both])
        self_r = pd.Series(np.diag(C.values), index=both)
        other = (C.values.sum(1) - np.diag(C.values)) / (len(both) - 1)
        rank_self = pd.Series([(C.loc[r] > C.loc[r, r]).sum() + 1 for r in both], index=both)
        for subset, idx in [("wszystkie RBP", both), (f"silne (>= {STRONG} genow w obu)", both.intersection(list(strong)))]:
            if len(idx) < 5:
                continue
            rowsA.append({"wariant": kname, "geny": gname, "RBP": subset, "n": len(idx), "mediana r_self": self_r[idx].median(),
                          "mediana r z innymi": np.median(other[[list(both).index(i) for i in idx]]),
                          "% sam na 1. miejscu": 100 * (rank_self[idx] == 1).mean(),
                          "% sam w top 5%": 100 * (rank_self[idx] <= max(1, int(0.05 * len(both)))).mean()})
tA = pd.DataFrame(rowsA)
print(tA.round(3).to_string(index=False))
tA.to_csv(OUT / "profile_replication.tsv", sep="\t", index=False)

# ================================================================ B. wspolodpowiedz z sasiadami
def binned(df, pad):
    d = df[["gene_id", "chr", "start", "end", "strand", "tss"]].copy()
    d["b0"], d["b1"] = (d["start"] - pad) // BIN, (d["end"] + pad) // BIN
    d = d.loc[d.index.repeat(d["b1"] - d["b0"] + 1)].copy()
    d["bin"] = d["b0"] + d.groupby(level=0).cumcount()
    return d.drop(columns=["b0", "b1"])


def neighbour_pairs(A, B, same_set):
    p = binned(A, 10_000).merge(binned(B, 0), on=["chr", "bin"], suffixes=("_a", "_b"))
    p = p[p["gene_id_a"] != p["gene_id_b"]].drop_duplicates(["gene_id_a", "gene_id_b"])
    if same_set:
        p = p[p["gene_id_a"] < p["gene_id_b"]]
    gap = np.maximum(0, np.maximum(p["start_b"] - p["end_a"], p["start_a"] - p["end_b"]))
    p = p[gap <= 10_000].copy()
    opp = p["strand_a"] != p["strand_b"]
    dt = p["tss_a"] - p["tss_b"]
    h2h = np.where(p["strand_a"] == "-", dt <= 500, dt >= -500)
    p["typ"] = "w poblizu"
    p.loc[opp & (p["start_a"] <= p["end_b"]) & (p["end_a"] >= p["start_b"]), "typ"] = "antysensowe"
    p.loc[opp & (dt.abs() <= 2_000) & h2h, "typ"] = "dywergentne"
    return p[["gene_id_a", "gene_id_b", "typ", "chr"]]


gl, gm = genes[genes["klasa"].isin(LNC)], genes[genes["klasa"] == "mRNA"]
P = pd.concat([neighbour_pairs(gl, gm, False).assign(para="lncRNA-mRNA"),
               neighbour_pairs(gm, gm, True).assign(para="mRNA-mRNA")])
P = P[~((P["para"] == "mRNA-mRNA") & (P["typ"] == "antysensowe"))]
chrom = genes.set_index("gene_id")["chr"]


def zrank(X):
    R = np.apply_along_axis(rankdata, 1, X.T.fillna(X.T.median()).T.values)
    R = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
    return pd.DataFrame(R, index=X.index)


res = []
for cell in ["K562", "HepG2"]:
    Zr = zrank(D[cell]["L"])
    e = np.log10(D[cell]["e"] + 1)
    have = set(Zr.index)
    pp = P[P["gene_id_a"].isin(have) & P["gene_id_b"].isin(have)].copy()
    A_, B_ = Zr.loc[pp["gene_id_a"]].values, Zr.loc[pp["gene_id_b"]].values
    pp["r"] = (A_ * B_).mean(1)
    # tlo: ten sam gen A, losowy mRNA z innego chromosomu o ekspresji jak prawdziwy partner
    mr = e[[g for g in e.index if kl.get(g) == "mRNA"]].sort_values()
    picks = []
    for a, b in zip(pp["gene_id_a"], pp["gene_id_b"]):
        i = int(np.searchsorted(mr.values, e[b]))
        for _ in range(20):
            j = int(np.clip(i + rng.integers(-40, 41), 0, len(mr) - 1))
            if chrom.get(mr.index[j]) != chrom.get(a):
                break
        picks.append(mr.index[j])
    pp["r_tlo"] = (Zr.loc[pp["gene_id_a"]].values * Zr.loc[picks].values).mean(1)
    pp["linia"] = cell
    res.append(pp)
res = pd.concat(res)
res.to_csv(OUT / "neighbour_coresponse.tsv.gz", sep="\t", index=False)

print("\nB. Wspolodpowiedz z sasiadami: korelacja zmian dwoch genow w knockdownach danej linii (Spearman)")
rowsB = []
for (cell, para, typ), g in res.groupby(["linia", "para", "typ"]):
    rowsB.append({"linia": cell, "para": para, "typ": typ, "pary": len(g), "mediana r": g["r"].median(),
                  "mediana r tla": g["r_tlo"].median(), "% r > 0.3": 100 * (g["r"] > 0.3).mean(),
                  "% r > 0.3 w tle": 100 * (g["r_tlo"] > 0.3).mean(), "p": mannwhitneyu(g["r"], g["r_tlo"]).pvalue})
tB = pd.DataFrame(rowsB)
print(tB.round(3).to_string(index=False, formatters={"p": "{:.1e}".format}))
tB.to_csv(OUT / "neighbour_coresponse_summary.tsv", sep="\t", index=False)

w = res.pivot_table(index=["gene_id_a", "gene_id_b", "para", "typ"], columns="linia", values="r").dropna().reset_index()
print("\n   Powtarzalnosc wspolodpowiedzi par miedzy liniami (korelacja r_K562 vs r_HepG2):")
for (para, typ), g in w.groupby(["para", "typ"]):
    print(f"   {para:<12} {typ:<12} {len(g):>6} par: rho = {spearmanr(g['K562'], g['HepG2'])[0]:.2f}")
w["r_min"] = w[["K562", "HepG2"]].min(axis=1)
nm = cls["gene_name"]
top = w[w["para"] == "lncRNA-mRNA"].nlargest(20, "r_min")
top = top.assign(lncRNA=nm.reindex(top["gene_id_a"]).values, sasiad=nm.reindex(top["gene_id_b"]).values)
print("\n   Pary lncRNA-sasiad o najsilniejszej wspolodpowiedzi w obu liniach (top 20):")
print(top[["lncRNA", "sasiad", "typ", "K562", "HepG2"]].round(2).to_string(index=False))
print(f"\nZapisano: {OUT}/profile_replication.tsv, neighbour_coresponse.tsv.gz, neighbour_coresponse_summary.tsv")
