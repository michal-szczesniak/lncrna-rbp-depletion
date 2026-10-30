"""
Faza 4, krok 4.1: odpornosc glownego wyniku (roznica lncRNA - dopasowane mRNA, macierz skorygowana, dopasowanie B).

Warianty:
  glowny               - jak w kroku 2.4 (przeliczony tu dla porownywalnosci)
  maska 50 kb CRISPR   - w knockdownach CRISPR dodatkowo maskowane geny w promieniu 50 kb od celu (krok 3.6)
  bez slabych shRNA    - bez knockdownow shRNA ze spadkiem celu > -0.5
  K = 5, K = 20        - liczba dopasowanych mRNA
  ekspresja >= 50      - tylko geny o medianie baseMean >= 50
  tylko shRNA / tylko CRISPR - replikacja liczona tylko dla RBP wyciszonych ta sama metoda w obu liniach
Dla kazdego: mediana |roznicy|, replikacja miedzy liniami (rho), zgodnosc z wariantem glownym, ile z 35 RBP z listy
(krok 2.5) spelnia nadal kryteria (|roznica| >= 0.1, q < 0.05 w obu liniach, ten sam znak).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors

M, OUT = Path("results/phase2/matrix"), Path("results/phase4")
OUT.mkdir(parents=True, exist_ok=True)
POOLED = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
COLS = ["log_ekspr", "log_dl", "splicing", "stab_6h"]
EFFECT, Q = 0.1, 0.05
pd.set_option("display.width", 220)

cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
kin = pd.read_csv("results/phase1/bru/gene_kinetics.tsv", sep="\t")
cis = pd.read_csv("data/meta/phase2_cis_genes.tsv", sep="\t")
cis50 = cis[cis["w50kb"]].groupby("kd_experiment")["gene_id"].apply(set)
hits = set(pd.read_csv("results/phase2/classes/rbp_hits.tsv", sep="\t", index_col=0).index)


def splicing_by_cell():
    reg = pd.read_csv("data/ref/eisa_regions_v29.tsv.gz", sep="\t")
    reg["L"] = reg["end"] - reg["start"]
    lens = reg.pivot_table(index="gene_id", columns="kind", values="L", aggfunc="sum")
    gene_ids = np.sort(reg["gene_id"].unique())
    smp = pd.read_csv("data/meta/pilot_samples.tsv", sep="\t", dtype=str)
    ctrl = smp[smp["condition"] == "control"].drop_duplicates(["sample_experiment", "bio_rep"])
    out = {}
    for cell, g in ctrl.groupby("cell"):
        vals = []
        for e, r in zip(g["sample_experiment"], g["bio_rep"]):
            f = Path(f"results/phase1/eisa/sums/{e}_{r}.npz")
            if f.exists():
                z = np.load(f)
                E, I = pd.Series(z["exon"], index=gene_ids), pd.Series(z["intron"], index=gene_ids)
                vals.append(np.log2(((I / lens["intron"].reindex(gene_ids)) / (E / lens["exon"].reindex(gene_ids))).where((E > 0) & (I > 0))))
        out[cell] = pd.concat(vals, axis=1).median(axis=1)
    return out


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def neighbours(feat, k, min_expr=None):
    f = feat.dropna(subset=COLS)
    if min_expr is not None:
        f = f[f["log_ekspr"] >= np.log10(min_expr + 1)]
    mr = f[f["klasa"] == "mRNA"]
    lo, hi = mr[COLS].quantile(0.01), mr[COLS].quantile(0.99)
    sel = f[f["klasa"].isin(POOLED) & ((f[COLS] >= lo) & (f[COLS] <= hi)).all(axis=1)]
    mu, sd = mr[COLS].mean().values, mr[COLS].std().values
    sd = np.where(sd > 0, sd, 1.0)
    _, idx = NearestNeighbors(n_neighbors=k).fit((mr[COLS].values - mu) / sd).kneighbors((sel[COLS].values - mu) / sd)
    return sel.index.values, mr.index.values[idx]


splice = splicing_by_cell()
VARIANTS = {"glowny": dict(k=10), "maska 50 kb CRISPR": dict(k=10, mask50=True), "bez slabych shRNA": dict(k=10, drop_weak=True),
            "K = 5": dict(k=5), "K = 20": dict(k=20), "ekspresja >= 50": dict(k=10, min_expr=50)}
rows = []
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    g = L.index
    feat = pd.DataFrame({"klasa": cls.reindex(g)["klasa"], "log_ekspr": np.log10(expr.reindex(g) + 1),
                         "log_dl": np.log10(cls.reindex(g)["width"]), "splicing": splice[cell].reindex(g),
                         "stab_6h": kin[kin["cell"] == cell].drop_duplicates("gene_id").set_index("gene_id")["stab_6h"].reindex(g)}, index=g)
    for vname, opt in VARIANTS.items():
        genes_, idx = neighbours(feat, opt["k"], opt.get("min_expr"))
        for kd in L.columns:
            meth, tl = summ.loc[kd, "method"], summ.loc[kd, "target_lfc"]
            if opt.get("drop_weak") and meth == "shRNA" and tl > -0.5:
                continue
            x = L[kd]
            if opt.get("mask50") and meth == "CRISPR" and kd in cis50.index:
                x = x.copy()
                x[x.index.isin(cis50[kd])] = np.nan
            d = x.reindex(genes_).values - np.nanmean(x.reindex(idx.ravel()).values.reshape(idx.shape), axis=1)
            d = d[~np.isnan(d)]
            if len(d) >= 20:
                rows.append({"wariant": vname, "linia": cell, "kd_experiment": kd, "target": summ.loc[kd, "target"],
                             "metoda": meth, "roznica": float(np.median(d)), "p": float(wilcoxon(d).pvalue)})
res = pd.DataFrame(rows)
res["q"] = res.groupby(["wariant", "linia"])["p"].transform(bh)
res.to_csv(OUT / "robustness_all.tsv.gz", sep="\t", index=False)

main = res[res["wariant"] == "glowny"].set_index(["linia", "kd_experiment"])["roznica"]
out = []


def summarize(name, r):
    w = r.groupby(["target", "linia"]).agg(roznica=("roznica", "mean"), q=("q", "max")).unstack("linia").dropna()
    rho = spearmanr(w[("roznica", "K562")], w[("roznica", "HepG2")])[0] if len(w) >= 10 else np.nan
    ok = lambda l: (w[("roznica", l)].abs() >= EFFECT) & (w[("q", l)] < Q)
    same = np.sign(w[("roznica", "K562")]) == np.sign(w[("roznica", "HepG2")])
    hit_now = set(w.index[ok("K562") & ok("HepG2") & same])
    cmp = r.set_index(["linia", "kd_experiment"])["roznica"]
    j = pd.concat([cmp, main], axis=1, join="inner").dropna()
    out.append({"wariant": name, "knockdowny": len(r), "RBP_w_obu": len(w), "mediana_|roznicy|": r["roznica"].abs().median(),
                "replikacja_rho": rho, "zgodnosc_z_glownym": spearmanr(j.iloc[:, 0], j.iloc[:, 1])[0] if len(j) > 10 else np.nan,
                "RBP_z_replikacja": len(hit_now), "z_35_z_listy": len(hit_now & hits)})


for v in VARIANTS:
    summarize(v, res[res["wariant"] == v])
g0 = res[res["wariant"] == "glowny"]
for met in ["shRNA", "CRISPR"]:
    same_m = g0.groupby("target")["metoda"].agg(lambda s: set(s) == {met})
    summarize(f"tylko {met}", g0[g0["target"].isin(same_m[same_m].index)])
t = pd.DataFrame(out)
t.to_csv(OUT / "robustness_summary.tsv", sep="\t", index=False)
print("Odpornosc glownego wyniku (lncRNA razem - dopasowane mRNA; macierz skorygowana, dopasowanie B)")
print(t.round(3).to_string(index=False))
print(f"\nZapisano: {OUT}/robustness_summary.tsv, {OUT}/robustness_all.tsv.gz")
