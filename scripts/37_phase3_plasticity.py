"""
Faza 3, krok 3.5: czy roznice lncRNA-mRNA wynikaja z plastycznosci transkrypcyjnej?

Hipoteza: wyciszenie RBP zmienia stan komorki, a geny regulacyjnie plastyczne (swoiste tkankowo, z promotorem bez
wyspy CpG) reaguja na takie zmiany silniej. lncRNA czesciej maja te cechy niz mRNA.
Cechy genu:
  tau  = swoistosc tkankowa z GTEx v8 (mediany TPM w tkankach, bez linii komorkowych; log2(TPM+1); 0 = wszedzie, 1 = jedna tkanka);
  CGI  = czy TSS genu (+-500 pz) naklada sie na wyspe CpG (UCSC cpgIslandExt, hg38).
1. Cechy wedlug klasy (kontrola: lncRNA powinny miec wyzsze tau i rzadziej CGI).
2. Co przewiduje "reaktywnosc" genu = mediana |log2FC| we wszystkich knockdownach (macierz skorygowana):
   regresja na rangach z ekspresja, dlugoscia, stabilnoscia, splicingiem, wskaznikiem jadrowym, tau i CGI.
3. Efekt klasy przy dopasowaniu C (ekspresja, dlugosc, splicing, stabilnosc, lokalizacja) i D = C + tau + CGI,
   na tym samym zbiorze genow: roznica ze znakiem (per knockdown) i roznica reaktywnosci.
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

M, OUT = Path("results/phase2/matrix"), Path("results/phase3/plasticity")
OUT.mkdir(parents=True, exist_ok=True)
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
ORDER = ["mRNA", "lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA", "pseudogen transkr."]
K_NN, TSS_WIN = 10, 500
pd.set_option("display.width", 230)

cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
cls["gene"] = cls["gene_id"].str.split(".").str[0]
genes = pd.read_csv("data/ref/genes_v29.tsv", sep="\t")
genes["gene"] = genes["gene_id"].str.split(".").str[0]
kin = pd.read_csv("results/phase1/bru/gene_kinetics.tsv", sep="\t")
nuc = pd.read_csv("results/phase3/localization/nuclear_index.tsv", sep="\t", index_col=0)

# ---------------------------------------------------------------- tau z GTEx
gt = pd.read_csv("data/ref/gtex/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz", sep="\t", skiprows=2)
tissues = [c for c in gt.columns[2:] if not c.startswith("Cells")]
x = np.log2(gt[tissues].values + 1)
mx = x.max(axis=1)
tau = np.where(mx >= 0.5, (1 - x / np.where(mx > 0, mx, 1)[:, None]).sum(axis=1) / (len(tissues) - 1), np.nan)
tau = pd.Series(tau, index=gt["Name"].str.split(".").str[0]).groupby(level=0).first()
print(f"0. GTEx: {len(tissues)} tkanek, tau dla {int(tau.notna().sum())} genow")

# ---------------------------------------------------------------- CGI przy TSS
cgi = pd.read_csv("data/ref/cpgIslandExt.txt.gz", sep="\t", header=None, usecols=[1, 2, 3], names=["chr", "s", "e"])
genes["tss"] = np.where(genes["strand"] == "+", genes["start"], genes["end"])
has = np.zeros(len(genes), dtype=bool)
for ch, gi in genes.groupby("chr").groups.items():
    c = cgi[cgi["chr"] == ch].sort_values("s")
    if c.empty:
        continue
    s, e = c["s"].values, c["e"].values
    emax = np.maximum.accumulate(e)
    t = genes.loc[gi, "tss"].values
    j = np.searchsorted(s, t + TSS_WIN, side="right") - 1          # ostatnia wyspa zaczynajaca sie przed koncem okna
    has[np.asarray(gi)] = (j >= 0) & (emax[np.clip(j, 0, None)] >= t - TSS_WIN)
cgi_gene = pd.Series(has, index=genes["gene"]).groupby(level=0).max()
print(f"   Wyspy CpG: {len(cgi)}; geny z CGI przy TSS: {int(cgi_gene.sum())} z {len(cgi_gene)}")

base = cls.drop_duplicates("gene").set_index("gene")[["klasa"]].copy()
base["tau"], base["CGI"] = tau.reindex(base.index), cgi_gene.reindex(base.index).astype(float)
t1 = base.groupby("klasa").agg(geny=("tau", "size"), tau_mediana=("tau", "median"), CGI_proc=("CGI", lambda s: 100 * s.mean()))
print("\n1. Cechy plastycznosci wedlug klasy")
print(t1.reindex(ORDER).round(2).to_string())


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


def rank_ols(df, y, xs):
    d = df[[y] + xs].dropna()
    Z = d.rank().apply(lambda c: (c - c.mean()) / c.std())
    X = np.c_[np.ones(len(Z)), Z[xs].values]
    beta, *_ = np.linalg.lstsq(X, Z[y].values, rcond=None)
    resid = Z[y].values - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (resid @ resid) / (len(Z) - X.shape[1]))
    return pd.DataFrame({"beta": beta[1:], "t": beta[1:] / se[1:]}, index=xs), len(Z)


splice = splicing_by_cell()
COLS_C = ["log_ekspr", "log_dl", "splicing", "stab_6h", "jadro"]
COLS_D = COLS_C + ["tau", "CGI"]
rows = []
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    g = L.index
    gk = g.str.split(".").str[0]
    feat = pd.DataFrame({"klasa": cls.set_index("gene_id").reindex(g)["klasa"].values,
                         "log_ekspr": np.log10(expr.reindex(g) + 1).values,
                         "log_dl": np.log10(cls.set_index("gene_id").reindex(g)["width"]).values,
                         "splicing": splice[cell].reindex(g).values,
                         "stab_6h": kin[kin["cell"] == cell].drop_duplicates("gene_id").set_index("gene_id")["stab_6h"].reindex(g).values,
                         "jadro": nuc[cell].reindex(gk).values, "tau": tau.reindex(gk).values,
                         "CGI": cgi_gene.reindex(gk).astype(float).values,
                         "reakt": L.abs().median(axis=1).values}, index=g)

    print(f"\n2. {cell}: co przewiduje reaktywnosc genu (mediana |log2FC| we wszystkich knockdownach)? Regresja na rangach")
    for name, sel in [("mRNA", feat["klasa"] == "mRNA"), ("lncRNA", feat["klasa"].isin(LNC))]:
        coef, n = rank_ols(feat[sel], "reakt", COLS_D)
        print(f"   {name} (n = {n}): " + "; ".join(f"{i} {r.beta:+.2f} (t {r.t:+.1f})" for i, r in coef.iterrows()))

    # 3. efekt klasy: C vs D na tym samym zbiorze genow
    f = feat.dropna(subset=COLS_D)
    mr = f[f["klasa"] == "mRNA"]
    lo, hi = mr[COLS_D].quantile(0.01), mr[COLS_D].quantile(0.99)
    sel = f[f["klasa"].isin(LNC) & ((f[COLS_D] >= lo) & (f[COLS_D] <= hi)).all(axis=1)]
    print(f"   Porownanie C i D na {len(sel)} lncRNA z pelnym zestawem cech")
    for v, cols in [("C", COLS_C), ("D", COLS_D)]:
        mu, sd = mr[cols].mean().values, mr[cols].std().values
        sd = np.where(sd > 0, sd, 1.0)
        _, idx = NearestNeighbors(n_neighbors=K_NN).fit((mr[cols].values - mu) / sd).kneighbors((sel[cols].values - mu) / sd)
        nb = mr.index.values[idx]
        react = float(np.median(sel["reakt"].values - feat.loc[nb.ravel(), "reakt"].values.reshape(nb.shape).mean(1)))
        print(f"   {v}: reaktywnosc lncRNA minus dopasowane mRNA = {react:+.4f}")
        for kd in L.columns:
            xk = L[kd]
            dd = xk.reindex(sel.index).values - np.nanmean(xk.reindex(nb.ravel()).values.reshape(nb.shape), axis=1)
            dd = dd[~np.isnan(dd)]
            if len(dd) >= 20:
                rows.append({"linia": cell, "kd_experiment": kd, "wariant": v, "roznica": float(np.median(dd))})

cr = pd.DataFrame(rows)
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
cr["target"] = summ.reindex(cr["kd_experiment"])["target"].values
cr.to_csv(OUT / "class_effect_C_vs_D.tsv", sep="\t", index=False)
print("\n3. Efekt klasy (lncRNA - dopasowane mRNA, per knockdown) przy dopasowaniu C i D, ten sam zbior genow")
for v in ["C", "D"]:
    ww = cr[cr["wariant"] == v].groupby(["target", "linia"])["roznica"].mean().unstack("linia").dropna()
    print(f"   {v}: mediana |roznicy| = {cr.loc[cr['wariant'] == v, 'roznica'].abs().median():.3f}; "
          f"replikacja miedzy liniami rho = {spearmanr(ww['K562'], ww['HepG2'])[0]:+.2f} ({len(ww)} RBP)")
cd = cr.pivot_table(index=["linia", "kd_experiment"], columns="wariant", values="roznica").dropna()
print(f"   Zgodnosc C i D miedzy knockdownami: rho = {spearmanr(cd['C'], cd['D'])[0]:.2f}")
base.to_csv(OUT / "plasticity_features.tsv", sep="\t")
print(f"\nZapisano: {OUT}/plasticity_features.tsv, {OUT}/class_effect_C_vs_D.tsv")
