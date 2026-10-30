"""
Faza 2, krok 2.3b: diagnostyka odpowiedzi ogolnej.

PCA na niesciagnietych log2FC okazalo sie zdominowane przez geny o niskiej ekspresji (skrajne, zaszumione log2FC).
Tutaj PCA liczymy na statystykach Walda (log2FC / blad standardowy, przyciete do +-10) i tylko dla genow
o przyzwoitej ekspresji (mediana baseMean >= MIN_EXPR), testowanych w >= 90% knockdownow.

Dla kazdej skladowej:
  - udzial wariancji; AUC CRISPR vs shRNA; korelacja z liczba zmienionych genow;
  - eta2 partii kontroli: jaka czesc zmiennosci skladowej wyjasnia to, ktora kontrola byla uzyta
    (dla porownania podana wartosc oczekiwana przy braku zwiazku);
  - sygnatury p53 i cyklu komorkowego oraz roznica ladunkow lncRNA - mRNA;
  - knockdowny o najwyzszych i najnizszych wartosciach (czy to znane grupy funkcjonalne?).
Na koncu: te same knockdowny skrajne dla skladowych PCA z kroku 2.3, ktore dotykaly lncRNA inaczej niz mRNA.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

M = Path("results/phase2/matrix")
N_PC, MIN_EXPR, MIN_FRAC, CLIP, N_SHOW = 8, 20, 0.9, 10, 8
LNC_ALL = {"lincRNA", "antysens", "dywergentny", "sense-intronic", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
P53 = ["CDKN1A", "MDM2", "GDF15", "BTG2", "TP53I3", "SESN1", "SESN2", "RRM2B", "FAS", "BBC3", "PMAIP1",
       "TNFRSF10B", "ZMAT3", "TP53INP1", "FDXR", "PLK3", "AEN", "DDB2", "XPC", "RPS27L", "SPATA18", "TIGAR"]
CYCLE = ["CCNB1", "CDK1", "PLK1", "AURKA", "AURKB", "TOP2A", "MKI67", "CENPF", "BUB1", "CCNA2", "CDC20",
         "KIF11", "CENPE", "TPX2", "NUSAP1", "UBE2C", "PRC1", "BIRC5", "CCNE2", "MCM2", "MCM10", "E2F1"]
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)

cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
kds = pd.read_csv("data/meta/phase2_kds.tsv", sep="\t", dtype=str).set_index("experiment")


def eta2(scores, groups):
    g = pd.DataFrame({"s": scores, "g": groups})
    tot = ((g["s"] - g["s"].mean()) ** 2).sum()
    between = g.groupby("g")["s"].apply(lambda x: len(x) * (x.mean() - g["s"].mean()) ** 2).sum()
    return between / tot, (g["g"].nunique() - 1) / (len(g) - 1)


def extremes(sc, meta):
    o = sc.sort_values()
    lab = lambda idx: ", ".join(f"{meta.loc[i, 'target']}({meta.loc[i, 'method'][0]})" for i in idx)
    return lab(o.index[:N_SHOW]), lab(o.index[::-1][:N_SHOW])


for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_{cell}.tsv.gz", sep="\t", index_col=0)
    S = pd.read_csv(M / f"se_{cell}.tsv.gz", sep="\t", index_col=0)
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    W = (L / S).clip(-CLIP, CLIP)
    keep = (W.notna().mean(axis=1) >= MIN_FRAC) & (expr.reindex(W.index) >= MIN_EXPR)
    X = W[keep].T.fillna(0.0)
    meta = summ.reindex(X.index)
    ctrl = kds.reindex(X.index)["possible_controls"]
    Xc = (X - X.mean()).values
    U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum()
    k = cls.reindex(X.columns)["klasa"]
    gname = cls.reindex(X.columns)["gene_name"]
    print(f"\n=== {cell}: {X.shape[0]} knockdownow x {X.shape[1]} genow (mRNA {int((k == 'mRNA').sum())}, "
          f"lncRNA {int(k.isin(LNC_ALL).sum())}); PCA na statystykach Walda; 5 pierwszych skladowych: {100 * var[:5].sum():.1f}% wariancji")
    rows, ext = [], []
    scores = pd.DataFrame(U[:, :N_PC] * sv[:N_PC], index=X.index, columns=[f"PC{i + 1}" for i in range(N_PC)])
    for i, pc in enumerate(scores.columns):
        sc, ld = scores[pc], pd.Series(Vt[i], index=X.columns)
        crispr = (meta["method"] == "CRISPR").values
        auc = mannwhitneyu(sc[crispr], sc[~crispr]).statistic / (crispr.sum() * (~crispr).sum())
        e2, e2_null = eta2(sc.values, ctrl.values)
        z = (ld - ld.mean()) / ld.std()
        top = ld.sort_values()
        rows.append({"PC": pc, "war_%": 100 * var[i], "AUC_metoda": auc,
                     "rho_istotne": spearmanr(sc, np.log10(meta["n_padj05"] + 1))[0],
                     "eta2_kontrola": e2, "eta2_oczek": e2_null,
                     "p53_z": z[gname.isin(P53).values].mean(), "cykl_z": z[gname.isin(CYCLE).values].mean(),
                     "lnc_minus_mRNA_z": z[k.isin(LNC_ALL).values].median() - z[(k == "mRNA").values].median(),
                     "geny_ujemne": ", ".join(gname[top.index[:4]].fillna("?")),
                     "geny_dodatnie": ", ".join(gname[top.index[-4:]].fillna("?"))})
        lo, hi = extremes(sc, meta)
        ext.append({"PC": pc, "najnizsze": lo, "najwyzsze": hi})
    t = pd.DataFrame(rows)
    scores.join(meta[["target", "method", "n_padj05"]]).to_csv(M / f"pca_wald_scores_{cell}.tsv", sep="\t")
    t.to_csv(M / f"pca_wald_summary_{cell}.tsv", sep="\t", index=False)
    print(t.round(2).to_string(index=False))
    print(f"\n   Knockdowny skrajne (C = CRISPR, s = shRNA), {cell}:")
    for e in ext:
        print(f"   {e['PC']} nisko: {e['najnizsze']}\n   {' ' * len(e['PC'])} wysoko: {e['najwyzsze']}")

    # skladowe z kroku 2.3 (log2FC), ktore dotykaly lncRNA inaczej niz mRNA
    old = pd.read_csv(M / f"pca_summary_{cell}.tsv", sep="\t")
    old_sc = pd.read_csv(M / f"pca_scores_{cell}.tsv", sep="\t", index_col=0)
    flagged = old[old["lncRNA_minus_mRNA_z"].abs() >= 0.5]
    if len(flagged):
        print(f"\n   Skladowe z kroku 2.3 (log2FC) z |lncRNA - mRNA| >= 0.5, {cell}:")
        for r in flagged.to_dict("records"):
            lo, hi = extremes(old_sc[r["skladowa"]], summ)
            print(f"   {r['skladowa']} (lncRNA - mRNA = {r['lncRNA_minus_mRNA_z']:+.2f}) nisko: {lo}")
            print(f"   {' ' * (len(r['skladowa']) + 30)} wysoko: {hi}")
print(f"\nZapisano: {M}/pca_wald_summary_*.tsv, {M}/pca_wald_scores_*.tsv")
