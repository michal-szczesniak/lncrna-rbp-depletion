"""
Faza 2, krok 2.3: kontrola jakosci wszystkich knockdownow, macierz efektow i odpowiedz ogolna.

1. Kontrola jakosci: skutecznosc wyciszenia (spadek wyciszanego genu) i liczba zmienionych genow,
   wedlug metody i linii. Uwaga: przy nokaucie CRISPR mRNA nie musi spadac (przesuniecie ramki bez rozpadu
   mRNA), wiec dla CRISPR spadek mRNA zaniza skutecznosc.
2. Macierz efektow (geny x knockdowny) osobno dla K562 i HepG2, tylko biblioteki poliA:
   niesciagniete log2FC i ich bledy standardowe; geny cis (okno 10 kb) zamaskowane jako brak danych.
3. Odpowiedz ogolna: analiza glownych skladowych (PCA) macierzy log2FC. Dla kazdej skladowej:
   udzial wariancji, zwiazek z metoda (CRISPR/shRNA) i z liczba zmienionych genow, geny o najwiekszych
   ladunkach, sygnatury p53 i cyklu komorkowego (K562 nie ma funkcjonalnego p53, HepG2 ma)
   oraz roznica ladunkow lncRNA i mRNA (czy skladowa dotyka lncRNA bardziej niz mRNA).
4. Macierz po odjeciu K pierwszych skladowych (do porownania w kroku 2.4).
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

OUT = Path("results/phase2/matrix")
OUT.mkdir(parents=True, exist_ok=True)
N_PC, K_REMOVE, MIN_FRAC_TESTED = 10, 5, 0.9
LNC_ALL = {"lincRNA", "antysens", "dywergentny", "sense-intronic", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
P53 = ["CDKN1A", "MDM2", "GDF15", "BTG2", "TP53I3", "SESN1", "SESN2", "RRM2B", "FAS", "BBC3", "PMAIP1",
       "TNFRSF10B", "ZMAT3", "TP53INP1", "FDXR", "PLK3", "AEN", "DDB2", "XPC", "RPS27L", "SPATA18", "TIGAR"]
CYCLE = ["CCNB1", "CDK1", "PLK1", "AURKA", "AURKB", "TOP2A", "MKI67", "CENPF", "BUB1", "CCNA2", "CDC20",
         "KIF11", "CENPE", "TPX2", "NUSAP1", "UBE2C", "PRC1", "BIRC5", "CCNE2", "MCM2", "MCM10", "E2F1"]
pd.set_option("display.width", 220)

# ---------------------------------------------------------------- 1. kontrola jakosci
summ = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob("results/phase2/deseq2/summary/*.tsv")], ignore_index=True)
summ.to_csv("results/phase2/deseq2/summary_all.tsv", sep="\t", index=False)
summ["wyciszenie"] = np.select(
    [summ["target_lfc"].isna(), summ["target_lfc"] <= -1, summ["target_lfc"] <= -0.5],
    ["cel niewyrazony", "silne (<= -1)", "umiarkowane (-1 do -0.5)"], "slabe (> -0.5)")
print("1. Kontrola jakosci (wszystkie knockdowny)")
qc = summ.groupby(["method", "cell", "biblioteka"]).agg(
    n=("kd_experiment", "size"), mediana_log2FC_celu=("target_lfc", "median"),
    mediana_genow_istotnych=("n_padj05", "median"))
print(qc.round(2).to_string())
print("\n   Skutecznosc wyciszenia na poziomie mRNA (liczba knockdownow):")
print(pd.crosstab([summ["method"], summ["cell"]], summ["wyciszenie"]).to_string())

# ---------------------------------------------------------------- 2. macierze
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").dropna(subset=["klasa"]).set_index("gene_id")
cis = pd.read_csv("data/meta/phase2_cis_genes.tsv", sep="\t")
cis = cis[cis["w10kb"]]
mats = {}
for cell in ["K562", "HepG2"]:
    sel = summ[(summ["cell"] == cell) & (summ["biblioteka"] == "poliA")].sort_values("kd_experiment")
    lfc, se, bm = {}, {}, {}
    for kd in sel["kd_experiment"]:
        d = pd.read_csv(f"results/phase2/deseq2/{kd}.tsv.gz", sep="\t",
                        usecols=["gene_id", "baseMean", "lfc_mle", "lfcSE_mle"]).set_index("gene_id")
        d = d[d.index.isin(cls.index)]
        lfc[kd], se[kd], bm[kd] = d["lfc_mle"], d["lfcSE_mle"], d["baseMean"]
    L, S, B = pd.DataFrame(lfc), pd.DataFrame(se), pd.DataFrame(bm)
    n_before = int(L.notna().sum().sum())
    c = cis[cis["kd_experiment"].isin(L.columns) & cis["gene_id"].isin(L.index)]
    for kd, g in c.groupby("kd_experiment"):
        L.loc[g["gene_id"], kd] = np.nan
        S.loc[g["gene_id"], kd] = np.nan
    L.to_csv(OUT / f"lfc_{cell}.tsv.gz", sep="\t")
    S.to_csv(OUT / f"se_{cell}.tsv.gz", sep="\t")
    B.median(axis=1).rename("baseMean_mediana").to_csv(OUT / f"expr_{cell}.tsv", sep="\t")
    frac = L.notna().mean(axis=1)
    k = cls.reindex(L.index)["klasa"]
    print(f"\n2. Macierz {cell}: {L.shape[0]} genow x {L.shape[1]} knockdownow; zamaskowano {n_before - int(L.notna().sum().sum())} "
          f"wartosci cis; geny testowane w >= {MIN_FRAC_TESTED:.0%} knockdownow: {int((frac >= MIN_FRAC_TESTED).sum())} "
          f"(mRNA {int(((frac >= MIN_FRAC_TESTED) & (k == 'mRNA')).sum())}, "
          f"lncRNA {int(((frac >= MIN_FRAC_TESTED) & k.isin(LNC_ALL)).sum())})")
    mats[cell] = (L, sel.set_index("kd_experiment"))

# ---------------------------------------------------------------- 3-4. odpowiedz ogolna
names = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")["gene_name"]
for cell, (L, meta) in mats.items():
    keep = L.notna().mean(axis=1) >= MIN_FRAC_TESTED
    X = L[keep].T.fillna(0.0)                      # knockdowny x geny
    mu = X.mean(axis=0)
    Xc = (X - mu).values
    U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum()
    scores = pd.DataFrame(U[:, :N_PC] * sv[:N_PC], index=X.index, columns=[f"PC{i + 1}" for i in range(N_PC)])
    load = pd.DataFrame(Vt[:N_PC].T, index=X.columns, columns=scores.columns)
    scores.join(meta[["target", "method", "n_padj05", "target_lfc"]]).to_csv(OUT / f"pca_scores_{cell}.tsv", sep="\t")
    load.to_csv(OUT / f"pca_loadings_{cell}.tsv.gz", sep="\t")
    k = cls.reindex(load.index)["klasa"]
    gname = names.reindex(load.index)
    rows = []
    for pc in scores.columns:
        sc, ld = scores[pc], load[pc]
        crispr = meta.reindex(sc.index)["method"] == "CRISPR"
        auc = mannwhitneyu(sc[crispr], sc[~crispr]).statistic / (crispr.sum() * (~crispr).sum())
        z = (ld - ld.mean()) / ld.std()
        top = ld.sort_values()
        rows.append({"skladowa": pc, "wariancja_%": 100 * var[int(pc[2:]) - 1],
                     "AUC_CRISPR_vs_shRNA": auc,
                     "rho_z_liczba_istotnych": spearmanr(sc, np.log10(meta.reindex(sc.index)["n_padj05"] + 1))[0],
                     "p53_z": z[gname.isin(P53).values].mean(), "cykl_z": z[gname.isin(CYCLE).values].mean(),
                     "lncRNA_minus_mRNA_z": z[k.isin(LNC_ALL).values].median() - z[(k == "mRNA").values].median(),
                     "ujemne": ", ".join(gname[top.index[:5]].fillna("?")),
                     "dodatnie": ", ".join(gname[top.index[-5:]].fillna("?"))})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"pca_summary_{cell}.tsv", sep="\t", index=False)
    print(f"\n3. Odpowiedz ogolna, {cell}: {X.shape[0]} knockdownow x {X.shape[1]} genow; "
          f"pierwsze {K_REMOVE} skladowych wyjasnia {100 * var[:K_REMOVE].sum():.1f}% wariancji")
    print("   AUC: 0.5 = brak zwiazku z metoda; p53_z, cykl_z, lncRNA_minus_mRNA_z: srednie/mediany ladunkow w odchyleniach std.")
    print(t.round(2).to_string(index=False, max_colwidth=45))
    resid = Xc - (U[:, :K_REMOVE] * sv[:K_REMOVE]) @ Vt[:K_REMOVE]
    R = pd.DataFrame(resid + mu.values, index=X.index, columns=X.columns).T
    R = R.where(L[keep].notna())                   # przywracamy braki danych (w tym maske cis)
    R.to_csv(OUT / f"lfc_resid{K_REMOVE}_{cell}.tsv.gz", sep="\t")
print(f"\nZapisano: {OUT}/ (lfc_*, se_*, expr_*, pca_*, lfc_resid{K_REMOVE}_*), results/phase2/deseq2/summary_all.tsv")
