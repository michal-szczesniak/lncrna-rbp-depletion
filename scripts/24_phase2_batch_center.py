"""
Faza 2, krok 2.3c: odjecie wplywu wspolnej kontroli (partii).

Kazdy log2FC to knockdown minus jego kontrola; knockdowny z ta sama kontrola dziela jej szum i odrebnosc.
Korekta: dla kazdego knockdownu odejmujemy, gen po genie, mediane log2FC POZOSTALYCH knockdownow z tej samej
partii (ta sama kontrola, bez niego samego). Zostaje to, co swoiste dla danego RBP wzgledem jego partii.
Partie mniejsze niz MIN_BATCH knockdownow pomijamy (mediana z 1-2 innych bylaby zbyt zaszumiona).

Sprawdzenia:
 1. rozmiary partii; ile knockdownow wypada;
 2. PCA (statystyki Walda) po korekcie: czy zwiazek skladowych z partia spadl do poziomu przypadku,
    czy zostaly osie funkcjonalne (knockdowny na krancach);
 3. kontrole pozytywne i wyniki pilotazu: lncRNA minus dopasowane mRNA przed i po korekcie,
    z informacja, z kim knockdown dzieli kontrole.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors

M = Path("results/phase2/matrix")
MIN_BATCH, N_PC, MIN_EXPR, MIN_FRAC, CLIP, N_SHOW, K_NN = 4, 6, 20, 0.9, 10, 6, 10
LNC_ALL = {"lincRNA", "antysens", "dywergentny", "sense-intronic", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
CHECK = ["MTREX", "EXOSC10", "EXOSC5", "EXOSC9", "UPF1", "XRN1", "AQR", "DDX3X", "RBM8A", "PABPN1"]
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)

cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
ctrl_of = pd.read_csv("data/meta/phase2_kds.tsv", sep="\t", dtype=str).set_index("experiment")["possible_controls"]


def eta2(scores, groups):
    g = pd.DataFrame({"s": scores, "g": groups})
    tot = ((g["s"] - g["s"].mean()) ** 2).sum()
    between = g.groupby("g")["s"].apply(lambda x: len(x) * (x.mean() - g["s"].mean()) ** 2).sum()
    return between / tot, (g["g"].nunique() - 1) / (len(g) - 1)


def pca_diag(L, S, expr, label):
    W = (L / S).clip(-CLIP, CLIP)
    keep = (W.notna().mean(axis=1) >= MIN_FRAC) & (expr.reindex(W.index) >= MIN_EXPR)
    X = W[keep].T.fillna(0.0)
    U, sv, Vt = np.linalg.svd((X - X.mean()).values, full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum()
    meta, ctrl = summ.reindex(X.index), ctrl_of.reindex(X.index)
    k = cls.reindex(X.columns)["klasa"]
    rows, ext = [], []
    for i in range(N_PC):
        sc, ld = pd.Series(U[:, i] * sv[i], index=X.index), pd.Series(Vt[i], index=X.columns)
        cr = (meta["method"] == "CRISPR").values
        e2, e0 = eta2(sc.values, ctrl.values)
        z = (ld - ld.mean()) / ld.std()
        rows.append({"PC": f"PC{i + 1}", "war_%": 100 * var[i],
                     "AUC_metoda": mannwhitneyu(sc[cr], sc[~cr]).statistic / (cr.sum() * (~cr).sum()),
                     "eta2_kontrola": e2, "eta2_oczek": e0,
                     "lnc_minus_mRNA_z": z[k.isin(LNC_ALL).values].median() - z[(k == "mRNA").values].median()})
        o = sc.sort_values()
        lab = lambda idx: ", ".join(f"{meta.loc[j, 'target']}({meta.loc[j, 'method'][0]})" for j in idx)
        ext.append((f"PC{i + 1}", lab(o.index[:N_SHOW]), lab(o.index[::-1][:N_SHOW])))
    print(f"\n   PCA {label}: {X.shape[0]} knockdownow x {X.shape[1]} genow; 5 skladowych: {100 * var[:5].sum():.1f}% wariancji")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    for pc, lo, hi in ext:
        print(f"   {pc} nisko: {lo}\n   {' ' * len(pc)} wysoko: {hi}")


def class_diff(lfc, expr, width):
    d = pd.DataFrame({"lfc": lfc, "e": np.log10(expr.reindex(lfc.index) + 1), "w": np.log10(width.reindex(lfc.index)),
                      "k": cls.reindex(lfc.index)["klasa"]}).dropna()
    m, s = (d["k"] == "mRNA").values, d["k"].isin(LNC_ALL).values
    X = d[["e", "w"]].values
    mu, sd = X[m].mean(0), X[m].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    _, idx = NearestNeighbors(n_neighbors=K_NN).fit((X[m] - mu) / sd).kneighbors((X[s] - mu) / sd)
    diff = d["lfc"].values[s] - d["lfc"].values[m][idx].mean(1)
    return float(np.median(diff)), float(wilcoxon(diff).pvalue)


for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_{cell}.tsv.gz", sep="\t", index_col=0)
    S = pd.read_csv(M / f"se_{cell}.tsv.gz", sep="\t", index_col=0)
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    batch = ctrl_of.reindex(L.columns)
    size = batch.map(batch.value_counts())
    print(f"\n=== {cell}: {L.shape[1]} knockdownow w {batch.nunique()} partiach; rozmiar partii: mediana {size.groupby(batch).first().median():.0f}, "
          f"zakres {size.min()}-{size.max()}; knockdowny w partiach < {MIN_BATCH}: {int((size < MIN_BATCH).sum())} (pomijane)")

    # ---------------------------------------------------------------- korekta
    C = pd.DataFrame(np.nan, index=L.index, columns=L.columns)
    for b, cols in batch.groupby(batch).groups.items():
        cols = list(cols)
        if len(cols) < MIN_BATCH:
            continue
        sub = L[cols].values
        for j, c in enumerate(cols):
            others = np.delete(sub, j, axis=1)
            C[c] = sub[:, j] - np.nanmedian(others, axis=1)
    C = C.dropna(axis=1, how="all")
    C.to_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t")

    # ---------------------------------------------------------------- PCA przed i po
    pca_diag(L[C.columns], S[C.columns], expr, "PRZED korekta (te same knockdowny)")
    pca_diag(C, S[C.columns], expr, "PO korekcie")

    # ---------------------------------------------------------------- kontrole pozytywne
    width = cls["width"]
    rows = []
    for kd in C.columns:
        t = summ.loc[kd, "target"]
        if t not in CHECK:
            continue
        mates = [summ.loc[x, "target"] for x in batch[batch == batch[kd]].index if x != kd]
        r0, p0 = class_diff(L[kd].dropna(), expr, width)
        r1, p1 = class_diff(C[kd].dropna(), expr, width)
        rows.append({"knockdown": f"{t} ({summ.loc[kd, 'method']})", "lnc-mRNA_przed": r0, "p_przed": p0,
                     "lnc-mRNA_po": r1, "p_po": p1, "partia (inne RBP z ta sama kontrola)": ", ".join(mates[:8])})
    print(f"\n   Wybrane RBP, {cell}: lncRNA minus dopasowane mRNA (log2FC bez sciagania), przed i po korekcie na partie")
    print(pd.DataFrame(rows).round(3).to_string(index=False, formatters={"p_przed": "{:.0e}".format, "p_po": "{:.0e}".format}))
print(f"\nZapisano: {M}/lfc_bc_K562.tsv.gz, {M}/lfc_bc_HepG2.tsv.gz")
