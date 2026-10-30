"""
Faza 3, krok 3.4B: bezposrednie wiazanie (eCLIP) a efekt wyciszenia RBP na lncRNA.

Piki: plik laczacy obie repliki dla kazdego eksperymentu eCLIP; prog jak w Van Nostrand i wsp. 2020:
log2 wzbogacenia >= 3 i -log10 p >= 3 (kolumny 7 i 8 pliku narrowPeak). Gen jest "zwiazany", jesli srodek
co najmniej jednego piku lezy w jego obrebie (eksony + introny) na tej samej nici.

1. Pokrycie: jaka czesc wyrazonych genow jest zwiazana, osobno dla mRNA i lncRNA, w przedzialach ekspresji
   (eCLIP wykrywa latwiej geny o wysokiej ekspresji).
2. Bezposredniosc: dla RBP z eCLIP i knockdownem w tej samej linii - czy geny zwiazane zmieniaja sie po wyciszeniu
   inaczej niz niezwiazane geny tej samej klasy o podobnej ekspresji i dlugosci (roznica ze znakiem i roznica |log2FC|).
3. Czy efekt klasy wynika z wiazania: roznica lncRNA - dopasowane mRNA (dopasowanie A) liczona na wszystkich genach
   i tylko na genach niezwiazanych przez dany RBP.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors

M, OUT = Path("results/phase2/matrix"), Path("results/phase3/eclip")
OUT.mkdir(parents=True, exist_ok=True)
MIN_L2FC, MIN_P, BIN, K_NN, MIN_BOUND = 3.0, 3.0, 100_000, 10, 20
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
ALIASES = {"SKIV2L2": "MTREX"}
pd.set_option("display.width", 220)

genes = pd.read_csv("data/ref/genes_v29.tsv", sep="\t")
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
genes = genes[genes["gene_id"].isin(cls.index[cls["klasa"].notna()])]
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
ef = pd.read_csv("data/meta/eclip_files.tsv", sep="\t", dtype=str)
man = pd.read_csv("data/raw/encode/download_manifest.tsv", sep="\t", dtype=str).drop_duplicates("file_accession", keep="last")
ef = ef.merge(man[["file_accession", "path"]], on="file_accession")
ef["target"] = ef["target"].replace(ALIASES)

# geny rozbite na przedzialy 100 kb (do szybkiego przypisania pikow)
g = genes[["gene_id", "chr", "start", "end", "strand"]].copy()
g["b0"], g["b1"] = g["start"] // BIN, g["end"] // BIN
g = g.loc[g.index.repeat(g["b1"] - g["b0"] + 1)].copy()
g["bin"] = g["b0"] + g.groupby(level=0).cumcount()
g = g.reset_index(drop=True)[["gene_id", "chr", "strand", "start", "end", "bin"]]

first = pd.read_csv(ef["path"].iloc[0], sep="\t", header=None, nrows=3, compression="infer")
print(f"Przykladowy plik pikow ({ef['file_accession'].iloc[0]}), pierwsze wiersze:\n{first.to_string(index=False, header=False)}")


def bound_genes(path):
    p = pd.read_csv(path, sep="\t", header=None, usecols=range(8), compression="infer")
    p.columns = ["chr", "start", "end", "name", "score", "strand", "l2fc", "mlog10p"]
    n_all = len(p)
    p = p[(p["l2fc"] >= MIN_L2FC) & (p["mlog10p"] >= MIN_P)]
    p["mid"] = (p["start"] + p["end"]) // 2
    p["bin"] = p["mid"] // BIN
    x = p[["chr", "strand", "mid", "bin"]].merge(g, on=["chr", "strand", "bin"])
    x = x[(x["mid"] >= x["start"]) & (x["mid"] <= x["end"])]
    return set(x["gene_id"]), n_all, len(p)


bound, stats = {}, []
for r in ef.itertuples():
    b, n_all, n_sig = bound_genes(r.path)
    bound[(r.cell, r.target)] = b
    stats.append({"linia": r.cell, "RBP": r.target, "piki": n_all, "piki_istotne": n_sig, "geny_zwiazane": len(b)})
st = pd.DataFrame(stats)
st.to_csv(OUT / "eclip_bound_counts.tsv", sep="\t", index=False)
print(f"\n0. Piki: mediana {st['piki'].median():.0f} na eksperyment, po progu {st['piki_istotne'].median():.0f}; "
      f"zwiazanych genow: mediana {st['geny_zwiazane'].median():.0f} (zakres {st['geny_zwiazane'].min()}-{st['geny_zwiazane'].max()})")

# ---------------------------------------------------------------- 1. pokrycie
print("\n1. Odsetek wyrazonych genow zwiazanych przez RBP (mediana z RBP), w przedzialach ekspresji wyznaczonych na mRNA")
cov_rows = []
for cell in ["K562", "HepG2"]:
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    expr = expr[expr >= 10]
    k = cls.reindex(expr.index)["klasa"]
    grp = pd.Series(np.where(k == "mRNA", "mRNA", np.where(k.isin(LNC), "lncRNA", "inne")), index=expr.index)
    le = np.log10(expr + 1)
    edges = np.quantile(le[grp == "mRNA"], [0, .25, .5, .75, 1])
    edges[0], edges[-1] = -np.inf, np.inf
    eb = pd.cut(le, edges, labels=["1 niska", "2", "3", "4 wysoka"])
    for (c, t), b in bound.items():
        if c != cell:
            continue
        isb = pd.Series(expr.index.isin(b), index=expr.index)
        for (gr, bn), s in isb.groupby([grp, eb], observed=True):
            if gr != "inne":
                cov_rows.append({"linia": cell, "RBP": t, "klasa": gr, "ekspresja": bn, "proc": 100 * s.mean(), "n": len(s)})
cov = pd.DataFrame(cov_rows)
print(cov.groupby(["linia", "ekspresja", "klasa"], observed=True)["proc"].median().unstack("klasa").round(1).to_string())
n_lnc = cov[cov["klasa"] == "lncRNA"].groupby(["linia", "ekspresja"], observed=True)["n"].first()
print("   liczba wyrazonych lncRNA w przedzialach:", n_lnc.to_dict())

# ---------------------------------------------------------------- 2-3. bezposredniosc i efekt klasy
rows = []
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    feat = pd.DataFrame({"klasa": cls.reindex(L.index)["klasa"], "e": np.log10(expr.reindex(L.index) + 1),
                         "w": np.log10(cls.reindex(L.index)["width"])}, index=L.index).dropna()
    feat = feat[feat["e"] >= 1]                                      # baseMean >= ~9
    for kd in L.columns:
        t = summ.loc[kd, "target"]
        if (cell, t) not in bound:
            continue
        b = bound[(cell, t)]
        x = L[kd].reindex(feat.index)
        d = feat.assign(lfc=x, b=feat.index.isin(b)).dropna(subset=["lfc"])
        row = {"linia": cell, "kd_experiment": kd, "RBP": t, "metoda": summ.loc[kd, "method"]}
        for name, sel in [("mRNA", d["klasa"] == "mRNA"), ("lncRNA", d["klasa"].isin(LNC))]:
            s = d[sel]
            bb, ub = s[s["b"]], s[~s["b"]]
            row[f"zwiazane_{name}"] = len(bb)
            if len(bb) < MIN_BOUND or len(ub) < 100:
                continue
            X, Y = ub[["e", "w"]].values, bb[["e", "w"]].values
            mu, sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1)
            _, idx = NearestNeighbors(n_neighbors=K_NN).fit((X - mu) / sd).kneighbors((Y - mu) / sd)
            diff = bb["lfc"].values - ub["lfc"].values[idx].mean(1)
            adiff = np.abs(bb["lfc"].values) - np.abs(ub["lfc"].values[idx]).mean(1)
            row[f"znak_{name}"], row[f"p_znak_{name}"] = float(np.median(diff)), float(wilcoxon(diff).pvalue)
            row[f"abs_{name}"], row[f"p_abs_{name}"] = float(np.median(adiff)), float(wilcoxon(adiff).pvalue)
        # 3. efekt klasy na wszystkich genach i tylko na niezwiazanych
        for tag, dd in [("wszystkie", d), ("niezwiazane", d[~d["b"]])]:
            m_, l_ = dd[dd["klasa"] == "mRNA"], dd[dd["klasa"].isin(LNC)]
            if len(l_) < 50:
                continue
            X, Y = m_[["e", "w"]].values, l_[["e", "w"]].values
            mu, sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1)
            _, idx = NearestNeighbors(n_neighbors=K_NN).fit((X - mu) / sd).kneighbors((Y - mu) / sd)
            row[f"klasa_{tag}"] = float(np.median(l_["lfc"].values - m_["lfc"].values[idx].mean(1)))
        rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(OUT / "eclip_directness.tsv", sep="\t", index=False)

print("\n2. Bezposredniosc: zwiazane geny wzgledem niezwiazanych genow tej samej klasy (dopasowanie: ekspresja, dlugosc)")
print("   'znak' = mediana roznicy log2FC (ujemna: zwiazane spadaja bardziej); 'abs' = roznica |log2FC| (dodatnia: zwiazane reaguja silniej)")
for cell, g_ in res.groupby("linia"):
    for name in ["mRNA", "lncRNA"]:
        s = g_.dropna(subset=[f"abs_{name}"])
        if s.empty:
            print(f"   {cell} {name:<6}: za malo zwiazanych genow")
            continue
        sig_abs = (s[f"p_abs_{name}"] < 0.01) & (s[f"abs_{name}"] > 0)
        print(f"   {cell} {name:<6}: RBP z >= {MIN_BOUND} zwiazanymi genami: {len(s)} z {len(g_)}; "
              f"zwiazane reaguja silniej (p < 0.01) w {100 * sig_abs.mean():.0f}%; mediana 'abs' {s[f'abs_{name}'].median():+.3f}; "
              f"mediana 'znak' {s[f'znak_{name}'].median():+.3f}; znak ujemny (p < 0.01) w "
              f"{100 * ((s[f'p_znak_{name}'] < 0.01) & (s[f'znak_{name}'] < 0)).mean():.0f}%, dodatni w "
              f"{100 * ((s[f'p_znak_{name}'] < 0.01) & (s[f'znak_{name}'] > 0)).mean():.0f}%")

print("\n3. Czy efekt klasy wynika z wiazania? lncRNA - dopasowane mRNA na wszystkich genach i tylko na niezwiazanych")
for cell, g_ in res.groupby("linia"):
    s = g_.dropna(subset=["klasa_wszystkie", "klasa_niezwiazane"])
    print(f"   {cell}: {len(s)} knockdownow; mediana |efektu| wszystkie {s['klasa_wszystkie'].abs().median():.3f}, "
          f"niezwiazane {s['klasa_niezwiazane'].abs().median():.3f}; zgodnosc rho = {spearmanr(s['klasa_wszystkie'], s['klasa_niezwiazane'])[0]:.2f}")
both = res.dropna(subset=["klasa_niezwiazane"]).groupby(["RBP", "linia"])["klasa_niezwiazane"].mean().unstack("linia").dropna()
if len(both) >= 20:
    print(f"   Replikacja efektu klasy na genach niezwiazanych miedzy liniami: rho = {spearmanr(both['K562'], both['HepG2'])[0]:+.2f} ({len(both)} RBP)")

print("\n   RBP z listy kandydatow (krok 2.5), jesli maja eCLIP:")
hits = pd.read_csv("results/phase2/classes/rbp_hits.tsv", sep="\t", index_col=0).index
cols = ["linia", "RBP", "zwiazane_lncRNA", "znak_lncRNA", "p_znak_lncRNA", "abs_lncRNA", "klasa_wszystkie", "klasa_niezwiazane"]
h = res[res["RBP"].isin(hits)][[c for c in cols if c in res.columns]].sort_values(["RBP", "linia"])
print(h.round(3).to_string(index=False) if len(h) else "   brak")
print(f"\nZapisano: {OUT}/eclip_bound_counts.tsv, {OUT}/eclip_directness.tsv")
