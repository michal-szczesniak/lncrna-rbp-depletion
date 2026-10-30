"""
Czesc 5.1: czy usuniecie nadzoru jadrowego (eksosom, NEXT, PAXT) podnosi szczegolnie lncRNA dywergentne i antysensowe?

Kategorie lncRNA z polozenia wzgledem genow kodujacych bialka (GENCODE v29; TSS = poczatek genu na jego nici):
  dywergentne   - przeciwna nic, TSS do 2 kb od TSS genu kodujacego, transkrypcja w przeciwnych kierunkach (glowa do glowy)
  antysensowe   - przeciwna nic, naklada sie na gen kodujacy (i nie jest dywergentne)
  w poblizu     - inny gen kodujacy w promieniu 10 kb (np. ta sama nic)
  miedzygenowe  - zaden gen kodujacy w promieniu 10 kb
Dla kazdego knockdownu (macierz skorygowana na kontrole): efekt kategorii = mediana roznic (lncRNA minus srednia z 10
mRNA o podobnej ekspresji i dlugosci). Porownanie: knockdowny nadzoru jadrowego vs wszystkie pozostale (Mann-Whitney, BH),
oraz ranking wszystkich RBP wedlug efektu na lncRNA dywergentne (srednia z linii).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.neighbors import NearestNeighbors

M, OUT = Path("results/phase2/matrix"), Path("results/phase4/network")
OUT.mkdir(parents=True, exist_ok=True)
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA", "sense-intronic"}
SURV = {"MTREX", "EXOSC1", "EXOSC2", "EXOSC3", "EXOSC4", "EXOSC5", "EXOSC6", "EXOSC7", "EXOSC8", "EXOSC9", "EXOSC10",
        "DIS3", "DIS3L", "ZCCHC8", "RBM7", "ZFC3H1", "PABPN1", "SKIV2L"}
DIV_KB, NEAR_KB, BIN, K_NN = 2_000, 10_000, 100_000, 10
CATS = ["dywergentne", "antysensowe", "w poblizu", "miedzygenowe"]
pd.set_option("display.width", 220)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


# ---------------------------------------------------------------- kategorie z polozenia
genes = pd.read_csv("data/ref/genes_v29.tsv", sep="\t")
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")["klasa"]
genes["klasa"] = cls.reindex(genes["gene_id"]).values
genes = genes[~genes["gene_id"].str.endswith("_PAR_Y")]
genes["tss"] = np.where(genes["strand"] == "+", genes["start"], genes["end"])
ln = genes[genes["klasa"].isin(LNC)].copy()
mr = genes[genes["klasa"] == "mRNA"].copy()


def binned(df, pad):
    d = df[["gene_id", "chr", "start", "end", "strand", "tss"]].copy()
    d["b0"], d["b1"] = (d["start"] - pad) // BIN, (d["end"] + pad) // BIN
    d = d.loc[d.index.repeat(d["b1"] - d["b0"] + 1)].copy()
    d["bin"] = d["b0"] + d.groupby(level=0).cumcount()
    return d.drop(columns=["b0", "b1"])


pairs = binned(ln, NEAR_KB).merge(binned(mr, 0), on=["chr", "bin"], suffixes=("_l", "_m")).drop_duplicates(["gene_id_l", "gene_id_m"])
gap = np.maximum(0, np.maximum(pairs["start_m"] - pairs["end_l"], pairs["start_l"] - pairs["end_m"]))
pairs = pairs[gap <= NEAR_KB].copy()
opp = pairs["strand_l"] != pairs["strand_m"]
dtss = pairs["tss_l"] - pairs["tss_m"]
h2h = np.where(pairs["strand_l"] == "-", dtss <= 500, dtss >= -500)          # glowa do glowy: TSS lncRNA nie w glebi genu
pairs["div"] = opp & (dtss.abs() <= DIV_KB) & h2h
pairs["anti"] = opp & (pairs["start_l"] <= pairs["end_m"]) & (pairs["end_l"] >= pairs["start_m"])
flag = pairs.groupby("gene_id_l").agg(div=("div", "any"), anti=("anti", "any"))
ln["kategoria"] = "miedzygenowe"
ln.loc[ln["gene_id"].isin(flag.index), "kategoria"] = "w poblizu"
ln.loc[ln["gene_id"].isin(flag.index[flag["anti"]]), "kategoria"] = "antysensowe"
ln.loc[ln["gene_id"].isin(flag.index[flag["div"]]), "kategoria"] = "dywergentne"
cat_of = ln.set_index("gene_id")["kategoria"]
cat_of.to_csv(OUT / "lncRNA_position_category.tsv", sep="\t")
print("Kategorie lncRNA z polozenia (wszystkie geny lncRNA w GENCODE v29):")
print(pd.crosstab(ln["klasa"], ln["kategoria"]).reindex(columns=CATS).fillna(0).astype(int).to_string())

# ---------------------------------------------------------------- efekt kategorii w kazdym knockdownie
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
rows = []
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    f = pd.DataFrame({"e": np.log10(expr.reindex(L.index) + 1), "w": np.log10(genes.set_index("gene_id")["end"].reindex(L.index)
                      - genes.set_index("gene_id")["start"].reindex(L.index) + 1), "kl": cls.reindex(L.index)}, index=L.index).dropna()
    f = f[f["e"] >= np.log10(20 + 1)]
    m_ = f[f["kl"] == "mRNA"]
    mu, sd = m_[["e", "w"]].mean().values, m_[["e", "w"]].std().values
    nn = NearestNeighbors(n_neighbors=K_NN).fit((m_[["e", "w"]].values - mu) / sd)
    nb = {}
    for c in CATS:
        sel = f[cat_of.reindex(f.index) == c]
        if len(sel) >= 20:
            _, idx = nn.kneighbors((sel[["e", "w"]].values - mu) / sd)
            nb[c] = (sel.index.values, m_.index.values[idx])
    print(f"   {cell}: lncRNA z ekspresja >= 20 w kategoriach: " + ", ".join(f"{c} {len(v[0])}" for c, v in nb.items()))
    for kd in L.columns:
        x = L[kd]
        r = {"linia": cell, "kd_experiment": kd, "target": summ.loc[kd, "target"] if kd in summ.index else kd}
        for c, (g, idx) in nb.items():
            d = x.reindex(g).values - np.nanmean(x.reindex(idx.ravel()).values.reshape(idx.shape), axis=1)
            r[c] = np.nanmedian(d)
        rows.append(r)
eff = pd.DataFrame(rows)
eff["nadzor"] = eff["target"].isin(SURV)
eff.to_csv(OUT / "category_effects_per_kd.tsv", sep="\t", index=False)

print("\n1. Efekt kategorii lncRNA (mediana z knockdownow): nadzor jadrowy vs pozostale knockdowny")
print("   knockdowny nadzoru: " + "; ".join(f"{c}: {', '.join(sorted(set(eff.loc[(eff['linia'] == c) & eff['nadzor'], 'target'])))}"
                                        for c in ["K562", "HepG2"]))
out = []
for cell, g in eff.groupby("linia"):
    for c in [c for c in CATS if c in g.columns]:
        a, b = g.loc[g["nadzor"], c].dropna(), g.loc[~g["nadzor"], c].dropna()
        out.append({"linia": cell, "kategoria": c, "nadzor (n)": len(a), "mediana nadzor": a.median(),
                    "mediana pozostale": b.median(), "p": mannwhitneyu(a, b).pvalue if len(a) >= 3 else np.nan})
t = pd.DataFrame(out)
t["q"] = bh(t["p"].fillna(1).values)
print(t.round({"mediana nadzor": 3, "mediana pozostale": 3}).to_string(index=False, formatters={"p": "{:.1e}".format, "q": "{:.1e}".format}))
t.to_csv(OUT / "surveillance_vs_others.tsv", sep="\t", index=False)

if "dywergentne" in eff.columns and "miedzygenowe" in eff.columns:
    eff["dyw_minus_miedzygen"] = eff["dywergentne"] - eff["miedzygenowe"]
    w = eff.groupby(["target", "linia"])[["dywergentne", "dyw_minus_miedzygen"]].mean().unstack("linia")
    w = w.dropna()
    w["srednia"] = w["dywergentne"].mean(axis=1)
    top = w.sort_values("srednia", ascending=False).head(15)
    print("\n2. RBP, po ktorych usunieciu najbardziej rosna lncRNA dywergentne (obie linie; srednia z linii)")
    print("   * = czynnik nadzoru jadrowego")
    for rbp, r in top.iterrows():
        print(f"   {rbp + ('*' if rbp in SURV else ''):<10} dywergentne K562 {r[('dywergentne', 'K562')]:+.2f}, HepG2 {r[('dywergentne', 'HepG2')]:+.2f}; "
              f"dywergentne minus miedzygenowe K562 {r[('dyw_minus_miedzygen', 'K562')]:+.2f}, HepG2 {r[('dyw_minus_miedzygen', 'HepG2')]:+.2f}")
print(f"\nZapisano: {OUT}/lncRNA_position_category.tsv, category_effects_per_kd.tsv, surveillance_vs_others.tsv")
