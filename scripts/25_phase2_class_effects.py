"""
Faza 2, krok 2.4: porownanie klas RNA we wszystkich knockdownach.

Dla kazdego knockdownu i kazdej klasy: mediana roznicy log2FC (gen klasy minus srednia z 10 najblizszych mRNA)
i test Wilcoxona. Warianty:
  macierz:      skorygowana na wspolna kontrole (glowna) i surowa (dla porownania);
  dopasowanie:  A = ekspresja + dlugosc genu;
                B = A + wydajnosc splicingu (kontrole z pilotazu, odczyty intron/ekson na pz) + stabilnosc bazowa (BruChase 6 h);
  wspolny zakres: geny klasy spoza zakresu 1.-99. percentyla mRNA dla ktorejkolwiek cechy sa pomijane.
lncRNA (razem) = lincRNA, antysens, dywergentny, inne lncRNA, gospodarze snoRNA i miRNA (bez sense-intronic,
ktorego odczyty pochodza glownie z intronow genu gospodarza; ta klasa jest raportowana osobno).
Geny cis wyciszanego genu sa juz zamaskowane w macierzach.

Wydruk: pokrycie klas, rozklad efektow, replikacja K562 vs HepG2 dla tych samych RBP (najwazniejszy test),
zgodnosc A i B, RBP o najsilniejszych efektach w obu liniach.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors

M, OUT = Path("results/phase2/matrix"), Path("results/phase2/classes")
OUT.mkdir(parents=True, exist_ok=True)
K_NN, MIN_N, EFFECT, ALPHA = 10, 20, 0.1, 0.01
CLASSES = ["lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA",
           "sense-intronic", "pseudogen transkr."]
POOLED = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
LNC = "lncRNA (razem)"
VARIANTS = {"A": ["log_ekspr", "log_dl"], "B": ["log_ekspr", "log_dl", "splicing", "stab_6h"]}
pd.set_option("display.width", 220)

cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
kin = pd.read_csv("results/phase1/bru/gene_kinetics.tsv", sep="\t")


def splicing_by_cell():
    """log2(gestosc odczytow w intronach / w eksonach) w kontrolach pilotazu, mediana z probek; per linia."""
    reg = pd.read_csv("data/ref/eisa_regions_v29.tsv.gz", sep="\t")
    reg["L"] = reg["end"] - reg["start"]
    lens = reg.pivot_table(index="gene_id", columns="kind", values="L", aggfunc="sum")
    gene_ids = np.sort(reg["gene_id"].unique())          # ta sama kolejnosc co w skrypcie EISA
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
                ratio = (I / lens["intron"].reindex(gene_ids)) / (E / lens["exon"].reindex(gene_ids))
                vals.append(np.log2(ratio.where((E > 0) & (I > 0))))
        out[cell] = pd.concat(vals, axis=1).median(axis=1) if vals else pd.Series(dtype=float)
        print(f"   splicing {cell}: {len(vals)} probek kontrolnych, {int(out[cell].notna().sum())} genow")
    return out


def neighbours(feat, cols):
    """Indeksy 10 najblizszych mRNA dla genow kazdej klasy; tylko geny we wspolnym zakresie cech."""
    f = feat.dropna(subset=cols)
    mr = f[f["klasa"] == "mRNA"]
    lo, hi = mr[cols].quantile(0.01), mr[cols].quantile(0.99)
    inside = ((f[cols] >= lo) & (f[cols] <= hi)).all(axis=1)
    mu, sd = mr[cols].mean().values, mr[cols].std().values
    sd = np.where(sd > 0, sd, 1.0)
    nn = NearestNeighbors(n_neighbors=K_NN).fit((mr[cols].values - mu) / sd)
    res = {}
    for c in [LNC] + CLASSES:
        sel = f[(f["klasa"].isin(POOLED) if c == LNC else f["klasa"] == c) & inside]
        if len(sel) < MIN_N:
            continue
        _, idx = nn.kneighbors((sel[cols].values - mu) / sd)
        res[c] = (sel.index.values, mr.index.values[idx])
    return res


print("0. Cechy metabolizmu bazowego")
splice = splicing_by_cell()
rows, cover = [], []
for cell in ["K562", "HepG2"]:
    mats = {"skorygowana": pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0),
            "surowa": pd.read_csv(M / f"lfc_{cell}.tsv.gz", sep="\t", index_col=0)}
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    g = mats["surowa"].index
    feat = pd.DataFrame({"klasa": cls.reindex(g)["klasa"], "log_ekspr": np.log10(expr.reindex(g) + 1),
                         "log_dl": np.log10(cls.reindex(g)["width"]),
                         "splicing": splice.get(cell, pd.Series(dtype=float)).reindex(g),
                         "stab_6h": kin[kin["cell"] == cell].drop_duplicates("gene_id").set_index("gene_id")["stab_6h"].reindex(g)},
                        index=g)
    nbr = {v: neighbours(feat, cols) for v, cols in VARIANTS.items()}
    for v in VARIANTS:
        for c, (genes, _) in nbr[v].items():
            cover.append({"linia": cell, "dopasowanie": v, "klasa": c, "geny": len(genes)})
    for mname, L in mats.items():
        for kd in L.columns:
            x = L[kd]
            for v in VARIANTS:
                for c, (genes, idx) in nbr[v].items():
                    own = x.reindex(genes).values
                    matched = np.nanmean(x.reindex(idx.ravel()).values.reshape(idx.shape), axis=1)
                    d = own - matched
                    d = d[~np.isnan(d)]
                    if len(d) < MIN_N:
                        continue
                    rows.append({"linia": cell, "kd_experiment": kd, "target": summ.loc[kd, "target"],
                                 "metoda": summ.loc[kd, "method"], "target_lfc": summ.loc[kd, "target_lfc"],
                                 "macierz": mname, "dopasowanie": v, "klasa": c, "n": len(d),
                                 "roznica": float(np.median(d)), "p": float(wilcoxon(d).pvalue)})
res = pd.DataFrame(rows)
res.to_csv(OUT / "class_effects_all.tsv.gz", sep="\t", index=False)

print("\n1. Liczba genow w porownaniach (po ograniczeniu do wspolnego zakresu cech z mRNA)")
print(pd.DataFrame(cover).pivot_table(index="klasa", columns=["linia", "dopasowanie"], values="geny").reindex([LNC] + CLASSES)
      .fillna(0).astype(int).to_string())

print(f"\n2. Rozklad efektu lncRNA (razem) w knockdownach; istotny = |roznica| >= {EFFECT} i p < {ALPHA}")
lnc = res[res["klasa"] == LNC].copy()
lnc["w_gore"] = (lnc["roznica"] >= EFFECT) & (lnc["p"] < ALPHA)
lnc["w_dol"] = (lnc["roznica"] <= -EFFECT) & (lnc["p"] < ALPHA)
dist = lnc.groupby(["linia", "macierz", "dopasowanie"]).agg(
    knockdowny=("roznica", "size"), mediana=("roznica", "median"),
    q25=("roznica", lambda s: s.quantile(0.25)), q75=("roznica", lambda s: s.quantile(0.75)),
    proc_w_gore=("w_gore", lambda s: 100 * s.mean()), proc_w_dol=("w_dol", lambda s: 100 * s.mean()))
print(dist.round(2).to_string())


def wide(df, klasa, macierz, dop):
    s = df[(df["klasa"] == klasa) & (df["macierz"] == macierz) & (df["dopasowanie"] == dop)]
    return s.groupby(["target", "linia"])["roznica"].mean().unstack("linia")


print("\n3. Replikacja: ten sam RBP w K562 i HepG2 (Spearman roznic lncRNA - mRNA)")
rep = []
for macierz in ["skorygowana", "surowa"]:
    for dop in VARIANTS:
        for klasa in [LNC] + CLASSES:
            w = wide(res, klasa, macierz, dop).dropna()
            if len(w) >= 20:
                r, p = spearmanr(w["K562"], w["HepG2"])
                rep.append({"macierz": macierz, "dopasowanie": dop, "klasa": klasa, "RBP_w_obu": len(w), "rho": r, "p": p})
rep = pd.DataFrame(rep)
rep.to_csv(OUT / "replication.tsv", sep="\t", index=False)
print(rep[rep["klasa"] == LNC].round(3).to_string(index=False, formatters={"p": "{:.1e}".format}))
print("\n   Wedlug klas (macierz skorygowana, dopasowanie B):")
print(rep[(rep["macierz"] == "skorygowana") & (rep["dopasowanie"] == "B")].round(3)
      .to_string(index=False, formatters={"p": "{:.1e}".format}))
w = wide(res[~res["kd_experiment"].isin(summ[(summ["method"] == "shRNA") & (summ["target_lfc"] > -0.5)].index)],
         LNC, "skorygowana", "B").dropna()
if len(w) >= 20:
    r, p = spearmanr(w["K562"], w["HepG2"])
    print(f"\n   Bez knockdownow shRNA ze slabym wyciszeniem (spadek celu > -0.5): rho = {r:.3f}, p = {p:.1e}, RBP = {len(w)}")

print("\n4. Zgodnosc dopasowan A i B (Spearman roznic lncRNA - mRNA miedzy knockdownami, macierz skorygowana)")
for cell in ["K562", "HepG2"]:
    s = lnc[(lnc["linia"] == cell) & (lnc["macierz"] == "skorygowana")].pivot_table(index="kd_experiment", columns="dopasowanie", values="roznica").dropna()
    print(f"   {cell}: rho = {spearmanr(s['A'], s['B'])[0]:.3f} ({len(s)} knockdownow); mediana |A| = {s['A'].abs().median():.3f}, mediana |B| = {s['B'].abs().median():.3f}")

print("\n5. RBP o najsilniejszym efekcie na lncRNA (razem), zgodnym w obu liniach (macierz skorygowana, dopasowanie B)")
w = wide(res, LNC, "skorygowana", "B").dropna()
w["srednia"] = w.mean(axis=1)
w["zgodny_znak"] = np.sign(w["K562"]) == np.sign(w["HepG2"])
meth = summ.groupby("target")["method"].agg(lambda s: "/".join(sorted(set(s))))
w["metoda"] = meth.reindex(w.index)
z = w[w["zgodny_znak"]].sort_values("srednia")
print("   lncRNA rosna (najwyzsze):")
print(z.tail(15).iloc[::-1].round(3).to_string())
print("   lncRNA spadaja (najnizsze):")
print(z.head(15).round(3).to_string())
w.to_csv(OUT / "lnc_effect_by_rbp.tsv", sep="\t")
print(f"\nZapisano: {OUT}/class_effects_all.tsv.gz, {OUT}/replication.tsv, {OUT}/lnc_effect_by_rbp.tsv")
