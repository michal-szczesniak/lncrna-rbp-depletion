"""
Faza 3, krok 3.3B: czy wplyw RBP na lncRNA zalezy od lokalizacji lncRNA w komorce?

1. Wskaznik jadrowy genu (osobno dla linii): log2((TPM jadro + 0.1) / (TPM cytoplazma + 0.1)) z frakcjonowania poliA
   (ten sam typ bibliotek co knockdowny); srednia z replik i eksperymentow. Dodatnie = gen wzbogacony w jadrze.
   Kontrola: lncRNA powinny byc bardziej jadrowe niz mRNA, MALAT1 i NEAT1 silnie jadrowe.
2. Dla kazdego knockdownu (macierz skorygowana na kontrole): korelacja Spearmana zmiany genu z wskaznikiem jadrowym,
   osobno w lncRNA i w mRNA, takze czastkowa (po odjeciu wplywu ekspresji i stabilnosci bazowej).
   rho > 0: po wyciszeniu rosna bardziej geny jadrowe (RBP usuwa RNA w jadrze, np. eksosom);
   rho < 0: rosna bardziej geny cytoplazmatyczne (RBP usuwa RNA w cytoplazmie, np. NMD).
3. Replikacja tej zaleznosci miedzy liniami i podsumowanie dla grup RBP.
4. Czy lokalizacja tlumaczy efekt klasy: roznica lncRNA - dopasowane mRNA przy dopasowaniu B oraz C = B + wskaznik
   jadrowy, na tym samym zbiorze genow.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors

M, OUT = Path("results/phase2/matrix"), Path("results/phase3/localization")
OUT.mkdir(parents=True, exist_ok=True)
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
ORDER = ["mRNA", "lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA", "sense-intronic", "pseudogen transkr."]
GROUPS = {"NMD": ["UPF1", "UPF2", "UPF3B", "SMG1", "SMG5", "SMG6", "SMG7", "SMG8", "SMG9"],
          "EJC": ["RBM8A", "MAGOH", "EIF4A3", "CASC3"],
          "eksosom i MTREX": ["MTREX", "EXOSC1", "EXOSC2", "EXOSC3", "EXOSC4", "EXOSC5", "EXOSC6", "EXOSC7", "EXOSC8",
                              "EXOSC9", "EXOSC10", "DIS3", "ZFC3H1", "ZCCHC8", "RBM7", "PABPN1"],
          "CCR4-NOT": ["CNOT1", "CNOT2", "CNOT3", "CNOT4", "CNOT6", "CNOT6L", "CNOT7", "CNOT8", "CNOT9", "CNOT10", "CNOT11"],
          "paraspeckle (DBHS)": ["NONO", "SFPQ", "PSPC1"],
          "degradacja cytoplazmatyczna": ["XRN1", "DCP2", "DDX6", "LSM1", "PATL1", "PAN2", "PAN3"]}
K_NN = 10
pd.set_option("display.width", 230)

cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
cls["gene"] = cls["gene_id"].str.split(".").str[0]
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
kin = pd.read_csv("results/phase1/bru/gene_kinetics.tsv", sep="\t")

# ---------------------------------------------------------------- 1. wskaznik jadrowy
ff = pd.read_csv("data/meta/fraction_files.tsv", sep="\t", dtype=str)
man = pd.read_csv("data/raw/encode/download_manifest.tsv", sep="\t", dtype=str).drop_duplicates("file_accession", keep="last")
ff = ff.merge(man[["file_accession", "path"]], on="file_accession")
ff = ff[ff["nucleic_acid"] == "polyadenylated mRNA"]
tpm = {}
for (cell, frac), g in ff.groupby(["cell", "subcellular_fraction"]):
    vals = []
    for p in g["path"]:
        q = pd.read_csv(p, sep="\t", usecols=["gene_id", "TPM"])
        q["gene"] = q["gene_id"].str.split(".").str[0]
        vals.append(np.log2(q.groupby("gene")["TPM"].sum() + 0.1))
    tpm[(cell, frac)] = pd.concat(vals, axis=1).mean(axis=1)
nuc = {}
print("1. Wskaznik jadrowy (log2 jadro/cytoplazma, poliA); mediany wedlug klasy")
rows = []
for cell in ["K562", "HepG2"]:
    n_, c_ = tpm[(cell, "nucleus")], tpm[(cell, "cytosol")]
    ok = np.maximum(n_, c_) >= np.log2(0.5 + 0.1)
    idx = (n_ - c_)[ok]
    nuc[cell] = idx
    k = cls.drop_duplicates("gene").set_index("gene").reindex(idx.index)["klasa"]
    rows.append(idx.groupby(k).median().rename(cell))
    named = cls.drop_duplicates("gene").set_index("gene_name")["gene"]
    ex = {n: idx.get(named.get(n), np.nan) for n in ["MALAT1", "NEAT1", "XIST", "GAPDH", "ACTB", "GAS5"]}
    print(f"   {cell}: {len(idx)} genow; przyklady: " + ", ".join(f"{n} {v:+.2f}" for n, v in ex.items() if pd.notna(v)))
print(pd.concat(rows, axis=1).reindex(ORDER).round(2).to_string())
both = pd.concat(nuc, axis=1).dropna()
print(f"   Zgodnosc wskaznika miedzy liniami: Spearman {spearmanr(both['K562'], both['HepG2'])[0]:.2f} ({len(both)} genow)")
pd.concat(nuc, axis=1).to_csv(OUT / "nuclear_index.tsv", sep="\t")


# ---------------------------------------------------------------- 2. zaleznosc efektu od lokalizacji
def resid_rank(df, y, covars):
    Z = df[[y] + covars].rank()
    X = np.c_[np.ones(len(Z)), Z[covars].values]
    beta, *_ = np.linalg.lstsq(X, Z[y].values, rcond=None)
    return Z[y].values - X @ beta


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


splice = splicing_by_cell()
per_kd, cls_rows = [], []
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    g = L.index
    gk = pd.Series(g.str.split(".").str[0], index=g)
    feat = pd.DataFrame({"klasa": cls.set_index("gene_id").reindex(g)["klasa"],
                         "log_ekspr": np.log10(expr.reindex(g) + 1), "log_dl": np.log10(cls.set_index("gene_id").reindex(g)["width"]),
                         "stab_6h": kin[kin["cell"] == cell].drop_duplicates("gene_id").set_index("gene_id")["stab_6h"].reindex(g),
                         "splicing": splice[cell].reindex(g), "jadro": nuc[cell].reindex(gk.values).values}, index=g)
    for kd in L.columns:
        row = {"linia": cell, "kd_experiment": kd, "target": summ.loc[kd, "target"], "metoda": summ.loc[kd, "method"]}
        for name, sel in [("lncRNA", feat["klasa"].isin(LNC)), ("mRNA", feat["klasa"] == "mRNA")]:
            d = feat[sel].assign(lfc=L[kd]).dropna(subset=["lfc", "jadro", "log_ekspr", "stab_6h"])
            if len(d) < 50:
                continue
            row[f"rho_{name}"] = spearmanr(d["lfc"], d["jadro"])[0]
            r1, r2 = resid_rank(d, "lfc", ["log_ekspr", "stab_6h"]), resid_rank(d, "jadro", ["log_ekspr", "stab_6h"])
            row[f"rho_czast_{name}"], row[f"p_czast_{name}"] = spearmanr(r1, r2)
            row[f"n_{name}"] = len(d)
        per_kd.append(row)

    # 4. czy lokalizacja tlumaczy efekt klasy: B vs C na tym samym zbiorze genow
    colsB, colsC = ["log_ekspr", "log_dl", "splicing", "stab_6h"], ["log_ekspr", "log_dl", "splicing", "stab_6h", "jadro"]
    f = feat.dropna(subset=colsC)
    mr = f[f["klasa"] == "mRNA"]
    lo, hi = mr[colsC].quantile(0.01), mr[colsC].quantile(0.99)
    sel = f[f["klasa"].isin(LNC) & ((f[colsC] >= lo) & (f[colsC] <= hi)).all(axis=1)]
    nbr = {}
    for v, cols in [("B", colsB), ("C", colsC)]:
        mu, sd = mr[cols].mean().values, mr[cols].std().values
        sd = np.where(sd > 0, sd, 1.0)
        _, idx = NearestNeighbors(n_neighbors=K_NN).fit((mr[cols].values - mu) / sd).kneighbors((sel[cols].values - mu) / sd)
        nbr[v] = mr.index.values[idx]
    for kd in L.columns:
        x = L[kd]
        for v, idx in nbr.items():
            dd = x.reindex(sel.index).values - np.nanmean(x.reindex(idx.ravel()).values.reshape(idx.shape), axis=1)
            dd = dd[~np.isnan(dd)]
            if len(dd) >= 20:
                cls_rows.append({"linia": cell, "kd_experiment": kd, "target": summ.loc[kd, "target"], "wariant": v,
                                 "roznica": float(np.median(dd)), "n": len(dd)})
    print(f"\n   {cell}: porownanie B i C na {len(sel)} lncRNA z pelnym zestawem cech")

pk = pd.DataFrame(per_kd)
pk.to_csv(OUT / "localization_dependence_per_kd.tsv", sep="\t", index=False)
print("\n2. Zaleznosc efektu knockdownu od lokalizacji (czastkowa rho, po odjeciu ekspresji i stabilnosci)")
for cell, g in pk.groupby("linia"):
    for name in ["lncRNA", "mRNA"]:
        r, p = g[f"rho_czast_{name}"], g[f"p_czast_{name}"]
        print(f"   {cell} {name:<6}: mediana {r.median():+.3f}; |rho| >= 0.1 i p < 0.01 w {100 * ((r.abs() >= 0.1) & (p < 0.01)).mean():.0f}% "
              f"knockdownow (dodatnich {100 * ((r >= 0.1) & (p < 0.01)).mean():.0f}%, ujemnych {100 * ((r <= -0.1) & (p < 0.01)).mean():.0f}%)")

print("\n3. Replikacja zaleznosci od lokalizacji miedzy liniami (ten sam RBP, Spearman)")
w = {n: pk.groupby(["target", "linia"])[f"rho_czast_{n}"].mean().unstack("linia").dropna() for n in ["lncRNA", "mRNA"]}
for n, t in w.items():
    print(f"   {n}: rho = {spearmanr(t['K562'], t['HepG2'])[0]:+.2f} ({len(t)} RBP)")
print("\n   Grupy RBP: mediana czastkowej rho dla lncRNA (w nawiasie liczba knockdownow)")
for gname, members in GROUPS.items():
    parts = []
    for cell in ["K562", "HepG2"]:
        s = pk[(pk["linia"] == cell) & pk["target"].isin(members)]["rho_czast_lncRNA"].dropna()
        parts.append(f"{cell} {s.median():+.3f} ({len(s)})" if len(s) else f"{cell} brak")
    print(f"   {gname:<28} " + "; ".join(parts))
t = w["lncRNA"].copy()
t["srednia"] = t.mean(axis=1)
t = t[np.sign(t["K562"]) == np.sign(t["HepG2"])].sort_values("srednia")
print("\n   RBP o najsilniejszej, zgodnej w obu liniach zaleznosci (lncRNA, czastkowa rho)")
print("   po wyciszeniu rosna bardziej lncRNA cytoplazmatyczne (rho < 0):")
print(t.head(12).round(3).to_string())
print("   po wyciszeniu rosna bardziej lncRNA jadrowe (rho > 0):")
print(t.tail(12).iloc[::-1].round(3).to_string())
t.to_csv(OUT / "localization_dependence_by_rbp.tsv", sep="\t")

cr = pd.DataFrame(cls_rows)
cr.to_csv(OUT / "class_effect_B_vs_C.tsv", sep="\t", index=False)
print("\n4. Czy lokalizacja tlumaczy efekt klasy? lncRNA - dopasowane mRNA, ten sam zbior genow")
for v in ["B", "C"]:
    ww = cr[cr["wariant"] == v].groupby(["target", "linia"])["roznica"].mean().unstack("linia").dropna()
    print(f"   {v}: mediana |roznicy| = {cr.loc[cr['wariant'] == v, 'roznica'].abs().median():.3f}; "
          f"replikacja miedzy liniami rho = {spearmanr(ww['K562'], ww['HepG2'])[0]:+.2f} ({len(ww)} RBP)")
bc = cr.pivot_table(index=["linia", "kd_experiment"], columns="wariant", values="roznica").dropna()
print(f"   Zgodnosc B i C miedzy knockdownami: rho = {spearmanr(bc['B'], bc['C'])[0]:.2f}")
print(f"\nZapisano: {OUT}/nuclear_index.tsv, localization_dependence_per_kd.tsv, localization_dependence_by_rbp.tsv, class_effect_B_vs_C.tsv")
