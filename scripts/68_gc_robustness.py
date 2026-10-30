"""
Test odpornosci class effect na zawartosc GC (Van Nostrand i wsp. 2020: zaleznosc zmian od GC w danych z wyciszen).

GC genu = srednia zawartosc GC transkryptow GENCODE v29 wazona dlugoscia (sekwencja egzonowa).
Warianty (macierz skorygowana na wspolna kontrole, jak w glownej analizie, krok 2.4):
  B     = ekspresja + dlugosc genu + wydajnosc splicingu + stabilnosc 6 h   (glowny wariant, odtworzony)
  C     = B + GC
  B_gc  = dopasowanie B, ale log2FC kazdego wyciszenia z odjeta mediana w 20 przedzialach GC (trend wspolny dla genow)
Dla kazdego wariantu: mediana class effect lncRNA (razem), powtarzalnosc miedzy liniami dla tych samych RBP (rho),
zgodnosc z wariantem B. Dodatkowo: GC lncRNA vs mRNA i sila zaleznosci log2FC od GC w wyciszeniach.
Wyjscie: results/phase4/followups/gc_*.tsv
"""
import gzip
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors

M, OUT = Path("results/phase2/matrix"), Path("results/phase4/followups")
OUT.mkdir(parents=True, exist_ok=True)
FA = Path("data/ref/gencode.v29.transcripts.fa.gz")
URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_29/gencode.v29.transcripts.fa.gz"
K_NN, MIN_N = 10, 20
POOLED = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
VARIANTS = {"B": ["log_ekspr", "log_dl", "splicing", "stab_6h"], "C": ["log_ekspr", "log_dl", "splicing", "stab_6h", "gc"]}
pd.set_option("display.width", 200)
strip = lambda x: str(x).split(".")[0]

# ---------------------------------------------------------------- GC genow
gc_file = Path("data/ref/gene_gc_v29.tsv")
if not gc_file.exists():
    if not FA.exists():
        print(f"Pobieram {URL} ...", flush=True)
        urllib.request.urlretrieve(URL, FA)
    acc = {}
    with gzip.open(FA, "rt") as fh:
        gid, n_gc, n = None, 0, 0
        for line in fh:
            if line.startswith(">"):
                if gid is not None:
                    a = acc.setdefault(gid, [0, 0]); a[0] += n_gc; a[1] += n
                gid, n_gc, n = strip(line[1:].split("|")[1]), 0, 0
            else:
                s = line.strip().upper()
                n_gc += s.count("G") + s.count("C"); n += len(s)
        if gid is not None:
            a = acc.setdefault(gid, [0, 0]); a[0] += n_gc; a[1] += n
    pd.DataFrame([(g, a[0] / a[1]) for g, a in acc.items() if a[1] > 0], columns=["gene", "gc"]).to_csv(gc_file, sep="\t", index=False)
GC = pd.read_csv(gc_file, sep="\t").set_index("gene")["gc"]
print(f"GC policzone dla {len(GC):,} genow")

cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
kin = pd.read_csv("results/phase1/bru/gene_kinetics.tsv", sep="\t")


def splicing_by_cell():
    """Jak w kroku 2.4: log2(gestosc odczytow w intronach / w eksonach) w kontrolach pilotazu, mediana z probek."""
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
                ratio = (I / lens["intron"].reindex(gene_ids)) / (E / lens["exon"].reindex(gene_ids))
                vals.append(np.log2(ratio.where((E > 0) & (I > 0))))
        out[cell] = pd.concat(vals, axis=1).median(axis=1) if vals else pd.Series(dtype=float)
    return out


def neighbours(feat, cols):
    """Jak w kroku 2.4: 10 najblizszych mRNA (cechy standaryzowane), tylko geny we wspolnym zakresie 1.-99. percentyla mRNA."""
    f = feat.dropna(subset=cols)
    mr = f[f["klasa"] == "mRNA"]
    lo, hi = mr[cols].quantile(0.01), mr[cols].quantile(0.99)
    inside = ((f[cols] >= lo) & (f[cols] <= hi)).all(axis=1)
    mu, sd = mr[cols].mean().values, mr[cols].std().values
    sd = np.where(sd > 0, sd, 1.0)
    nn = NearestNeighbors(n_neighbors=K_NN).fit((mr[cols].values - mu) / sd)
    sel = f[f["klasa"].isin(POOLED) & inside]
    _, idx = nn.kneighbors((sel[cols].values - mu) / sd)
    return sel.index.values, mr.index.values[idx]


def class_effect(L, genes, idx):
    out = {}
    for kd in L.columns:
        x = L[kd]
        d = x.reindex(genes).values - np.nanmean(x.reindex(idx.ravel()).values.reshape(idx.shape), axis=1)
        d = d[~np.isnan(d)]
        if len(d) >= MIN_N:
            out[kd] = (float(np.median(d)), float(wilcoxon(d).pvalue))
    return out


splice = splicing_by_cell()
rows, gc_dep, gc_cls = [], [], []
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    g = L.index
    feat = pd.DataFrame({"klasa": cls.reindex(g)["klasa"], "log_ekspr": np.log10(expr.reindex(g) + 1),
                         "log_dl": np.log10(cls.reindex(g)["width"]),
                         "splicing": splice.get(cell, pd.Series(dtype=float)).reindex(g),
                         "stab_6h": kin[kin["cell"] == cell].drop_duplicates("gene_id").set_index("gene_id")["stab_6h"].reindex(g),
                         "gc": pd.Series([GC.get(strip(x), np.nan) for x in g], index=g)}, index=g)
    # 1. GC lncRNA vs mRNA; zaleznosc log2FC od GC wsrod mRNA
    for k, sel in [("mRNA", feat["klasa"] == "mRNA"), ("lncRNA (razem)", feat["klasa"].isin(POOLED))]:
        gc_cls.append({"linia": cell, "klasa": k, "geny": int(sel.sum()), "mediana_GC": feat.loc[sel, "gc"].median()})
    mr = feat.index[feat["klasa"] == "mRNA"]
    for kd in L.columns:
        x = L.loc[mr, kd]
        ok = x.notna() & feat.loc[mr, "gc"].notna()
        if ok.sum() > 100:
            gc_dep.append({"linia": cell, "kd_experiment": kd, "rho_lfc_GC_mRNA": spearmanr(x[ok], feat.loc[mr, "gc"][ok])[0]})
    # 2. warianty
    Lgc = L.copy()
    bins = pd.qcut(feat["gc"], 20, labels=False, duplicates="drop")
    Lgc = Lgc.sub(Lgc.groupby(bins).transform("median"))          # odjety trend GC wspolny dla wszystkich genow
    for v, cols, mat in [("B", VARIANTS["B"], L), ("C", VARIANTS["C"], L), ("B_gc", VARIANTS["B"], Lgc)]:
        genes, idx = neighbours(feat, cols)
        for kd, (eff, p) in class_effect(mat, genes, idx).items():
            rows.append({"linia": cell, "kd_experiment": kd, "target": summ.loc[kd, "target"], "wariant": v,
                         "geny_lncRNA": len(genes), "roznica": eff, "p": p})
    print(f"   {cell}: gotowe ({len(L.columns)} wyciszen)", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "gc_class_effects.tsv.gz", sep="\t", index=False)
dep = pd.DataFrame(gc_dep)
dep.to_csv(OUT / "gc_dependence_per_kd.tsv", sep="\t", index=False)

print("\n1. Zawartosc GC (sekwencja egzonowa)")
print(pd.DataFrame(gc_cls).round(3).to_string(index=False))
print("\n   Zaleznosc log2FC od GC wsrod mRNA w wyciszeniach (Spearman, po korekcie na wspolna kontrole):")
print(dep.groupby("linia")["rho_lfc_GC_mRNA"].describe(percentiles=[.1, .5, .9]).round(3).to_string())
print(f"   wyciszenia z |rho| > 0,2: {int((dep['rho_lfc_GC_mRNA'].abs() > 0.2).sum())} z {len(dep)}")

print("\n2. Class effect lncRNA (razem) w trzech wariantach")
summary = []
for v in ["B", "C", "B_gc"]:
    s = res[res["wariant"] == v]
    w = s.groupby(["target", "linia"])["roznica"].mean().unstack("linia").dropna()
    r, p = spearmanr(w["K562"], w["HepG2"])
    rb = np.random.default_rng(0)
    boots = [spearmanr(*w.sample(len(w), replace=True, random_state=int(rb.integers(1e9)))[["K562", "HepG2"]].values.T)[0] for _ in range(1000)]
    summary.append({"wariant": v, "geny_lncRNA_K562": int(s[s["linia"] == "K562"]["geny_lncRNA"].iloc[0]),
                    "mediana_class_effect": s["roznica"].median(), "RBP_w_obu": len(w), "rho_miedzy_liniami": r,
                    "CI95": f"{np.percentile(boots, 2.5):.2f}-{np.percentile(boots, 97.5):.2f}", "p": p})
summary = pd.DataFrame(summary)
summary.to_csv(OUT / "gc_summary.tsv", sep="\t", index=False)
print(summary.round(3).to_string(index=False, formatters={"p": "{:.1e}".format}))

print("\n3. Zgodnosc z wariantem B (Spearman class effect miedzy wyciszeniami)")
for cell in ["K562", "HepG2"]:
    t = res[res["linia"] == cell].pivot_table(index="kd_experiment", columns="wariant", values="roznica").dropna()
    print(f"   {cell}: B vs C rho = {spearmanr(t['B'], t['C'])[0]:.3f}; B vs B_gc rho = {spearmanr(t['B'], t['B_gc'])[0]:.3f} "
          f"({len(t)} wyciszen)")
print(f"\nZapisano: {OUT}/gc_summary.tsv, gc_class_effects.tsv.gz, gc_dependence_per_kd.tsv; GC genow: {gc_file}")
