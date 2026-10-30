"""
Faza 1, krok 4: stabilnosc bazowa klas z Bru-seq i BruChase-seq.

Dla kazdej linii (K562, HepG2):
  synteza      = gestosc odczytow Bru-seq bez traktowania (RNA powstajace w czasie znakowania)
  stabilnosc   = log2( BruChase po 2 h lub 6 h / Bru-seq ), czyli jaka czesc znakowanego RNA przetrwala
1. Porownanie klas w stanie podstawowym: stabilnosc klasy minus stabilnosc mRNA o podobnej syntezie
   i dlugosci (10 najblizszych mRNA). Oczekiwanie z literatury: lncRNA mniej stabilne niz mRNA.
2. Polaczenie z pilotazem: czy geny mniej stabilne bazowo rosna silniej po knockdownie
   (korelacja Spearmana stabilnosci 6 h z niesciagnietym log2FC, osobno dla mRNA i lncRNA).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.neighbors import NearestNeighbors

OUT = Path("results/phase1/bru")
OUT.mkdir(parents=True, exist_ok=True)
MIN_READS = 20          # minimalna liczba odczytow Bru-seq (w kazdej replice), zeby liczyc stabilnosc
K_NEIGHBOURS = 10
EXPERIMENTS = {          # eksperyment -> (linia, warunek); Bru-seq po napromienianiu pomijamy
    "ENCSR729WFH": ("K562", "bru"), "ENCSR633UIR": ("K562", "chase2h"), "ENCSR762OPQ": ("K562", "chase6h"),
    "ENCSR974AQD": ("HepG2", "bru"), "ENCSR295FEH": ("HepG2", "chase2h"), "ENCSR135DZR": ("HepG2", "chase6h"),
}
COLS = ["chr", "start", "end", "name", "score", "strand", "gene_len", "feature", "c9",
        "eff_len", "reads", "density", "norm_density"]
ORDER = ["lncRNA (wszystkie)", "lincRNA", "antysens", "dywergentny", "sense-intronic", "inne lncRNA",
         "gospodarz snoRNA", "gospodarz miRNA", "pseudogen transkr."]
LNC_ALL = {"lincRNA", "antysens", "dywergentny", "sense-intronic", "inne lncRNA",
           "gospodarz snoRNA", "gospodarz miRNA"}

files = pd.read_csv("data/meta/files_to_download_v29.tsv", sep="\t", dtype=str).fillna("")
man = pd.read_csv("data/raw/encode/download_manifest.tsv", sep="\t", dtype=str).fillna("")
files = files.merge(man[["file_accession", "path"]], on="file_accession")
files = files[files["experiment"].isin(EXPERIMENTS) & (files["output_type"] == "gene quantifications")]

# ---------------------------------------------------------------- 1. wczytanie i RPKM
libs = []
for f in files.to_dict("records"):
    cell, cond = EXPERIMENTS[f["experiment"]]
    d = pd.read_csv(f["path"], sep="\t", header=None, names=COLS)
    d = d[d["feature"] == "gene_sense"].copy()
    d["gene"] = d["name"].str.split("/").str[0].str.split(".").str[0]
    d = d.groupby("gene", as_index=False)[["reads", "eff_len"]].sum()
    d["rpkm"] = d["reads"] / d["eff_len"].clip(lower=1) * 1e3 / (d["reads"].sum() / 1e6)
    d["cell"], d["cond"], d["rep"] = cell, cond, f["bio_rep"]
    libs.append(d)
libs = pd.concat(libs)
print("1. Biblioteki Bru (liczba, mln odczytow na niciach sensownych genow):")
print(libs.groupby(["cell", "cond", "rep"])["reads"].sum().div(1e6).round(1).to_string())

wide = libs.pivot_table(index=["cell", "gene"], columns=["cond", "rep"], values=["reads", "rpkm"])
kin = []
for cell in ["K562", "HepG2"]:
    w = wide.loc[cell].dropna(axis=1, how="all")   # tylko repliki tej linii
    ok = (w["reads"]["bru"] >= MIN_READS).all(axis=1)
    rp = {c: w["rpkm"][c].mean(axis=1) for c in ["bru", "chase2h", "chase6h"]}
    eps = 0.01
    k = pd.DataFrame({"synteza": np.log2(rp["bru"] + eps),
                      "stab_2h": np.log2((rp["chase2h"] + eps) / (rp["bru"] + eps)),
                      "stab_6h": np.log2((rp["chase6h"] + eps) / (rp["bru"] + eps))})[ok]
    k["cell"] = cell
    kin.append(k.reset_index())
kin = pd.concat(kin)

genes = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
genes = genes[~genes["gene_id"].str.endswith("_PAR_Y")].copy()
genes["gene"] = genes["gene_id"].str.split(".").str[0]
kin = kin.merge(genes[["gene", "gene_id", "gene_name", "klasa", "width"]], on="gene", how="inner")
kin.to_csv(OUT / "gene_kinetics.tsv", sep="\t", index=False)
print(f"\n   Geny z wiarygodna stabilnoscia (>= {MIN_READS} odczytow Bru w kazdej replice):")
print(kin.groupby(["cell", "klasa"]).size().unstack("cell").reindex(["mRNA"] + ORDER[1:]).fillna(0).astype(int).to_string())

# ---------------------------------------------------------------- 2. klasy w stanie podstawowym
rows = []
for cell in ["K562", "HepG2"]:
    k = kin[kin["cell"] == cell].dropna(subset=["klasa"]).reset_index(drop=True)
    X = np.c_[k["synteza"], np.log10(k["width"])]
    mrna = k["klasa"].values == "mRNA"
    mu, sd = X[mrna].mean(0), X[mrna].std(0)
    nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS).fit((X[mrna] - mu) / sd)
    for cls in ORDER:
        sel = k["klasa"].isin(LNC_ALL).values if cls == "lncRNA (wszystkie)" else (k["klasa"].values == cls)
        if sel.sum() < 20:
            continue
        _, idx = nn.kneighbors((X[sel] - mu) / sd)
        row = {"cell": cell, "klasa": cls, "n": int(sel.sum())}
        for col in ["stab_2h", "stab_6h"]:
            diff = k.loc[sel, col].values - k.loc[mrna, col].values[idx].mean(1)
            row[f"{col}_roznica"] = float(np.median(diff))
            row[f"{col}_p"] = float(wilcoxon(diff).pvalue)
        rows.append(row)
base = pd.DataFrame(rows)
base.to_csv(OUT / "class_stability_baseline.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print("\n2. Stabilnosc bazowa: klasa minus mRNA o podobnej syntezie i dlugosci (mediana log2; ujemna = mniej stabilna)")
t = base.pivot(index="klasa", columns="cell", values=["stab_2h_roznica", "stab_6h_roznica"]).reindex(ORDER)
print(t.round(2).to_string())

# ---------------------------------------------------------------- 3. polaczenie z knockdownami pilotazu
kds = pd.read_csv("results/phase1/deseq2/summary.tsv", sep="\t")
rows = []
for kd in kds.to_dict("records"):
    r = pd.read_csv(f"results/phase1/deseq2/{kd['kd_experiment']}.tsv.gz", sep="\t")
    r["gene"] = r["gene_id"].str.split(".").str[0]
    m = r.merge(kin[kin["cell"] == kd["cell"]][["gene", "klasa", "stab_6h"]], on="gene").dropna(subset=["lfc_mle"])
    row = {"kd": f"{kd['target']} {kd['cell']}", "rola": kd["pilot_role"]}
    for name, sel in [("mRNA", m["klasa"] == "mRNA"), ("lncRNA", m["klasa"].isin(LNC_ALL))]:
        rho, p = spearmanr(m.loc[sel, "stab_6h"], m.loc[sel, "lfc_mle"])
        row[f"rho_{name}"], row[f"p_{name}"], row[f"n_{name}"] = rho, p, int(sel.sum())
    rows.append(row)
link = pd.DataFrame(rows).sort_values(["rola", "kd"])
link.to_csv(OUT / "stability_vs_kd.tsv", sep="\t", index=False)
print("\n3. Korelacja stabilnosci bazowej (6 h) z log2FC po knockdownie (Spearman)")
print("   ujemna = mniej stabilne geny rosna silniej po knockdownie")
print(link.round({"rho_mRNA": 2, "rho_lncRNA": 2}).to_string(index=False, formatters={
    "p_mRNA": "{:.1e}".format, "p_lncRNA": "{:.1e}".format}))
print(f"\nZapisano: {OUT}/gene_kinetics.tsv, {OUT}/class_stability_baseline.tsv, {OUT}/stability_vs_kd.tsv")
