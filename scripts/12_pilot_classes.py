"""
Faza 1, krok 3: klasy genow i kontrola pozytywna.

1. Przypisuje kazdemu genowi GENCODE v29 klase (mRNA, podklasy lncRNA, geny gospodarze snoRNA i miRNA,
   pseudogeny transkrybowane) -> data/ref/gene_classes_v29.tsv
2. Dla kazdego knockdownu porownuje zmiane (lfc_shr) genow z danej klasy ze zmiana mRNA dopasowanych
   pod wzgledem ekspresji (baseMean) i dlugosci genu: kazdy gen klasy dostaje srednia z 10 najblizszych mRNA.
   Test: Wilcoxon (pary: gen klasy - dopasowane mRNA).
3. Sprawdza kryterium A pilotazu: po wyciszeniu MTREX i EXOSC10 lncRNA rosna silniej niz dopasowane mRNA.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/phase1/classes")
OUT.mkdir(parents=True, exist_ok=True)
K_NEIGHBOURS = 10

# ---------------------------------------------------------------- 1. klasy
genes = pd.read_csv("data/ref/genes_v29.tsv", sep="\t")
LNC = {"lincRNA": "lincRNA", "antisense": "antysens", "bidirectional_promoter_lncRNA": "dywergentny",
       "sense_intronic": "sense-intronic", "sense_overlapping": "inne lncRNA",
       "processed_transcript": "inne lncRNA", "3prime_overlapping_ncRNA": "inne lncRNA",
       "macro_lncRNA": "inne lncRNA", "non_coding": "inne lncRNA"}
PSEUDO = {"transcribed_processed_pseudogene", "transcribed_unprocessed_pseudogene",
          "transcribed_unitary_pseudogene"}
genes["klasa"] = None
genes.loc[genes["gene_type"] == "protein_coding", "klasa"] = "mRNA"
genes.loc[genes["gene_type"].isin(PSEUDO), "klasa"] = "pseudogen transkr."
is_lnc = genes["gene_type"].isin(LNC)
genes.loc[is_lnc, "klasa"] = genes.loc[is_lnc, "gene_type"].map(LNC)


def hosts(small_types):
    """lncRNA, ktore na tej samej nici w calosci zawieraja gen malego RNA danego typu."""
    small = genes[genes["gene_type"].isin(small_types)][["chr", "strand", "start", "end"]]
    lnc = genes[is_lnc][["gene_id", "chr", "strand", "start", "end"]]
    m = lnc.merge(small, on=["chr", "strand"], suffixes=("", "_s"))
    m = m[(m["start"] <= m["start_s"]) & (m["end"] >= m["end_s"])]
    return set(m["gene_id"])


snhg = hosts({"snoRNA", "scaRNA"})
mirhg = hosts({"miRNA"}) - snhg
genes.loc[genes["gene_id"].isin(mirhg), "klasa"] = "gospodarz miRNA"
genes.loc[genes["gene_id"].isin(snhg), "klasa"] = "gospodarz snoRNA"
genes[["gene_id", "gene_name", "gene_type", "klasa", "width"]].to_csv(
    "data/ref/gene_classes_v29.tsv", sep="\t", index=False)

ORDER = ["lncRNA (wszystkie)", "lincRNA", "antysens", "dywergentny", "sense-intronic", "inne lncRNA",
         "gospodarz snoRNA", "gospodarz miRNA", "pseudogen transkr."]
LNC_ALL = {"lincRNA", "antysens", "dywergentny", "sense-intronic", "inne lncRNA",
           "gospodarz snoRNA", "gospodarz miRNA"}
print("1. Geny w klasach (GENCODE v29):")
print(genes["klasa"].value_counts().reindex(["mRNA"] + ORDER[1:]).fillna(0).astype(int).to_string())

# ---------------------------------------------------------------- 2. porownanie klas
kds = pd.read_csv("results/phase1/deseq2/summary.tsv", sep="\t")
rows = []
for kd in kds.to_dict("records"):
    r = pd.read_csv(f"results/phase1/deseq2/{kd['kd_experiment']}.tsv.gz", sep="\t")
    r = r.merge(genes[["gene_id", "klasa", "width"]], on="gene_id", how="left")
    r = r.dropna(subset=["lfc_shr", "klasa"]).reset_index(drop=True)
    X = np.c_[np.log10(r["baseMean"] + 1), np.log10(r["width"])]
    mrna = r["klasa"].values == "mRNA"
    mu, sd = X[mrna].mean(0), X[mrna].std(0)
    nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS).fit((X[mrna] - mu) / sd)
    lfc_mrna = r.loc[mrna, "lfc_shr"].values
    for cls in ORDER:
        if cls == "lncRNA (wszystkie)":
            sel = r["klasa"].isin(LNC_ALL).values
        else:
            sel = r["klasa"].values == cls
        if sel.sum() < 20:
            continue
        _, idx = nn.kneighbors((X[sel] - mu) / sd)
        matched = lfc_mrna[idx].mean(1)
        diff = r.loc[sel, "lfc_shr"].values - matched
        rows.append({**{k: kd[k] for k in ["kd_experiment", "target", "cell", "method", "pilot_role"]},
                     "klasa": cls, "n": int(sel.sum()),
                     "mediana_lfc_klasy": float(np.median(r.loc[sel, "lfc_shr"])),
                     "mediana_lfc_mRNA_dopas": float(np.median(matched)),
                     "mediana_roznicy": float(np.median(diff)), "p": float(wilcoxon(diff).pvalue)})
res = pd.DataFrame(rows)
res.to_csv(OUT / "class_effects.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- 3. wydruk i wykres
def star(p):
    return "" if pd.isna(p) else ("**" if p < 0.001 else ("*" if p < 0.01 else ""))


res["kd"] = res["target"] + " " + res["cell"]
order_kd = (kds.sort_values(["pilot_role", "target", "cell"])
            .assign(kd=lambda d: d["target"] + " " + d["cell"])["kd"].tolist())
piv = res.pivot(index="kd", columns="klasa", values="mediana_roznicy").reindex(index=order_kd, columns=ORDER)
pv = res.pivot(index="kd", columns="klasa", values="p").reindex(index=order_kd, columns=ORDER)
stars = pd.DataFrame([[star(p) for p in row] for row in pv.values], index=pv.index, columns=pv.columns)
txt = piv.round(2).astype(str).replace("nan", "-") + stars
pd.set_option("display.width", 250)
print("\n2. Mediana roznicy log2FC: klasa minus dopasowane mRNA (* p<0.01, ** p<0.001)")
print("   dodatnia = klasa rosnie po knockdownie bardziej niz porownywalne mRNA")
print(txt.to_string())

print("\n3. Kryterium A (kontrola pozytywna): lncRNA (wszystkie) po MTREX i EXOSC10")
pc = res[(res["klasa"] == "lncRNA (wszystkie)") & res["target"].isin(["MTREX", "EXOSC10"])]
for r in pc.to_dict("records"):
    ok = r["mediana_roznicy"] > 0 and r["p"] < 0.01
    print(f"   {r['kd']:<14} roznica {r['mediana_roznicy']:+.3f}, p = {r['p']:.1e}, n = {r['n']}  -> "
          f"{'spelnione' if ok else 'NIE spelnione'}")

fig, ax = plt.subplots(figsize=(11, 0.45 * len(order_kd) + 2))
lim = np.nanmax(np.abs(piv.values))
im = ax.imshow(piv.values.astype(float), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(len(ORDER)), ORDER, rotation=40, ha="right")
ax.set_yticks(range(len(order_kd)), order_kd)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        if pd.notna(piv.iat[i, j]):
            ax.text(j, i, f"{piv.iat[i, j]:.2f}{stars.iat[i, j]}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="klasa - mRNA (log2FC)")
ax.set_title("Pilotaz: zmiana klas po knockdownie wzgledem dopasowanych mRNA")
fig.tight_layout()
fig.savefig(OUT / "class_effects_heatmap.png", dpi=150)
print(f"\nZapisano: {OUT}/class_effects.tsv, {OUT}/class_effects_heatmap.png, data/ref/gene_classes_v29.tsv")
