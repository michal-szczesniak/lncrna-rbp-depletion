"""
Faza 1, krok 3b: czy wynik kroku 3 to biologia, czy efekt sciagania apeglm?

1. Diagnostyka: jak mocno apeglm sciagnal log2FC w kazdym knockdownie (osobno mRNA i lncRNA).
2. Porownanie klas na NIESCIAGNIETYCH log2FC (lfc_mle), z tym samym dopasowaniem mRNA co w kroku 3.
3. Drugi, niezalezny odczyt: odsetek genow istotnie w gore / w dol (padj < 0.05) w klasie
   wobec dopasowanych mRNA.
4. Kryterium A w wersji poprawionej: mediana roznicy >= 0.1 log2 i p < 0.01.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/phase1/classes_mle")
OUT.mkdir(parents=True, exist_ok=True)
K_NEIGHBOURS, MIN_EFFECT = 10, 0.1
ORDER = ["lncRNA (wszystkie)", "lincRNA", "antysens", "dywergentny", "sense-intronic", "inne lncRNA",
         "gospodarz snoRNA", "gospodarz miRNA", "pseudogen transkr."]
LNC_ALL = {"lincRNA", "antysens", "dywergentny", "sense-intronic", "inne lncRNA",
           "gospodarz snoRNA", "gospodarz miRNA"}

genes = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
kds = pd.read_csv("results/phase1/deseq2/summary.tsv", sep="\t")
kds["kd"] = kds["target"] + " " + kds["cell"]
diag, rows = [], []
for kd in kds.to_dict("records"):
    r = pd.read_csv(f"results/phase1/deseq2/{kd['kd_experiment']}.tsv.gz", sep="\t")
    r = r.merge(genes[["gene_id", "klasa", "width"]], on="gene_id", how="left")
    r = r.dropna(subset=["lfc_mle", "lfc_shr", "klasa"]).reset_index(drop=True)
    r["up"] = ((r["padj"] < 0.05) & (r["lfc_mle"] > 0)).astype(float)
    r["down"] = ((r["padj"] < 0.05) & (r["lfc_mle"] < 0)).astype(float)
    mrna = r["klasa"].values == "mRNA"
    lnc = r["klasa"].isin(LNC_ALL).values

    def shrink(sel):  # ile zostaje z |log2FC| po sciaganiu (1 = brak sciagania, 0 = wszystko do zera)
        return np.median(np.abs(r.loc[sel, "lfc_shr"])) / np.median(np.abs(r.loc[sel, "lfc_mle"]))
    diag.append({"kd": kd["kd"], "geny_istotne": kd["n_padj05"],
                 "SD_mle_mRNA": r.loc[mrna, "lfc_mle"].std(), "SD_mle_lncRNA": r.loc[lnc, "lfc_mle"].std(),
                 "zostaje_po_sciaganiu_mRNA": shrink(mrna), "zostaje_po_sciaganiu_lncRNA": shrink(lnc)})

    X = np.c_[np.log10(r["baseMean"] + 1), np.log10(r["width"])]
    mu, sd = X[mrna].mean(0), X[mrna].std(0)
    nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS).fit((X[mrna] - mu) / sd)
    m_lfc, m_up, m_down = (r.loc[mrna, c].values for c in ["lfc_mle", "up", "down"])
    for cls in ORDER:
        sel = lnc if cls == "lncRNA (wszystkie)" else (r["klasa"].values == cls)
        if sel.sum() < 20:
            continue
        _, idx = nn.kneighbors((X[sel] - mu) / sd)
        diff = r.loc[sel, "lfc_mle"].values - m_lfc[idx].mean(1)
        rows.append({"kd_experiment": kd["kd_experiment"], "kd": kd["kd"], "pilot_role": kd["pilot_role"],
                     "klasa": cls, "n": int(sel.sum()),
                     "mediana_roznicy_mle": float(np.median(diff)), "p": float(wilcoxon(diff).pvalue),
                     "proc_w_gore_klasa": 100 * r.loc[sel, "up"].mean(), "proc_w_gore_mRNA": 100 * m_up[idx].mean(),
                     "proc_w_dol_klasa": 100 * r.loc[sel, "down"].mean(), "proc_w_dol_mRNA": 100 * m_down[idx].mean()})
diag = pd.DataFrame(diag)
res = pd.DataFrame(rows)
diag.to_csv(OUT / "shrinkage_diagnostic.tsv", sep="\t", index=False)
res.to_csv(OUT / "class_effects_mle.tsv", sep="\t", index=False)

pd.set_option("display.width", 250)
order_kd = kds.sort_values(["pilot_role", "target", "cell"])["kd"].tolist()
print("1. Diagnostyka sciagania: jaka czesc |log2FC| zostaje po apeglm (mediana; 1 = brak sciagania)")
print(diag.set_index("kd").reindex(order_kd).round(2).to_string())


def star(p):
    return "" if pd.isna(p) else ("**" if p < 0.001 else ("*" if p < 0.01 else ""))


piv = res.pivot(index="kd", columns="klasa", values="mediana_roznicy_mle").reindex(index=order_kd, columns=ORDER)
pv = res.pivot(index="kd", columns="klasa", values="p").reindex(index=order_kd, columns=ORDER)
stars = pd.DataFrame([[star(p) for p in row] for row in pv.values], index=pv.index, columns=pv.columns)
print("\n2. Mediana roznicy NIESCIAGNIETYCH log2FC: klasa minus dopasowane mRNA (* p<0.01, ** p<0.001)")
print((piv.round(2).astype(str).replace("nan", "-") + stars).to_string())

print("\n3. Odsetek genow istotnie zmienionych (padj<0.05): lncRNA (wszystkie) wobec dopasowanych mRNA")
t = res[res["klasa"] == "lncRNA (wszystkie)"].set_index("kd").reindex(order_kd)
print(t[["proc_w_gore_klasa", "proc_w_gore_mRNA", "proc_w_dol_klasa", "proc_w_dol_mRNA"]].round(1).to_string())

print(f"\n4. Kryterium A, wersja poprawiona: mediana roznicy >= {MIN_EFFECT} log2 i p < 0.01")
for r in res[(res["klasa"] == "lncRNA (wszystkie)") & res["kd"].str.match(r"^(MTREX|EXOSC10) ")].to_dict("records"):
    ok = r["mediana_roznicy_mle"] >= MIN_EFFECT and r["p"] < 0.01
    print(f"   {r['kd']:<14} roznica {r['mediana_roznicy_mle']:+.3f}, p = {r['p']:.1e}  -> "
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
fig.colorbar(im, ax=ax, label="klasa - mRNA (log2FC, bez sciagania)")
ax.set_title("Pilotaz: zmiana klas wzgledem dopasowanych mRNA, niesciagniete log2FC")
fig.tight_layout()
fig.savefig(OUT / "class_effects_mle_heatmap.png", dpi=150)
print(f"\nZapisano: {OUT}/shrinkage_diagnostic.tsv, {OUT}/class_effects_mle.tsv, {OUT}/class_effects_mle_heatmap.png")
