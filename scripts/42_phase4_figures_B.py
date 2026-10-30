"""
Faza 4, krok 4.2B: ryciny 3-5 i S1.

Rycina 3. Czego efekt klasy NIE tlumaczy.
  a, b) Replikacja miedzy liniami i mediana |efektu| przed i po uwzglednieniu kolejnych cech: metabolizm (A -> B na tych
        samych genach), lokalizacja (B -> C), plastycznosc (C -> D), bezposrednie wiazanie (wszystkie -> niezwiazane geny).
  c) Czy geny zwiazane (eCLIP) reaguja na wyciszenie silniej niz niezwiazane (rozklad miedzy RBP).
Rycina 4. Spliceosom i NMD.
  a) Kategorie funkcjonalne RBP a efekt na lncRNA.
  b) EISA, wyciszenie spliceosomu: bezwzgledna zmiana intronow mRNA i lncRNA w przedzialach ekspresji.
  c) Wyciszenie NMD (UPF1, UPF2, SMG): bezwzgledna zmiana eksonow mRNA i lncRNA w przedzialach ekspresji.
Rycina 5. Geny gospodarze snoRNA jako osobna podklasa: lokalizacja, swoistosc tkankowa, promotory CpG, odpowiedz na NMD.
Rycina S1. Efekty cis nokautu Cas9 i shRNA na sasiadow wyciszanego genu.
Wyjscie: results/phase4/figures/fig3..fig5, figS1 (.png i .pdf)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

F = Path("results/phase4/figures")
F.mkdir(parents=True, exist_ok=True)
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz miRNA"}
NMD = {"UPF1", "UPF2", "UPF3B", "SMG1", "SMG5", "SMG6", "SMG7", "SMG8", "SMG9"}
EJC = {"RBM8A", "MAGOH", "EIF4A3", "CASC3"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
COL = {"K562": "#3b6ea8", "HepG2": "#c8553d", "mRNA": "#7f7f7f", "lncRNA": "#2a9d8f", "sno": "#e9a03b"}


def letters(axes):
    for a, l in zip(axes, "abcdefgh"):
        a.text(-0.16, 1.08, l, transform=a.transAxes, fontweight="bold", fontsize=11)


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(F / f"{name}.{ext}")
    plt.close(fig)


def place_labels(ax, xs, ys, names, fontsize=6, min_dy=0.045, min_dx=0.16, dx=0.012):
    """Etykiety punktow bez nakladania: rozsuwanie w pionie (uklad osi 0-1), cienka linia do punktu."""
    pts = ax.transAxes.inverted().transform(ax.transData.transform(np.c_[xs, ys]))
    lab = pts + np.array([dx, 0.0])
    for _ in range(200):
        moved = False
        for i in range(len(lab)):
            for j in range(i + 1, len(lab)):
                if abs(lab[i, 0] - lab[j, 0]) < min_dx and abs(lab[i, 1] - lab[j, 1]) < min_dy:
                    d = (min_dy - abs(lab[i, 1] - lab[j, 1])) / 2 + 1e-3
                    sgn = 1 if (lab[i, 1], i) >= (lab[j, 1], j) else -1
                    lab[i, 1] += sgn * d
                    lab[j, 1] -= sgn * d
                    moved = True
        if not moved:
            break
    for (x, y), (lx, ly), n in zip(pts, lab, names):
        far = np.hypot(lx - x, ly - y) > 0.03
        ax.annotate(n, xy=(x, y), xycoords="axes fraction", xytext=(lx, ly), textcoords="axes fraction",
                    fontsize=fontsize, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.3, color="0.5", shrinkA=0, shrinkB=1) if far else None)


def stats(df, v):
    s = df[df["wariant"] == v]
    w = s.groupby(["target", "linia"])["roznica"].mean().unstack("linia").dropna()
    return spearmanr(w["K562"], w["HepG2"])[0], s["roznica"].abs().median()


# ---------------------------------------------------------------- Rycina 3
rv = pd.read_csv("results/phase2/classes/robustness_variants.tsv.gz", sep="\t")
loc = pd.read_csv("results/phase3/localization/class_effect_B_vs_C.tsv", sep="\t")
pla = pd.read_csv("results/phase3/plasticity/class_effect_C_vs_D.tsv", sep="\t")
ecl = pd.read_csv("results/phase3/eclip/eclip_directness.tsv", sep="\t")
e2 = pd.concat([ecl[["linia", "RBP", "klasa_wszystkie"]].rename(columns={"RBP": "target", "klasa_wszystkie": "roznica"}).assign(wariant="przed"),
                ecl[["linia", "RBP", "klasa_niezwiazane"]].rename(columns={"RBP": "target", "klasa_niezwiazane": "roznica"}).assign(wariant="po")])
tests = [("metabolizm\n(A → B)", rv, "A_na_zbiorze_B", "B"), ("lokalizacja\n(B → C)", loc, "B", "C"),
         ("plastyczność\n(C → D)", pla, "C", "D"), ("wiązanie\n(geny niezwiązane)", e2, "przed", "po")]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
for k, (ax, what, lab) in enumerate(zip(axes[:2], [0, 1], ["replikacja K562–HepG2 (rho)", "mediana |lncRNA − mRNA| (log2)"])):
    for i, (name, df, a, b) in enumerate(tests):
        va, vb = stats(df, a)[what], stats(df, b)[what]
        ax.plot([i - 0.15, i + 0.15], [va, vb], color="0.6", lw=1)
        ax.scatter([i - 0.15], [va], facecolor="white", edgecolor="k", s=28, zorder=3)
        ax.scatter([i + 0.15], [vb], color="k", s=28, zorder=3)
    ax.set_xticks(range(len(tests)), [t[0] for t in tests], fontsize=7)
    ax.set_ylabel(lab)
    lo_ = min(0, ax.get_ylim()[0])
    ax.set_ylim(lo_, ax.get_ylim()[1] * 1.15)
    ax.set_title("Replikacja między liniami" if k == 0 else "Wielkość efektu klasy")
    ax.text(0.02, 0.97, "○ bez cechy   ● z cechą", transform=ax.transAxes, fontsize=7, va="top")
ax = axes[2]
data, labs = [], []
for cell in ["K562", "HepG2"]:
    for g in ["mRNA", "lncRNA"]:
        x = ecl.loc[ecl["linia"] == cell, f"abs_{g}"].dropna()
        data.append(x.values); labs.append(f"{g}\n{cell}")
ax.boxplot(data, widths=0.5, showfliers=False)
for i, x in enumerate(data, 1):
    ax.scatter(np.random.default_rng(i).normal(i, 0.06, len(x)), x, s=4, color="0.5", alpha=0.6)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(range(1, len(labs) + 1), labs, fontsize=7)
ax.set_ylabel("|log2FC| związane − niezwiązane")
ax.set_title("Czy związane geny reagują silniej?")
letters(axes)
save(fig, "fig3")

# ---------------------------------------------------------------- Rycina 4
cat = pd.read_csv("results/phase2/classes/functional_categories.tsv", sep="\t")
cat = cat[cat["typ"] == "funkcja"].sort_values("mediana")
smp = pd.read_csv("data/meta/phase3_samples.tsv", sep="\t", dtype=str).drop_duplicates("kd_experiment").set_index("kd_experiment")
rows = []
for kd, info in smp.iterrows():
    f = Path(f"results/phase3/eisa/{kd}.tsv.gz")
    if not f.exists():
        continue
    grupa = "EJC" if info["target"] in EJC else ("NMD" if info["target"] in NMD else ("spliceosom" if info["grupa"] == "spliceosom" else None))
    if grupa is None:
        continue
    d = pd.read_csv(f, sep="\t", index_col=0)
    d["k"] = np.where(d["klasa"] == "mRNA", "mRNA", np.where(d["klasa"].isin(LNC), "lncRNA", "inne"))
    x = np.log10(d["ctrl_ekson"] + 1)
    edges = np.quantile(x[d["k"] == "mRNA"], [0, .2, .4, .6, .8, 1])
    edges[0], edges[-1] = -np.inf, np.inf
    d["bin"] = pd.cut(x, edges, labels=[1, 2, 3, 4, 5])
    t = d[d["k"] != "inne"].groupby(["bin", "k"], observed=True)[["d_ekson", "d_intron"]].median().reset_index()
    rows.append(t.assign(kd=kd, grupa=grupa, linia=info["cell"]))
eb = pd.concat(rows).groupby(["grupa", "linia", "bin", "k"], observed=True)[["d_ekson", "d_intron"]].median().reset_index()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.3, 1, 1]})
ax = axes[0]
y = np.arange(len(cat))
ax.hlines(y, cat["mediana_pozostalych"], cat["mediana"], color="0.75", lw=1)
ax.scatter(cat["mediana_pozostalych"], y, s=10, color="0.6")
ax.scatter(cat["mediana"], y, s=np.clip(cat["n"], 5, 80), color=np.where(cat["q"] < 0.05, "#c8553d", "#3b6ea8"), zorder=3)
from matplotlib.lines import Line2D
leg_h = [Line2D([], [], marker="o", ls="", color="0.6", ms=3, label="pozostałe RBP"),
         Line2D([], [], marker="o", ls="", color="#3b6ea8", ms=5, label="RBP z kategorii"),
         Line2D([], [], marker="o", ls="", color="#c8553d", ms=5, label="RBP z kategorii, q < 0,05")]
ax.set_yticks(y, [f"{c} ({n})" for c, n in zip(cat["kategoria"], cat["n"])], fontsize=6)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("mediana lncRNA − mRNA (log2)")
ax.set_title("Kategorie funkcjonalne RBP")
ax.legend(handles=leg_h, frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
for ax, (grupa, comp, title) in zip(axes[1:], [("spliceosom", "d_intron", "Spliceosom: zmiana intronów"),
                                                ("NMD", "d_ekson", "NMD (UPF, SMG): zmiana eksonów")]):
    for cell, ls in [("K562", "-"), ("HepG2", "--")]:
        for k in ["mRNA", "lncRNA"]:
            s = eb[(eb["grupa"] == grupa) & (eb["linia"] == cell) & (eb["k"] == k)]
            if len(s):
                ax.plot(s["bin"].astype(int), s[comp], ls=ls, marker="o", ms=3, color=COL[k], label=f"{k}, {cell}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks([1, 2, 3, 4, 5], ["niska", "2", "3", "4", "wysoka"])
    ax.set_xlabel("przedział ekspresji (wg mRNA)")
    ax.set_ylabel("mediana zmiany (log2)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=6)
letters(axes)
save(fig, "fig4")

# ---------------------------------------------------------------- Rycina 5
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
cls["gene"] = cls["gene_id"].str.split(".").str[0]
order = ["mRNA", "lincRNA", "antysens", "inne lncRNA", "gospodarz miRNA", "gospodarz snoRNA"]
short = ["mRNA", "lincRNA", "antysens", "inne", "gosp.\nmiRNA", "gosp.\nsnoRNA"]
colors = [COL["mRNA"]] + [COL["lncRNA"]] * 4 + [COL["sno"]]
nuc = pd.read_csv("results/phase3/localization/nuclear_index.tsv", sep="\t", index_col=0)
pf = pd.read_csv("results/phase3/plasticity/plasticity_features.tsv", sep="\t", index_col=0)
k_of = cls.drop_duplicates("gene").set_index("gene")["klasa"]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, (vals, lab, title) in zip(axes[:2], [(nuc["HepG2"], "log2 jądro/cytoplazma (HepG2)", "Lokalizacja"),
                                             (pf["tau"], "tau (GTEx)", "Swoistość tkankowa")]):
    kk = k_of.reindex(vals.index)
    data = [vals[kk == c].dropna().values for c in order]
    b = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True)
    for patch, c in zip(b["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_xticks(range(1, len(order) + 1), short, fontsize=7)
    ax.set_ylabel(lab); ax.set_title(title)
ax = axes[2]
cg = pf.groupby("klasa")["CGI"].mean().reindex(order) * 100
ax.bar(range(len(order)), cg.values, color=colors, alpha=0.8)
ax.set_xticks(range(len(order)), short, fontsize=7)
ax.set_ylabel("% promotorów z wyspą CpG"); ax.set_title("Promotory")
ax = axes[3]
known = ["ZFAS1", "SNHG5", "SNHG6", "SNHG8", "SNHG11", "GAS5", "SNHG1", "SNHG7"]
lab5 = []
for i, cell in enumerate(["K562", "HepG2"]):
    c = pd.read_csv(f"results/phase3/splice_nmd/consensus_{cell}.tsv.gz", sep="\t", index_col=0)
    c = c[c["klasa"].notna()]
    lnc = c.loc[c["klasa"].isin(LNC), "nmd"].dropna()
    sno = c[c["klasa"] == "gospodarz snoRNA"].dropna(subset=["nmd"])
    ax.boxplot([lnc.values], positions=[i * 2], widths=0.6, showfliers=False)
    ax.scatter(np.random.default_rng(i).normal(i * 2 + 0.8, 0.07, len(sno)), sno["nmd"], s=10, color=COL["sno"])
    kn = sno[sno["gen"].isin(known)]
    lab5.append(pd.DataFrame({"x": i * 2 + 0.85, "y": kn["nmd"].values, "n": kn["gen"].values}))
ax.set_xticks([0, 0.8, 2, 2.8], ["lncRNA\nK562", "gosp. snoRNA\nK562", "lncRNA\nHepG2", "gosp. snoRNA\nHepG2"], fontsize=6)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("odpowiedź na NMD/EJC (mediana log2FC)"); ax.set_title("Wrażliwość na NMD")
ax.set_xlim(-0.6, 3.6)
if lab5:
    l5 = pd.concat(lab5)
    place_labels(ax, l5["x"].values, l5["y"].values, l5["n"].tolist(), fontsize=5, min_dy=0.04, min_dx=0.05, dx=0.03)
letters(axes)
save(fig, "fig5")

# ---------------------------------------------------------------- Rycina S1
cis = pd.read_csv("results/phase3/cis_nat/cis_effects.tsv", sep="\t")
cats = ["NAT", "naklad. ta sama nic", "sasiad <=10 kb, ta sama nic", "sasiad <=10 kb, przeciwna nic", "sasiad 10-50 kb"]
labels = ["NAT", "nakładający się,\nta sama nić", "≤10 kb,\nta sama nić", "≤10 kb,\nprzeciwna nić", "10–50 kb"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, grp in zip(axes, ["mRNA", "lncRNA"]):
    for j, (met, col) in enumerate([("CRISPR", "#6a4c93"), ("shRNA", "#8ac926")]):
        s = cis[(cis["grupa"] == grp) & (cis["metoda"] == met)].groupby("kategoria")["delta"]
        med, n = s.median().reindex(cats), s.size().reindex(cats).fillna(0)
        x = np.arange(len(cats)) + (j - 0.5) * 0.38
        ax.bar(x, med.where(n >= 10).values, 0.36, color=col, label=met)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(cats)), labels, fontsize=7)
    ax.set_ylabel("mediana zmiany względem tła (log2)")
    ax.set_title(f"Geny cis: {grp}")
    ax.legend(frameon=False, fontsize=7)
letters(axes)
save(fig, "figS1")
print(f"Zapisano: {F}/fig3, fig4, fig5, figS1 (.png i .pdf)")
