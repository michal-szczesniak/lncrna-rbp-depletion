"""
Faza 4, krok 4.2A: ryciny 1 i 2.

Rycina 1. Dane i korekta na wspolna kontrole.
  a) Zwiazek glownych skladowych (PCA na statystykach Walda) z partia kontroli (eta2), przed i po korekcie, obie linie.
  b) Kontrole pozytywne: roznica lncRNA - dopasowane mRNA przed i po korekcie (dopasowanie A).
Rycina 2. Glowny wynik.
  a) Rozklad roznicy lncRNA - dopasowane mRNA w knockdownach (macierz skorygowana, dopasowanie B).
  b) Ten sam RBP w K562 i HepG2; zaznaczone RBP z lista (krok 2.5).
  c) Replikacja miedzy liniami w kolejnych wariantach analizy i testach odpornosci.
Wyjscie: results/phase4/figures/fig1.(png|pdf), fig2.(png|pdf)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

M, C, F = Path("results/phase2/matrix"), Path("results/phase2/classes"), Path("results/phase4/figures")
F.mkdir(parents=True, exist_ok=True)
LNC = "lncRNA (razem)"
CTRL_RBP = ["MTREX", "EXOSC10", "EXOSC5", "EXOSC9", "UPF1", "AQR", "DDX3X", "XRN1"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
COL = {"K562": "#3b6ea8", "HepG2": "#c8553d", "przed": "#9e9e9e", "po": "#3b6ea8", "up": "#c8553d", "down": "#3b6ea8"}

summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
ctrl_of = pd.read_csv("data/meta/phase2_kds.tsv", sep="\t", dtype=str).set_index("experiment")["possible_controls"]


def eta2_pcs(L, S, expr, n_pc=6):
    W = (L / S).clip(-10, 10)
    keep = (W.notna().mean(axis=1) >= 0.9) & (expr.reindex(W.index) >= 20)
    X = W[keep].T.fillna(0.0)
    U, sv, _ = np.linalg.svd((X - X.mean()).values, full_matrices=False)
    grp = ctrl_of.reindex(X.index).values
    out = []
    for i in range(n_pc):
        s = pd.Series(U[:, i] * sv[i])
        tot = ((s - s.mean()) ** 2).sum()
        betw = s.groupby(grp).apply(lambda x: len(x) * (x.mean() - s.mean()) ** 2).sum()
        out.append(betw / tot)
    null = (pd.Series(grp).nunique() - 1) / (len(grp) - 1)
    return np.array(out), null


def place_labels(ax, xs, ys, names, fontsize=6, min_dy=0.045, min_dx=0.16):
    """Etykiety punktow bez nakladania: rozsuwanie w pionie (uklad osi 0-1), cienka linia do punktu."""
    pts = ax.transAxes.inverted().transform(ax.transData.transform(np.c_[xs, ys]))
    lab = pts + np.array([0.012, 0.012])
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


# ---------------------------------------------------------------- Rycina 1
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.3]})
for ax, cell in zip(axes[:2], ["K562", "HepG2"]):
    Lb = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    L = pd.read_csv(M / f"lfc_{cell}.tsv.gz", sep="\t", index_col=0)[Lb.columns]
    S = pd.read_csv(M / f"se_{cell}.tsv.gz", sep="\t", index_col=0)[Lb.columns]
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    e0, null = eta2_pcs(L, S, expr)
    e1, _ = eta2_pcs(Lb, S, expr)
    x = np.arange(1, len(e0) + 1)
    ax.bar(x - 0.2, e0, 0.4, color=COL["przed"], label="przed korektą")
    ax.bar(x + 0.2, e1, 0.4, color=COL["po"], label="po korekcie")
    ax.axhline(null, ls="--", lw=0.8, color="k", label="oczekiwane przy braku związku")
    ax.set_xticks(x, [f"PC{i}" for i in x])
    ax.set_ylim(0, 1)
    ax.set_title(f"{cell}: składowe a partia kontroli")
    ax.set_ylabel("eta² (partia kontroli)")
axes[0].legend(frameon=False, fontsize=7, loc="upper right")

res = pd.read_csv(C / "class_effects_all.tsv.gz", sep="\t")
pc = res[(res["klasa"] == LNC) & (res["dopasowanie"] == "A") & res["target"].isin(CTRL_RBP)]
pc = pc.groupby(["target", "linia", "macierz"])["roznica"].mean().unstack("macierz").reset_index()
ax = axes[2]
order = [t for t in CTRL_RBP if t in set(pc["target"])]
for j, cell in enumerate(["K562", "HepG2"]):
    s = pc[pc["linia"] == cell].set_index("target").reindex(order)
    y = np.arange(len(order)) + (j - 0.5) * 0.3
    ax.hlines(y, s["surowa"], s["skorygowana"], color=COL[cell], lw=1)
    ax.scatter(s["surowa"], y, facecolor="white", edgecolor=COL[cell], s=18, zorder=3)
    ax.scatter(s["skorygowana"], y, color=COL[cell], s=18, zorder=3, label=cell)
ax.axvline(0, color="k", lw=0.6)
ax.set_yticks(np.arange(len(order)), order)
ax.invert_yaxis()
ax.set_xlabel("lncRNA − dopasowane mRNA (log2)\n○ przed korektą   ● po korekcie")
ax.set_title("Wybrane RBP przed i po korekcie")
ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
for a, l in zip(axes, "abc"):
    a.text(-0.16, 1.08, l, transform=a.transAxes, fontweight="bold", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(F / f"fig1.{ext}")
plt.close(fig)

# ---------------------------------------------------------------- Rycina 2
main = res[(res["klasa"] == LNC) & (res["macierz"] == "skorygowana") & (res["dopasowanie"] == "B")]
w = main.groupby(["target", "linia"])["roznica"].mean().unstack("linia").dropna()
hits = pd.read_csv(C / "rbp_hits.tsv", sep="\t", index_col=0)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1.1, 1.2]})
ax = axes[0]
bins = np.linspace(-0.6, 0.6, 49)
for cell in ["K562", "HepG2"]:
    ax.hist(main.loc[main["linia"] == cell, "roznica"].clip(-0.6, 0.6), bins=bins, histtype="step", lw=1.2, color=COL[cell], label=cell)
for v in (-0.1, 0.1):
    ax.axvline(v, ls=":", color="k", lw=0.7)
ax.set_xlabel("lncRNA − dopasowane mRNA (log2)")
ax.set_ylabel("liczba knockdownów")
ax.set_title("Efekt klasy w knockdownach")
ax.legend(frameon=False, fontsize=7)

ax = axes[1]
is_hit = w.index.isin(hits.index)
ax.scatter(w.loc[~is_hit, "K562"], w.loc[~is_hit, "HepG2"], s=7, color="#bdbdbd", lw=0)
hh = w[is_hit]
ax.scatter(hh["K562"], hh["HepG2"], s=14, color=np.where(hh.mean(axis=1) > 0, COL["up"], COL["down"]), lw=0)
lim = max(w.abs().max().max(), 0.3) * 1.05
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
lab = hh.assign(m=hh.mean(axis=1).abs()).sort_values("m", ascending=False).head(14)
place_labels(ax, lab["K562"].values, lab["HepG2"].values, lab.index.tolist())
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COL["up"], ms=4, label="lista: lncRNA rosną"),
                   Line2D([], [], marker="o", ls="", color=COL["down"], ms=4, label="lista: lncRNA spadają"),
                   Line2D([], [], marker="o", ls="", color="#bdbdbd", ms=3, label="pozostałe RBP")],
          frameon=False, fontsize=6, loc="upper left")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
r = spearmanr(w["K562"], w["HepG2"])[0]
ax.set_xlabel("K562"); ax.set_ylabel("HepG2")
ax.set_title(f"Ten sam RBP w dwóch liniach\nrho = {r:.2f}, {len(w)} RBP")

ax = axes[2]
rep = pd.read_csv(C / "replication.tsv", sep="\t")
rep = rep[rep["klasa"] == LNC].set_index(["macierz", "dopasowanie"])["rho"]
rob = pd.read_csv(C / "robustness_variants.tsv.gz", sep="\t")
rob_rho = {v: spearmanr(*rob[rob["wariant"] == v].groupby(["target", "linia"])["roznica"].mean().unstack("linia").dropna()[["K562", "HepG2"]].T.values)[0]
           for v in ["A_na_zbiorze_B", "B_mRNA_bez_RI"]}
r41 = pd.read_csv("results/phase4/robustness_summary.tsv", sep="\t").set_index("wariant")["replikacja_rho"]
bars = [("surowa, A", rep.get(("surowa", "A"))), ("surowa, B", rep.get(("surowa", "B"))),
        ("skorygowana, A", rep.get(("skorygowana", "A"))), ("skorygowana, A na genach z B", rob_rho["A_na_zbiorze_B"]),
        ("skorygowana, B (główny)", rep.get(("skorygowana", "B"))), ("B, mRNA bez izoform RI", rob_rho["B_mRNA_bez_RI"]),
        ("B, 5 dopasowanych mRNA", r41.get("K = 5")), ("B, 20 dopasowanych mRNA", r41.get("K = 20")),
        ("B, geny z baseMean ≥ 50", r41.get("ekspresja >= 50")), ("B, tylko shRNA", r41.get("tylko shRNA")),
        ("B, tylko CRISPR", r41.get("tylko CRISPR"))]
names, vals = zip(*bars)
y = np.arange(len(names))
ax.barh(y, vals, color=["#9e9e9e"] * 2 + ["#7aa6d6"] * 2 + [COL["po"]] + ["#7aa6d6"] * 6)
ax.set_yticks(y, names, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("replikacja K562–HepG2 (rho Spearmana)")
ax.set_title("Replikacja w wariantach analizy")
for yy, v in zip(y, vals):
    ax.text(v + 0.005, yy, f"{v:.2f}", va="center", fontsize=6)
for a, l in zip(axes, "abc"):
    a.text(-0.16, 1.08, l, transform=a.transAxes, fontweight="bold", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(F / f"fig2.{ext}")
plt.close(fig)
print(f"Zapisano: {F}/fig1.png, fig1.pdf, fig2.png, fig2.pdf")
