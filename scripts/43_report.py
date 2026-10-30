"""
Phase 4, step 4.3: plain-language HTML report with English figures and tables.

Run from the project directory ($PROJ). Reads the result files produced by steps 2.x-4.1, redraws all figures with
English labels (results/phase4/figures_en/), and writes one self-contained HTML file with the figures embedded:
    results/phase4/report/rbp_lncRNA_report.html
The file needs no internet connection except for web fonts (it falls back to system fonts).
"""
import base64
import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

M, C = Path("results/phase2/matrix"), Path("results/phase2/classes")
FIG, REP = Path("results/phase4/figures_en"), Path("results/phase4/report")
FIG.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)
LNC_ALL = "lncRNA (razem)"
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz miRNA"}
NMD = {"UPF1", "UPF2", "UPF3B", "SMG1", "SMG5", "SMG6", "SMG7", "SMG8", "SMG9"}
EJC = {"RBM8A", "MAGOH", "EIF4A3", "CASC3"}
EN = {"mRNA": "mRNA", "lincRNA": "lincRNA", "antysens": "antisense", "dywergentny": "divergent", "inne lncRNA": "other lncRNA",
      "gospodarz snoRNA": "snoRNA host", "gospodarz miRNA": "miRNA host", "sense-intronic": "sense-intronic",
      "pseudogen transkr.": "transcribed pseudogene", LNC_ALL: "lncRNA (all)"}
COL = {"K562": "#3b6ea8", "HepG2": "#c8553d", "before": "#9e9e9e", "after": "#3b6ea8", "up": "#c8553d", "down": "#3b6ea8",
       "mRNA": "#7f7f7f", "lncRNA": "#2a9d8f", "sno": "#e9a03b"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})


def letters(axes):
    for a, l in zip(axes, "abcdefgh"):
        a.text(-0.16, 1.08, l, transform=a.transAxes, fontweight="bold", fontsize=11)


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def place_labels(ax, xs, ys, names, fontsize=6, min_dy=0.045, min_dx=0.16, dx=0.012, dy=0.012):
    """Non-overlapping point labels: vertical repulsion in axes coordinates, thin leader line to the point."""
    pts = ax.transAxes.inverted().transform(ax.transData.transform(np.c_[xs, ys]))
    lab = pts + np.array([dx, dy])
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
        ax.annotate(n, xy=(x, y), xycoords="axes fraction", xytext=(lx, ly), textcoords="axes fraction", fontsize=fontsize,
                    va="center", arrowprops=dict(arrowstyle="-", lw=0.3, color="0.5", shrinkA=0, shrinkB=1) if far else None)


def rep_stats(df, v):
    s = df[df["wariant"] == v]
    w = s.groupby(["target", "linia"])["roznica"].mean().unstack("linia").dropna()
    return spearmanr(w["K562"], w["HepG2"])[0], s["roznica"].abs().median(), len(w)


# ====================================================================== inputs
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
ctrl_of = pd.read_csv("data/meta/phase2_kds.tsv", sep="\t", dtype=str).set_index("experiment")["possible_controls"]
res = pd.read_csv(C / "class_effects_all.tsv.gz", sep="\t")
main = res[(res["klasa"] == LNC_ALL) & (res["macierz"] == "skorygowana") & (res["dopasowanie"] == "B")]
w_main = main.groupby(["target", "linia"])["roznica"].mean().unstack("linia").dropna()
rho_main = spearmanr(w_main["K562"], w_main["HepG2"])[0]
hits = pd.read_csv(C / "rbp_hits.tsv", sep="\t", index_col=0)
robust = pd.read_csv(C / "rbp_hits_robustness.tsv", sep="\t", index_col=0)
r41 = pd.read_csv("results/phase4/robustness_summary.tsv", sep="\t")
rep = pd.read_csv(C / "replication.tsv", sep="\t")
rep_l = rep[rep["klasa"] == LNC_ALL].set_index(["macierz", "dopasowanie"])["rho"]
rv = pd.read_csv(C / "robustness_variants.tsv.gz", sep="\t")
loc = pd.read_csv("results/phase3/localization/class_effect_B_vs_C.tsv", sep="\t")
pla = pd.read_csv("results/phase3/plasticity/class_effect_C_vs_D.tsv", sep="\t")
ecl = pd.read_csv("results/phase3/eclip/eclip_directness.tsv", sep="\t")
cat = pd.read_csv(C / "functional_categories.tsv", sep="\t")
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
cls["gene"] = cls["gene_id"].str.split(".").str[0]
nuc = pd.read_csv("results/phase3/localization/nuclear_index.tsv", sep="\t", index_col=0)
pf = pd.read_csv("results/phase3/plasticity/plasticity_features.tsv", sep="\t", index_col=0)
cis = pd.read_csv("results/phase3/cis_nat/cis_effects.tsv", sep="\t")
smp3 = pd.read_csv("data/meta/phase3_samples.tsv", sep="\t", dtype=str).drop_duplicates("kd_experiment").set_index("kd_experiment")


# ====================================================================== Figure 1
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
        out.append(s.groupby(grp).apply(lambda x: len(x) * (x.mean() - s.mean()) ** 2).sum() / tot)
    return np.array(out), (pd.Series(grp).nunique() - 1) / (len(grp) - 1)


print("Figure 1 (recomputes PCA, a few minutes)...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.3]})
eta = {}
for ax, cell in zip(axes[:2], ["K562", "HepG2"]):
    Lb = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    L = pd.read_csv(M / f"lfc_{cell}.tsv.gz", sep="\t", index_col=0)[Lb.columns]
    S = pd.read_csv(M / f"se_{cell}.tsv.gz", sep="\t", index_col=0)[Lb.columns]
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    e0, null = eta2_pcs(L, S, expr)
    e1, _ = eta2_pcs(Lb, S, expr)
    eta[cell] = (e0, e1, null)
    x = np.arange(1, len(e0) + 1)
    ax.bar(x - 0.2, e0, 0.4, color=COL["before"], label="before correction")
    ax.bar(x + 0.2, e1, 0.4, color=COL["after"], label="after correction")
    ax.axhline(null, ls="--", lw=0.8, color="k", label="expected by chance alone")
    ax.set_xticks(x, [f"{i}" for i in x])
    ax.set_xlabel("shared pattern (principal component)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{cell}: shared patterns vs control sample")
    ax.set_ylabel("share of pattern explained\nby the control sample used")
axes[0].legend(frameon=False, fontsize=7, loc="upper right")
CTRL_RBP = ["MTREX", "EXOSC10", "EXOSC5", "EXOSC9", "UPF1", "AQR", "DDX3X", "XRN1"]
pc = res[(res["klasa"] == LNC_ALL) & (res["dopasowanie"] == "A") & res["target"].isin(CTRL_RBP)]
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
ax.set_xlabel("class effect (log2)\n○ before   ● after control correction")
ax.set_title("Well-studied RBPs before and after correction")
ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
letters(axes)
save(fig, "fig1")

# ====================================================================== Figure 2
print("Figure 2...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.1, 1.3]})
ax = axes[0]
bins = np.linspace(-0.6, 0.6, 49)
for cell in ["K562", "HepG2"]:
    ax.hist(main.loc[main["linia"] == cell, "roznica"].clip(-0.6, 0.6), bins=bins, histtype="step", lw=1.2, color=COL[cell], label=cell)
for v in (-0.1, 0.1):
    ax.axvline(v, ls=":", color="k", lw=0.7)
ax.set_xlabel("class effect (log2)")
ax.set_ylabel("number of removal experiments")
ax.set_title("Class effects of all experiments")
ax.legend(frameon=False, fontsize=7)
ax = axes[1]
is_hit = w_main.index.isin(hits.index)
ax.scatter(w_main.loc[~is_hit, "K562"], w_main.loc[~is_hit, "HepG2"], s=7, color="#bdbdbd", lw=0)
hh = w_main[is_hit]
ax.scatter(hh["K562"], hh["HepG2"], s=14, color=np.where(hh.mean(axis=1) > 0, COL["up"], COL["down"]), lw=0)
lim = max(w_main.abs().max().max(), 0.3) * 1.05
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
lab = hh.assign(m=hh.mean(axis=1).abs()).sort_values("m", ascending=False).head(14)
place_labels(ax, lab["K562"].values, lab["HepG2"].values, lab.index.tolist())
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COL["up"], ms=4, label="lncRNAs clearly higher after removal"),
                   Line2D([], [], marker="o", ls="", color=COL["down"], ms=4, label="lncRNAs clearly lower after removal"),
                   Line2D([], [], marker="o", ls="", color="#bdbdbd", ms=3, label="other RBPs")],
          frameon=False, fontsize=6, loc="upper left")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("class effect in K562 (log2)"); ax.set_ylabel("class effect in HepG2 (log2)")
ax.set_title(f"Each dot: one RBP removed in both lines\nagreement rho = {rho_main:.2f} ({len(w_main)} RBPs)")
ax = axes[2]
rob_rho = {v: rep_stats(rv, v)[0] for v in ["A_na_zbiorze_B", "B_mRNA_bez_RI"]}
rr = r41.set_index("wariant")["replikacja_rho"]
bars = [("raw data, basic matching", rep_l.get(("surowa", "A"))), ("raw data, full matching", rep_l.get(("surowa", "B"))),
        ("control-corrected, basic matching", rep_l.get(("skorygowana", "A"))),
        ("control-corrected, basic matching,\nsame genes as full matching", rob_rho["A_na_zbiorze_B"]),
        ("MAIN: control-corrected, full matching", rep_l.get(("skorygowana", "B"))),
        ("main, reference mRNAs without\nretained-intron forms", rob_rho["B_mRNA_bez_RI"]),
        ("main, 5 reference mRNAs", rr.get("K = 5")), ("main, 20 reference mRNAs", rr.get("K = 20")),
        ("main, well-expressed genes only", rr.get("ekspresja >= 50")),
        ("main, shRNA experiments only", rr.get("tylko shRNA")), ("main, CRISPR experiments only", rr.get("tylko CRISPR"))]
names, vals = zip(*bars)
yy = np.arange(len(names))
ax.barh(yy, vals, color=["#9e9e9e"] * 2 + ["#7aa6d6"] * 2 + [COL["after"]] + ["#7aa6d6"] * 6)
ax.set_yticks(yy, names, fontsize=6.5)
ax.invert_yaxis()
ax.set_xlabel("agreement between the two cell lines (rho)")
ax.set_title("Agreement in different versions of the analysis")
for y_, v in zip(yy, vals):
    ax.text(v + 0.005, y_, f"{v:.2f}", va="center", fontsize=6)
letters(axes)
save(fig, "fig2")

# ====================================================================== Figure 3
print("Figure 3...", flush=True)
e2 = pd.concat([ecl[["linia", "RBP", "klasa_wszystkie"]].rename(columns={"RBP": "target", "klasa_wszystkie": "roznica"}).assign(wariant="before"),
                ecl[["linia", "RBP", "klasa_niezwiazane"]].rename(columns={"RBP": "target", "klasa_niezwiazane": "roznica"}).assign(wariant="after")])
tests = [("splicing &\nstability", rv, "A_na_zbiorze_B", "B"), ("nucleus vs\ncytoplasm", loc, "B", "C"),
         ("tissue specif.\n+ CpG promoter", pla, "C", "D"), ("direct binding\n(eCLIP)", e2, "before", "after")]
feat_tab = []
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
for k, (ax, what, ylab) in enumerate(zip(axes[:2], [0, 1], ["agreement between cell lines (rho)", "typical class effect, sign ignored (log2)"])):
    for i, (name, df, a, b) in enumerate(tests):
        sa, sb = rep_stats(df, a), rep_stats(df, b)
        if k == 0:
            feat_tab.append({"Property added to the comparison": {"splicing &\nstability": "Splicing efficiency and stability",
                     "nucleus vs\ncytoplasm": "Location in the cell (nucleus vs cytoplasm)",
                     "tissue specif.\n+ CpG promoter": "Tissue specificity and CpG-island promoter",
                     "direct binding\n(eCLIP)": "Direct binding (only genes not bound by the protein)"}[name], "Replication without": sa[0],
                             "Replication with": sb[0], "Effect size without": sa[1], "Effect size with": sb[1], "RBPs": sb[2]})
        va, vb = sa[what], sb[what]
        ax.plot([i - 0.15, i + 0.15], [va, vb], color="0.6", lw=1)
        ax.scatter([i - 0.15], [va], facecolor="white", edgecolor="k", s=28, zorder=3)
        ax.scatter([i + 0.15], [vb], color="k", s=28, zorder=3)
    ax.set_xticks(range(len(tests)), [t[0] for t in tests], fontsize=7)
    ax.set_ylabel(ylab)
    ax.set_ylim(min(0, ax.get_ylim()[0]), ax.get_ylim()[1] * 1.15)
    ax.set_title("Agreement between the two cell lines" if k == 0 else "Typical size of the class effect")
    ax.text(0.02, 0.97, "○ property not used   ● property added to matching", transform=ax.transAxes, fontsize=7, va="top")
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
ax.set_ylabel("size of change: bound minus\nunbound genes (log2)")
ax.set_title("Do bound genes respond more strongly?")
letters(axes)
save(fig, "fig3")

# ====================================================================== Figure 4
print("Figure 4...", flush=True)
catf = cat[cat["typ"] == "funkcja"].sort_values("mediana")
rows = []
for kd, info in smp3.iterrows():
    f = Path(f"results/phase3/eisa/{kd}.tsv.gz")
    if not f.exists():
        continue
    grupa = "EJC" if info["target"] in EJC else ("NMD" if info["target"] in NMD else ("spliceosome" if info["grupa"] == "spliceosom" else None))
    if grupa is None:
        continue
    d = pd.read_csv(f, sep="\t", index_col=0)
    d["k"] = np.where(d["klasa"] == "mRNA", "mRNA", np.where(d["klasa"].isin(LNC), "lncRNA", "other"))
    x = np.log10(d["ctrl_ekson"] + 1)
    edges = np.quantile(x[d["k"] == "mRNA"], [0, .2, .4, .6, .8, 1])
    edges[0], edges[-1] = -np.inf, np.inf
    d["bin"] = pd.cut(x, edges, labels=[1, 2, 3, 4, 5])
    t = d[d["k"] != "other"].groupby(["bin", "k"], observed=True)[["d_ekson", "d_intron"]].median().reset_index()
    rows.append(t.assign(kd=kd, grupa=grupa, linia=info["cell"]))
eb_all = pd.concat(rows)
n_kd_grp = eb_all.groupby(["grupa", "linia"])["kd"].nunique()
eb = eb_all.groupby(["grupa", "linia", "bin", "k"], observed=True)[["d_ekson", "d_intron"]].median().reset_index()
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.3, 1, 1]})
ax = axes[0]
y = np.arange(len(catf))
ax.hlines(y, catf["mediana_pozostalych"], catf["mediana"], color="0.75", lw=1)
ax.scatter(catf["mediana_pozostalych"], y, s=10, color="0.6")
ax.scatter(catf["mediana"], y, s=np.clip(catf["n"], 5, 80), color=np.where(catf["q"] < 0.05, "#c8553d", "#3b6ea8"), zorder=3)
ax.set_yticks(y, [f"{c} ({n})" for c, n in zip(catf["kategoria"], catf["n"])], fontsize=6)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("median class effect (log2)")
ax.set_title("RBPs grouped by known function")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color="0.6", ms=3, label="all other RBPs"),
                   Line2D([], [], marker="o", ls="", color="#3b6ea8", ms=5, label="RBPs in category"),
                   Line2D([], [], marker="o", ls="", color="#c8553d", ms=5, label="RBPs in category, q < 0.05")],
          frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
for ax, (grupa, comp, title) in zip(axes[1:], [("spliceosome", "d_intron", "Spliceosome removed: unspliced RNA"),
                                                ("NMD", "d_ekson", "NMD removed (UPF, SMG): mature RNA")]):
    for cell, ls in [("K562", "-"), ("HepG2", "--")]:
        for k in ["mRNA", "lncRNA"]:
            s = eb[(eb["grupa"] == grupa) & (eb["linia"] == cell) & (eb["k"] == k)]
            if len(s):
                ax.plot(s["bin"].astype(int), s[comp], ls=ls, marker="o", ms=3, color=COL[k],
                        label=f"{k}, {cell} ({n_kd_grp.get((grupa, cell), 0)} KD)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks([1, 2, 3, 4, 5], ["low", "2", "3", "4", "high"])
    ax.set_xlabel("expression bin (defined on mRNA)")
    ax.set_ylabel("median change after removal (log2)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=6)
letters(axes)
save(fig, "fig4")

# ====================================================================== Figure 5
print("Figure 5...", flush=True)
order = ["mRNA", "lincRNA", "antysens", "inne lncRNA", "gospodarz miRNA", "gospodarz snoRNA"]
short = ["mRNA", "lincRNA", "antisense", "other", "miRNA\nhost", "snoRNA\nhost"]
colors = [COL["mRNA"]] + [COL["lncRNA"]] * 4 + [COL["sno"]]
k_of = cls.drop_duplicates("gene").set_index("gene")["klasa"]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.7))
for ax, (vals, ylab, title) in zip(axes[:2], [(nuc["HepG2"], "log2 nucleus / cytoplasm (HepG2)", "Localization"),
                                              (pf["tau"], "tau (GTEx; 1 = one tissue only)", "Tissue specificity")]):
    kk = k_of.reindex(vals.index)
    b = ax.boxplot([vals[kk == c].dropna().values for c in order], widths=0.6, showfliers=False, patch_artist=True)
    for patch, c_ in zip(b["boxes"], colors):
        patch.set_facecolor(c_); patch.set_alpha(0.6)
    ax.set_xticks(range(1, len(order) + 1), short, fontsize=7)
    ax.set_ylabel(ylab); ax.set_title(title)
ax = axes[2]
cg = pf.groupby("klasa")["CGI"].mean().reindex(order) * 100
ax.bar(range(len(order)), cg.values, color=colors, alpha=0.8)
ax.set_xticks(range(len(order)), short, fontsize=7)
ax.set_ylabel("% promoters with a CpG island"); ax.set_title("Promoters")
ax = axes[3]
known = ["ZFAS1", "SNHG5", "SNHG6", "SNHG8", "SNHG11", "GAS5", "SNHG1", "SNHG7"]
lab5 = []
for i, cell in enumerate(["K562", "HepG2"]):
    c_ = pd.read_csv(f"results/phase3/splice_nmd/consensus_{cell}.tsv.gz", sep="\t", index_col=0)
    c_ = c_[c_["klasa"].notna()]
    ln = c_.loc[c_["klasa"].isin(LNC), "nmd"].dropna()
    sno = c_[c_["klasa"] == "gospodarz snoRNA"].dropna(subset=["nmd"])
    ax.boxplot([ln.values], positions=[i * 2], widths=0.6, showfliers=False)
    ax.scatter(np.random.default_rng(i).normal(i * 2 + 0.8, 0.07, len(sno)), sno["nmd"], s=10, color=COL["sno"])
    kn = sno[sno["gen"].isin(known)]
    lab5.append(pd.DataFrame({"x": i * 2 + 0.85, "y": kn["nmd"].values, "n": kn["gen"].values}))
ax.set_xticks([0, 0.8, 2, 2.8], ["lncRNA\nK562", "snoRNA host\nK562", "lncRNA\nHepG2", "snoRNA host\nHepG2"], fontsize=6)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("change after removing NMD/EJC\nproteins (median log2)", fontsize=8); ax.set_title("Sensitivity to NMD")
ax.set_xlim(-0.6, 3.6)
l5 = pd.concat(lab5)
if len(l5):
    place_labels(ax, l5["x"].values, l5["y"].values, l5["n"].tolist(), fontsize=5, min_dy=0.04, min_dx=0.05, dx=0.03, dy=0.0)
letters(axes)
save(fig, "fig5")

# ====================================================================== Figure S1
print("Figure S1...", flush=True)
cats = ["NAT", "naklad. ta sama nic", "sasiad <=10 kb, ta sama nic", "sasiad <=10 kb, przeciwna nic", "sasiad 10-50 kb"]
cats_en = ["antisense partner (NAT)", "overlapping, same strand", "≤10 kb, same strand", "≤10 kb, opposite strand", "10–50 kb"]
labels = ["NAT", "overlapping,\nsame strand", "≤10 kb,\nsame strand", "≤10 kb,\nopposite strand", "10–50 kb"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, grp in zip(axes, ["mRNA", "lncRNA"]):
    for j, (met, col) in enumerate([("CRISPR", "#6a4c93"), ("shRNA", "#8ac926")]):
        s = cis[(cis["grupa"] == grp) & (cis["metoda"] == met)].groupby("kategoria")["delta"]
        med, n = s.median().reindex(cats), s.size().reindex(cats).fillna(0)
        ax.bar(np.arange(len(cats)) + (j - 0.5) * 0.38, med.where(n >= 10).values, 0.36, color=col, label=met)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(cats)), labels, fontsize=7)
    ax.set_ylabel("median change compared with\nsimilar genes elsewhere (log2)")
    ax.set_title(f"Genes near the removed RBP gene: {grp}")
    ax.legend(frameon=False, fontsize=7)
letters(axes)
save(fig, "figS1")

# ====================================================================== tables
def table(df, num_fmt=None, cls_=""):
    num_fmt = num_fmt or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body_ = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (int, np.integer)):
                cells.append(f'<td class="n">{v:,}</td>')
            elif isinstance(v, (float, np.floating)):
                cells.append(f'<td class="n">{"" if pd.isna(v) else num_fmt.get(c, "{:.2f}").format(v)}</td>')
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        body_.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tbl {cls_}"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_)}</tbody></table></div>'


def pct(l2):
    """log2 change -> percent change, rounded."""
    return f"{100 * (2 ** l2 - 1):+.0f}%"


# T1 data
t1 = summ.groupby(["method", "cell", "biblioteka"]).size().unstack("cell").fillna(0).astype(int).reset_index()
t1["biblioteka"] = t1["biblioteka"].map({"poliA": "poly(A) RNA (main analysis)", "calkowite RNA": "total RNA (not used)"}).fillna(t1["biblioteka"])
t1["method"] = t1["method"].map({"CRISPR": "CRISPR: the RBP gene is destroyed", "shRNA": "shRNA: the RBP mRNA is degraded"}).fillna(t1["method"])
t1 = t1.rename(columns={"method": "How the RBP was removed", "biblioteka": "RNA sequenced",
                        "HepG2": "HepG2 (liver cancer cells)", "K562": "K562 (leukaemia cells)"})
n_kd, n_rbp_total = len(summ), summ["target"].nunique()
n_both_lines = int((summ.groupby("target")["cell"].nunique() == 2).sum())
# T2 robustness
vmap = {"glowny": "Main analysis",
        "maska 50 kb CRISPR": "Also ignore genes within 50 kb of the gene targeted by CRISPR",
        "bez slabych shRNA": "Leave out shRNA experiments in which the RBP mRNA dropped by less than 30%",
        "K = 5": "Compare each lncRNA with 5 instead of 10 reference mRNAs",
        "K = 20": "Compare each lncRNA with 20 instead of 10 reference mRNAs",
        "ekspresja >= 50": "Use only well-expressed genes",
        "tylko shRNA": "Use only RBPs removed by shRNA in both cell lines",
        "tylko CRISPR": "Use only RBPs removed by CRISPR in both cell lines"}
t2 = r41.assign(wariant=r41["wariant"].map(vmap).fillna(r41["wariant"]))[
    ["wariant", "RBP_w_obu", "replikacja_rho", "mediana_|roznicy|", "RBP_z_replikacja", "z_35_z_listy"]]
t2.columns = ["Version of the analysis", "RBPs removed in both lines", "Agreement between lines (rho)",
              "Typical class effect, sign ignored (log2)", "RBPs with a clear effect in both lines",
              f"of these, also among the main {len(hits)}"]
# T3 features
t3 = pd.DataFrame(feat_tab)
t3.columns = ["Property added to the matching", "Agreement between lines, before", "Agreement between lines, after",
              "Typical class effect before (log2)", "Typical class effect after (log2)", "RBPs compared"]
# T4 candidates
prof_cols = [c for c in ["lincRNA", "antysens", "gospodarz snoRNA", "gospodarz miRNA"] if c in hits.columns]
prof_en = {"lincRNA": "lincRNAs", "antysens": "antisense RNAs", "gospodarz snoRNA": "snoRNA hosts", "gospodarz miRNA": "miRNA hosts"}
t4 = hits[["K562", "HepG2", "metoda"] + prof_cols].copy()
t4["dir"] = np.where(t4[["K562", "HepG2"]].mean(axis=1) > 0, "higher", "lower")
t4["Robust"] = robust.reindex(t4.index)["odporny"].map({True: "yes", False: "no"}).fillna("–")
t4 = t4.assign(m=t4[["K562", "HepG2"]].mean(axis=1)).sort_values("m").drop(columns="m").reset_index()
t4 = t4.rename(columns={t4.columns[0]: "RBP"})
t4 = t4[["RBP", "dir", "K562", "HepG2", "metoda"] + prof_cols + ["Robust"]]
t4.columns = (["RBP", "lncRNAs after removal", "Class effect K562", "Class effect HepG2", "How removed"]
              + [f"{prof_en[c]} (both lines)" for c in prof_cols] + ["Passes extra check"])
t4["How removed"] = t4["How removed"].str.replace("/", " and ")
# T5 categories
t5 = cat[cat["typ"] == "funkcja"].sort_values("p")[["kategoria", "n", "mediana", "mediana_pozostalych", "q"]]
t5.columns = ["Function of the RBPs (ENCODE annotation)", "RBPs in group", "Median class effect in group",
              "Median class effect of all other RBPs", "Adjusted p-value"]
# T6 snoRNA hosts
k_n = k_of.reindex(nuc.index)
t6 = pd.DataFrame({"Gene class": [EN[c] for c in order],
                   "Nuclear enrichment, K562": [nuc.loc[k_n == c, "K562"].median() for c in order],
                   "Nuclear enrichment, HepG2": [nuc.loc[k_n == c, "HepG2"].median() for c in order],
                   "Tissue specificity (0–1)": [pf.loc[pf["klasa"] == c, "tau"].median() for c in order],
                   "% with CpG-island promoter": [100 * pf.loc[pf["klasa"] == c, "CGI"].mean() for c in order]})
# T7 cis
g7 = cis[cis["grupa"] == "mRNA"].groupby(["kategoria", "metoda"]).agg(
    med=("delta", "median"),
    down=("padj", lambda x: 100 * ((x < 0.05) & (cis.loc[x.index, "lfc"] < 0)).mean())).reset_index()
t7 = g7.pivot_table(index="kategoria", columns="metoda", values=["med", "down"]).reindex(cats)
t7.columns = [f"{'Median change' if a == 'med' else '% clearly lower'}, {b}" for a, b in t7.columns]
t7.index = cats_en
t7 = t7.reset_index().rename(columns={"index": "Protein-coding gene near the removed RBP gene"})

# key numbers for the text
spl = eb[(eb["grupa"] == "spliceosome") & (eb["k"].isin(["mRNA", "lncRNA"]))].groupby(["k"])["d_intron"].agg(["min", "max"])
rob_cnt = int((robust["odporny"] == True).sum()) if "odporny" in robust else len(hits)
r_sh, r_cr = rr.get("tylko shRNA", np.nan), rr.get("tylko CRISPR", np.nan)
n_sh = int(r41.set_index("wariant").get("RBP_w_obu", pd.Series()).get("tylko shRNA", 0))
n_cr = int(r41.set_index("wariant").get("RBP_w_obu", pd.Series()).get("tylko CRISPR", 0))
up_n, down_n = int((t4["lncRNAs after removal"] == "higher").sum()), int((t4["lncRNAs after removal"] == "lower").sum())
typ = main["roznica"].abs().median()
n_rest = len(w_main) - len(hits)
p_main = spearmanr(w_main["K562"], w_main["HepG2"])[1]


def p_words(p):
    if p < 1e-6:
        return "the probability of agreement this strong arising by chance is below one in a million"
    if p < 1e-3:
        return "the probability of agreement this strong arising by chance is below one in a thousand"
    return f"the probability of agreement this strong arising by chance is about {p:.3f}"
rho_raw = rep_l.get(("surowa", "A"), np.nan)


def img(name):
    return base64.b64encode((FIG / f"{name}.png").read_bytes()).decode()


def figure(name, number, title, read, mean):
    return f'''<figure id="{name}">
  <img src="data:image/png;base64,{img(name)}" alt="Figure {number}: {html.escape(title)}">
  <figcaption><p class="ft">Figure {number}. {title}</p>
  <p><span class="lead">How to read it.</span> {read}</p>
  <p><span class="lead">What it means.</span> {mean}</p></figcaption>
</figure>'''


SCHEME = '''<svg class="scheme" viewBox="0 0 900 190" role="img" aria-label="How the class effect is calculated">
<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#5b6675"/></marker></defs>
<g font-family="IBM Plex Sans, Segoe UI, Helvetica, Arial, sans-serif" font-size="14" fill="#1d2733">
<rect x="10" y="40" width="190" height="96" rx="6" fill="#eef2f6" stroke="#c9d3de"/>
<text x="105" y="70" text-anchor="middle" font-weight="600">1. Remove one RBP</text>
<text x="105" y="94" text-anchor="middle" font-size="12" fill="#5b6675">by shRNA or CRISPR,</text>
<text x="105" y="112" text-anchor="middle" font-size="12" fill="#5b6675">in K562 or HepG2 cells</text>
<rect x="240" y="40" width="190" height="96" rx="6" fill="#eef2f6" stroke="#c9d3de"/>
<text x="335" y="70" text-anchor="middle" font-weight="600">2. Measure all RNAs</text>
<text x="335" y="94" text-anchor="middle" font-size="12" fill="#5b6675">change of every gene</text>
<text x="335" y="112" text-anchor="middle" font-size="12" fill="#5b6675">vs untreated cells</text>
<rect x="462" y="22" width="246" height="132" rx="6" fill="#e7f4f2" stroke="#2a9d8f"/>
<text x="585" y="50" text-anchor="middle" font-weight="600">3. Fair comparison</text>
<circle cx="500" cy="88" r="9" fill="#2a9d8f"/><text x="516" y="93" font-size="12">change of one lncRNA</text>
<circle cx="496" cy="118" r="5" fill="#7f7f7f"/><circle cx="510" cy="118" r="5" fill="#7f7f7f"/><circle cx="524" cy="118" r="5" fill="#7f7f7f"/>
<text x="536" y="123" font-size="12">minus average change of</text>
<text x="585" y="140" text-anchor="middle" font-size="11.5">its 10 most similar mRNAs</text>
<rect x="740" y="40" width="150" height="96" rx="6" fill="#eef2f6" stroke="#c9d3de"/>
<text x="815" y="66" text-anchor="middle" font-weight="600">4. Class effect</text>
<text x="815" y="90" text-anchor="middle" font-size="12" fill="#5b6675">median of step 3</text>
<text x="815" y="108" text-anchor="middle" font-size="12" fill="#5b6675">over all lncRNAs:</text>
<text x="815" y="126" text-anchor="middle" font-size="12" fill="#5b6675">one number per RBP</text>
<line x1="202" y1="88" x2="236" y2="88" stroke="#5b6675" stroke-width="1.6" marker-end="url(#ar)"/>
<line x1="432" y1="88" x2="458" y2="88" stroke="#5b6675" stroke-width="1.6" marker-end="url(#ar)"/>
<line x1="710" y1="88" x2="736" y2="88" stroke="#5b6675" stroke-width="1.6" marker-end="url(#ar)"/>
</g></svg>'''

LOG2 = '''<div class="tbl narrow"><table><thead><tr><th>Change in log2 units</th><th>Change in percent</th></tr></thead><tbody>
<tr><td class="n">+0.1</td><td class="n">about +7%</td></tr><tr><td class="n">+0.2</td><td class="n">about +15%</td></tr>
<tr><td class="n">+0.4</td><td class="n">about +32%</td></tr><tr><td class="n">+1</td><td class="n">doubling (+100%)</td></tr>
<tr><td class="n">−0.1</td><td class="n">about −7%</td></tr><tr><td class="n">−0.4</td><td class="n">about −24%</td></tr>
<tr><td class="n">−1</td><td class="n">halving (−50%)</td></tr></tbody></table></div>'''

body = f'''
<div class="col">
<h1>How do long non-coding RNAs respond when RNA-binding proteins are removed?</h1>
<p class="sub">A re-analysis of {n_kd} experiments from the ENCODE project, written for readers outside the field.</p>
<p class="answer">Cells contain two kinds of long RNA: protein-coding messenger RNAs (<span class="mrna">mRNAs</span>) and long non-coding RNAs (<span class="lnc">lncRNAs</span>). Both are handled by RNA-binding proteins (RBPs). We removed RBPs one at a time and asked whether lncRNAs change in the same way as mRNAs that resemble them. For many RBPs they do not, and an RBP that pushes lncRNAs in one direction in one cell type tends to do the same in another. This difference is not explained by any of the RNA properties we could measure.</p>

<p>Main findings, in plain words:</p>
<ul class="find">
<li><strong>lncRNAs often change differently from comparable mRNAs.</strong> For each RBP we calculated one number, the <em>class effect</em> (explained below): whether, after removing that RBP, lncRNAs as a group rose or fell more than mRNAs that are similar to them. Most class effects are small, typically about {pct(typ).lstrip('+')} in either direction.</li>
<li><strong>The class effect is a property of the RBP, not noise.</strong> {n_both_lines} RBPs were removed in both cell lines we studied. RBPs with the most negative class effects in one cell line tended to have the most negative class effects in the other, and likewise for positive ones. On a scale from 0 (no agreement) to 1 (identical ranking), the agreement is {rho_main:.2f}. This is moderate, but {p_words(p_main)}.</li>
<li><strong>{len(hits)} RBPs have a clear class effect in both cell lines, pointing the same way.</strong> After removing {up_n} of them, lncRNAs end up higher than comparable mRNAs; after removing {down_n}, lower. For the other {n_rest} RBPs the class effect is small, seen in only one cell line, or points in opposite directions in the two lines. That does not mean those RBPs leave lncRNAs untouched; their effects are too small to call with confidence. By chance alone about 12 RBPs would pass these criteria, so roughly one in three of the {len(hits)} may be a false positive.</li>
<li><strong>The agreement does not depend on how the RBP was removed.</strong> Using only RBPs removed by shRNA in both cell lines ({n_sh} RBPs), the agreement is {r_sh:.2f}; using only RBPs removed by CRISPR ({n_cr} RBPs), it is {r_cr:.2f}. It also holds in six other versions of the analysis.</li>
<li><strong>Known differences between lncRNAs and mRNAs do not explain it.</strong> We compared lncRNAs only with mRNAs that resemble them in expression level, length, splicing efficiency, stability, location in the cell, tissue specificity and promoter type. We also checked whether the RBP binds the RNA directly. None of these makes the difference disappear.</li>
<li><strong>The splicing machinery is the one group of RBPs with a consistent effect.</strong> Removing it disturbs the processing of mRNAs much more than that of lncRNAs, which are spliced weakly anyway.</li>
<li><strong>Removing the RNA-decay pathway called NMD raises lncRNAs</strong> more than mRNAs of the same expression level, consistent with many lncRNAs being destroyed by this pathway.</li>
<li><strong>snoRNA host genes, a subgroup of lncRNAs, behave unlike the rest.</strong> They look like always-on housekeeping genes and are destroyed by NMD in the cytoplasm.</li>
<li><strong>Side finding about the method:</strong> removing an RBP with CRISPR also lowers the activity of neighbouring genes, up to at least 50,000 DNA letters away. Removing it with shRNA does not.</li>
</ul>

<nav aria-label="Contents"><strong>Contents</strong><ol>
<li><a href="#why">Why we did this</a></li><li><a href="#data">The data</a></li><li><a href="#how">The class effect</a></li>
<li><a href="#agree">Checking that a result is real</a></li><li><a href="#batch">A technical problem</a></li><li><a href="#main">Main result</a></li>
<li><a href="#notexpl">What does not explain it</a></li><li><a href="#splice">Splicing and NMD</a></li><li><a href="#sno">snoRNA host genes</a></li>
<li><a href="#cis">Side finding: CRISPR</a></li><li><a href="#cands">The {len(hits)} RBPs</a></li><li><a href="#limits">Limitations</a></li>
</ol></nav>

<h2 id="why">Why we did this</h2>
<p>A gene is first copied into RNA. Messenger RNAs (mRNAs) are then read to make proteins. Long non-coding RNAs (lncRNAs) are not read into proteins; many of them help control other genes. Throughout its life every RNA is handled by RNA-binding proteins (RBPs). RBPs cut out pieces of the RNA that are not needed (splicing), move the RNA out of the nucleus, protect it, or destroy it.</p>
<p>lncRNAs are known to differ from mRNAs. They are usually present in fewer copies, are spliced less efficiently, are destroyed faster and stay in the nucleus more often. We asked whether lncRNAs also depend on different RBPs than mRNAs. If they do, is that only a consequence of these known differences, or is there something more?</p>

<h2 id="data">The data</h2>
<p>The ENCODE project removed RBPs one at a time from human cells and then measured the amount of every RNA by sequencing (RNA-seq). Each removal experiment was done twice and compared with untreated control cells grown at the same time. We used {n_kd} such experiments, covering {n_rbp_total} different RBPs in two cell lines: K562 (derived from leukaemia) and HepG2 (derived from liver cancer). {n_both_lines} RBPs were removed in both cell lines; these are the key to checking whether a result is real.</p>
<p>The RBPs were removed in two ways. shRNA makes the cell destroy the RBP's mRNA, so less RBP is made. CRISPR cuts the RBP's gene in the DNA so that it no longer works. We also used other ENCODE and public data: maps of where RBPs bind RNA (eCLIP), measurements of how fast each RNA is made and destroyed (Bru-seq and BruChase), RNA amounts in the nucleus and in the cytoplasm, and gene activity in human tissues (GTEx).</p>
{table(t1)}

<h2 id="how">How we measured the response of lncRNAs: the class effect</h2>
<p>After an RBP is removed, some RNAs go up and some go down. A simple comparison of lncRNAs with mRNAs would be unfair, because lncRNAs differ from mRNAs in ways that alone can change how they react. For example, RNAs present in few copies give noisier measurements, and short-lived RNAs react faster to any change in the cell. So we compared each lncRNA only with mRNAs that resemble it.</p>
<p><strong>A worked example.</strong> Suppose that after removing RBP X, one lncRNA drops by 20%. We find the 10 mRNAs most similar to that lncRNA in expression level, gene length, splicing efficiency and stability. Suppose these 10 mRNAs drop by 5% on average. Then this lncRNA fell about 15% more than comparable mRNAs. We repeat this for every lncRNA we can measure (typically 1,000 to 4,000 of them) and take the middle value (the median). That single number is the <strong>class effect</strong> of removing RBP X:</p>
<ul class="find">
<li>a <strong>class effect near zero</strong> means lncRNAs reacted like comparable mRNAs;</li>
<li>a <strong>negative class effect</strong> means lncRNAs ended up lower than comparable mRNAs (they fell more or rose less). This suggests the RBP normally supports lncRNAs more than mRNAs;</li>
<li>a <strong>positive class effect</strong> means lncRNAs ended up higher than comparable mRNAs. This suggests the RBP normally holds lncRNAs back, for example by destroying them.</li>
</ul>
</div>
<div class="wide">{SCHEME}</div>
<div class="col">
<p>Changes are reported on the log2 scale, which is standard in this field. The table shows how to translate log2 values into percentages.</p>
{LOG2}
<p>We used two levels of similarity when choosing the 10 reference mRNAs. <em>Basic matching</em> uses expression level and gene length only. <em>Full matching</em> also uses splicing efficiency and stability, measured in separate experiments. Full matching is our main analysis, because it removes the best-known differences between lncRNAs and mRNAs.</p>

<h2 id="agree">How we check that a result is real: agreement between the two cell lines</h2>
<p>A single class effect could be noise. But if a class effect reflects what an RBP actually does, the same RBP should give a similar class effect in both cell lines. For each of the {n_both_lines} RBPs removed in both lines we therefore have two numbers: its class effect in K562 and its class effect in HepG2.</p>
<p>We measure how well these two lists agree with the <strong>Spearman rank correlation, rho</strong>. We rank the RBPs from the most negative to the most positive class effect, once using K562 and once using HepG2, and compare the two rankings. rho = 1 means the two rankings are identical, rho = 0 means they are unrelated, and negative values would mean they are reversed. Throughout this report, “agreement between the two cell lines” means this rho. Two different cell types, different experiment days and a modest effect size all lower rho, so even a perfect biological effect would not give 1. A rho of about 0.3 based on several hundred RBPs is a clear signal.</p>

<h2 id="batch">A technical problem we had to fix first: shared control samples</h2>
<p>Each removal experiment is compared with a control sample, untreated cells grown on the same day. In ENCODE, about ten removal experiments share one control sample. If that control sample happened to differ slightly, for example in how many lncRNA reads it contained, all ten experiments would show the same apparent change, even though their RBPs are unrelated. We found that this was the strongest pattern in the raw data.</p>
<p>We corrected for it as follows. For each experiment we subtracted, gene by gene, the typical change seen in the other experiments that share its control sample. What remains is what is specific to the removed RBP. We call the data after this step <em>control-corrected</em>.</p>
</div>
<div class="wide">{figure("fig1", 1, "Shared control samples dominated the raw data; the correction removes their influence.",
 "Panels a and b: we searched the data for the strongest patterns of change shared by many experiments (in statistical terms, principal components; numbered 1 to 6 from the strongest). Each bar shows what share of a pattern is explained simply by which control sample an experiment was compared with. Grey bars are before the correction, blue bars after it. The dashed line is the share expected by chance alone. Panel c shows the class effect for eight well-studied RBPs before the correction (open circles) and after it (filled circles), in K562 (blue) and HepG2 (red).",
 f"Before the correction, the control sample explained up to {max(eta['K562'][0].max(), eta['HepG2'][0].max()):.0%} of the strongest patterns, which are therefore largely technical. After the correction, it explains no more than chance would. Well-known biology survives the correction. MTREX, a protein that delivers RNAs to the RNA-destroying machinery in the nucleus, still raises lncRNAs when removed, in both cell lines. Some apparent effects disappear: EXOSC10 in K562 looked like it raised lncRNAs only because it shared a control sample with other experiments.")}</div>
<div class="col">

<h2 id="main">Main result: lncRNAs respond differently from comparable mRNAs, and the same RBP does so in both cell lines</h2>
</div>
<div class="wide">{figure("fig2", 2, "Class effects are mostly small, but they are reproducible between the two cell lines.",
 f"Panel a: how many experiments had each class effect, separately for K562 (blue) and HepG2 (red); the dotted lines mark ±0.1 log2 (about ±7%). Panel b: each dot is one of the {len(w_main)} RBPs removed in both cell lines, placed by its class effect in K562 (horizontal axis) and in HepG2 (vertical axis). Dots in the lower-left and upper-right quarters are RBPs with the same direction of effect in both lines. Coloured dots are the {len(hits)} RBPs with a clear effect in both lines; the strongest are named. Panel c: agreement between the two cell lines (rho) in different versions of the analysis. Grey bars use the raw data, blue bars the control-corrected data; the dark bar is the main analysis.",
 f"Most class effects are small (panel a). Still, an RBP's class effect in K562 predicts its class effect in HepG2 (panel b). The agreement grows from {rho_raw:.2f} in the raw data to {rho_main:.2f} after the control correction and full matching (panel c). If the corrections had removed real biology, the agreement would have fallen instead. The agreement stays between {t2['Agreement between lines (rho)'].min():.2f} and {t2['Agreement between lines (rho)'].max():.2f} in every check we tried (table below).")}</div>
<div class="col">
<h3>Does the result survive different choices in the analysis?</h3>
<p>We repeated the main analysis eight times, each time changing one choice. A robust result should look similar every time. Columns: the number of RBPs removed in both cell lines in that version; their agreement between the two lines (rho); the typical size of the class effect; and how many RBPs have a clear class effect in both lines and in the same direction. A clear class effect means a change of at least 0.1 log2 (about 7%) that is statistically significant after correcting for the number of tests. The last column shows how many of these are also among the {len(hits)} RBPs found in the main analysis.</p>
{table(t2, {"Typical class effect, sign ignored (log2)": "{:.3f}", "Agreement between lines (rho)": "{:.2f}"}, "w")}

<h2 id="notexpl">What the difference between lncRNAs and mRNAs is not explained by</h2>
<p>If the class effect only reflected a known property of lncRNAs, then also requiring the reference mRNAs to share that property should make the class effect shrink or stop agreeing between the cell lines. We tested this for each property in turn.</p>
</div>
<div class="wide">{figure("fig3", 3, "Adding RNA properties to the comparison barely changes the class effect.",
 "Panels a and b: for each group of properties, the open circle is the analysis without that property and the filled circle the analysis in which the reference mRNAs must also match the lncRNA in that property. Panel a shows the agreement between the two cell lines (rho), panel b the typical size of the class effect. Panel c uses maps of where each RBP binds RNA (eCLIP). For each RBP (one dot), it shows whether the RNAs that the RBP binds changed more after its removal than similar RNAs it does not bind. Positive values would mean bound RNAs react more strongly.",
 "None of the properties makes the class effect disappear or stop agreeing between the cell lines. Splicing efficiency and stability explain a small part of its size (about 14%); location in the cell, tissue specificity and promoter type explain nothing. Direct binding does not explain it either: RNAs bound by an RBP do not react more strongly to its removal (panel c, values at or just below zero), and the class effect is the same when bound RNAs are left out. We also checked that lncRNAs are not missing from the binding maps. At the same expression level they are bound at least as often as mRNAs; they are simply present in few copies.")}</div>
<div class="col">
<p>The table gives the numbers behind panels a and b. “Before” is the analysis without the property and “after” the analysis with it. For the binding row, “after” means only RNAs that the removed RBP does not bind were used.</p>
{table(t3, {"Typical class effect before (log2)": "{:.3f}", "Typical class effect after (log2)": "{:.3f}", "Agreement between lines, before": "{:.2f}", "Agreement between lines, after": "{:.2f}"}, "w")}

<h2 id="splice">Two biological threads: splicing and nonsense-mediated decay</h2>
<p>ENCODE sorts RBPs by their known functions. We asked whether RBPs with the same function push lncRNAs in the same direction. Only one group did so consistently: the spliceosome, the machinery that cuts out introns (the parts of an RNA that are removed before it is used). Its removal lowered lncRNAs relative to comparable mRNAs.</p>
<p>A closer look showed what this means. We separately measured unspliced RNA (reads in introns) and mature RNA (reads in exons). Without the spliceosome, introns stay in the RNA, and this happens much more in mRNAs than in lncRNAs. So the lncRNA–mRNA difference comes mainly from mRNAs being more dependent on the spliceosome, not from lncRNAs being destroyed.</p>
<p>A second pathway, nonsense-mediated decay (NMD), destroys certain RNAs in the cytoplasm. Removing its main proteins raised lncRNAs more than mRNAs of the same expression level.</p>
</div>
<div class="wide">{figure("fig4", 4, "Of all functional groups of RBPs, only the spliceosome has a consistent effect, and it acts mainly through mRNAs.",
 "Panel a: RBPs grouped by known function, with the number of RBPs in brackets. Each coloured dot is the median class effect of the RBPs in that group, and each grey dot the median of all other RBPs; red marks a statistically significant difference. Panels b and c show changes measured separately for mRNAs (grey) and lncRNAs (green), without subtracting one from the other. Genes are split into five groups from low to high expression. Panel b: change in unspliced RNA after removing spliceosome proteins. Panel c: change in mature RNA after removing NMD proteins. Solid lines are K562, dashed lines HepG2; the legend gives the number of removal experiments.",
 f"After spliceosome removal, unspliced RNA builds up by about {spl.loc['mRNA', 'min']:.2f} to {spl.loc['mRNA', 'max']:.2f} log2 in mRNAs but only {spl.loc['lncRNA', 'min']:.2f} to {spl.loc['lncRNA', 'max']:.2f} in lncRNAs, at every expression level and in both cell lines. lncRNA processing therefore depends much less on the spliceosome. After NMD removal in K562, lncRNAs rise more than mRNAs at every expression level, which fits with many lncRNAs being NMD targets. The HepG2 lines in panel c come from only two experiments and should not be trusted.")}</div>
<div class="col">
<details><summary>Show all functional groups</summary>
<p class="note">The adjusted p-value takes into account that many groups were tested; values below 0.05 are considered significant.</p>
{table(t5, {"Median class effect in group": "{:.3f}", "Median class effect of all other RBPs": "{:.3f}", "Adjusted p-value": "{:.3f}"})}</details>

<h2 id="sno">snoRNA host genes behave like a separate group</h2>
<p>Some lncRNA genes carry small nucleolar RNAs (snoRNAs) inside their introns; when the intron is cut out, the snoRNA is released. These lncRNAs are called snoRNA host genes. They kept standing out in our analyses, so we compared their basic properties with those of other lncRNAs.</p>
</div>
<div class="wide">{figure("fig5", 5, "snoRNA host genes differ from other lncRNAs and resemble always-on housekeeping genes.",
 "Panel a: how much of each gene's RNA is in the nucleus compared with the cytoplasm (higher values mean more nuclear). Panel b: how tissue-specific each gene is across 52 human tissues (0 = equally active everywhere, 1 = active in one tissue only). Panel c: the share of genes whose promoter, the switch that turns the gene on, contains a CpG island, which is typical of genes that are always on. Panel d: how much each snoRNA host gene (orange dot) changed after its NMD proteins were removed, next to the range for other lncRNAs (box). Well-known snoRNA host genes are named.",
 "Compared with other lncRNAs, snoRNA host genes are more cytoplasmic, are active in most tissues and usually have CpG-island promoters, so in these respects they resemble mRNAs. Well-known snoRNA hosts (ZFAS1, SNHG5, SNHG6, SNHG8, SNHG11) rise strongly when NMD is removed, which confirms that NMD destroys their RNA after the snoRNA has been released. snoRNA host genes should therefore be treated as a separate subgroup rather than as typical lncRNAs.")}</div>
<div class="col">
{table(t6, {"Nuclear enrichment, K562": "{:.2f}", "Nuclear enrichment, HepG2": "{:.2f}", "Tissue specificity (0–1)": "{:.2f}", "% with CpG-island promoter": "{:.0f}"})}
<p class="note">Nuclear enrichment is log2 of the ratio of RNA in the nucleus to RNA in the cytoplasm, as a median for each class. Because the two compartments were measured separately, only the order of the classes is meaningful, not the absolute values.</p>

<h2 id="cis">Side finding: CRISPR also lowers neighbouring genes</h2>
<p>Removing an RBP might also affect genes that lie right next to its gene in the DNA, including its antisense partner (natural antisense transcript, NAT), an RNA made from the opposite DNA strand. Such local effects have nothing to do with what the RBP does. That is why we excluded neighbouring genes from all analyses above. Here we looked at them directly, comparing each neighbouring gene with genes of similar expression elsewhere in the same experiment.</p>
</div>
<div class="wide">{figure("figS1", "S1", "CRISPR, but not shRNA, lowers the activity of genes near the removed RBP's gene.",
 "Each bar is the median change of genes at a given position relative to the removed RBP's gene, after CRISPR (purple) or shRNA (green), compared with similar genes elsewhere. Categories range from genes that overlap the RBP gene to genes 10,000 to 50,000 DNA letters (10–50 kb) away. Left: protein-coding neighbours; right: lncRNA neighbours. Missing bars mean there were too few genes.",
 "After CRISPR, protein-coding neighbours drop by about 0.1 to 0.3 log2 (7–20%), even 10–50 kb away. This matches known side effects of the DNA cut made by CRISPR: genes near a DNA break are silenced, and repairing the break sometimes deletes large pieces of DNA. shRNA lowers only genes that share part of their sequence with the RBP's mRNA, which is expected because shRNA recognises sequence. Antisense lncRNA partners were not generally lowered by either method. A few individual partners changed strongly; for example, a lncRNA antisense to DDX20 rose about 8-fold in both cell lines after CRISPR.")}</div>
<div class="col">
{table(t7, {c: "{:.1f}" if c.startswith("%") else "{:.2f}" for c in t7.columns}, "w")}
<p class="note">For comparison, in a typical experiment about 4% of all genes were clearly lower after CRISPR and about 11% after shRNA. “Clearly lower” means statistically significant after correcting for the number of genes tested.</p>

<h2 id="cands">The {len(hits)} RBPs with a clear class effect in both cell lines</h2>
<p>Criteria: in each cell line, the class effect is at least 0.1 log2 (about 7%) in size and statistically significant after correcting for the number of RBPs tested, and it points the same way in both lines. Columns: whether lncRNAs end up higher or lower than comparable mRNAs after the RBP is removed; the class effect in each cell line (log2); how the RBP was removed; the class effect calculated separately for four types of lncRNA (average of both lines); and whether the RBP passes an extra check. The extra check repeats the analysis using only reference mRNAs whose genes have no annotated forms with unspliced introns, to rule out a known annotation artefact. {rob_cnt} of the {len(hits)} RBPs pass it. Remember that roughly one in three of these RBPs may be a false positive.</p>
{table(t4, {c: "{:.2f}" for c in t4.columns if c not in ("RBP", "lncRNAs after removal", "How removed", "Passes extra check")}, "w")}
<p class="note">RBPs whose removal leaves lncRNAs higher include known RNA-destroying factors (MTREX, STAU1, KHSRP, CNOT10). RBPs whose removal leaves lncRNAs lower are mostly splicing factors (SRSF1, WBP11, PUF60, GPKOW, AQR, SF1, RBM22). They also include NONO, a protein of nuclear bodies called paraspeckles, which had the strongest effect of all.</p>

<h2 id="limits">Limitations</h2>
<p>All data come from two cancer cell lines, which may not represent normal cells. For most RBPs the class effect is small. Sequencing measures the share of each RNA among all RNAs, not absolute amounts, so a class effect can arise from changes in the reference mRNAs as well as in the lncRNAs. We checked this for the splicing result, where it turned out to matter. About one in three of the {len(hits)} RBPs may be false positives. The measurements of splicing efficiency and stability come from a small number of control samples. Finally, removing an RBP changes many genes, so we cannot tell whether an effect on lncRNAs is direct or passes through other genes.</p>

<details><summary>Glossary</summary><dl>
<dt>RBP (RNA-binding protein)</dt><dd>A protein that attaches to RNA and controls how it is processed, where it goes, how long it lasts or whether it is used to make protein.</dd>
<dt>mRNA (messenger RNA)</dt><dd>RNA copied from a gene and read by the cell to make a protein.</dd>
<dt>lncRNA (long non-coding RNA)</dt><dd>RNA longer than 200 letters that is not used to make a protein. Types include lincRNAs (lying between other genes), antisense RNAs (made from the opposite DNA strand of another gene) and host genes of small RNAs (snoRNA and miRNA host genes).</dd>
<dt>Removal experiment (knockdown or knockout)</dt><dd>An experiment in which one RBP is removed from cells, by shRNA (the RBP's mRNA is destroyed) or by CRISPR (the RBP's gene is cut).</dd>
<dt>Class effect</dt><dd>For one removal experiment: the median, over all measurable lncRNAs, of how much each lncRNA changed minus how much its 10 most similar mRNAs changed. Negative values mean lncRNAs ended up lower than comparable mRNAs, positive values higher.</dd>
<dt>Agreement between the two cell lines (rho)</dt><dd>The Spearman rank correlation between the class effects of the same RBPs in K562 and in HepG2: 1 means identical ranking, 0 means no relation.</dd>
<dt>log2 change</dt><dd>A way of writing fold changes: +1 means doubling, −1 halving, ±0.1 about ±7%.</dd>
<dt>Splicing, spliceosome, intron</dt><dd>Splicing cuts out parts of an RNA (introns) that are not needed; the spliceosome is the machinery that does it.</dd>
<dt>NMD (nonsense-mediated decay)</dt><dd>A quality-control pathway in the cytoplasm that destroys RNAs with certain features.</dd>
<dt>eCLIP</dt><dd>An experiment that maps where on RNA a given RBP binds.</dd>
<dt>NAT (natural antisense transcript)</dt><dd>An RNA made from the opposite DNA strand of another gene and overlapping it.</dd>
<dt>Statistically significant, p-value, adjusted p-value</dt><dd>A p-value is the probability of seeing a result at least this strong by chance. An adjusted p-value also accounts for having tested many genes, RBPs or groups.</dd>
</dl></details>
</div>'''


CSS = '''
:root{--ink:#1d2733;--muted:#5b6675;--paper:#ffffff;--panel:#f4f6f8;--rule:#dde3ea;--lnc:#2a9d8f;--mrna:#7f7f7f;--k562:#3b6ea8;--hepg2:#c8553d;--sno:#e9a03b}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"Source Serif 4",Georgia,"Times New Roman",serif;font-size:18px;line-height:1.62}
main{padding:2.5rem 1.25rem 4rem}
.col{max-width:42rem;margin:0 auto}
.wide{max-width:72rem;margin:2.25rem auto}
h1,h2,h3,.ft,.lead,th,.note,nav,.kicker{font-family:"IBM Plex Sans","Segoe UI",Helvetica,Arial,sans-serif}
h1{font-size:2.15rem;line-height:1.18;font-weight:600;margin:0 0 .6rem;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:1.02rem;margin:0 0 2rem}
.answer{font-size:1.32rem;line-height:1.5;border-left:4px solid var(--lnc);padding:.2rem 0 .2rem 1.1rem;margin:0 0 2rem}
h2{font-size:1.38rem;font-weight:600;margin:3.2rem 0 .8rem;padding-top:.8rem;border-top:1px solid var(--rule)}
h3{font-size:1.05rem;font-weight:600;margin:1.8rem 0 .4rem}
p{margin:0 0 1rem}
ul.find{padding-left:1.1rem;margin:0 0 1.4rem}
ul.find li{margin:0 0 .65rem}
strong{font-weight:600}
.lnc{color:var(--lnc);font-weight:600}.mrna{color:var(--mrna);font-weight:600}
nav{font-size:.92rem;background:var(--panel);border-radius:6px;padding:1rem 1.25rem;margin:0 0 2rem}
nav ol{margin:.3rem 0 0;padding-left:0;list-style-position:inside;columns:2;column-gap:2rem}
nav li{break-inside:avoid;margin:.1rem 0}
nav a{color:var(--ink)}
figure{margin:0}
figure img{display:block;width:100%;height:auto;background:#fff;border:1px solid var(--rule);border-radius:6px}
figcaption{max-width:52rem;margin:.9rem auto 0;font-size:.97rem;color:#2c3744}
.ft{font-weight:600;font-size:1rem;margin-bottom:.4rem}
.lead{font-weight:600}
.scheme{width:100%;height:auto;display:block}
.tbl{overflow-x:auto;margin:1.2rem 0 1.8rem;border:1px solid var(--rule);border-radius:6px}
.tbl.narrow{max-width:22rem}
.tbl.w{width:min(62rem,calc(100vw - 2.5rem));position:relative;left:50%;transform:translateX(-50%)}
table{border-collapse:collapse;width:100%;font-size:.86rem;font-family:"IBM Plex Sans","Segoe UI",Helvetica,Arial,sans-serif}
th{text-align:left;font-weight:600;background:var(--panel);padding:.55rem .7rem;border-bottom:1px solid var(--rule);vertical-align:bottom}
td{padding:.45rem .7rem;border-bottom:1px solid #eef1f4;vertical-align:top}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
.note{font-size:.88rem;color:var(--muted)}
details{border:1px solid var(--rule);border-radius:6px;padding:.7rem 1rem;margin:1rem 0}
summary{cursor:pointer;font-family:"IBM Plex Sans","Segoe UI",Helvetica,Arial,sans-serif;font-weight:600}
dl{margin:.6rem 0 0}dt{font-weight:600;font-family:"IBM Plex Sans","Segoe UI",Helvetica,Arial,sans-serif;font-size:.95rem;margin-top:.7rem}dd{margin:.1rem 0 0}
a:focus-visible,summary:focus-visible{outline:2px solid var(--k562);outline-offset:2px}
@media (max-width:640px){body{font-size:17px}h1{font-size:1.7rem}nav ol{columns:1}.answer{font-size:1.15rem}}
'''


doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>lncRNA responses to RBP knockdowns: plain-language report</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><main>{body}</main></body></html>'''
out = REP / "rbp_lncRNA_report.html"
out.write_text(doc, encoding="utf-8")
print(f"Saved: {out} ({out.stat().st_size / 1e6:.1f} MB); English figures in {FIG}/")