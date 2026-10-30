"""
Raport v2, czesc A: ryciny 3-7 (angielskie podpisy, wersja podstawowa).
Ryciny 1, 2 i S1 pochodza z results/phase4/figures_en (skrypt 43), ryciny locus z results/phase4/cases (skrypt 47).
Wyjscie: results/phase4/report_v2/figs/fig3..fig7.png (+ .pdf)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr

F = Path("results/phase4/report_v2/figs")
F.mkdir(parents=True, exist_ok=True)
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz miRNA"}
NMD = {"UPF1", "UPF2", "UPF3B", "SMG1", "SMG5", "SMG6", "SMG7", "SMG8", "SMG9"}
EJC = {"RBM8A", "MAGOH", "EIF4A3", "CASC3"}
COL = {"K562": "#3b6ea8", "HepG2": "#c8553d", "mRNA": "#7f7f7f", "lncRNA": "#2a9d8f", "sno": "#e9a03b", "surv": "#6a4c93"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})


def letters(axes):
    for a, l in zip(axes, "abcdefgh"):
        a.text(-0.16, 1.08, l, transform=a.transAxes, fontweight="bold", fontsize=11)


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(F / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"   {name}", flush=True)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def stars(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "n.s."


def bracket(ax, x1, x2, y, q, h=0.02):
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.plot([x1, x1, x2, x2], [y, y + h * yr, y + h * yr, y], color="k", lw=0.7)
    ax.text((x1 + x2) / 2, y + 1.2 * h * yr, stars(q), ha="center", va="bottom", fontsize=7)


def rep_stats(df, v):
    s = df[df["wariant"] == v]
    w = s.groupby(["target", "linia"])["roznica"].mean().unstack("linia").dropna()
    return spearmanr(w["K562"], w["HepG2"])[0], s["roznica"].abs().median()


# ================================================================ Rycina 3
rv = pd.read_csv("results/phase2/classes/robustness_variants.tsv.gz", sep="\t")
loc = pd.read_csv("results/phase3/localization/class_effect_B_vs_C.tsv", sep="\t")
pla = pd.read_csv("results/phase3/plasticity/class_effect_C_vs_D.tsv", sep="\t")
ecl = pd.read_csv("results/phase3/eclip/eclip_directness.tsv", sep="\t")
e2 = pd.concat([ecl[["linia", "RBP", "klasa_wszystkie"]].rename(columns={"RBP": "target", "klasa_wszystkie": "roznica"}).assign(wariant="before"),
                ecl[["linia", "RBP", "klasa_niezwiazane"]].rename(columns={"RBP": "target", "klasa_niezwiazane": "roznica"}).assign(wariant="after")])
tests = [("+ splicing &\nstability", rv, "A_na_zbiorze_B", "B"), ("+ nucleus vs\ncytoplasm", loc, "B", "C"),
         ("+ tissue specif.\n& CpG promoter", pla, "C", "D"), ("unbound genes\nonly (eCLIP)", e2, "before", "after")]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.7), gridspec_kw={"width_ratios": [1.15, 1.15, 1, 0.9]})
for k, (ax, what, ylab) in enumerate(zip(axes[:2], [0, 1], ["agreement between cell lines (rho)", "typical class effect, sign ignored (log2)"])):
    for i, (name, df, a, b) in enumerate(tests):
        va, vb = rep_stats(df, a)[what], rep_stats(df, b)[what]
        ax.plot([i - 0.15, i + 0.15], [va, vb], color="0.6", lw=1)
        ax.scatter([i - 0.15], [va], facecolor="white", edgecolor="k", s=28, zorder=3)
        ax.scatter([i + 0.15], [vb], color="k", s=28, zorder=3)
    ax.set_xticks(range(len(tests)), [t[0] for t in tests], fontsize=7)
    ax.set_ylabel(ylab)
    ax.set_ylim(min(0, ax.get_ylim()[0]), ax.get_ylim()[1] * 1.15)
    ax.set_title("Agreement between the two cell lines" if k == 0 else "Typical size of the class effect")
    ax.text(0.02, 0.97, "○ before   ● after adding the property", transform=ax.transAxes, fontsize=7, va="top")
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
ax = axes[3]
rt = pd.read_csv("results/phase4/stats/replication_tests.tsv", sep="\t")
lab = {"wszystkie RBP": "all RBPs", "bez splicingu i NMD": "without splicing\n& NMD RBPs"}
for i, r in enumerate(rt.itertuples()):
    ax.errorbar(r.rho, i, xerr=[[r.rho - r.ci_lo], [r.ci_hi - r.rho]], fmt="o", color="k", capsize=3)
    ax.text(r.ci_hi + 0.01, i, f"n = {r.n}\np = {r.p:.0e}", va="center", fontsize=6.5)
ax.set_yticks(range(len(rt)), [lab.get(z, z) for z in rt["zestaw"]], fontsize=7.5)
ax.set_ylim(-0.6, len(rt) - 0.4)
ax.invert_yaxis()
ax.axvline(0, color="k", lw=0.6)
ax.set_xlim(-0.05, max(0.6, rt["ci_hi"].max() + 0.25))
ax.set_xlabel("agreement between lines (rho, 95% CI)")
ax.set_title("Without splicing and NMD RBPs")
letters(axes)
save(fig, "fig3")

# ================================================================ Rycina 4
cat = pd.read_csv("results/phase2/classes/functional_categories.tsv", sep="\t")
catf = cat[cat["typ"] == "funkcja"].sort_values("mediana")
smp3 = pd.read_csv("data/meta/phase3_samples.tsv", sep="\t", dtype=str).drop_duplicates("kd_experiment").set_index("kd_experiment")
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
n_kd = eb_all.groupby(["grupa", "linia"])["kd"].nunique()
eb = eb_all.groupby(["grupa", "linia", "bin", "k"], observed=True)[["d_ekson", "d_intron"]].median().reset_index()
sp = pd.read_csv("results/phase4/stats/spliceosome_by_splicing_efficiency.tsv", sep="\t")
fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), gridspec_kw={"width_ratios": [1.3, 1, 1, 1]})
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
                   Line2D([], [], marker="o", ls="", color="#3b6ea8", ms=5, label="RBPs in group"),
                   Line2D([], [], marker="o", ls="", color="#c8553d", ms=5, label="RBPs in group, q < 0.05")],
          frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
for ax, (grupa, comp, title) in zip(axes[1:3], [("spliceosome", "d_intron", "Spliceosome removed: unspliced RNA"),
                                                 ("NMD", "d_ekson", "NMD removed (UPF, SMG): mature RNA")]):
    for cell, ls in [("K562", "-"), ("HepG2", "--")]:
        for k in ["mRNA", "lncRNA"]:
            s = eb[(eb["grupa"] == grupa) & (eb["linia"] == cell) & (eb["k"] == k)]
            if len(s):
                ax.plot(s["bin"].astype(int), s[comp], ls=ls, marker="o", ms=3, color=COL[k], label=f"{k}, {cell} ({n_kd.get((grupa, cell), 0)} KD)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks([1, 2, 3, 4, 5], ["low", "2", "3", "4", "high"])
    ax.set_xlabel("expression bin (defined on mRNA)")
    ax.set_ylabel("median change after removal (log2)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=6)
ax = axes[3]
for cell, ls in [("K562", "-"), ("HepG2", "--")]:
    s = sp[sp["linia"] == cell].reset_index(drop=True)
    xs = np.arange(1, len(s) + 1)
    ax.plot(xs, s["kumulacja intronow mRNA"], ls=ls, marker="o", ms=3, color=COL["mRNA"], label=f"mRNA, {cell}")
    ax.plot(xs, s["kumulacja intronow lncRNA"], ls=ls, marker="o", ms=3, color=COL["lncRNA"], label=f"lncRNA, {cell}")
    for x_, r in zip(xs, s.itertuples()):
        ax.text(x_, max(r[5], r[6]) + (0.012 if cell == "K562" else 0.032), stars(r.q), ha="center", fontsize=6, color=COL[cell])
ax.set_xticks([1, 2, 3, 4, 5], ["fastest", "2", "3", "4", "slowest"])
ax.set_xlabel("splicing in control cells (bin, defined on mRNA)")
ax.set_ylabel("unspliced RNA after spliceosome\nremoval (median log2)")
ax.set_title("Same splicing speed, different response")
ax.legend(frameon=False, fontsize=6)
letters(axes)
save(fig, "fig4")

# ================================================================ Rycina 5 (wykresy skrzypcowe + istotnosc)
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
cls["gene"] = cls["gene_id"].str.split(".").str[0]
k_of = cls.drop_duplicates("gene").set_index("gene")["klasa"]
order = ["mRNA", "lincRNA", "antysens", "inne lncRNA", "gospodarz miRNA", "gospodarz snoRNA"]
short = ["mRNA", "lincRNA", "NAT", "other", "miRNA\nhost", "snoRNA\nhost"]
colors = [COL["mRNA"]] + [COL["lncRNA"]] * 4 + [COL["sno"]]
nuc = pd.read_csv("results/phase3/localization/nuclear_index.tsv", sep="\t", index_col=0)
pf = pd.read_csv("results/phase3/plasticity/plasticity_features.tsv", sep="\t", index_col=0)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
tests5, draw = [], []
for j, (vals, ylab, title) in enumerate([(nuc["HepG2"], "log2 nucleus / cytoplasm (HepG2)", "Localization"),
                                          (pf["tau"], "tau (GTEx; 1 = one tissue only)", "Tissue specificity")]):
    ax = axes[j]
    kk = k_of.reindex(vals.index)
    data = [vals[kk == c].dropna().values for c in order]
    parts = ax.violinplot([d if len(d) > 1 else np.array([0, 0]) for d in data], showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(order) + 1), short, fontsize=7)
    ax.set_ylabel(ylab); ax.set_title(title)
    for ref in ["mRNA", "lincRNA"]:
        i = order.index(ref)
        tests5.append((j, i + 1, len(order), mannwhitneyu(data[i], data[-1]).pvalue))
ax = axes[2]
cg = pf.groupby("klasa")["CGI"].agg(["sum", "count"]).reindex(order)
ax.bar(range(1, len(order) + 1), 100 * cg["sum"] / cg["count"], color=colors, alpha=0.8)
ax.set_xticks(range(1, len(order) + 1), short, fontsize=7)
ax.set_ylabel("% promoters with a CpG island"); ax.set_title("Promoters")
for ref in ["mRNA", "lincRNA"]:
    i = order.index(ref)
    a, b = cg.loc[ref], cg.loc["gospodarz snoRNA"]
    tests5.append((2, i + 1, len(order), fisher_exact([[a["sum"], a["count"] - a["sum"]], [b["sum"], b["count"] - b["sum"]]])[1]))
ax = axes[3]
for i, cell in enumerate(["K562", "HepG2"]):
    c_ = pd.read_csv(f"results/phase3/splice_nmd/consensus_{cell}.tsv.gz", sep="\t", index_col=0)
    c_ = c_[c_["klasa"].notna()]
    ln = c_.loc[c_["klasa"].isin(LNC), "nmd"].dropna().values
    sn = c_.loc[c_["klasa"] == "gospodarz snoRNA", "nmd"].dropna().values
    parts = ax.violinplot([ln, sn if len(sn) > 1 else np.array([0, 0])], positions=[2 * i + 1, 2 * i + 1.8], showmedians=True, widths=0.7)
    for pc, c in zip(parts["bodies"], [COL["lncRNA"], COL["sno"]]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    tests5.append((3, 2 * i + 1, 2 * i + 1.8, mannwhitneyu(ln, sn).pvalue))
ax.set_xticks([1, 1.8, 3, 3.8], ["lncRNA\nK562", "snoRNA host\nK562", "lncRNA\nHepG2", "snoRNA host\nHepG2"], fontsize=6.5)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("change after removing NMD/EJC\nproteins (median log2)", fontsize=8); ax.set_title("Sensitivity to NMD")
q5 = bh([t[3] for t in tests5])
for ax in axes:
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.22 * (hi - lo))
level = {}
for (j, x1, x2, _), q in zip(tests5, q5):
    ax = axes[j]
    lo, hi = ax.get_ylim()
    k = level.get(j, 0)
    bracket(ax, x1, x2, hi - (0.17 - 0.08 * k) * (hi - lo), q)
    level[j] = k + 1
letters(axes)
fig.text(0.5, -0.02, "Brackets: Mann-Whitney (a, b, d) or Fisher (c) test, Benjamini-Hochberg correction across the 8 tests: "
         "* q < 0.05, ** q < 0.01, *** q < 0.001, n.s. not significant", ha="center", fontsize=7)
save(fig, "fig5")

# ================================================================ Rycina 6 (nadzor jadrowy i lncRNA dywergentne)
ce = pd.read_csv("results/phase4/network/category_effects_per_kd.tsv", sep="\t")
sv = pd.read_csv("results/phase4/network/surveillance_vs_others.tsv", sep="\t")
cats = ["dywergentne", "antysensowe", "w poblizu", "miedzygenowe"]
cat_en = ["divergent", "NAT", "nearby", "intergenic"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 1, 1.1]})
for ax, cell in zip(axes[:2], ["K562", "HepG2"]):
    g = ce[ce["linia"] == cell]
    ymax = max(g[c].quantile(0.98) for c in cats)
    ymax = max(ymax, g.loc[g["nadzor"], cats].max().max() if g["nadzor"].any() else ymax)
    for i, c in enumerate(cats):
        oth = g.loc[~g["nadzor"], c].dropna()
        sur = g.loc[g["nadzor"], [c, "target"]].dropna()
        ax.boxplot([oth.values], positions=[i], widths=0.5, showfliers=False)
        ax.scatter(np.full(len(sur), i + 0.3), sur[c], color=COL["surv"], s=18, zorder=3)
        for _, r in sur.iterrows():
            ax.annotate(r["target"], (i + 0.3, r[c]), fontsize=5, xytext=(3, 0), textcoords="offset points", va="center")
        qq = sv[(sv["linia"] == cell) & (sv["kategoria"] == c)]["q"]
        if len(qq):
            ax.text(i + 0.15, ymax + 0.06, stars(qq.iloc[0]), ha="center", fontsize=7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(top=ymax + 0.15)
    ax.set_xticks(range(len(cats)), cat_en, fontsize=7.5)
    ax.set_ylabel("class effect of the category (log2)")
    ax.set_title(f"{cell}: box = all other RBPs, dots = nuclear surveillance", fontsize=8.5)
ax = axes[2]
ce["excess"] = ce["dywergentne"] - ce["miedzygenowe"]
w = ce.groupby(["target", "linia"])["excess"].mean().unstack("linia").dropna()
w["mean"] = w.mean(axis=1)
top = w.sort_values("mean", ascending=False).head(15)[::-1]
yy = np.arange(len(top))
ax.barh(yy, top["mean"], color=[COL["surv"] if t in {"MTREX", "EXOSC2", "EXOSC5", "EXOSC9", "EXOSC10"} else "#9e9e9e" for t in top.index])
ax.scatter(top["K562"], yy, color=COL["K562"], s=12, zorder=3, label="K562")
ax.scatter(top["HepG2"], yy, color=COL["HepG2"], s=12, zorder=3, label="HepG2")
ax.set_yticks(yy, top.index, fontsize=7)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("divergent minus intergenic lncRNA effect (log2)")
ax.set_title("RBPs whose removal raises divergent lncRNAs most")
ax.legend(frameon=False, fontsize=7, loc="lower right")
letters(axes)
save(fig, "fig6")

# ================================================================ Rycina 7 (poziom genow)
nb = pd.read_csv("results/phase4/network/neighbour_coresponse_summary.tsv", sep="\t")
hub = pd.read_csv("results/phase4/specific/specific_hub_rbps.tsv", sep="\t", index_col=0)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
types = [("lncRNA-mRNA", "dywergentne", "lncRNA-mRNA\ndivergent"), ("lncRNA-mRNA", "antysensowe", "lncRNA-mRNA\nNAT"),
         ("lncRNA-mRNA", "w poblizu", "lncRNA-mRNA\nnearby"), ("mRNA-mRNA", "dywergentne", "mRNA-mRNA\ndivergent"),
         ("mRNA-mRNA", "w poblizu", "mRNA-mRNA\nnearby")]
for j, cell in enumerate(["K562", "HepG2"]):
    for i, (pa, ty, _) in enumerate(types):
        r = nb[(nb["linia"] == cell) & (nb["para"] == pa) & (nb["typ"] == ty)]
        if r.empty:
            continue
        x = i + (j - 0.5) * 0.35
        ax.bar(x, r["mediana r"].iloc[0], 0.33, color=COL[cell], label=cell if i == 0 else None)
        ax.scatter(x, r["mediana r tla"].iloc[0], marker="_", s=120, color="k", zorder=3, label="background" if (i == 0 and j == 0) else None)
ax.set_xticks(range(len(types)), [t[2] for t in types], fontsize=7)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("median co-response r across knockdowns")
ax.set_title("Do neighbouring genes change together?")
ax.legend(frameon=False, fontsize=7)
ax = axes[1]
h = hub.head(10)[::-1]
yy = np.arange(len(h))
ax.barh(yy - 0.18, h["lncRNA"], 0.35, color=COL["lncRNA"], label="lncRNAs")
ax.barh(yy + 0.18, h.iloc[:, 1], 0.35, color=COL["mRNA"], label="mRNAs, same expression")
ax.set_yticks(yy, h.index, fontsize=7.5)
ax.set_xlabel("genes for which this RBP is the dominant\nregulator in both cell lines")
ax.set_title("Which RBPs control specific lncRNAs?")
ax.legend(frameon=False, fontsize=7, loc="lower right")
letters(axes)
save(fig, "fig7")
print(f"Zapisano ryciny w {F}/")
