"""
Rycina 2 (publikacja): czego roznica lncRNA-mRNA nie tlumaczy. Panele w results/figures_pub/fig2/:
  a_dopasowania  - cztery porownania jako osobne male wykresy (gora: zgodnosc miedzy liniami, rho z 95% CI;
                   dol: typowa wielkosc efektu), z liczba lncRNA i RBP oraz cechami, wedlug ktorych dobrano mRNA
  b1_wiazanie    - |log2FC| genow zwiazanych minus niezwiazanych (eCLIP), mRNA i lncRNA, obie linie
  b2_najsilniejsze - iloraz szans: geny zwiazane wsrod 5% najsilniej reagujacych (na RBP)
  c_polozenie    - replikacja class effect wg polozenia lncRNA (95% CI)
  d_profile      - jak czesto profil RBP w K562 najbardziej przypomina ten sam RBP w HepG2 (lncRNA vs mRNA)
Runda 2 (19.10): w A tytuly = cecha dodana do dopasowania (jak w podpisie), wspolna skala osi y, pod panelem tylko
liczby lncRNA i RBP (cechy "przed" opisuje podpis); wezsze panele B-E, czcionki >= 6 pt (figlib.MIN_FS).
"""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from figlib import MM, PAL, bh, bracket, panel_fixed, rho_ci, save, fmt_p, stars  # noqa: E402
from scipy.stats import binomtest, fisher_exact, wilcoxon  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIG = "fig2"
M = Path("results/phase2/matrix")
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}


def per_rbp(df, col="roznica"):
    return df.groupby(["target", "linia"])[col].mean().unstack("linia").dropna()


# ---------------------------------------------------------------- liczby lncRNA w porownaniach (te same kryteria co analizy)
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


def n_lnc(cols, f):
    g = f.dropna(subset=cols)
    mr = g[g["k"] == "mRNA"]
    lo, hi = mr[cols].quantile(0.01), mr[cols].quantile(0.99)
    return int((g["k"].isin(LNC) & ((g[cols] >= lo) & (g[cols] <= hi)).all(axis=1)).sum())


counts = {}
try:
    cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
    kin = pd.read_csv("results/phase1/bru/gene_kinetics.tsv", sep="\t")
    nuc = pd.read_csv("results/phase3/localization/nuclear_index.tsv", sep="\t", index_col=0)
    pf = pd.read_csv("results/phase3/plasticity/plasticity_features.tsv", sep="\t", index_col=0)
    spl = splicing_by_cell()
    for cell in ["K562", "HepG2"]:
        L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
        g = L.index
        gk = g.str.split(".").str[0]
        expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
        f = pd.DataFrame({"k": cls.reindex(g)["klasa"].values, "e": np.log10(expr.reindex(g) + 1).values,
                          "w": np.log10(cls.reindex(g)["width"]).values, "s": spl[cell].reindex(g).values,
                          "t": kin[kin["cell"] == cell].drop_duplicates("gene_id").set_index("gene_id")["stab_6h"].reindex(g).values,
                          "n": nuc[cell].reindex(gk).values, "tau": pf["tau"].reindex(gk).values,
                          "cgi": pf["CGI"].reindex(gk).values}, index=g)
        counts[cell] = {"B": n_lnc(["e", "w", "s", "t"], f), "C": n_lnc(["e", "w", "s", "t", "n"], f),
                        "D": n_lnc(["e", "w", "s", "t", "n", "tau", "cgi"], f), "A": n_lnc(["e", "w"], f)}
except Exception as e:  # noqa: BLE001
    print(f"   (liczby genow pominiete: {e})")


def nlab(key):
    return f"{counts['K562'][key]:,} / {counts['HepG2'][key]:,} lncRNAs" if counts else ""


# ---------------------------------------------------------------- a: male wykresy dla czterech porownan
rv = pd.read_csv("results/phase2/classes/robustness_variants.tsv.gz", sep="\t")
loc = pd.read_csv("results/phase3/localization/class_effect_B_vs_C.tsv", sep="\t")
pla = pd.read_csv("results/phase3/plasticity/class_effect_C_vs_D.tsv", sep="\t")
ecl = pd.read_csv("results/phase3/eclip/eclip_directness.tsv", sep="\t")
e2 = pd.concat([ecl[["linia", "RBP", "klasa_wszystkie"]].rename(columns={"RBP": "target", "klasa_wszystkie": "roznica"}).assign(wariant="b"),
                ecl[["linia", "RBP", "klasa_niezwiazane"]].rename(columns={"RBP": "target", "klasa_niezwiazane": "roznica"}).assign(wariant="a")])
# (nazwa w tsv, dane, wariant przed, wariant po, klucz liczby lncRNA, tytul = cecha dodana do dopasowania)
COMP = [("Metabolism", rv, "A_na_zbiorze_B", "B", "B", "+ splicing,\nstability"),
        ("Localization", loc, "B", "C", "C", "+ nuclear\nenrichment"),
        ("Tissue specificity", pla, "C", "D", "D", "+ tissue specificity,\nCpG promoter"),
        ("Direct binding", e2, "b", "a", "A", "bound lncRNAs\nremoved (eCLIP)")]
fig = plt.figure(figsize=(112 * MM, 66 * MM))
gs = fig.add_gridspec(2, 4, left=0.1, right=0.99, top=0.86, bottom=0.2, wspace=0.55, hspace=0.3)
rows_a = []
for j, (name, df, a, b, key, title) in enumerate(COMP):
    wa, wb = per_rbp(df[df["wariant"] == a]), per_rbp(df[df["wariant"] == b])
    ra, rb = rho_ci(wa["K562"], wa["HepG2"]), rho_ci(wb["K562"], wb["HepG2"])
    ea, eb = df.loc[df["wariant"] == a, "roznica"].abs().median(), df.loc[df["wariant"] == b, "roznica"].abs().median()
    rows_a.append({"porownanie": name, "rho_przed": ra[0], "ci_przed": f"{ra[2]:.2f}-{ra[3]:.2f}", "rho_po": rb[0],
                   "ci_po": f"{rb[2]:.2f}-{rb[3]:.2f}", "efekt_przed": ea, "efekt_po": eb, "RBP": len(wb)})
    ax = fig.add_subplot(gs[0, j])
    for x, (r_, p_, lo_, hi_), filled in [(0, ra, False), (1, rb, True)]:
        ax.errorbar(x, r_, yerr=[[r_ - lo_], [hi_ - r_]], fmt="o", ms=3, mfc="k" if filled else "white", mec="k",
                    color="k", elinewidth=0.6, capsize=1.2, capthick=0.5)
    ax.plot([0, 1], [ra[0], rb[0]], color="0.6", lw=0.5, zorder=0)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.1, 0.55)                    # ta sama skala we wszystkich porownaniach
    ax.axhline(0, color="0.6", lw=0.4, zorder=0)
    ax.set_xticks([0, 1], ["before", "after"])
    ax.set_title(title, fontsize=6.5, pad=3)
    if j == 0:
        ax.set_ylabel("agreement between\ncell lines (ρ, 95% CI)")
    ax2 = fig.add_subplot(gs[1, j])
    ax2.bar([0, 1], [ea, eb], width=0.55, color=["white", "0.35"], edgecolor="k", lw=0.5)
    ax2.set_xlim(-0.6, 1.6); ax2.set_ylim(0, 0.11)
    ax2.set_xticks([0, 1], ["before", "after"])
    if j == 0:
        ax2.set_ylabel("typical class effect,\n|log2|")
    nrbp = f"{len(wb)} RBPs" if name != "Direct binding" else f"{len(wb)} RBPs (eCLIP)"
    ax2.text(0.5, -0.36, f"{nlab(key)}\n{nrbp}", transform=ax2.transAxes, ha="center", va="top", fontsize=6, linespacing=1.2)
save(fig, FIG, "a_dopasowania")
pd.DataFrame(rows_a).to_csv(Path("results/figures_pub") / FIG / "a_dopasowania.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- b1: zwiazane vs niezwiazane
fig, ax = panel_fixed(34, 36)
data, labs, cols = [], [], []
for cell in ["K562", "HepG2"]:
    for g_ in ["mRNA", "lncRNA"]:
        x = ecl.loc[ecl["linia"] == cell, f"abs_{g_}"].dropna().values
        data.append(x); labs.append(f"{g_}\n{cell}\n({len(x)})"); cols.append(PAL[g_])
pos = np.arange(1, len(data) + 1)
parts = ax.violinplot(data, positions=pos, widths=0.75, showextrema=False)
for pc, c in zip(parts["bodies"], cols):
    pc.set_facecolor(c); pc.set_alpha(0.5); pc.set_edgecolor("none")
for x_, d in zip(pos, data):
    q1, md, q3 = np.percentile(d, [25, 50, 75])
    ax.plot([x_, x_], [q1, q3], color="k", lw=1.2)
    ax.scatter([x_], [md], color="white", edgecolor="k", s=8, zorder=3, lw=0.5)
ax.axhline(0, color="0.4", lw=0.5, ls=(0, (2, 2)))
qb1 = bh([wilcoxon(d).pvalue for d in data])
top = max(np.percentile(d, 99) for d in data)
for x_, q_ in zip(pos, qb1):
    ax.text(x_, top + 0.02, stars(q_), ha="center", fontsize=6)
ax.set_ylim(top=top + 0.07)
ax.set_xticks(pos, labs, fontsize=6)
ax.set_ylabel("|log2FC|: bound minus matched\nunbound genes (per RBP)")
print("   b1 (2C) mediany na RBP: " + "; ".join(f"{l.replace(chr(10), ' ')}: {np.median(d):+.3f}" for l, d in zip(labs, data)))
save(fig, FIG, "b1_wiazanie")

# ---------------------------------------------------------------- b2: najsilniej reagujace
ta = pd.read_csv("results/phase4/followups/eclip_top_responders.tsv", sep="\t")
fig, ax = panel_fixed(34, 36)
k = 0
for cell in ["K562", "HepG2"]:
    for g_ in ["mRNA", "lncRNA"]:
        s = ta[(ta["linia"] == cell) & (ta["geny"] == g_)]
        orv = pd.Series(np.where(s["OR_MH"] > 0, np.log2(s["OR_MH"].clip(lower=1 / 16, upper=64)), -5.5), index=s.index)
        sig = (s["q"] < 0.05) & (s["OR_MH"] > 1)
        jit = np.random.default_rng(k).normal(0, 0.08, len(s))
        ax.scatter(k + 1 + jit[~sig.values], orv[~sig.values], s=3, color=PAL[g_], alpha=0.5, lw=0)
        ax.scatter(k + 1 + jit[sig.values], orv[sig.values], s=6, color="k", lw=0)
        ax.text(k + 1, 6.6, f"{100 * sig.mean():.0f}%", ha="center", fontsize=6)
        print(f"   b2 (2D) {g_} {cell}: wzbogacenie istotne dla {100 * sig.mean():.1f}% RBP ({int(sig.sum())} z {len(s)})")
        k += 1
ax.axhline(0, color="0.4", lw=0.5, ls=(0, (2, 2)))
ax.axhline(-4.6, color="0.7", lw=0.4)
ax.set_xticks(range(1, 5), ["mRNA\nK562", "lncRNA\nK562", "mRNA\nHepG2", "lncRNA\nHepG2"], fontsize=6)
ax.set_ylim(-6.3, 7.3)
ax.set_yticks([-5.5, -4, -2, 0, 2, 4, 6], ["none in\ntop 5%", "−4", "−2", "0", "2", "4", "6"])
ax.set_ylabel("bound genes among top 5%\nresponders (log2 odds ratio)")
save(fig, FIG, "b2_najsilniejsze")

# ---------------------------------------------------------------- c: replikacja wg polozenia
cp = pd.read_csv("results/phase4/followups/class_effect_by_position.tsv", sep="\t")
EN = {"dywergentne": "divergent", "antysensowe": "NAT", "w poblizu": "nearby", "miedzygenowe": "intergenic"}
fig, ax = panel_fixed(30, 36)
for i, r_ in enumerate(cp.itertuples()):
    ax.errorbar(i, r_.rho, yerr=[[r_.rho - r_.CI_lo], [r_.CI_hi - r_.rho]], fmt="o", ms=3, color=PAL["lncRNA"],
                elinewidth=0.6, capsize=1.2, capthick=0.5)
ax.axhline(0, color="0.4", lw=0.5)
ax.set_xticks(range(len(cp)), [EN.get(k_, k_) for k_ in cp["kategoria"]], fontsize=6, rotation=35, ha="right", rotation_mode="anchor")
ax.set_xlim(-0.6, len(cp) - 0.4)
ax.set_ylim(-0.05, 0.5)
ax.set_ylabel("agreement between\ncell lines (ρ, 95% CI)")
ax.set_xlabel("lncRNA position relative to\nprotein-coding genes")
save(fig, FIG, "c_polozenie")

# ---------------------------------------------------------------- d: powtarzalnosc profili
pr = pd.read_csv("results/phase4/network/profile_replication.tsv", sep="\t")
pr = pr[pr["wariant"] == "log2FC"]
fig, ax = panel_fixed(34, 36)
labels = {"wszystkie RBP": "all RBPs", "silne (>= 2000 genow w obu)": "strong\ndepletions\n(≥ 2,000 genes)"}
for i, (subset, g_) in enumerate(pr.groupby("RBP", sort=False)):
    for j, gname in enumerate(["lncRNA", "mRNA"]):
        r_ = g_[g_["geny"] == gname]
        if r_.empty:
            continue
        v = r_["% sam na 1. miejscu"].iloc[0]
        ax.bar(i + (j - 0.5) * 0.36, v, 0.34, color=PAL[gname], label=gname if i == 0 else None)
        nn = int(r_["n"].iloc[0]); kk = int(round(v * nn / 100))
        ax.text(i + (j - 0.5) * 0.36, v + 0.5, stars(binomtest(kk, nn, 1 / 342, alternative="greater").pvalue), ha="center", fontsize=6)
    a_, b_ = g_[g_["geny"] == "lncRNA"].iloc[0], g_[g_["geny"] == "mRNA"].iloc[0]
    ka, kb = int(round(a_["% sam na 1. miejscu"] * a_["n"] / 100)), int(round(b_["% sam na 1. miejscu"] * b_["n"] / 100))
    pf_ = fisher_exact([[ka, int(a_["n"]) - ka], [kb, int(b_["n"]) - kb]])[1]
    ytop = max(a_["% sam na 1. miejscu"], b_["% sam na 1. miejscu"]) + 3
    ax.plot([i - 0.18, i - 0.18, i + 0.18, i + 0.18], [ytop, ytop + 0.8, ytop + 0.8, ytop], color="k", lw=0.5)
    ax.text(i, ytop + 1.1, stars(pf_), ha="center", fontsize=6)
ax.axhline(100 / max(pr["n"].max(), 1), color="0.3", lw=0.5, ls=(0, (2, 2)))
ax.text(1.55, 100 / max(pr["n"].max(), 1) + 0.6, "chance", fontsize=6, ha="right")
ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
ax.set_xticks(range(pr["RBP"].nunique()), [labels.get(s_, s_) for s_ in pr["RBP"].unique()], fontsize=6)
ax.set_ylabel("RBPs whose best match\nin HepG2 is itself (%)")
ax.legend(loc="upper left", fontsize=6, handlelength=1.0)
save(fig, FIG, "d_profile")

# ---------------------------------------------------------------- podglad
try:
    from PIL import Image
    d = Path("results/figures_pub") / FIG
    a = Image.open(d / "a_dopasowania.png")
    row = [Image.open(d / f"{n}.png") for n in ["b1_wiazanie", "b2_najsilniejsze", "c_polozenie", "d_profile"]]
    W = max(a.width, sum(i.width for i in row) + 30)
    H = a.height + max(i.height for i in row) + 30
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(a, (0, 0))
    x = 0
    for im in row:
        canvas.paste(im, (x, a.height + 30)); x += im.width + 10
    canvas.save(d / "podglad2.png")
    print(f"   {FIG}/podglad2.png")
except Exception as e:  # noqa: BLE001
    print(f"   (podglad pominiety: {e})")
