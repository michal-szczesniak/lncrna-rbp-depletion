"""
Rycina 3 (publikacja): obrobka RNA - spliceosom, NMD i gospodarze snoRNA. Panele w results/figures_pub/fig3/:
  a_kategorie     - class effect grup funkcjonalnych RBP (mediana grupy vs pozostale RBP; q)
  b_spliceosom    - niewyciete RNA (introny) po wyciszeniu spliceosomu, mRNA vs lncRNA w 5 przedzialach ekspresji;
                    gwiazdki: test Wilcoxona par (mediany mRNA i lncRNA tego samego eksperymentu), korekta BH
  c_szybkosc      - to samo w przedzialach szybkosci splicingu w kontrolach (z kroku 4.5)
  d_nmd           - dojrzale RNA (eksony) po wyciszeniu NMD (UPF, SMG), mRNA vs lncRNA wg ekspresji
  e1..e4_sno      - gospodarze snoRNA: lokalizacja, swoistosc tkankowa, promotory CpG, wrazliwosc na NMD/EJC;
                    nawiasy: gospodarze snoRNA vs mRNA i vs lincRNA (Mann-Whitney / Fisher), BH lacznie dla e1-e4
Runda 2 (19.10): w A kolory niezalezne od linii komorkowych (czarny = istotna kategoria, bialy = nieistotna, szary =
pozostale RBP); B i C na tej samej skali osi y; testy z B zapisane w b_testy.tsv (z medianami); czcionki >= 6 pt.
Runda 8 (28.10): w e3 (Ryc. 3G) slupki maja 95% przedzialy ufnosci Wilsona dla odsetka promotorow z wyspa CpG;
liczby genow, odsetki i przedzialy zapisane w e3_cpg_ci.tsv.
"""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from figlib import MM, PAL, bh, panel_fixed, save, stars  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import fisher_exact, mannwhitneyu, wilcoxon  # noqa: E402

FIG = "fig3"
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz miRNA"}
NMD = {"UPF1", "UPF2", "UPF3B", "SMG1", "SMG5", "SMG6", "SMG7", "SMG8", "SMG9"}
EJC = {"RBM8A", "MAGOH", "EIF4A3", "CASC3"}
LS = {"K562": "-", "HepG2": "--"}


def class_leg(ax, cells):
    """Legenda: kolor = klasa RNA, styl linii = linia komorkowa."""
    h = [plt.Line2D([], [], color=PAL["mRNA"], lw=1.2, label="mRNA"), plt.Line2D([], [], color=PAL["lncRNA"], lw=1.2, label="lncRNA")]
    h += [plt.Line2D([], [], color="k", lw=0.8, ls=LS[c], label=lab) for c, lab in cells.items()]
    ax.legend(handles=h, fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=2.2, columnspacing=0.8,
              borderaxespad=0.1)


def brk(ax, x1, x2, y, text, dy):
    ax.plot([x1, x1, x2, x2], [y, y + dy, y + dy, y], color="k", lw=0.5)
    ax.text((x1 + x2) / 2, y + dy * 1.3, text, ha="center", va="bottom", fontsize=6)


# ---------------------------------------------------------------- a: kategorie funkcjonalne
cat = pd.read_csv("results/phase2/classes/functional_categories.tsv", sep="\t")
cf = cat[cat["typ"] == "funkcja"].sort_values("mediana").reset_index(drop=True)
fig, ax = panel_fixed(36, 4.2 * len(cf) + 4)
y = np.arange(len(cf))
ax.hlines(y, cf["mediana_pozostalych"], cf["mediana"], color="0.75", lw=0.6)
ax.scatter(cf["mediana_pozostalych"], y, s=5, color="0.6", lw=0, zorder=2)
sig = cf["q"] < 0.05
# kolory niezalezne od linii komorkowych: czarny = mediana kategorii rozna od pozostalych RBP (q < 0,05), bialy = nie
ax.scatter(cf.loc[~sig, "mediana"], y[~sig], s=12, facecolor="white", edgecolor="k", lw=0.6, zorder=3)
ax.scatter(cf.loc[sig, "mediana"], y[sig], s=14, color="k", lw=0, zorder=3)
for yi, r_ in zip(y, cf.itertuples()):
    if r_.q < 0.05:
        ax.text(max(r_.mediana, r_.mediana_pozostalych) + 0.004, yi, stars(r_.q), va="center", fontsize=6)
ax.set_yticks(y, [f"{c} ({n})" for c, n in zip(cf["kategoria"], cf["n"])], fontsize=6)
ax.axvline(0, color="0.4", lw=0.4)
ax.set_xlabel("median class effect (log2)")
ax.set_ylim(-0.7, len(cf) - 0.3)
save(fig, FIG, "a_kategorie")

# ---------------------------------------------------------------- dane EISA na eksperyment
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
eb = pd.concat(rows)


def bins_panel(grupa, comp, ylab, name, ylim=None):
    fig, ax = panel_fixed(40, 36)
    tests = []
    nk = {}
    for cell in ["K562", "HepG2"]:
        s = eb[(eb["grupa"] == grupa) & (eb["linia"] == cell)]
        nkd = s["kd"].nunique()
        nk[cell] = nkd
        if nkd == 0:
            continue
        for k in ["mRNA", "lncRNA"]:
            m = s[s["k"] == k].groupby("bin", observed=True)[comp].median()
            ax.plot(m.index.astype(int), m.values, ls=LS[cell], marker="o", ms=2.5, lw=0.8, color=PAL[k],
                    )
        w = s.pivot_table(index=["kd", "bin"], columns="k", values=comp, observed=True).dropna().reset_index()
        for b, g in w.groupby("bin", observed=True):
            if len(g) >= 5:
                tests.append((cell, int(b), wilcoxon(g["mRNA"], g["lncRNA"]).pvalue,
                              max(g["mRNA"].median(), g["lncRNA"].median()), g["mRNA"].median(), g["lncRNA"].median(),
                              (g["mRNA"] - g["lncRNA"]).median(), len(g)))
    if ylim is not None:
        ax.set_ylim(*ylim)
    if tests:
        q = bh([t[2] for t in tests])
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        for (cell, b, _, ytop, *_r), q_ in zip(tests, q):
            ax.text(b + (0.12 if cell == "HepG2" else -0.12), ytop + 0.06 * span, stars(q_), ha="center", fontsize=6, color=PAL[cell])
        if ylim is None:
            ax.set_ylim(top=ax.get_ylim()[1] + 0.1 * span)
        pd.DataFrame([{"linia": t[0], "przedzial": t[1], "p": t[2], "q": q_, "mediana_mRNA": t[4], "mediana_lncRNA": t[5],
                       "mediana_roznicy_mRNA_minus_lncRNA": t[6], "eksperymenty": t[7]} for t, q_ in zip(tests, q)]).to_csv(
            Path("results/figures_pub") / FIG / f"{name}_testy.tsv", sep="\t", index=False)
    ax.axhline(0, color="0.4", lw=0.4)
    ax.set_xticks([1, 2, 3, 4, 5], ["low", "2", "3", "4", "high"])
    ax.set_xlabel("expression in control cells (mRNA quintiles)")
    ax.set_ylabel(ylab)
    class_leg(ax, {c: f"{c} ({n} exp.)" for c, n in nk.items() if n})
    ylim_out = ax.get_ylim()
    save(fig, FIG, name)
    return tests, ylim_out


tests_b, YLIM_B = bins_panel("spliceosome", "d_intron", "unspliced RNA after spliceosome\ndepletion (median log2 change)", "b_spliceosom")
print("   testy b:", len(tests_b), "; zakres osi y (wspolny z c):", np.round(YLIM_B, 3))
# d: kazde wyciszenie czynnika NMD osobno (wynik kroku 4.26), z oznaczeniem skutecznosci dla NMD
nm = pd.read_csv("results/phase4/followups/nmd_skuteczne.tsv", sep="\t")
nm["linia"] = nm["wyciszenie"].str.split().str[-1]
nm = nm.sort_values(["skuteczne_NMD", "srednia_roznica"], ascending=[True, True]).reset_index(drop=True)
fig, ax = panel_fixed(36, 3.6 * len(nm) + 4)
bcols = [c for c in nm.columns if c.startswith("lnc-mRNA_przedz")]
for i, r in nm.iterrows():
    v = r[bcols].astype(float).values
    ax.scatter(v, np.full(len(v), i), s=3, color="0.65", lw=0, zorder=2)
    col = PAL[r["linia"]] if r["linia"] in PAL else "k"
    ax.scatter([r["srednia_roznica"]], [i], s=16, zorder=3, lw=0.8,
               facecolor=col if r["skuteczne_NMD"] else "white", edgecolor=col)
    ax.text(1.02, i, f"{int(r['przedzialy_lnc>mRNA_q<0.05'])}/{len(bcols)}", transform=ax.get_yaxis_transform(), va="center", fontsize=6)
ax.axvline(0, color="0.4", lw=0.4)
ax.set_yticks(range(len(nm)), nm["wyciszenie"], fontsize=6)
ax.set_ylim(-0.7, len(nm) - 0.3)
ax.set_xlabel("mature RNA after depletion:\nlncRNA minus mRNA (log2)")
ax.text(1.02, len(nm) - 0.2, "bins\nq<0.05", transform=ax.get_yaxis_transform(), va="bottom", fontsize=6)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", mfc="k", mec="k", markersize=3.5, label="raises NMD targets"),
                   plt.Line2D([], [], marker="o", ls="", mfc="white", mec="k", markersize=3.5, label="does not"),
                   plt.Line2D([], [], marker="o", ls="", color="0.65", markersize=2, label="expression bins"),
                   plt.Line2D([], [], marker="o", ls="", color=PAL["K562"], markersize=3.5, label="K562"),
                   plt.Line2D([], [], marker="o", ls="", color=PAL["HepG2"], markersize=3.5, label="HepG2")],
          fontsize=6, loc="lower left", bbox_to_anchor=(-0.45, 1.1), ncol=2, handletextpad=0.1, columnspacing=0.8, borderaxespad=0.1)
save(fig, FIG, "d_nmd")

# ---------------------------------------------------------------- c: ta sama szybkosc splicingu
sp = pd.read_csv("results/phase4/stats/spliceosome_by_splicing_efficiency.tsv", sep="\t")
fig, ax = panel_fixed(40, 36)
for cell in ["K562", "HepG2"]:
    s = sp[sp["linia"] == cell].reset_index(drop=True)
    xs = np.arange(1, len(s) + 1)
    ax.plot(xs, s["kumulacja intronow mRNA"], ls=LS[cell], marker="o", ms=2.5, lw=0.8, color=PAL["mRNA"], label=f"mRNA, {cell}")
    ax.plot(xs, s["kumulacja intronow lncRNA"], ls=LS[cell], marker="o", ms=2.5, lw=0.8, color=PAL["lncRNA"], label=f"lncRNA, {cell}")
span = ax.get_ylim()[1] - ax.get_ylim()[0]
for cell in ["K562", "HepG2"]:
    s = sp[sp["linia"] == cell].reset_index(drop=True)
    for x_, r_ in zip(np.arange(1, len(s) + 1), s.itertuples()):
        ax.text(x_ + (0.12 if cell == "HepG2" else -0.12), max(r_[5], r_[6]) + 0.06 * span, stars(r_.q), ha="center", fontsize=6, color=PAL[cell])
ax.set_ylim(*YLIM_B)                                   # ta sama skala co panel B
ax.set_xticks([1, 2, 3, 4, 5], ["most\nefficient", "2", "3", "4", "least\nefficient"])
ax.set_xlabel("splicing efficiency in control cells\n(mRNA quintiles)")
ax.set_ylabel("unspliced RNA after spliceosome\ndepletion (median log2 change)")
class_leg(ax, {"K562": "K562", "HepG2": "HepG2"})
save(fig, FIG, "c_szybkosc")

# ---------------------------------------------------------------- e: gospodarze snoRNA
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
cls["gene"] = cls["gene_id"].str.split(".").str[0]
k_of = cls.drop_duplicates("gene").set_index("gene")["klasa"]
order = ["mRNA", "lincRNA", "antysens", "inne lncRNA", "gospodarz miRNA", "gospodarz snoRNA"]
short = ["mRNA", "lincRNA", "NAT", "other", "miRNA\nhost", "snoRNA\nhost"]
colors = [PAL["mRNA"]] + [PAL["lncRNA"]] * 4 + [PAL["sno"]]
nuc = pd.read_csv("results/phase3/localization/nuclear_index.tsv", sep="\t", index_col=0)
pf = pd.read_csv("results/phase3/plasticity/plasticity_features.tsv", sep="\t", index_col=0)
REF = ["mRNA", "lincRNA"]
panels, tests = [], []


def violins(ax, data, pos, cols):
    parts = ax.violinplot([d if len(d) > 1 else np.array([0, 0]) for d in data], positions=pos, widths=0.8, showextrema=False)
    for pc, c in zip(parts["bodies"], cols):
        pc.set_facecolor(c); pc.set_alpha(0.55); pc.set_edgecolor("none")
    for x_, d in zip(pos, data):
        if len(d) > 1:
            q1, md, q3 = np.percentile(d, [25, 50, 75])
            ax.plot([x_, x_], [q1, q3], color="k", lw=1.0)
            ax.scatter([x_], [md], color="white", edgecolor="k", s=6, lw=0.4, zorder=3)


for key, vals, ylab in [("e1_lokalizacja", nuc["HepG2"], "log2 nucleus / cytoplasm\n(HepG2)"),
                        ("e2_tkanki", pf["tau"], "tissue specificity, tau\n(GTEx; 1 = one tissue)")]:
    kk = k_of.reindex(vals.index)
    data = [vals[(kk == c).values].dropna().values for c in order]
    fig, ax = panel_fixed(36, 36)
    violins(ax, data, np.arange(1, 7), colors)
    ax.set_xticks(np.arange(1, 7), [x.replace("\n", " ") for x in short], fontsize=6, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylab)
    for ref in REF:
        i = order.index(ref)
        tests.append((key, i + 1, 6, mannwhitneyu(data[i], data[-1]).pvalue))
    panels.append((key, fig, ax))
def wilson(k, n, z=1.959964):
    """95% przedzial ufnosci Wilsona dla odsetka k/n (k, n: tablice)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


cg = pf.groupby("klasa")["CGI"].agg(["sum", "count"]).reindex(order)
pct = (100 * cg["sum"] / cg["count"]).values
ci_lo, ci_hi = wilson(cg["sum"].values, cg["count"].values)
fig, ax = panel_fixed(36, 36)
ax.bar(np.arange(1, 7), pct, color=colors, alpha=0.85, width=0.7,
       yerr=[pct - 100 * ci_lo, 100 * ci_hi - pct], error_kw=dict(ecolor="k", elinewidth=0.6, capsize=1.5, capthick=0.6))
pd.DataFrame({"klasa": order, "geny": cg["count"].values, "z_wyspa_CpG": cg["sum"].values, "procent": pct,
              "CI95_dolny": 100 * ci_lo, "CI95_gorny": 100 * ci_hi}).to_csv(
    Path("results/figures_pub") / FIG / "e3_cpg_ci.tsv", sep="\t", index=False)
print("   e3_cpg: odsetek promotorow z wyspa CpG (95% CI Wilsona):")
for k_, n_, s_, p_, l_, h_ in zip(order, cg["count"].values, cg["sum"].values, pct, 100 * ci_lo, 100 * ci_hi):
    print(f"      {k_:18s} {int(s_):6d} / {int(n_):6d} = {p_:5.1f}%  (95% CI {l_:5.1f}-{h_:5.1f})")
ax.set_xticks(np.arange(1, 7), [x.replace("\n", " ") for x in short], fontsize=6, rotation=40, ha="right", rotation_mode="anchor")
ax.set_ylabel("promoters with a\nCpG island (%)")
for ref in REF:
    i = order.index(ref)
    a, b = cg.loc[ref], cg.loc["gospodarz snoRNA"]
    tests.append(("e3_cpg", i + 1, 6, fisher_exact([[a["sum"], a["count"] - a["sum"]], [b["sum"], b["count"] - b["sum"]]])[1]))
panels.append(("e3_cpg", fig, ax))
fig, ax = panel_fixed(36, 36)
data, pos, cols, labs = [], [], [], []
for i, cell in enumerate(["K562", "HepG2"]):
    c_ = pd.read_csv(f"results/phase3/splice_nmd/consensus_{cell}.tsv.gz", sep="\t", index_col=0)
    c_ = c_[c_["klasa"].notna()]
    ln = c_.loc[c_["klasa"].isin(LNC), "nmd"].dropna().values
    sn = c_.loc[c_["klasa"] == "gospodarz snoRNA", "nmd"].dropna().values
    data += [ln, sn]; pos += [2 * i + 1, 2 * i + 1.9]; cols += [PAL["lncRNA"], PAL["sno"]]
    labs += [f"other\nlncRNAs\n{cell}", f"snoRNA\nhosts\n{cell}"]
    tests.append(("e4_nmd", 2 * i + 1, 2 * i + 1.9, mannwhitneyu(ln, sn).pvalue))
violins(ax, data, pos, cols)
ax.axhline(0, color="0.4", lw=0.4)
ax.set_xticks(pos, labs, fontsize=6)
ax.set_ylabel("change after NMD/EJC factor\ndepletion (median log2)")
panels.append(("e4_nmd", fig, ax))
q = bh([t[3] for t in tests])
for key, fig, ax in panels:
    lo, hi = ax.get_ylim()
    own = [(t, q_) for t, q_ in zip(tests, q) if t[0] == key]
    ax.set_ylim(lo, hi + 0.12 * (hi - lo) * (len(own) + 1))
    for lev, (t, q_) in enumerate(own):
        brk(ax, t[1], t[2], hi + (0.05 + 0.12 * lev) * (hi - lo), stars(q_), 0.025 * (hi - lo))
    save(fig, FIG, key)
pd.DataFrame([{"panel": t[0], "porownanie": f"{t[1]} vs {t[2]}", "p": t[3], "q": q_} for t, q_ in zip(tests, q)]).to_csv(
    Path("results/figures_pub") / FIG / "e_testy.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- podglad
try:
    from PIL import Image
    d = Path("results/figures_pub") / FIG
    r1 = [Image.open(d / f"{n}.png") for n in ["a_kategorie", "b_spliceosom", "c_szybkosc", "d_nmd"]]
    r2 = [Image.open(d / f"{n}.png") for n in ["e1_lokalizacja", "e2_tkanki", "e3_cpg", "e4_nmd"]]
    W = max(sum(i.width for i in r1), sum(i.width for i in r2)) + 40
    h1, h2 = max(i.height for i in r1), max(i.height for i in r2)
    canvas = Image.new("RGB", (W, h1 + h2 + 30), "white")
    x = 0
    for im in r1:
        canvas.paste(im, (x, h1 - im.height)); x += im.width + 10
    x = 0
    for im in r2:
        canvas.paste(im, (x, h1 + 30 + h2 - im.height)); x += im.width + 10
    canvas.save(d / "podglad3.png")
    print(f"   {FIG}/podglad3.png")
except Exception as e:  # noqa: BLE001
    print(f"   (podglad pominiety: {e})")
