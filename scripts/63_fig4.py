"""
Rycina 4 (publikacja): nadzor jadrowy i lncRNA dywergentne. Panele w results/figures_pub/fig4/:
  a1_K562, a2_HepG2 - class effect kategorii polozenia lncRNA: wszystkie pozostale RBP (pudelko) vs nadzor jadrowy (punkty);
                      gwiazdki: Mann-Whitney z korekta BH (z kroku 5.1)
  b_nadmiar         - RBP, ktorych wyciszenie najbardziej podnosi lncRNA dywergentne wzgledem miedzygenowych
  c_regulatory      - RBP najczesciej bedace wspolnym regulatorem (top 10) lncRNA vs mRNA o tej samej ekspresji;
                      gwiazdki: Fisher (lncRNA vs mRNA), BH
  d_KCNQ1OT1        - locus KCNQ1OT1 (modul figlocus): pokrycie i poziom genu w replikach
  e1/e2_sasiedzi    - wspolodpowiedz sasiadow (r) vs losowe pary o tej samej ekspresji, wykresy skrzypcowe, Mann-Whitney z BH;
                      powtarzalnosc miedzy liniami zapisana w e_powtarzalnosc.tsv
Runda 2 (19.10): A i B na wspolnej skali osi y; czcionki >= 6 pt (figlib.MIN_FS).
Runda 3 (20.10): uklad bez zmniejszania w druku: D (wspolne regulatory) z legenda nad osia (nie zaslania slupkow),
E (locus) wezszy (110 mm) obok D, F i G szersze (podpisy osi x nie nachodza na siebie).
"""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from figlib import PAL, panel_fixed, save, stars  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

FIG = "fig4"
SURV = {"MTREX", "EXOSC1", "EXOSC2", "EXOSC3", "EXOSC4", "EXOSC5", "EXOSC6", "EXOSC7", "EXOSC8", "EXOSC9", "EXOSC10",
        "DIS3", "ZCCHC8", "RBM7", "ZFC3H1", "PABPN1"}
CATS = ["dywergentne", "antysensowe", "w poblizu", "miedzygenowe"]
CAT_EN = ["divergent", "NAT", "nearby", "intergenic"]

# ---------------------------------------------------------------- a: kategorie, nadzor vs pozostale (wykresy skrzypcowe)
ce = pd.read_csv("results/phase4/network/category_effects_per_kd.tsv", sep="\t")
sv = pd.read_csv("results/phase4/network/surveillance_vs_others.tsv", sep="\t")
# wspolna skala osi y dla K562 i HepG2 (porownanie miedzy liniami)
YMAX = max(max(ce.loc[ce["linia"] == c_, k_].quantile(0.995) for k_ in CATS) for c_ in ["K562", "HepG2"])
if ce["nadzor"].any():
    YMAX = max(YMAX, ce.loc[ce["nadzor"], CATS].max().max())
YMIN = min(ce[CATS].min().min(), -0.05)
for key, cell in [("a1_K562", "K562"), ("a2_HepG2", "HepG2")]:
    g = ce[ce["linia"] == cell]
    fig, ax = panel_fixed(40, 36)
    ymax = YMAX
    for i, c in enumerate(CATS):
        oth = g.loc[~g["nadzor"], c].dropna().values
        parts = ax.violinplot([oth], positions=[i], widths=0.7, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#CFCFCF"); pc.set_alpha(0.9); pc.set_edgecolor("none")
        q1, md, q3 = np.percentile(oth, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color="k", lw=0.9)
        ax.scatter([i], [md], color="white", edgecolor="k", s=5, lw=0.4, zorder=3)
        sur = g.loc[g["nadzor"], c].dropna().values
        ax.scatter(i + np.linspace(-0.12, 0.12, len(sur)) if len(sur) > 1 else np.full(len(sur), i), sur,
                   s=8, color=PAL["surv"], edgecolor="white", lw=0.3, zorder=4)
        q = sv[(sv["linia"] == cell) & (sv["kategoria"] == c)]["q"]
        if len(q):
            ax.text(i, ymax * 1.08 + 0.02, stars(q.iloc[0]), ha="center", fontsize=6)
    ax.axhline(0, color="0.4", lw=0.4)
    ax.set_ylim(YMIN * 1.05, ymax * 1.08 + 0.1)
    ax.set_xticks(range(4), CAT_EN, fontsize=6)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel(f"class effect of lncRNA\ncategory, {cell} (log2)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#CFCFCF"),
                       plt.Line2D([], [], marker="o", ls="", color=PAL["surv"], markersize=3)],
              labels=["all other RBPs", "nuclear surveillance RBPs"], fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=1, handlelength=1.0, columnspacing=0.6, borderaxespad=0.1)
    save(fig, FIG, key)

# ---------------------------------------------------------------- b: nadmiar dywergentnych
ce["nadmiar"] = ce["dywergentne"] - ce["miedzygenowe"]
w = ce.groupby(["target", "linia"])["nadmiar"].mean().unstack("linia").dropna()
w["mean"] = w.mean(axis=1)
top = w.sort_values("mean", ascending=False).head(15)[::-1]
fig, ax = panel_fixed(32, 56)
yy = np.arange(len(top))
ax.barh(yy, top["mean"], height=0.65, color=[PAL["surv"] if t in SURV else "#BDBDBD" for t in top.index], lw=0)
ax.scatter(top["K562"], yy, s=6, color=PAL["K562"], lw=0, zorder=3, label="K562")
ax.scatter(top["HepG2"], yy, s=6, color=PAL["HepG2"], lw=0, zorder=3, label="HepG2")
ax.set_yticks(yy, top.index, fontsize=6)
ax.axvline(0, color="0.4", lw=0.4)
ax.set_xlabel("divergent minus intergenic\nlncRNA effect (log2)")
ax.legend(fontsize=6, loc="lower right", handletextpad=0.1, borderaxespad=0.1)
save(fig, FIG, "b_nadmiar")

# ---------------------------------------------------------------- c: wspolne regulatory (top 10)
h = pd.read_csv("results/phase4/followups/topk_hubs_k10.tsv", sep="\t", index_col=0).head(12)[::-1]
fig, ax = panel_fixed(32, 46)
yy = np.arange(len(h))
ax.barh(yy + 0.2, h["lncRNA"], height=0.38, color=PAL["lncRNA"], lw=0, label="lncRNAs")
ax.barh(yy - 0.2, h["mRNA"], height=0.38, color=PAL["mRNA"], lw=0, label="mRNAs, matched")
for y_, r_ in zip(yy, h.itertuples()):
    if r_.q < 0.05:
        ax.text(max(r_.lncRNA, r_.mRNA) + 1.5, y_, stars(r_.q), va="center", fontsize=6)
ax.set_yticks(yy, h.index, fontsize=6)
ax.set_xlabel("genes with this RBP among their\ntop 10 regulators in both lines")
ax.legend(fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, handlelength=1.0, borderaxespad=0.1)
save(fig, FIG, "c_regulatory")

# ---------------------------------------------------------------- e: wspolodpowiedz sasiadow (wykresy skrzypcowe)
from figlib import bh  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
TYPES = [("lncRNA-mRNA", "dywergentne", "lncRNA–mRNA\ndivergent"), ("lncRNA-mRNA", "antysensowe", "lncRNA–mRNA\nNAT"),
         ("mRNA-mRNA", "dywergentne", "mRNA–mRNA\ndivergent")]
# runda 3: os 56 mm (ok. 18 mm na kategorie; podpis "lncRNA–mRNA" w 6 pt ma ok. 15 mm)
pr = pd.read_csv("results/phase4/network/neighbour_coresponse.tsv.gz", sep="\t")
wv = pr.pivot_table(index=["gene_id_a", "gene_id_b", "para", "typ"], columns="linia", values="r").dropna().reset_index()
rep = {(pa, ty): spearmanr(g["K562"], g["HepG2"])[0] for (pa, ty), g in wv.groupby(["para", "typ"])}
for key, cell in [("e1_sasiedzi_K562", "K562"), ("e2_sasiedzi_HepG2", "HepG2")]:
    fig, ax = panel_fixed(56, 36)
    tests = []
    for i, (pa, ty, lab) in enumerate(TYPES):
        g = pr[(pr["linia"] == cell) & (pr["para"] == pa) & (pr["typ"] == ty)]
        if g.empty:
            continue
        for x_, vals, col in [(i - 0.2, g["r"].dropna().values, PAL[cell]), (i + 0.2, g["r_tlo"].dropna().values, "#BDBDBD")]:
            parts = ax.violinplot([vals], positions=[x_], widths=0.36, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(col); pc.set_alpha(0.7); pc.set_edgecolor("none")
            q1, md, q3 = np.percentile(vals, [25, 50, 75])
            ax.plot([x_, x_], [q1, q3], color="k", lw=0.9)
            ax.scatter([x_], [md], color="white", edgecolor="k", s=5, lw=0.4, zorder=3)
        tests.append((i, mannwhitneyu(g["r"].dropna(), g["r_tlo"].dropna()).pvalue, len(g)))
    for (i, _, n), q_ in zip(tests, bh([t[1] for t in tests])):
        ax.plot([i - 0.2, i - 0.2, i + 0.2, i + 0.2], [0.88, 0.91, 0.91, 0.88], color="k", lw=0.5)
        ax.text(i, 0.93, stars(q_), ha="center", fontsize=6)
        ax.text(i, -0.97, f"n={n}", ha="center", fontsize=6, color="0.3")
    ax.axhline(0, color="0.4", lw=0.4)
    ax.set_ylim(-1.02, 1.08); ax.set_xlim(-0.55, 2.55)
    ax.set_xticks(range(len(TYPES)), [t[2] for t in TYPES], fontsize=6)
    ax.set_ylabel(f"co-response of neighbours\nacross depletions, {cell} (r)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PAL[cell], alpha=0.7), plt.Rectangle((0, 0), 1, 1, color="#BDBDBD", alpha=0.7)],
              labels=["neighbour pairs", "random pairs, matched expression"], fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=2, handlelength=1.0, columnspacing=0.6, borderaxespad=0.1)
    save(fig, FIG, key)
pd.DataFrame([{"para": pa, "typ": ty, "powtarzalnosc_rho": r_} for (pa, ty), r_ in rep.items()]).to_csv(
    Path("results/figures_pub") / FIG / "e_powtarzalnosc.tsv", sep="\t", index=False)
print("   powtarzalnosc wspolodpowiedzi miedzy liniami: " + "; ".join(f"{pa} {ty} rho={r_:.2f}" for (pa, ty), r_ in rep.items()))

# ---------------------------------------------------------------- d: locus KCNQ1OT1
try:
    from figlocus import draw_locus
    TRK = [("MTREX", "HepG2"), ("EXOSC2", "HepG2"), ("EXOSC5", "HepG2"), ("MTREX", "K562")]
    draw_locus(FIG, "d_KCNQ1OT1", ["KCNQ1OT1"], "KCNQ1OT1", TRK, width_mm=110, track_mm=8.0, track_gap=3.0)
    from figlocus import gene_profile
    prof = gene_profile("KCNQ1OT1", TRK, nbins=10)
    prof.to_csv(Path("results/figures_pub") / FIG / "d_KCNQ1OT1_profil.tsv", sep="\t", index=False)
    print("   Profil KCNQ1OT1 od TSS (odc1) do konca (odc10), sredni sygnal RPM:")
    print(prof.round(2).to_string(index=False))
except Exception as e:  # noqa: BLE001
    print(f"   (locus pominiety: {e})")

# ---------------------------------------------------------------- podglad
try:
    from PIL import Image
    d = Path("results/figures_pub") / FIG
    r1 = [Image.open(d / f"{n}.png") for n in ["a1_K562", "a2_HepG2", "b_nadmiar", "c_regulatory", "e1_sasiedzi_K562", "e2_sasiedzi_HepG2"] if (d / f"{n}.png").exists()]
    loc = Image.open(d / "d_KCNQ1OT1.png") if (d / "d_KCNQ1OT1.png").exists() else None
    W = max(sum(i.width for i in r1) + 50, loc.width if loc else 0)
    h1 = max(i.height for i in r1)
    canvas = Image.new("RGB", (W, h1 + (loc.height + 30 if loc else 0)), "white")
    x = 0
    for im in r1:
        canvas.paste(im, (x, h1 - im.height)); x += im.width + 10
    if loc:
        canvas.paste(loc, (0, h1 + 30))
    canvas.save(d / "podglad4.png")
    print(f"   {FIG}/podglad4.png")
except Exception as e:  # noqa: BLE001
    print(f"   (podglad pominiety: {e})")
