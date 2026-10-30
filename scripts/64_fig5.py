"""
Rycina 5 (publikacja): swoista regulacja pojedynczych lncRNA. Panele w results/figures_pub/fig5/:
  a_topk       - odsetek genow ze wspolnym regulatorem w obu liniach (top 1 / 3 / 10), lncRNA vs mRNA o tej samej
                 ekspresji, z tlem permutacyjnym; nad grupa: iloraz szans i p (Fisher, lncRNA vs mRNA)
  b_mapa       - mapa ciepla: 22 lncRNA o swoistym, powtarzalnym regulatorze x RBP dominujace, nietypowosc zmiany (z)
                 osobno w K562 i HepG2; obwiedziona komorka = dominujacy regulator tego lncRNA
  c_NORAD      - locus NORAD (modul figlocus): SRSF1, RCC2, EIF3D w obu liniach
Runda 2 (19.10): w B komorki bez wartosci (geny w promieniu 50 kb od wyciszanego genu) szare, nie biale; czcionki >= 6 pt.
Runda 3 (20.10): locus NORAD w tym samym stylu co Ryc. 4E (sciezki 8 mm, modele genow upakowane - figlocus).
"""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from figlib import MM, PAL, panel_fixed, save, fmt_p  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402

FIG = "fig5"
M = Path("results/phase2/matrix")
MAD_FLOOR = 0.05
EN_POS = {"dywergentne": "div.", "antysensowe": "NAT", "w poblizu": "near", "miedzygenowe": "interg.", "-": ""}

# ---------------------------------------------------------------- a: top 1 / 3 / 10 (linie, skala logarytmiczna)
tk = pd.read_csv("results/phase4/followups/topk_regulators.tsv", sep="\t")
fig, ax = panel_fixed(40, 36)
ks = sorted(tk["k"].unique())
xs = np.arange(len(ks))
for g, dx in [("lncRNA", -0.06), ("mRNA", 0.06)]:
    t = tk[tk["geny"] == g].set_index("k").reindex(ks)
    ax.plot(xs + dx, t["% genow ze wspolnym regulatorem"], "-o", color=PAL[g], lw=1.0, ms=3.2, label=f"{g}s, observed")
    ax.plot(xs + dx, t["% w tle (permutacja)"], "--o", color=PAL[g], lw=0.7, ms=3.0, mfc="white", label=f"{g}s, permuted")
for i, k in enumerate(ks):
    a_ = tk[(tk["k"] == k) & (tk["geny"] == "lncRNA")].iloc[0]; b_ = tk[(tk["k"] == k) & (tk["geny"] == "mRNA")].iloc[0]
    o, p = fisher_exact([[a_["n_wspolne"], a_["n"] - a_["n_wspolne"]], [b_["n_wspolne"], b_["n"] - b_["n_wspolne"]]])
    ax.text(i, a_["% genow ze wspolnym regulatorem"] * 1.45, f"OR {o:.2f}\np = {fmt_p(p)}", ha="center", va="bottom", fontsize=6)
ax.set_yscale("log")
ax.set_ylim(0.2, 400)
ax.set_yticks([0.3, 1, 3, 10, 30, 100], ["0.3", "1", "3", "10", "30", "100"])
ax.yaxis.set_minor_locator(plt.NullLocator())
ax.set_xticks(xs, [f"top {k}" for k in ks])
ax.set_xlim(-0.4, len(ks) - 0.6)
ax.set_xlabel("regulators considered per gene")
ax.set_ylabel("genes with the same regulator\nin both cell lines (%)")
ax.legend(fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.6, columnspacing=0.6, borderaxespad=0.1)
save(fig, FIG, "a_topk")

# ---------------------------------------------------------------- b: mapa ciepla 22 lncRNA x RBP dominujace
spec = pd.read_csv("results/phase4/specific/specific_lncRNA.tsv", sep="\t", index_col=0)
spec = spec.sort_values(["rbp1_K562", "dom_min"], ascending=[True, False])
rbps = list(dict.fromkeys(spec["rbp1_K562"]))                  # kolejnosc kolumn = kolejnosc grup wierszy
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
cis = pd.read_csv("data/meta/phase2_cis_genes.tsv", sep="\t")
cis50 = cis[cis["w50kb"]].groupby("kd_experiment")["gene_id"].apply(set)
Z = {}
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    L = L.loc[L.index.intersection(spec.index)]
    for kd in L.columns.intersection(cis50.index):
        L.loc[L.index.isin(cis50[kd]), kd] = np.nan
    med = L.median(axis=1)
    mad = (1.4826 * L.sub(med, axis=0).abs().median(axis=1)).clip(lower=MAD_FLOOR)
    z = L.sub(med, axis=0).div(mad, axis=0)
    t = summ.reindex(L.columns)["target"].values
    Z[cell] = z.T.groupby(t).mean().T.reindex(index=spec.index, columns=rbps)
lim = 12
fig = plt.figure(figsize=(118 * MM, (12 + 3.0 * len(spec)) * MM))
cw, ch = 2.6, 3.0                                              # rozmiar komorki (mm)
left = 30
for p_, cell in enumerate(["K562", "HepG2"]):
    x0 = left + p_ * (len(rbps) * cw + 6)
    Htot = 12 + ch * len(spec)
    ax = fig.add_axes([x0 / 118, 6 / Htot, len(rbps) * cw / 118, len(spec) * ch / Htot])
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#BDBDBD")                                    # brak wartosci (gen w promieniu 50 kb od wyciszanego) = szary
    im = ax.imshow(np.ma.masked_invalid(Z[cell].clip(-lim, lim).values), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto",
                   interpolation="nearest")
    for i, (gid, r) in enumerate(spec.iterrows()):
        j = rbps.index(r["rbp1_K562"])
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=0.7))
    ax.set_xticks(range(len(rbps)), rbps, rotation=90, fontsize=6)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=1)
    if p_ == 0:
        ax.set_yticks(range(len(spec)), [f"{r['lncRNA']} ({EN_POS.get(r['polozenie'], '')})" if EN_POS.get(r["polozenie"], "") else r["lncRNA"]
                                         for _, r in spec.iterrows()], fontsize=6)
    else:
        ax.set_yticks([])
    ax.set_xlabel(cell, fontsize=6, labelpad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
cax = fig.add_axes([(left + 2 * len(rbps) * cw + 10) / 118, 6 / (12 + 3.0 * len(spec)), 2.2 / 118, 0.35])
cb = fig.colorbar(im, cax=cax)
cb.set_label("change vs all depletions (z)", fontsize=6)
cb.ax.tick_params(labelsize=6, length=1.5)
cb.outline.set_linewidth(0.4)
save(fig, FIG, "b_mapa")
pd.concat({c: Z[c] for c in Z}, axis=1).round(2).assign(lncRNA=spec["lncRNA"]).to_csv(Path("results/figures_pub") / FIG / "b_mapa.tsv", sep="\t")

# ---------------------------------------------------------------- c: locus NORAD
try:
    from figlocus import draw_locus, genes as G_
    names = [n for n in ["NORAD", "LINC00657"] if n in set(G_["gene_name"])]
    draw_locus(FIG, "c_NORAD", names, names[0],
               [("SRSF1", "K562"), ("SRSF1", "HepG2"), ("RCC2", "K562"), ("RCC2", "HepG2"), ("EIF3D", "K562"), ("EIF3D", "HepG2")],
               track_mm=8.0, track_gap=3.0)          # runda 3: jak locus na Ryc. 4E
except Exception as e:  # noqa: BLE001
    print(f"   (locus pominiety: {e})")

# ---------------------------------------------------------------- podglad
try:
    from PIL import Image
    d = Path("results/figures_pub") / FIG
    a, b = Image.open(d / "a_topk.png"), Image.open(d / "b_mapa.png")
    c = Image.open(d / "c_NORAD.png") if (d / "c_NORAD.png").exists() else None
    W = max(a.width + b.width + 20, c.width if c else 0)
    h1 = max(a.height, b.height)
    canvas = Image.new("RGB", (W, h1 + (c.height + 30 if c else 0)), "white")
    canvas.paste(a, (0, 0)); canvas.paste(b, (a.width + 20, 0))
    if c:
        canvas.paste(c, (0, h1 + 30))
    canvas.save(d / "podglad5.png")
    print(f"   {FIG}/podglad5.png")
except Exception as e:  # noqa: BLE001
    print(f"   (podglad pominiety: {e})")
