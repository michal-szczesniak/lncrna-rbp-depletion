"""
Ryciny uzupelniajace S3 i S4 (wyniki negatywne).
  figS3/a_korekta    - pary RBP-lncRNA z literatury: log2FC w DESeq2 (wzgledem wlasnej kontroli) vs log2FC po korekcie
                       na wspolna kontrole; kazdy punkt = para w jednym eksperymencie; kolor = padj < 0.05 w DESeq2
  figS3/b_centyle    - pozycja kazdej pary wsrod wszystkich wyciszen w linii (centyl), z zaznaczonymi skrajnymi 5%
  figS4/a_lnc, b_mrna - podobienstwo par RBP w K562 vs HepG2 (siec z lncRNA / z mRNA o tej samej ekspresji), gestosc
  figS4/c_wlinii     - podobienstwo par RBP z lncRNA vs z mRNA w tej samej linii (K562)
Runda 2 (19.10): bez par z XIST w HepG2 (linia meska, XIST nieobecny); czcionki >= 6 pt (figlib.MIN_FS).
"""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from figlib import MM, PAL, panel_fixed, place_labels, save, fmt_p  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

# ================================================================ S3
kp = pd.read_csv("results/phase4/cases/known_pairs.tsv", sep="\t")
EXCLUDE = {("ELAVL1", "NORAD"), ("STAU1", "TUG1"), ("IGF2BP2", "H19"), ("IGF2BP3", "H19")}   # brak zrodla w literaturze (rejestr X07, X08; tabela S2)
kp = kp[~kp.apply(lambda r: (r["RBP"], r["lncRNA"]) in EXCLUDE, axis=1)]
kp = kp[~((kp["lncRNA"] == "XIST") & (kp["linia"] == "HepG2"))]                 # HepG2 to linia meska: XIST nie jest wyrazany
kp = kp.dropna(subset=["log2FC_DESeq2", "log2FC_skoryg"]).copy()
print(f"   S3: {kp.groupby(['RBP', 'lncRNA']).ngroups} par po wykluczeniu par bez zrodla")
kp["para"] = kp["RBP"] + " → " + kp["lncRNA"]
kp["sig"] = kp["padj_DESeq2"] < 0.05
fig, ax = panel_fixed(42, 42)
lim = np.ceil(max(kp[["log2FC_DESeq2", "log2FC_skoryg"]].abs().max().max(), 0.5) * 2) / 2 + 0.25
ax.axhspan(-0.15, 0.15, color="#F0F0F0", lw=0, zorder=0)
ax.plot([-lim, lim], [-lim, lim], color="0.7", lw=0.5, ls=(0, (2, 2)), zorder=1)
ax.axhline(0, color="0.55", lw=0.4, zorder=1); ax.axvline(0, color="0.55", lw=0.4, zorder=1)
for cell, mk in [("K562", "o"), ("HepG2", "s")]:
    s_ = kp[kp["linia"] == cell]
    ax.scatter(s_.loc[~s_["sig"], "log2FC_DESeq2"], s_.loc[~s_["sig"], "log2FC_skoryg"], s=9, marker=mk, facecolor="white",
               edgecolor=PAL[cell], lw=0.6, zorder=3)
    ax.scatter(s_.loc[s_["sig"], "log2FC_DESeq2"], s_.loc[s_["sig"], "log2FC_skoryg"], s=9, marker=mk, color=PAL[cell], lw=0, zorder=3)
art = kp[(kp["sig"]) & (kp["log2FC_DESeq2"].abs() >= 0.4) & (kp["log2FC_skoryg"].abs() < 0.15)]       # efekt znika po korekcie
strong = kp.reindex(kp["log2FC_skoryg"].abs().sort_values(ascending=False).index).head(3)            # efekt zostaje
lab = pd.concat([art.reindex(art["log2FC_DESeq2"].abs().sort_values(ascending=False).index).head(4), strong]).drop_duplicates()
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
# etykiety w kolumnie po prawej stronie, z liniami do punktow
if len(lab):
    pts = ax.transAxes.inverted().transform(ax.transData.transform(lab[["log2FC_DESeq2", "log2FC_skoryg"]].values))
    order_ = np.argsort(-pts[:, 1])
    ys = pts[order_, 1].copy()
    for _ in range(200):
        for i in range(1, len(ys)):
            if ys[i - 1] - ys[i] < 0.075:
                ys[i] = ys[i - 1] - 0.075
        ys = np.clip(ys, -0.05, 1.05)
    for (x0, y0), yl, idx in zip(pts[order_], ys, order_):
        r_ = lab.iloc[idx]
        ax.annotate(f"{r_['para']} ({r_['linia']})", xy=(x0, y0), xycoords="axes fraction", xytext=(1.04, yl), textcoords="axes fraction",
                    fontsize=6, va="center", ha="left", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.3, color="0.55", shrinkA=0, shrinkB=1.5))
ax.set_xlabel("log2FC vs own control (DESeq2)")
ax.set_ylabel("log2FC after shared-\ncontrol correction")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=PAL["K562"], markersize=3, label="K562"),
                   plt.Line2D([], [], marker="s", ls="", color=PAL["HepG2"], markersize=3, label="HepG2"),
                   plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.4", markersize=3, label="padj ≥ 0.05")],
          fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, handletextpad=0.1, columnspacing=0.8, borderaxespad=0.1)
save(fig, "figS3", "a_korekta")

kc = kp.dropna(subset=["centyl_wsrod_knockdownow"]).copy()
wide = kc.pivot_table(index="para", columns="linia", values="centyl_wsrod_knockdownow", aggfunc="mean")
order = wide.sub(50).abs().max(axis=1).sort_values().index
fig, ax = panel_fixed(40, 3.0 * len(order) + 4)
ax.axvspan(0, 5, color="#EFEFEF", lw=0); ax.axvspan(95, 100, color="#EFEFEF", lw=0)
ax.axvline(50, color="0.6", lw=0.4, ls=(0, (2, 2)))
for i, para in enumerate(order):
    v = wide.loc[para].dropna()
    if len(v) == 2:
        ax.plot(v.values, [i, i], color="0.75", lw=0.8, zorder=2)
    for cell, mk in [("K562", "o"), ("HepG2", "s")]:
        if cell in v.index:
            ax.scatter(v[cell], i, s=10, marker=mk, color=PAL[cell], lw=0, zorder=3)
ax.set_yticks(range(len(order)), order, fontsize=6)
ax.set_ylim(-0.7, len(order) - 0.3)
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
ax.tick_params(axis="y", length=0)
ax.set_xlabel("percentile among all depletions\nin the cell line (after correction)")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=PAL["K562"], markersize=3, label="K562"),
                   plt.Line2D([], [], marker="s", ls="", color=PAL["HepG2"], markersize=3, label="HepG2")],
          fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.1)
save(fig, "figS3", "b_centyle")
ext = kc[(kc["centyl_wsrod_knockdownow"] <= 5) | (kc["centyl_wsrod_knockdownow"] >= 95)]
print(f"   S3b: {wide.shape[0]} par; w skrajnych 5% w co najmniej jednej linii: {ext['para'].nunique()} par:")
print(ext[["para", "linia", "centyl_wsrod_knockdownow"]].round(1).to_string(index=False))
kp.to_csv(Path("results/figures_pub/figS3") / "pary_z_literatury.tsv", sep="\t", index=False)

# ================================================================ S4
pr = pd.read_csv("results/phase4/network/rbp_pairs_lnc_vs_mrna.tsv.gz", sep="\t")
PANELS = [("a_lnc", "lnc_K562", "lnc_HepG2", "similarity in K562\n(lncRNA responses)", "similarity in HepG2\n(lncRNA responses)", PAL["lncRNA"]),
          ("b_mrna", "mRNA_K562", "mRNA_HepG2", "similarity in K562\n(mRNA responses)", "similarity in HepG2\n(mRNA responses)", "#777777"),
          ("c_wlinii", "lnc_K562", "mRNA_K562", "similarity in K562\n(lncRNA responses)", "similarity in K562\n(mRNA responses)", PAL["K562"])]
rows = []
for name, xa, ya, xl, yl, col in PANELS:
    d = pr[[xa, ya]].dropna()
    r, p = spearmanr(d[xa], d[ya])
    rows.append({"panel": name, "x": xa, "y": ya, "pary": len(d), "rho": r, "p": p})
    fig, ax = panel_fixed(36, 36)
    hb = ax.hexbin(d[xa], d[ya], gridsize=40, bins="log", cmap="Greys", mincnt=1, linewidths=0, extent=(-1, 1, -1, 1))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.axhline(0, color=col, lw=0.4); ax.axvline(0, color=col, lw=0.4)
    ax.text(0.03, 0.97, f"ρ = {r:.2f}\n{len(d):,} RBP pairs", transform=ax.transAxes, va="top", fontsize=6)
    ax.set_xlabel(xl, fontsize=6); ax.set_ylabel(yl, fontsize=6)
    save(fig, "figS4", name)
pd.DataFrame(rows).to_csv(Path("results/figures_pub/figS4") / "podobienstwa.tsv", sep="\t", index=False)
print("   S4: " + "; ".join(f"{r['panel']} rho={r['rho']:.2f}" for r in rows))

# ================================================================ podglady
try:
    from PIL import Image
    for fig_, names, out in [("figS3", ["a_korekta", "b_centyle"], "podgladS3.png"), ("figS4", ["a_lnc", "b_mrna", "c_wlinii"], "podgladS4.png")]:
        d = Path("results/figures_pub") / fig_
        ims = [Image.open(d / f"{n}.png") for n in names]
        W, H = sum(i.width for i in ims) + 20 * len(ims), max(i.height for i in ims)
        canvas = Image.new("RGB", (W, H), "white")
        x = 0
        for im in ims:
            canvas.paste(im, (x, H - im.height)); x += im.width + 20
        canvas.save(d / out)
        print(f"   {fig_}/{out}")
except Exception as e:  # noqa: BLE001
    print(f"   (podglad pominiety: {e})")
