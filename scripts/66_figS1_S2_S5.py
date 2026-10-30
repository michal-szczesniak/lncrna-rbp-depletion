"""
Ryciny uzupelniajace S1, S2 i S5.
  figS1/a_sasiedzi   - zmiana genow kodujacych w poblizu genu wyciszanego RBP (wzgledem podobnych genow gdzie indziej),
                       wg typu sasiada, CRISPR vs shRNA; gwiazdki: Wilcoxon vs 0 z korekta BH; pod spodem % wyraznie obnizonych
  figS2/a_spadek     - spadek mRNA wyciszanego RBP (log2FC), shRNA i CRISPR w obu liniach; prog slabego wyciszenia (spadek < 30%)
  figS2/b_skutecznosc - spadek mRNA RBP vs liczba zmienionych genow (padj < 0.05)
  figS2/c_NMD        - sprawdzenie UPF1: zmiana znanych celow NMD po wyciszeniu UPF1 i innych czynnikow NMD (log2FC DESeq2)
  figS5/<locus>      - pozostale ryciny locus (modul figlocus)
Runda 2 (19.10): S2C obwodzi wyciszenia skuteczne dla NMD wg nmd_skuteczne.tsv (UPF1 K562, UPF2 w obu liniach), kropka =
cel istotnie w gore (padj < 0,05), szare = sam wyciszany gen (nie oceniany); czcionki >= 6 pt (figlib.MIN_FS).
Runda 3 (20.10): ryciny locus mniejsze (sciezki 7 mm, modele genow upakowane - zmiana w figlocus), by S5 i S6 zmiescily
sie na stronie; dodany locus KCNQ1OT1 w K562 (Ryc. S6C), dotad plik z 6.10 rysowany starsza wersja modulu.
"""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from figlib import MM, PAL, bh, panel_fixed, save, stars  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

COLM = {"CRISPR": "#56B4E9", "shRNA": "#CC79A7"}
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t")

# ================================================================ S1: CRISPR a sasiedzi
cis = pd.read_csv("results/phase3/cis_nat/cis_effects.tsv", sep="\t")
cis = cis[cis["grupa"] == "mRNA"]
CATS = ["NAT", "naklad. ta sama nic", "sasiad <=10 kb, ta sama nic", "sasiad <=10 kb, przeciwna nic", "sasiad 10-50 kb"]
CEN = ["NAT", "overlapping,\nsame strand", "≤10 kb,\nsame strand", "≤10 kb,\nopposite strand", "10–50 kb"]
fig, ax = panel_fixed(70, 36)
tests, lows = [], {}
for i, c in enumerate(CATS):
    for j, m in enumerate(["CRISPR", "shRNA"]):
        d = cis[(cis["kategoria"] == c) & (cis["metoda"] == m)]
        v = d["delta"].dropna().values
        if len(v) < 3:
            continue
        x = i + (j - 0.5) * 0.36
        parts = ax.violinplot([np.clip(v, -3, 3)], positions=[x], widths=0.34, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(COLM[m]); pc.set_alpha(0.7); pc.set_edgecolor("none")
        q1, md, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([x, x], [q1, q3], color="k", lw=0.9)
        ax.scatter([x], [md], color="white", edgecolor="k", s=5, lw=0.4, zorder=3)
        low = 100 * ((d["padj"] < 0.05) & (d["lfc"] < 0)).mean()
        lows[(i, m)] = low
        tests.append((x, wilcoxon(v).pvalue))
for (x, _), q in zip(tests, bh([t[1] for t in tests])):
    ax.text(x, 2.55, stars(q), ha="center", fontsize=6)
ax.axhline(0, color="0.4", lw=0.4)
ax.set_ylim(-3.1, 2.9)
ax.set_xticks(range(len(CATS)), [f"{c}\n{lows.get((i, 'CRISPR'), float('nan')):.0f}% | {lows.get((i, 'shRNA'), float('nan')):.0f}%"
                                   for i, c in enumerate(CEN)], fontsize=6)
ax.set_xlabel("% clearly lower (padj < 0.05, log2FC < 0): CRISPR | shRNA", fontsize=6)
ax.set_ylabel("change of neighbouring protein-coding\ngene vs similar genes (log2)")
ax.set_ylim(-3.1, 2.9)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLM[m], alpha=0.7) for m in COLM], labels=["CRISPR knockout", "shRNA knockdown"],
          fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.1)
save(fig, "figS1", "a_sasiedzi")

# ================================================================ S2: skutecznosc wyciszen
fig, ax = panel_fixed(34, 36)
k = 0
for cell in ["K562", "HepG2"]:
    for m in ["shRNA", "CRISPR"]:
        v = summ.loc[(summ["cell"] == cell) & (summ["method"] == m), "target_lfc"].dropna().values
        if len(v) > 1:
            parts = ax.violinplot([np.clip(v, -6, 3)], positions=[k], widths=0.75, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(COLM[m]); pc.set_alpha(0.7); pc.set_edgecolor("none")
            md = np.median(v)
            ax.scatter([k], [md], color="white", edgecolor="k", s=6, lw=0.4, zorder=3)
            ax.text(k, 2.6, f"{100 * (v > np.log2(0.7)).mean():.0f}%", ha="center", fontsize=6)
        k += 1
ax.axhline(np.log2(0.7), color="0.3", lw=0.5, ls=(0, (2, 2)))
ax.axhline(0, color="0.6", lw=0.4)
ax.set_xticks(range(4), ["shRNA\nK562", "CRISPR\nK562", "shRNA\nHepG2", "CRISPR\nHepG2"], fontsize=6)
ax.set_ylim(-6.3, 3.2)
ax.set_ylabel("change of the depleted\nRBP's mRNA (log2FC)")
ax.text(3.45, np.log2(0.7) + 0.15, "−30%", fontsize=6, ha="right", color="0.3")
for m in ["shRNA", "CRISPR"]:
    v_ = summ.loc[summ["method"] == m, "target_lfc"].dropna()
    print(f"   S2a: {m}: mRNA RBP spadlo o mniej niz 30% w {100 * (v_ > np.log2(0.7)).mean():.1f}% eksperymentow ({int((v_ > np.log2(0.7)).sum())} z {len(v_)})")
save(fig, "figS2", "a_spadek")

fig, ax = panel_fixed(34, 36)
for m in ["shRNA", "CRISPR"]:
    s_ = summ[summ["method"] == m]
    ax.scatter(s_["target_lfc"].clip(-6, 3), np.log10(s_["n_padj05"] + 1), s=3, color=COLM[m], lw=0, alpha=0.7, label=m)
ax.axvline(np.log2(0.7), color="0.3", lw=0.5, ls=(0, (2, 2)))
ax.set_xlabel("RBP mRNA change (log2FC)")
ax.set_ylabel("genes changed (padj < 0.05)")
ax.set_yticks([0, 1, 2, 3, 4], ["0", "10", "100", "1,000", "10,000"])
ax.legend(fontsize=6, loc="lower left", handletextpad=0.1, markerscale=2)
save(fig, "figS2", "b_skutecznosc")

# ---------------------------------------------------------------- c: UPF1 i inne czynniki NMD
NMDF = ["UPF1", "UPF2", "UPF3B", "SMG1", "SMG5", "SMG6", "SMG7", "SMG8", "SMG9"]
TARGETS = ["GAS5", "SNHG1", "SNHG5", "SNHG6", "SNHG8", "ZFAS1", "GADD45A", "GADD45B", "ATF4", "SMG5", "SMG7", "UPF3B"]
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
name2id = cls.drop_duplicates("gene_name").set_index("gene_name")["gene_id"]
tg = {n: name2id[n] for n in TARGETS if n in name2id.index}
kds = summ[summ["target"].isin(NMDF)].sort_values(["cell", "target"])
eff = set()
try:                                                              # wyciszenia skuteczne dla NMD (krok 4.26, 67_nmd_skuteczne.py)
    nsk = pd.read_csv("results/phase4/followups/nmd_skuteczne.tsv", sep="\t")
    eff = set(nsk.loc[nsk["skuteczne_NMD"].astype(str) == "True", "kd_experiment"])
except Exception as e:  # noqa: BLE001
    print(f"   (S2c: brak nmd_skuteczne.tsv: {e})")
mat, info, sigm, kd_of = {}, [], {}, {}
for _, r in kds.iterrows():
    f = Path(f"results/phase2/deseq2/{r['kd_experiment']}.tsv.gz")
    if not f.exists():
        continue
    d = pd.read_csv(f, sep="\t", usecols=["gene_id", "lfc_mle", "padj"]).set_index("gene_id")
    col = f"{r['target']} {r['cell']} ({r['method'][0]})"
    # sam wyciszany gen (np. SMG5 po wyciszeniu SMG5) nie jest oceniany jako cel NMD: brak wartosci (szary)
    mat[col] = pd.Series({n: (np.nan if n == r["target"] else d["lfc_mle"].get(g, np.nan)) for n, g in tg.items()})
    sigm[col] = pd.Series({n: (n != r["target"]) and (d["lfc_mle"].get(g, 0) > 0) and (d["padj"].get(g, 1) < 0.05) for n, g in tg.items()})
    kd_of[col] = r["kd_experiment"]
    info.append({"wyciszenie": col, "kd_experiment": r["kd_experiment"], "spadek_mRNA_RBP": r["target_lfc"], "geny_zmienione": r["n_padj05"],
                 "mediana_zmiany_celow_NMD": np.nanmedian(mat[col].values),
                 "celow_NMD_w_gore_padj05": int(sigm[col].sum()), "skuteczne_NMD": r["kd_experiment"] in eff})
M_ = pd.DataFrame(mat).reindex(list(tg))
inf = pd.DataFrame(info)
inf.to_csv(Path("results/figures_pub/figS2") / "c_NMD_wyciszenia.tsv", sep="\t", index=False)
M_.round(2).to_csv(Path("results/figures_pub/figS2") / "c_NMD_cele.tsv", sep="\t")
print("S2c: wyciszenia czynnikow NMD i znane cele NMD (log2FC DESeq2 wzgledem wlasnej kontroli)")
print(inf.round(2).to_string(index=False))
if not M_.empty:
    W_ = 30 + 2.8 * M_.shape[1]
    fig = plt.figure(figsize=(W_ * MM, (14 + 3.0 * M_.shape[0]) * MM))
    H_ = 14 + 3.0 * M_.shape[0]
    ax = fig.add_axes([22 / W_, 4 / H_, 2.8 * M_.shape[1] / W_, 3.0 * M_.shape[0] / H_])
    lim = 2
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#BDBDBD")
    im = ax.imshow(np.ma.masked_invalid(M_.clip(-lim, lim).values), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(M_.shape[1]), M_.columns, rotation=90, fontsize=6)
    ax.xaxis.tick_top()
    ax.set_yticks(range(M_.shape[0]), M_.index, fontsize=6)
    ax.tick_params(length=0, pad=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    S_ = pd.DataFrame(sigm).reindex(index=M_.index, columns=M_.columns).fillna(False)
    yy, xx = np.nonzero(S_.values.astype(bool))
    ax.scatter(xx, yy, s=2.5, color="k", lw=0, zorder=3)                 # cel istotnie w gore (padj < 0,05)
    for j, c in enumerate(M_.columns):
        if kd_of.get(c) in eff:                                           # wyciszenie skuteczne dla NMD
            ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, M_.shape[0], fill=False, ec="k", lw=0.9))
    cax = fig.add_axes([(25 + 2.8 * M_.shape[1]) / W_, 4 / H_, 1.8 / W_, min(0.5, 3.0 * M_.shape[0] / H_)])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("log2FC", fontsize=6); cb.ax.tick_params(labelsize=6, length=1.5); cb.outline.set_linewidth(0.4)
    save(fig, "figS2", "c_NMD")

# ================================================================ S5: pozostale ryciny locus
LOCI = [("NEAT1_eksosom", ["NEAT1"], "NEAT1", [("MTREX", "K562"), ("MTREX", "HepG2"), ("EXOSC2", "HepG2"), ("EXOSC5", "HepG2")]),
        ("ZFAS1_EJC", ["ZFAS1"], "ZFAS1", [("MAGOH", "K562"), ("MAGOH", "HepG2"), ("EIF4A3", "K562"), ("EIF4A3", "HepG2")]),
        ("SNHG5_EJC", ["SNHG5"], "SNHG5", [("MAGOH", "K562"), ("MAGOH", "HepG2"), ("EIF4A3", "K562"), ("EIF4A3", "HepG2")]),
        ("XIST_CIZ1", ["XIST"], "XIST", [("CIZ1", "K562")]),
        ("OIP5-AS1_HuR", ["OIP5-AS1"], "OIP5-AS1", [("ELAVL1", "K562"), ("ELAVL1", "HepG2")]),
        ("KCNQ1OT1_K562_eksosom", ["KCNQ1OT1"], "KCNQ1OT1",
         [("MTREX", "K562"), ("EXOSC2", "K562"), ("EXOSC5", "K562"), ("EXOSC9", "K562"), ("EXOSC10", "K562")])]
LOCUS_KW = dict(track_mm=7.0, track_gap=3.0)          # runda 3: mniejsze sciezki (S5 i S6 na jednej stronie)
try:
    from figlocus import draw_locus, genes as G_
    for name, gn, focus, tr in LOCI:
        if focus not in set(G_["gene_name"]):
            print(f"   (locus {name} pominiety: brak genu {focus})")
            continue
        try:
            draw_locus("figS5", name, gn, focus, tr, **LOCUS_KW)
            if name == "KCNQ1OT1_K562_eksosom":
                from figlocus import gene_profile
                gene_profile(focus, tr, nbins=10).to_csv(Path("results/figures_pub/figS5") / f"{name}_profil.tsv", sep="\t", index=False)
        except Exception as e:  # noqa: BLE001
            print(f"   (locus {name} pominiety: {e})")
except Exception as e:  # noqa: BLE001
    print(f"   (ryciny locus pominiete: {e})")

# ================================================================ podglady
try:
    from PIL import Image
    for fig_, names, out in [("figS1", ["a_sasiedzi"], "podgladS1.png"), ("figS2", ["a_spadek", "b_skutecznosc", "c_NMD"], "podgladS2.png")]:
        d = Path("results/figures_pub") / fig_
        ims = [Image.open(d / f"{n}.png") for n in names if (d / f"{n}.png").exists()]
        W, H = sum(i.width for i in ims) + 20 * len(ims), max(i.height for i in ims)
        canvas = Image.new("RGB", (W, H), "white")
        x = 0
        for im in ims:
            canvas.paste(im, (x, H - im.height)); x += im.width + 20
        canvas.save(d / out)
        print(f"   {fig_}/{out}")
except Exception as e:  # noqa: BLE001
    print(f"   (podglad pominiety: {e})")
