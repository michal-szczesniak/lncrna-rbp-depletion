"""
Rycina 1 (publikacja): podejscie i glowny wynik. Kazdy panel osobno: results/figures_pub/fig1/<panel>.pdf|png
  a_przyklad     - czym jest class effect: rozklad roznic lncRNA minus dopasowane mRNA w dwoch przykladowych eksperymentach
  b_kontrole     - artefakt wspolnej kontroli przed i po korekcie (eta2 szesciu glownych skladowych, obie linie)
  c_rozklad      - rozklad class effect w eksperymentach
  d_replikacja   - class effect tego samego RBP w K562 i HepG2 (342 RBP), rho z 95% CI
  e_odpornosc    - tylko wiersze bazowe (e_odpornosc_przed_audytem.tsv); panel 1E rysuje 94_fig1e_audit.py
  podglad1.png    - zlozony podglad (tylko do oceny)
Runda 2 (19.10): kolory w B i D niezalezne od kolorow linii komorkowych, wiekszy panel D z poprawionymi etykietami,
czcionki >= 6 pt (figlib.MIN_FS). Wybor paneli: python scripts/60_fig1.py a d  (bez argumentow: wszystkie).
Runda 3 (20.10): panel D z niesymetryczna osia x, etykiety RBP obnizajacych lncRNA odsuniete od osi y.
Runda 4 (20.10): panel C - legenda i mediana |class effect| nad osiami (nie nachodza na linie +-0,1 log2).
Runda 5 (20.10): w panelu a (Ryc. 1B) przyklad MTREX zamiast HNRNPK (HNRNPK nie jest odporny na GC, MTREX jest).
"""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from figlib import *  # noqa: E402,F401,F403
from figlib import MM, PAL, W1, W2, WF, panel, panel_fixed, place_labels, rho_ci, save, fmt_p  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

M, C = Path("results/phase2/matrix"), Path("results/phase2/classes")
LNC_ALL = "lncRNA (razem)"
FIG = "fig1"
res = pd.read_csv(C / "class_effects_all.tsv.gz", sep="\t")
lnc = res[res["klasa"] == LNC_ALL]
main = lnc[(lnc["macierz"] == "skorygowana") & (lnc["dopasowanie"] == "B")]
hits = pd.read_csv(C / "rbp_hits.tsv", sep="\t", index_col=0)
RUN = set(sys.argv[1:]) or set("abcde")


def per_rbp(df):
    return df.groupby(["target", "linia"])["roznica"].mean().unstack("linia").dropna()


# ---------------------------------------------------------------- a: przyklad (czym jest class effect)
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


EXAMPLES = [("HepG2", "NONO", "#1A1A1A"), ("HepG2", "MTREX", "#8C8C8C")]   # nie niebieski/pomaranczowy (te = linie komorkowe)
try:
    if "a" not in RUN:
        raise RuntimeError("pominiety na zyczenie")
    summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
    cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
    kin = pd.read_csv("results/phase1/bru/gene_kinetics.tsv", sep="\t")
    spl = splicing_by_cell()
    LNCS = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
    COLS = ["e", "w", "s", "t"]
    fig, ax = panel_fixed(40, 36)
    for cell, rbp, col in EXAMPLES:
        L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
        kds = [k for k in summ.index[(summ["target"] == rbp) & (summ["cell"] == cell)] if k in L.columns]
        if not kds:
            continue
        expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
        f = pd.DataFrame({"k": cls.reindex(L.index)["klasa"], "e": np.log10(expr.reindex(L.index) + 1),
                          "w": np.log10(cls.reindex(L.index)["width"]), "s": spl[cell].reindex(L.index),
                          "t": kin[kin["cell"] == cell].drop_duplicates("gene_id").set_index("gene_id")["stab_6h"].reindex(L.index)}).dropna()
        mr = f[f["k"] == "mRNA"]
        lo_, hi_ = mr[COLS].quantile(0.01), mr[COLS].quantile(0.99)
        sel = f[f["k"].isin(LNCS) & ((f[COLS] >= lo_) & (f[COLS] <= hi_)).all(axis=1)]
        mu, sd = mr[COLS].mean().values, mr[COLS].std().values
        sd = np.where(sd > 0, sd, 1.0)
        _, idx = NearestNeighbors(n_neighbors=10).fit((mr[COLS].values - mu) / sd).kneighbors((sel[COLS].values - mu) / sd)
        x = L[kds[0]]
        d = x.reindex(sel.index).values - np.nanmean(x.reindex(mr.index.values[idx].ravel()).values.reshape(idx.shape), axis=1)
        d = d[np.isfinite(d)]
        print(f"   panel a: {rbp} {cell} {kds[0]}: {len(d)} lncRNA, mediana {np.median(d):+.4f}", flush=True)
        ax.hist(np.clip(d, -2, 2), bins=np.linspace(-2, 2, 61), histtype="step", lw=0.8, color=col, density=True,
                label=f"{rbp}: class effect {np.median(d):+.2f}".replace("-", "−"))
        ax.axvline(np.median(d), color=col, lw=0.8)
    ax.axvline(0, color="0.5", lw=0.4, ls=(0, (2, 2)))
    ax.set_xlabel("lncRNA change minus mean change\nof its 10 matched mRNAs (log2)")
    ax.set_ylabel("density of lncRNAs")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=6, handlelength=1.0, borderaxespad=0.2,
              title="depletion in HepG2", title_fontsize=6, alignment="left")
    save(fig, FIG, "a_przyklad")
except Exception as e:  # noqa: BLE001
    print(f"   (panel a pominiety: {e})")

# ---------------------------------------------------------------- b: wspolne kontrole (eta2)
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
        out.append(s.groupby(grp).apply(lambda x: len(x) * (x.mean() - s.mean()) ** 2).sum() / tot)
    return np.array(out), (pd.Series(grp).nunique() - 1) / (len(grp) - 1)


def panel_b():
    print("   (panel b: PCA, kilka minut)", flush=True)
    fig, ax = panel_fixed(40, 36)
    for j, cell in enumerate(["K562", "HepG2"]):
        Lb = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
        L = pd.read_csv(M / f"lfc_{cell}.tsv.gz", sep="\t", index_col=0)[Lb.columns]
        S = pd.read_csv(M / f"se_{cell}.tsv.gz", sep="\t", index_col=0)[Lb.columns]
        expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
        e0, null = eta2_pcs(L, S, expr)
        e1, _ = eta2_pcs(Lb, S, expr)
        x = np.arange(1, 7) + (j - 0.5) * 0.22
        for xi, a, b in zip(x, e0, e1):
            ax.plot([xi, xi], [a, b], color=PAL[cell], lw=0.6, alpha=0.6)
        ax.scatter(x, e0, s=10, facecolor="white", edgecolor=PAL[cell], lw=0.7, zorder=3, label=f"{cell}, before")
        ax.scatter(x, e1, s=10, color=PAL[cell], zorder=3, label=f"{cell}, after")
        ax.axhline(null, ls=(0, (2, 2)), lw=0.5, color=PAL[cell])
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance explained by\ncontrol sample (η²)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, loc="upper right", ncol=2, handletextpad=0.2, columnspacing=0.6, borderaxespad=0.1)
    save(fig, FIG, "b_kontrole")


if "b" in RUN:
    panel_b()

# ---------------------------------------------------------------- c: rozklad class effect
def panel_c():
    fig, ax = panel_fixed(40, 36)
    bins = np.linspace(-0.6, 0.6, 41)
    for cell in ["K562", "HepG2"]:
        v = main.loc[main["linia"] == cell, "roznica"].clip(-0.6, 0.6)
        ax.hist(v, bins=bins, histtype="step", lw=0.8, color=PAL[cell], label=f"{cell}, {len(v)}")
    for v in (-0.1, 0.1):
        ax.axvline(v, ls=(0, (2, 2)), lw=0.5, color="0.4")
    ax.set_xlabel("class effect (log2)")
    ax.set_ylabel("experiments")
    # runda 4 (20.10): legenda i mediana nad osiami (wewnatrz nachodzily na linie +-0,1)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=1, fontsize=6, handlelength=1.0, borderaxespad=0.1)
    ax.set_xlim(-0.6, 0.6)
    med_abs = main["roznica"].abs().median()
    ax.text(1.0, 1.03, f"median |class effect|\n{med_abs:.3f} log2 (~{(2 ** med_abs - 1) * 100:.0f}%)", transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
    save(fig, FIG, "c_rozklad")


if "c" in RUN:
    panel_c()

# ---------------------------------------------------------------- d: replikacja
w = per_rbp(main)
# rho, p i 95% CI z jednego zrodla (te same liczby co w tekscie)
rtm = pd.read_csv("results/phase4/stats/replication_tests.tsv", sep="\t").iloc[0]
r, p, lo, hi = float(rtm["rho"]), float(rtm["p"]), float(rtm["ci_lo"]), float(rtm["ci_hi"])
if int(rtm["n"]) != len(w):
    print(f"   UWAGA: replication_tests.tsv ma n = {int(rtm['n'])}, a panel d {len(w)} RBP")
gc35 = pd.read_csv("results/phase4/followups/gc_35rbp.tsv", sep="\t")
robust = set(gc35.loc[gc35["w_B"].astype(str).eq("True") & gc35["w_C"].astype(str).eq("True"), "RBP"])
CLR = "#1A1A1A"                    # wyrazne RBP: czern (kierunek widac z polozenia; niebieski/pomaranczowy = linie komorkowe)


def panel_d():
    from matplotlib.lines import Line2D
    fig, ax = panel_fixed(46, 44)
    is_hit = w.index.isin(hits.index)
    is_rob = is_hit & w.index.isin(robust)
    ax.scatter(w.loc[~is_hit, "K562"], w.loc[~is_hit, "HepG2"], s=5, color=PAL["grey"], lw=0)
    hs = w[is_hit & ~is_rob]
    ax.scatter(hs["K562"], hs["HepG2"], s=11, facecolor="white", edgecolor=CLR, lw=0.6, zorder=3)
    hh = w[is_rob]
    ax.scatter(hh["K562"], hh["HepG2"], s=11, lw=0, zorder=4, color=CLR)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.2, color=CLR, label=f"robust to GC ({len(hh)})"),
                       Line2D([], [], marker="o", ls="", ms=3.2, mfc="white", mec=CLR, mew=0.6, label=f"GC-sensitive ({len(hs)})")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=6, title="clear effect in both lines:", title_fontsize=6,
              alignment="left", handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1, frameon=False)
    print(f"   panel d: {len(hh)} RBP odpornych na GC, {len(hs)} wrazliwych; rho = {r:.3f} ({lo:.2f}-{hi:.2f})")
    # runda 3 (20.10): os x niesymetryczna, z miejscem na kolumne etykiet po lewej (etykiety nie dotykaja osi y);
    # skala obu osi prawie rowna (ok. 0.024 log2/mm), bez wymuszania set_aspect
    lim = max(w.abs().max().max(), 0.3) * 1.12
    xmax = max(w["K562"].max(), 0.3) * 1.15
    ax.set_xlim(-lim - 0.11, xmax); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="0.6", lw=0.4); ax.axvline(0, color="0.6", lw=0.4)
    up_ = hh[hh.mean(axis=1) > 0].sort_values("HepG2", ascending=False)
    # RBP podnoszace lncRNA leza tuz obok otwartych punktow: etykiety z lewej i wyzej, linie do wlasnych punktow
    for i, (name, row) in enumerate(up_.iterrows()):
        ax.annotate(name, xy=(row["K562"], row["HepG2"]), xytext=(row["K562"] - 0.06 - 0.065 * i, row["HepG2"] + 0.12 - 0.04 * i),
                    ha="right", va="center", fontsize=6,
                    arrowprops=dict(arrowstyle="-", color="0.35", lw=0.4, shrinkA=0.5, shrinkB=2.2))
    # RBP obnizajace lncRNA: stale miejsca etykiet (dane sie nie zmieniaja; ustawione tak, by etykiety nie nachodzily
    # na punkty), kolumna z lewej strony skupiska i dwie etykiety z prawej
    DOWN_POS = {"RPS19": (-0.40, -0.10, "right"), "WBP11": (-0.40, -0.19, "right"), "SRSF1": (-0.40, -0.28, "right"),
                "CNOT2": (-0.40, -0.37, "right"), "NONO": (-0.40, -0.46, "right"), "PPP1R10": (-0.46, -0.02, "left"),
                "PUF60": (-0.10, -0.43, "left"), "AQR": (-0.07, -0.33, "left")}
    for name, (tx, ty, ha) in DOWN_POS.items():
        if name in hh.index:
            ax.annotate(name, xy=(hh.loc[name, "K562"], hh.loc[name, "HepG2"]), xytext=(tx, ty), ha=ha, va="center", fontsize=6,
                        arrowprops=dict(arrowstyle="-", color="0.35", lw=0.4, shrinkA=0.5, shrinkB=2.2))
    ax.text(0.02, 0.98, f"ρ = {r:.2f} ({lo:.2f}–{hi:.2f})\np = {fmt_p(p)}\nn = {len(w)} RBPs", transform=ax.transAxes,
            va="top", fontsize=6)
    ax.set_xlabel("class effect in K562 (log2)")
    ax.set_ylabel("class effect in HepG2 (log2)")
    save(fig, FIG, "d_replikacja")


if "d" in RUN:
    panel_d()

# ---------------------------------------------------------------- e: odpornosc (forest)
def panel_e():
    rows = []


    def add(group, label, wdf=None, stored=None):
        if stored is not None:
            rows.append({"grupa": group, "wariant": label, **stored})
        elif wdf is not None and len(wdf) >= 10:
            r_, p_, lo_, hi_ = rho_ci(wdf["K562"], wdf["HepG2"])
            rows.append({"grupa": group, "wariant": label, "rho": r_, "lo": lo_, "hi": hi_, "n": len(wdf)})


    add("Correction and matching", "raw data, basic matching", per_rbp(lnc[(lnc["macierz"] == "surowa") & (lnc["dopasowanie"] == "A")]))
    add("Correction and matching", "corrected, basic matching", per_rbp(lnc[(lnc["macierz"] == "skorygowana") & (lnc["dopasowanie"] == "A")]))
    add("Correction and matching", "corrected, full matching (main)", stored={"rho": r, "lo": lo, "hi": hi, "n": len(w)})
    gs = pd.read_csv("results/phase4/followups/gc_summary.tsv", sep="\t").set_index("wariant")


    def add_gc(group, key, label):
        if key in gs.index:
            import re
            m_ = re.match(r"\s*(-?\d*\.?\d+)\s*[-–]\s*(-?\d*\.?\d+)", str(gs.loc[key, "CI95"]))
            lo_, hi_ = float(m_.group(1)), float(m_.group(2))
            add(group, label, stored={"rho": float(gs.loc[key, "rho_miedzy_liniami"]), "lo": lo_, "hi": hi_,
                                      "n": int(gs.loc[key, "RBP_w_obu"])})


    add_gc("Correction and matching", "C", "full matching + GC content")
    ra = pd.read_csv("results/phase4/robustness_all.tsv.gz", sep="\t")
    vlab = {"maska 50 kb CRISPR": "50 kb mask around CRISPR targets", "bez slabych shRNA": "weak shRNA depletions removed",
            "K = 5": "5 reference mRNAs", "K = 20": "20 reference mRNAs", "ekspresja >= 50": "well-expressed genes only"}
    for v, l in vlab.items():
        add("Robustness", l, per_rbp(ra[ra["wariant"] == v]))
    rv = pd.read_csv(C / "robustness_variants.tsv.gz", sep="\t")
    add("Robustness", "reference mRNAs without retained-intron isoforms", per_rbp(rv[rv["wariant"] == "B_mRNA_bez_RI"]))
    add_gc("Robustness", "B_gc", "GC trend removed from every experiment")
    g0 = ra[ra["wariant"] == "glowny"]
    for met in ["shRNA", "CRISPR"]:
        same = g0.groupby("target")["metoda"].agg(lambda s: set(s) == {met})
        add("Subsets", f"{met} depletions only", per_rbp(g0[g0["target"].isin(same[same].index)]))
    rt = pd.read_csv("results/phase4/stats/replication_tests.tsv", sep="\t")
    r2 = rt[rt["zestaw"] == "bez splicingu i NMD"].iloc[0]
    add("Subsets", "without splicing and NMD RBPs", stored={"rho": r2["rho"], "lo": r2["ci_lo"], "hi": r2["ci_hi"], "n": int(r2["n"])})
    cp = pd.read_csv("results/phase4/followups/class_effect_by_position.tsv", sep="\t")
    ri = cp[cp["kategoria"] == "miedzygenowe"].iloc[0]
    add("Subsets", "intergenic lncRNAs only (basic matching)", stored={"rho": ri["rho"], "lo": ri["CI_lo"], "hi": ri["CI_hi"], "n": int(ri["RBP"])})
    fe = pd.DataFrame(rows)
    # wiersze bazowe dla 94_fig1e_audit.py (ten skrypt rysuje panel 1E razem z wariantami z audytu)
    fe.to_csv(Path("results/figures_pub") / FIG / "e_odpornosc_przed_audytem.tsv", sep="\t", index=False)
    print("   e: wiersze bazowe zapisane w fig1/e_odpornosc_przed_audytem.tsv; panel 1E rysuje scripts/94_fig1e_audit.py")


if "e" in RUN:
    panel_e()

# ---------------------------------------------------------------- podglad
try:
    from PIL import Image
    d = Path("results/figures_pub") / FIG
    names = [n for n in ["a_przyklad", "c_rozklad", "d_replikacja"] if (d / f"{n}.png").exists()]
    row1 = [Image.open(d / f"{n}.png") for n in names]
    row2 = [Image.open(d / f"{n}.png") for n in ["b_kontrole", "e_odpornosc"]]
    W = max(sum(i.width for i in row1), sum(i.width for i in row2))
    H = max(i.height for i in row1) + max(i.height for i in row2) + 30
    canvas = Image.new("RGB", (W, H), "white")
    x = 0
    for im in row1:
        canvas.paste(im, (x, max(i.height for i in row1) - im.height)); x += im.width + 10
    x, y0 = 0, max(i.height for i in row1) + 30
    for im in row2:
        canvas.paste(im, (x, y0)); x += im.width + 10
    canvas.save(d / "podglad1.png")
    print(f"   {FIG}/podglad1.png")
except Exception as e:  # noqa: BLE001
    print(f"   (podglad pominiety: {e})")
