"""
94_fig1e_audit.py (wersja 3, 19.10) - panel 1E po audycie: zgodnosc miedzy liniami (rho, 95% CI) przy roznych sposobach
korekty wspolnych kontroli, dopasowaniu i GC, po usunieciu sily odpowiedzi, wg metody wyciszenia, w wariantach
odpornosci i podzbiorach oraz dla pseudo-lncRNA (mRNA analizowane jak lncRNA).
Wersja 3: panel wezszy (104 mm), by zmiescil sie obok powiekszonego panelu 1D w 180 mm; krotsze etykiety.
Zmiany wersji 2 wzgledem wersji 1: wraca wiersz "intergenic lncRNAs only" (w wersji 1 wypadal przez blad filtra), krotsze
etykiety, osobna kolumna n, czcionki 6.5-7 pt, szerokosc dobrana tak, by wiersz D + E miescil sie w 180 mm bez
zmniejszania.
Czyta: results/figures_pub/fig1/e_odpornosc_przed_audytem.tsv (wiersze sprzed audytu; tworzona przy pierwszym
       uruchomieniu), results/audit/A1_A4_rho.tsv, results/audit/A2_pseudo_lncRNA.tsv; sile odpowiedzi liczy od nowa.
Zapisuje: results/figures_pub/fig1/e_odpornosc.{pdf,png,tsv}.
Opcja --tylko-rysunek: rysuje z istniejacego e_odpornosc.tsv, bez liczenia.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from figlib import MM, PAL, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtrans  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIG = "fig1"
D = Path("results/figures_pub") / FIG
A = Path("results/audit")
OUT = D / "e_odpornosc.tsv"
BAK = D / "e_odpornosc_przed_audytem.tsv"

# etykiety na rycinie (klucz = nazwa wariantu w tsv)
LAB = {
    "corrected, full matching (main)": "main analysis (batch-mate correction)",
    "no correction": "no correction",
    "correction without batch-mates shared between lines": "correction without shared batch-mates",
    "virtual control (no other depletions used)": "virtual control (median of controls)",
    "basic matching (expression, length; all lncRNAs)": "basic matching, all lncRNAs",
    "full matching + GC content": "GC content added to matching",
    "GC trend removed from every experiment": "GC trend removed from each experiment",
    "effect of response strength removed": "response strength regressed out",
    "same method in both lines": "same depletion method in both lines",
    "different methods in the two lines": "different methods in the two lines",
    "50 kb mask around CRISPR targets": "50 kb around CRISPR targets masked",
    "weak shRNA depletions removed": "weak shRNA depletions removed",
    "5 reference mRNAs": "5 matched mRNAs",
    "20 reference mRNAs": "20 matched mRNAs",
    "well-expressed genes only": "well-expressed genes only",
    "reference mRNAs without retained-intron isoforms": "no mRNAs with retained-intron isoforms",
    "shRNA depletions only": "shRNA depletions only",
    "CRISPR depletions only": "CRISPR depletions only",
    "without splicing and NMD RBPs": "splicing and NMD RBPs excluded",
    "intergenic lncRNAs only (basic matching)": "intergenic lncRNAs (basic matching)",
}
GROUPS = {"Shared controls": "Shared controls", "Matching and GC content": "Matching and GC content",
          "Other confounders": "Other confounders", "Robustness": "Robustness variants", "Subsets": "Subsets",
          "Null: mRNAs analysed as lncRNAs": "Null: mRNAs analysed as lncRNAs"}


def build_rows():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(4)
    if not BAK.exists():
        shutil.copy(OUT, BAK)
    old = pd.read_csv(BAK, sep="\t")
    print("Wiersze sprzed audytu:")
    print(old.to_string(index=False))
    a = pd.read_csv(A / "A1_A4_rho.tsv", sep="\t").set_index("wariant")
    ps = pd.read_csv(A / "A2_pseudo_lncRNA.tsv", sep="\t")

    def boot_ci(w, nb=2000):
        x, y = w["K562"].to_numpy(), w["HepG2"].to_numpy()
        bs = [spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, len(x), len(x)) for _ in range(nb))]
        return spearmanr(x, y).statistic, *np.nanpercentile(bs, [2.5, 97.5])

    # sila odpowiedzi (jak w 92_audit_quick.py)
    summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
    res = pd.read_csv("results/phase2/classes/class_effects_all.tsv.gz", sep="\t")
    main = res[(res["klasa"] == "lncRNA (razem)") & (res["macierz"] == "skorygowana") & (res["dopasowanie"] == "B")].set_index("kd_experiment")["roznica"]
    x = np.log10(pd.to_numeric(summ.reindex(main.index)["n_padj05"], errors="coerce") + 1)
    cell = summ.reindex(main.index)["cell"]
    resid = pd.Series(np.nan, index=main.index)
    for c in ["K562", "HepG2"]:
        ok = (cell == c) & x.notna() & main.notna()
        b = np.polyfit(x[ok], main[ok], 2)
        resid[ok] = main[ok] - np.polyval(b, x[ok])
    t = summ.reindex(resid.index)
    w = resid.groupby([t["target"].values, t["cell"].values]).mean().unstack().dropna()
    r_s, lo_s, hi_s = boot_ci(w)

    def row(group, label, rho, lo, hi, n, kind="point"):
        return {"grupa": group, "wariant": label, "rho": rho, "lo": lo, "hi": hi, "n": int(n), "rodzaj": kind}

    def from_a(key):
        r = a.loc[key]
        return r["rho"], r["lo"], r["hi"], r["n"]

    def from_old(name):
        r = old[old["wariant"] == name]
        if len(r) != 1:
            sys.exit(f"Brak wiersza '{name}' w {BAK}")
        r = r.iloc[0]
        return r["rho"], r["lo"], r["hi"], r["n"]

    rows = []
    G = "Shared controls"
    rows.append(row(G, "corrected, full matching (main)", *from_old("corrected, full matching (main)")))
    rows.append(row(G, "no correction", *from_a("bez korekty (te same RBP)")))
    rows.append(row(G, "correction without batch-mates shared between lines", *from_a("korekta bez wspolnych sasiadow")))
    rows.append(row(G, "virtual control (no other depletions used)", *from_a("wirtualna kontrola (bez innych wyciszen)")))
    G = "Matching and GC content"
    rows.append(row(G, "basic matching (expression, length; all lncRNAs)", *from_old("corrected, basic matching")))
    rows.append(row(G, "full matching + GC content", *from_old("full matching + GC content")))
    rows.append(row(G, "GC trend removed from every experiment", *from_old("GC trend removed from every experiment")))
    G = "Other confounders"
    rows.append(row(G, "effect of response strength removed", r_s, lo_s, hi_s, len(w)))
    rows.append(row(G, "same method in both lines", *from_a("glowny, ta sama metoda")))
    rows.append(row(G, "different methods in the two lines", *from_a("glowny, rozne metody")))
    moved = {"corrected, full matching (main)", "corrected, basic matching", "full matching + GC content",
             "GC trend removed from every experiment", "raw data, basic matching"}
    for r in old.itertuples():
        if r.wariant not in moved:
            rows.append(row(r.grupa, r.wariant, r.rho, r.lo, r.hi, r.n))
    rows.append(row("Null: mRNAs analysed as lncRNAs", f"pseudo-lncRNAs, {len(ps)} draws (median, range)",
                    ps["rho_miedzy_liniami"].median(), ps["rho_miedzy_liniami"].min(), ps["rho_miedzy_liniami"].max(),
                    ps["n_rbp"].iloc[0], "null"))
    fe = pd.DataFrame(rows)
    fe.to_csv(OUT, sep="\t", index=False)
    print("\nWiersze panelu 1E:")
    print(fe.round(3).to_string(index=False))
    missing = set(old["wariant"]) - moved - set(fe["wariant"])
    if missing:
        sys.exit(f"Zgubione wiersze: {missing}")
    return fe


def plot(fe):
    FS, FSB = 6.5, 7
    pitch, gap = 3.05, 1.2                     # mm na wiersz, odstep miedzy grupami
    W = 104                                    # szerokosc panelu w mm
    x_lab, x_n, x0, x1 = 1.0, 52.5, 55.5, 102.0  # mm: poczatek etykiet, prawa krawedz kolumny n, os x
    top, bottom = 4.0, 10.5
    ngr = fe["grupa"].nunique()
    H = top + bottom + pitch * (len(fe) + ngr) + gap * (ngr - 1)
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([x0 / W, bottom / H, (x1 - x0) / W, (H - top - bottom) / H])
    tr = mtrans.blended_transform_factory(fig.transFigure, ax.transData)
    y = 0.0
    ax.text(x_n / W, -0.9, "RBPs", transform=tr, ha="right", va="center", fontsize=FS, color="0.35")
    for gname, g in fe.groupby("grupa", sort=False):
        ax.text(x_lab / W, y, GROUPS.get(gname, gname), transform=tr, ha="left", va="center", fontsize=FS,
                fontweight="bold")
        y += 1
        for r in g.itertuples():
            main_row = "main" in r.wariant
            null_row = r.rodzaj == "null" or str(r.grupa).startswith("Null")
            col = PAL["lncRNA"] if main_row else ("0.5" if null_row else "k")
            ax.errorbar(r.rho, y, xerr=[[r.rho - r.lo], [r.hi - r.rho]], fmt="o", ms=3.4 if main_row else 2.6,
                        color=col, elinewidth=0.6, capsize=1.1, capthick=0.5, clip_on=False)
            lab = r.wariant.split(" (")[0] if null_row else LAB.get(r.wariant, r.wariant)
            ax.text((x_lab + 2.2) / W, y, lab, transform=tr, ha="left", va="center",
                    fontsize=FS, color=col if (main_row or null_row) else "k")
            ax.text(x_n / W, y, f"{r.n}", transform=tr, ha="right", va="center", fontsize=FS, color="0.35")
            y += 1
        y += gap / pitch
    ax.set_ylim(y - gap / pitch - 0.4, -1.5)
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_xlim(-0.22, 0.52)
    ax.set_xticks(np.arange(-0.2, 0.51, 0.1))
    ax.set_xticklabels([f"{v:.1f}".replace("-", "−") for v in np.arange(-0.2, 0.51, 0.1)], fontsize=FS)
    ax.axvline(0, color="0.6", lw=0.4, zorder=0)
    ax.axvline(fe.loc[fe["wariant"].str.contains("main"), "rho"].iloc[0], color=PAL["lncRNA"], lw=0.4,
               ls=(0, (2, 2)), zorder=0)
    ax.set_xlabel("agreement between cell lines (Spearman ρ, 95% CI)", fontsize=FSB, labelpad=2)
    ax.tick_params(axis="x", length=2, width=0.5, pad=1.5)
    ax.spines["bottom"].set_linewidth(0.6)
    save(fig, FIG, "e_odpornosc")
    print(f"\nZapisano {D / 'e_odpornosc.png'} i .pdf ({W} x {H:.0f} mm)")


if __name__ == "__main__":
    if not Path("results/figures_pub").is_dir():
        sys.exit("Uruchom z katalogu projektu rbp_classes.")
    fe = pd.read_csv(OUT, sep="\t", keep_default_na=False) if "--tylko-rysunek" in sys.argv else build_rows()
    plot(fe)
