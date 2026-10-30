"""
Ryciny locus do publikacji (wspolne dla ryciny 4, 5 i suplementu).

draw_locus(figure, name, gene_names, focus, tracks, ...) rysuje:
  - modele genow w regionie (GENCODE v29; scalone eksony na pelnej dlugosci genu; bez przewidywanych malych RNA z Rfam),
  - po jednej sciezce na wyciszenie: pokrycie RNA-seq na nici genu (sygnal unikalnych odczytow ENCODE, RPM, srednia z replik),
    kontrola jako szare pole, wyciszenie jako linia w kolorze linii komorkowej,
  - z prawej przy kazdej sciezce: poziom badanego genu w kazdej replice (srednie pokrycie eksonow, RPM), kontrola vs
    wyciszenie, z log2FC (lfc_mle) i padj z DESeq2 wzgledem wlasnej kontroli oraz centylem zmiany PO korekcie na wspolna
    kontrole (lfc_bc) wsrod wszystkich wyciszen w linii.
Runda 3 (20.10): ryciny locus mniejsze, by zmiescic je na stronie:
  - modele genow upakowane w wiersze: geny i ich etykiety nie nachodza na siebie, kilka genow moze dzielic wiersz;
    etykieta obok genu (z lewej od poczatku, a gdy brak miejsca - z prawej od konca), nie nad nim, wiec nie dotyka genu
    w wierszu wyzej (blad z rundy 2: np. KCNQ1 i SNORD12C nachodzace na model genu powyzej),
  - parametry gene_mm (wysokosc wiersza modelu genu, domyslnie 3.4 mm) i track_gap (odstep miedzy sciezkami, 3.0 mm),
  - opis sciezki 0.4 mm nad sciezka (zamiast 3% jej wysokosci).
Runda 4 (20.10): opis osi y krotszy przy 1-2 sciezkach (Ryc. S6A, S6B).
"""
import gzip
import re
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from figlib import MM, PAL, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyBigWig  # noqa: E402

NBINS = 500
M = Path("results/phase2/matrix")
LNC_SET = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA", "sense-intronic"}

genes = pd.read_csv("data/ref/genes_v29.tsv", sep="\t")
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")["klasa"]
cs = pd.read_csv("data/meta/case_samples.tsv", sep="\t", dtype=str)
man = pd.read_csv("data/raw/encode/download_manifest.tsv", sep="\t", dtype=str).drop_duplicates("file_accession", keep="last")
bwf = pd.read_csv("data/meta/case_bigwig_files.tsv", sep="\t", dtype=str).merge(man[["file_accession", "path"]], on="file_accession")
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
_LBC, _EX, _ORIENT = {}, {}, {}


def _exons(chrom, s, e):
    key = (chrom, s, e)
    if key not in _EX:
        rows = []
        with gzip.open("data/ref/gencode.v29.annotation.gtf.gz", "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.split("\t", 9)
                if f[2] != "exon" or f[0] != chrom:
                    continue
                a, b = int(f[3]), int(f[4])
                if b < s or a > e:
                    continue
                rows.append((a, b, f[6], re.search(r'gene_name "([^"]+)"', f[8]).group(1),
                             re.search(r'gene_id "([^"]+)"', f[8]).group(1)))
        _EX[key] = pd.DataFrame(rows, columns=["start", "end", "strand", "gene_name", "gene_id"])
    return _EX[key]


def _files(exp, rep):
    g = bwf[(bwf["experiment"] == exp) & (bwf["bio_rep"] == rep)].set_index("output_type")["path"]
    return g.get("plus strand signal of unique reads"), g.get("minus strand signal of unique reads")


def _strand_file(exp, rep, strand):
    p, m = _files(exp, rep)
    if p is None or m is None:
        return None
    if (exp, rep) not in _ORIENT:
        gapdh = genes[genes["gene_name"] == "GAPDH"].iloc[0]
        vals = []
        for path in (p, m):
            bw = pyBigWig.open(path)
            v = bw.stats(gapdh["chr"], int(gapdh["start"]), int(gapdh["end"]), type="mean")[0] if gapdh["chr"] in bw.chroms() else 0
            vals.append(v or 0)
        _ORIENT[(exp, rep)] = vals[0] >= vals[1]          # GAPDH lezy na nici +
    plus_is_plus = _ORIENT[(exp, rep)]
    return (p if plus_is_plus else m) if strand == "+" else (m if plus_is_plus else p)


def _coverage(exp, rep, strand, chrom, s, e):
    path = _strand_file(exp, rep, strand)
    if path is None:
        return None
    bw = pyBigWig.open(path)
    if chrom not in bw.chroms():
        return None
    return np.nan_to_num(np.array(bw.stats(chrom, s, min(e, bw.chroms()[chrom]), nBins=NBINS, type="mean"), dtype=float))


def _gene_signal(exp, rep, strand, chrom, exons):
    path = _strand_file(exp, rep, strand)
    if path is None:
        return np.nan
    bw = pyBigWig.open(path)
    tot, L = 0.0, 0
    for a, b in exons:
        v = bw.stats(chrom, int(a), int(b), type="mean")[0]
        tot += (v or 0) * (b - a); L += b - a
    return tot / L if L else np.nan


def _tick(v):
    return f"{v:.0f}" if v >= 10 else f"{v:.2g}"


def _merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _values(kd, gid, cell):
    out = {}
    f = Path(f"results/phase2/deseq2/{kd}.tsv.gz")
    if f.exists():
        d = pd.read_csv(f, sep="\t", usecols=["gene_id", "lfc_mle", "padj"]).set_index("gene_id")
        if gid in d.index:
            out["lfc"], out["padj"] = d.loc[gid, "lfc_mle"], d.loc[gid, "padj"]
    if cell not in _LBC and (M / f"lfc_bc_{cell}.tsv.gz").exists():
        _LBC[cell] = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    L = _LBC.get(cell)
    if L is not None and gid in L.index and kd in L.columns and pd.notna(L.loc[gid, kd]):
        row = L.loc[gid].dropna()
        out["cent"] = 100 * (row < L.loc[gid, kd]).mean()
    return out


def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n.a."
    if p == 0:
        return "<10$^{-300}$"
    m, e = f"{p:.0e}".split("e")
    return f"{m}×10$^{{{int(e)}}}$" if p < 1e-3 else f"{p:.2g}"


def _ordinal(x):
    n = int(round(x))
    suf = "th" if 10 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def draw_locus(figure, name, gene_names, focus, tracks, width_mm=150, track_mm=9.0, pad_frac=0.06, gene_mm=3.4, track_gap=3.0):
    gsel = genes[genes["gene_name"].isin(gene_names)]
    fg = genes[genes["gene_name"] == focus].iloc[0]
    chrom, strand, gid = fg["chr"], fg["strand"], fg["gene_id"]
    s0, e0 = int(gsel["start"].min()), int(gsel["end"].max())
    pad = int(pad_frac * (e0 - s0)) + 500
    s, e = s0 - pad, e0 + pad
    xs = np.linspace(s, e, NBINS)
    ex = _exons(chrom, s, e)
    ex = ex[~ex["gene_name"].str.match(r"^RF\d{5}$")]
    fex = _merge(ex.loc[ex["gene_id"] == gid, ["start", "end"]].values.tolist())
    tr = []
    for rbp, cell in tracks:
        k = cs[(cs["target"] == rbp) & (cs["cell"] == cell)]
        for kd, kk in k.groupby("kd_experiment"):
            tr.append((rbp, cell, kd, kk))
    order = ex.groupby("gene_name")["start"].min().sort_values().index.tolist()
    W = width_mm
    L0, Wm, Ws = 16, W - 16 - 26, 14                     # lewy margines, szerokosc sciezek, szerokosc panelu bocznego (mm)
    # modele genow: pakowanie w wiersze (zachlannie, w kolejnosci poczatkow); kazdy gen zajmuje przedzial razem z etykieta
    FS_G = 6.0
    bpmm = (e - s) / Wm                                    # par zasad na mm
    pad_bp = 0.8 * bpmm
    gitems = []
    for gname in order:
        gx = ex[ex["gene_name"] == gname]
        gt = genes[genes["gene_id"] == gx["gene_id"].iloc[0]]
        gs_, ge_ = (int(gt["start"].iloc[0]), int(gt["end"].iloc[0])) if len(gt) else (gx["start"].min(), gx["end"].max())
        st_ = gx["strand"].iloc[0]
        txt = f"{gname} {'→' if st_ == '+' else '←'}"
        lw_bp = (len(txt) * 0.56 * FS_G * 0.3528 + 0.6) * bpmm   # przyblizona szerokosc etykiety (Arial, 6 pt) + odstep
        gitems.append({"name": gname, "gx": gx, "a": max(gs_, s), "b": min(ge_, e), "txt": txt, "lw": lw_bp})
    grows = []
    for it in gitems:
        cands = [("left", (it["a"] - it["lw"], it["b"]))]
        if it["b"] + it["lw"] <= e:
            cands.append(("right", (it["a"], it["b"] + it["lw"])))
        done = False
        for side, iv in cands:
            for r_, occ in enumerate(grows):
                if all(iv[1] + pad_bp < o[0] or iv[0] - pad_bp > o[1] for o in occ):
                    occ.append(iv); it["row"], it["side"] = r_, side; done = True
                    break
            if done:
                break
        if not done:
            grows.append([cands[0][1]]); it["row"], it["side"] = len(grows) - 1, "left"
    nrow = max(len(grows), 1)
    gm_mm = gene_mm * (nrow + 0.1)
    GAP = 4.0                                             # odstep miedzy modelem genow a pierwsza sciezka (mm)
    H = gm_mm + GAP + track_mm * len(tr) + track_gap * len(tr) + 10
    fig = plt.figure(figsize=(W * MM, H * MM))
    y_top = H - 2
    axg = fig.add_axes([L0 / W, (y_top - gm_mm) / H, Wm / W, gm_mm / H])
    for it in gitems:
        gx, y = it["gx"], -it["row"]
        kl = cls.get(gx["gene_id"].iloc[0], None)
        gname = it["name"]
        col = PAL["dark"] if gname == focus else ("#7A7A7A" if kl == "mRNA" else (PAL["lncRNA"] if kl in LNC_SET else "#555555"))
        axg.plot([it["a"], it["b"]], [y, y], color=col, lw=0.6, solid_capstyle="butt")
        for a, b in _merge(gx[["start", "end"]].values.tolist()):
            axg.add_patch(Rectangle((max(a, s), y - 0.25), min(b, e) - max(a, s), 0.5, color=col, lw=0))
        if it["side"] == "left":
            axg.text(it["a"] - 0.6 * bpmm, y, it["txt"], fontsize=FS_G, color=col, ha="right", va="center")
        else:
            axg.text(it["b"] + 0.6 * bpmm, y, it["txt"], fontsize=FS_G, color=col, ha="left", va="center")
    axg.set_xlim(s, e); axg.set_ylim(-(nrow - 1) - 0.55, 0.55)
    axg.axis("off")
    rows = []
    for j, (rbp, cell, kd, kk) in enumerate(tr):
        yb = y_top - gm_mm - GAP - (j + 1) * (track_mm + track_gap)
        ax = fig.add_axes([L0 / W, yb / H, Wm / W, track_mm / H])
        axs = fig.add_axes([(L0 + Wm + 6) / W, yb / H, Ws / W, track_mm / H])
        ctrl, kdr = kk[kk["condition"] == "control"], kk[kk["condition"] == "kd"]
        cc = [c_ for c_ in (_coverage(r.sample_experiment, r.bio_rep, strand, chrom, s, e) for r in ctrl.itertuples()) if c_ is not None]
        kc = [c_ for c_ in (_coverage(kd, r.bio_rep, strand, chrom, s, e) for r in kdr.itertuples()) if c_ is not None]
        c_cov = np.mean(cc, axis=0) if cc else np.zeros(NBINS)
        k_cov = np.mean(kc, axis=0) if kc else np.zeros(NBINS)
        ax.fill_between(xs, 0, c_cov, color="#C8C8C8", lw=0, step="mid")
        ax.plot(xs, k_cov, color=PAL[cell], lw=0.6, drawstyle="steps-mid")
        top = max(c_cov.max(), k_cov.max(), 1e-3) * 1.15
        ax.set_xlim(s, e); ax.set_ylim(0, top)
        ax.set_yticks([0, top / 1.15]); ax.set_yticklabels(["0", _tick(top / 1.15)], fontsize=5)
        ax.tick_params(axis="y", length=1.5, pad=1)
        ax.spines["bottom"].set_visible(j == len(tr) - 1)
        if j < len(tr) - 1:
            ax.set_xticks([])
        meth = summ.loc[kd, "method"] if kd in summ.index else ""
        ax.text(0.0, 1 + 0.4 / track_mm, f"{rbp} · {cell} · {meth}", transform=ax.transAxes, fontsize=6, va="bottom")
        # panel boczny: poziom genu w replikach
        cv = [_gene_signal(r.sample_experiment, r.bio_rep, strand, chrom, fex) for r in ctrl.itertuples()]
        kv = [_gene_signal(kd, r.bio_rep, strand, chrom, fex) for r in kdr.itertuples()]
        for x_, vals, colr in [(0, cv, "#9A9A9A"), (1, kv, PAL[cell])]:
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                axs.scatter(np.full(len(vals), x_) + np.linspace(-0.08, 0.08, len(vals)), vals, s=5, color=colr, lw=0, zorder=3)
                axs.plot([x_ - 0.22, x_ + 0.22], [np.mean(vals)] * 2, color="k", lw=0.6)
        hi = max([v for v in cv + kv if np.isfinite(v)] + [1e-3]) * 1.25
        axs.set_xlim(-0.5, 1.5); axs.set_ylim(0, hi)
        axs.set_yticks([0, hi / 1.25]); axs.set_yticklabels(["0", _tick(hi / 1.25)], fontsize=5)
        axs.tick_params(axis="y", length=1.5, pad=1)
        axs.set_xticks([0, 1]); axs.set_xticklabels(["ctrl", "KD"] if j == len(tr) - 1 else ["", ""], fontsize=5)
        v = _values(kd, gid, cell)
        lab = []
        if "lfc" in v:
            lab.append(f"{v['lfc']:+.2f}, {_fmt_p(v['padj'])}")
        if "cent" in v:
            lab.append(f"{_ordinal(v['cent'])} pct")
        axs.text(1.08, 0.5, "\n".join(lab), transform=axs.transAxes, fontsize=4.8, va="center")
        rows.append({"wyciszenie": kd, "RBP": rbp, "linia": cell, "metoda": meth, "log2FC": v.get("lfc"), "padj": v.get("padj"),
                     "centyl": v.get("cent"), "kontrola_RPM": np.nanmean(cv) if cv else np.nan, "wyciszenie_RPM": np.nanmean(kv) if kv else np.nan})
    dec = max(2, int(np.ceil(-np.log10((e - s) / 1e6))) + 1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda val, _: f"{val / 1e6:.{dec}f}"))
    ax.set_xlabel(f"{chrom} (Mb)", fontsize=6)
    # opis osi dopasowany do wysokosci sciezek (runda 4): przy 1-2 sciezkach krotszy, by nie wystawal ponad sciezki
    ylab = (f"RNA-seq signal, {strand} strand (RPM)" if len(tr) >= 3 else
            f"RNA-seq signal,\n{strand} strand (RPM)" if len(tr) == 2 else f"RPM\n({strand} strand)")
    fig.text(4 / W, (y_top - gm_mm - GAP - (len(tr) * (track_mm + track_gap)) / 2) / H, ylab,
             rotation=90, va="center", ha="center", fontsize=6, linespacing=1.0)
    fig.text((L0 + Wm + 6 + Ws / 2) / W, (y_top - gm_mm - GAP + 1) / H, f"{focus} (RPM)", ha="center", va="bottom", fontsize=5.5)
    save(fig, figure, name)
    pd.DataFrame(rows).to_csv(Path("results/figures_pub") / figure / f"{name}.tsv", sep="\t", index=False)


def gene_profile(focus, tracks, nbins=10):
    """Sredni sygnal (RPM) w nbins rownych odcinkach genu od TSS do konca (5' -> 3'), kontrola i wyciszenie, dla kazdej sciezki."""
    fg = genes[genes["gene_name"] == focus].iloc[0]
    chrom, strand, a, b = fg["chr"], fg["strand"], int(fg["start"]), int(fg["end"])
    edges = np.linspace(a, b, nbins + 1).astype(int)
    out = []
    for rbp, cell in tracks:
        k = cs[(cs["target"] == rbp) & (cs["cell"] == cell)]
        for kd, kk in k.groupby("kd_experiment"):
            for cond, sub in [("kontrola", kk[kk["condition"] == "control"]), ("wyciszenie", kk[kk["condition"] == "kd"])]:
                vals = []
                for r in sub.itertuples():
                    path = _strand_file(r.sample_experiment, r.bio_rep, strand)
                    if path is None:
                        continue
                    bw = pyBigWig.open(path)
                    v = [bw.stats(chrom, int(edges[i]), int(edges[i + 1]), type="mean")[0] or 0 for i in range(nbins)]
                    vals.append(v[::-1] if strand == "-" else v)       # zawsze od TSS (5') do konca (3')
                if vals:
                    out.append({"RBP": rbp, "linia": cell, "wyciszenie": kd, "warunek": cond,
                                **{f"odc{i + 1}": x for i, x in enumerate(np.mean(vals, axis=0))}})
    return pd.DataFrame(out)
