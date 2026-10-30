"""
Czesc 4B-2: ryciny locus dla studiow przypadkow (podpisy po angielsku, do publikacji).

Kazda rycina: model genow w regionie (GENCODE v29, scalone eksony; dla NEAT1 zaznaczony region obecny tylko
w dlugiej izoformie NEAT1_2), a ponizej po jednej sciezce na knockdown: pokrycie RNA-seq na nici genu
(sygnal unikalnych odczytow ENCODE, srednia z dwoch replik) w kontroli (szare pole) i po usunieciu RBP (linia),
oraz piki eCLIP tego RBP w tej samej linii, jesli sa. W podpisie sciezki: log2FC genu w DESeq2 (wzgledem wlasnej
kontroli) z padj, log2FC po korekcie na wspolna kontrole i centyl wsrod wszystkich knockdownow w linii.
Wyjscie: results/phase4/cases/locus_<nazwa>.(png|pdf)
"""
import gzip
import re
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import pyBigWig

OUT, M = Path("results/phase4/cases"), Path("results/phase2/matrix")
OUT.mkdir(parents=True, exist_ok=True)
NBINS = 600
COL = {"K562": "#3b6ea8", "HepG2": "#c8553d"}
LNC_SET = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA", "sense-intronic"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220})

# (nazwa pliku, geny do pokazania, gen badany, sciezki (RBP, linia), tytul)
LOCI = [("NEAT1_MALAT1_exosome", ["NEAT1", "MALAT1"], "NEAT1",
         [("MTREX", "K562"), ("MTREX", "HepG2"), ("EXOSC2", "HepG2"), ("EXOSC5", "HepG2")],
         "NEAT1 rises after removal of nuclear exosome components (neighbouring MALAT1 shown for comparison)"),
        ("KCNQ1OT1_exosome", ["KCNQ1OT1"], "KCNQ1OT1",
         [("MTREX", "HepG2"), ("EXOSC2", "HepG2"), ("EXOSC5", "HepG2"), ("MTREX", "K562")],
         "KCNQ1OT1 rises after removal of three nuclear exosome components (HepG2)"),
        ("ZFAS1_EJC", ["ZFAS1"], "ZFAS1",
         [("MAGOH", "K562"), ("MAGOH", "HepG2"), ("EIF4A3", "K562"), ("EIF4A3", "HepG2")],
         "The snoRNA host ZFAS1 rises after removal of exon junction complex proteins"),
        ("SNHG5_EJC", ["SNHG5"], "SNHG5",
         [("MAGOH", "K562"), ("MAGOH", "HepG2"), ("EIF4A3", "K562"), ("EIF4A3", "HepG2")],
         "The snoRNA host SNHG5 rises after removal of exon junction complex proteins"),
        ("NORAD", ["NORAD", "LINC00657"], "NORAD",
         [("SRSF1", "K562"), ("SRSF1", "HepG2"), ("RCC2", "K562"), ("RCC2", "HepG2"), ("EIF3D", "K562"), ("EIF3D", "HepG2")],
         "NORAD falls after removal of the same three RBPs in both cell lines"),
        ("XIST_CIZ1", ["XIST"], "XIST", [("CIZ1", "K562")],
         "XIST rises after knockout of CIZ1, a known XIST-anchoring protein (K562)"),
        ("NEAT1_NONO", ["NEAT1"], "NEAT1", [("NONO", "K562"), ("NONO", "HepG2")],
         "NONO and NEAT1: the long isoform NEAT1_2 is barely captured by poly(A) RNA-seq"),
        ("OIP5-AS1_HuR", ["OIP5-AS1"], "OIP5-AS1", [("ELAVL1", "K562"), ("ELAVL1", "HepG2")],
         "OIP5-AS1 does not change after knockout of its binding partner HuR (ELAVL1)")]

EXTRA = {"NEAT1_MALAT1_exosome": ["MALAT1"]}   # dodatkowe geny, dla ktorych podajemy liczby w sciezkach


def fmt_p(p):
    return "< 1e-300" if p == 0 else f"{p:.0e}"


genes = pd.read_csv("data/ref/genes_v29.tsv", sep="\t")
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")["klasa"]
cs = pd.read_csv("data/meta/case_samples.tsv", sep="\t", dtype=str)
man = pd.read_csv("data/raw/encode/download_manifest.tsv", sep="\t", dtype=str).drop_duplicates("file_accession", keep="last")
bwf = pd.read_csv("data/meta/case_bigwig_files.tsv", sep="\t", dtype=str).merge(man[["file_accession", "path"]], on="file_accession")
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
ef = pd.read_csv("data/meta/eclip_files.tsv", sep="\t", dtype=str).merge(man[["file_accession", "path"]], on="file_accession", how="left")
ef["target"] = ef["target"].replace({"SKIV2L2": "MTREX"})

# ---------------------------------------------------------------- modele genow (eksony z GTF dla potrzebnych chromosomow)
need = []
for _, names, focus, _, _ in LOCI:
    g = genes[genes["gene_name"].isin(names)]
    if not g.empty:
        need.append((g["chr"].iloc[0], int(g["start"].min()), int(g["end"].max())))
print("Czytam eksony z GTF...", flush=True)
ex = []
with gzip.open("data/ref/gencode.v29.annotation.gtf.gz", "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.split("\t", 9)
        if f[2] != "exon":
            continue
        c, s, e = f[0], int(f[3]), int(f[4])
        if any(c == nc and e >= ns - 200_000 and s <= ne + 200_000 for nc, ns, ne in need):
            ex.append((c, s, e, f[6], re.search(r'gene_name "([^"]+)"', f[8]).group(1),
                       re.search(r'gene_id "([^"]+)"', f[8]).group(1), re.search(r'transcript_id "([^"]+)"', f[8]).group(1)))
ex = pd.DataFrame(ex, columns=["chr", "start", "end", "strand", "gene_name", "gene_id", "tx"])

# ---------------------------------------------------------------- orientacja plikow nici (raz na probke, na GAPDH)
gapdh = genes[genes["gene_name"] == "GAPDH"].iloc[0]
orient = {}


def files_for(exp, rep):
    g = bwf[(bwf["experiment"] == exp) & (bwf["bio_rep"] == rep)].set_index("output_type")["path"]
    return g.get("plus strand signal of unique reads"), g.get("minus strand signal of unique reads")


def strand_file(exp, rep, strand):
    p, m = files_for(exp, rep)
    if p is None or m is None:
        return None
    if (exp, rep) not in orient:
        vals = []
        for path in (p, m):
            bw = pyBigWig.open(path)
            v = bw.stats(gapdh["chr"], int(gapdh["start"]), int(gapdh["end"]), type="mean")[0] if gapdh["chr"] in bw.chroms() else 0
            vals.append(v or 0)
        orient[(exp, rep)] = "same" if vals[0] >= vals[1] else "flipped"   # GAPDH lezy na nici +
    plus_is_plus = orient[(exp, rep)] == "same"
    return (p if plus_is_plus else m) if strand == "+" else (m if plus_is_plus else p)


def coverage(exp, reps, strand, chrom, s, e):
    tr = []
    for rep in reps:
        path = strand_file(exp, rep, strand)
        if path is None:
            continue
        bw = pyBigWig.open(path)
        if chrom not in bw.chroms():
            continue
        v = np.array(bw.stats(chrom, s, min(e, bw.chroms()[chrom]), nBins=NBINS, type="mean"), dtype=float)
        tr.append(np.nan_to_num(v))
    return np.mean(tr, axis=0) if tr else np.zeros(NBINS)


def gene_values(kd, gid, cell):
    """log2FC i padj z DESeq2, log2FC po korekcie i centyl wsrod knockdownow w linii."""
    out = {}
    f = Path(f"results/phase2/deseq2/{kd}.tsv.gz")
    if f.exists():
        d = pd.read_csv(f, sep="\t", usecols=["gene_id", "lfc_mle", "padj"]).set_index("gene_id")
        if gid in d.index:
            out["raw"], out["padj"] = d.loc[gid, "lfc_mle"], d.loc[gid, "padj"]
    Lf = M / f"lfc_bc_{cell}.tsv.gz"
    if Lf.exists():
        row = LBC[cell].loc[gid] if gid in LBC[cell].index else None
        if row is not None and kd in row.index and pd.notna(row[kd]):
            x = row.dropna()
            out["bc"], out["cent"] = row[kd], 100 * (x < row[kd]).mean()
    return out


LBC = {c: pd.read_csv(M / f"lfc_bc_{c}.tsv.gz", sep="\t", index_col=0) for c in ["K562", "HepG2"] if (M / f"lfc_bc_{c}.tsv.gz").exists()}


def eclip_peaks(rbp, cell, chrom, s, e, strand):
    f = ef[(ef["target"] == rbp) & (ef["cell"] == cell)]
    if f.empty or pd.isna(f["path"].iloc[0]):
        return None
    p = pd.read_csv(f["path"].iloc[0], sep="\t", header=None, usecols=[0, 1, 2, 5, 6, 7])
    p.columns = ["chr", "s", "e", "strand", "l2fc", "mlog10p"]
    return p[(p["chr"] == chrom) & (p["strand"] == strand) & (p["e"] >= s) & (p["s"] <= e) & (p["l2fc"] >= 3) & (p["mlog10p"] >= 3)]


# ---------------------------------------------------------------- rysowanie
for fname, names, focus, tracks, title in LOCI:
    gsel = genes[genes["gene_name"].isin(names)]
    if gsel.empty:
        print(f"   [pomijam] {fname}: brak genu {names}")
        continue
    fg = genes[genes["gene_name"] == focus] if focus in set(genes["gene_name"]) else gsel.iloc[[0]]
    fg = fg.iloc[0]
    chrom, strand = fg["chr"], fg["strand"]
    s0, e0 = int(gsel["start"].min()), int(gsel["end"].max())
    pad = int(0.08 * (e0 - s0)) + 1000
    s, e = s0 - pad, e0 + pad
    xs = np.linspace(s, e, NBINS)
    tr_ok = []
    for rbp, cell in tracks:
        k = cs[(cs["target"] == rbp) & (cs["cell"] == cell)]
        if k.empty:
            print(f"   [uwaga] {fname}: brak knockdownu {rbp} {cell}")
            continue
        for kd, kk in k.groupby("kd_experiment"):
            tr_ok.append((rbp, cell, kd, kk))
    nt = len(tr_ok)
    if nt == 0:
        print(f"   [pomijam] {fname}: brak sciezek")
        continue
    has_ec = {(r, c): eclip_peaks(r, c, chrom, s, e, strand) for r, c in tracks}
    n_ec = sum(1 for v in has_ec.values() if v is not None)
    hr = [1.1] + [1.0] * nt + ([0.25] * n_ec)
    fig, axes = plt.subplots(len(hr), 1, figsize=(8.5, 1.0 + 1.05 * nt + 0.3 * n_ec + 0.8), sharex=True,
                             gridspec_kw={"height_ratios": hr, "hspace": 0.35})
    axes = np.atleast_1d(axes)
    # model genow
    ax = axes[0]
    reg = ex[(ex["chr"] == chrom) & (ex["end"] >= s) & (ex["start"] <= e) & ~ex["gene_name"].str.match(r"^RF\d{5}$")]
    order = reg.groupby("gene_name")["start"].min().sort_values().index
    for i, gname in enumerate(order):
        gx = reg[reg["gene_name"] == gname]
        y = -1.5 * i
        gt = genes[genes["gene_id"] == gx["gene_id"].iloc[0]]
        gs_, ge_ = (int(gt["start"].iloc[0]), int(gt["end"].iloc[0])) if len(gt) else (gx["start"].min(), gx["end"].max())
        st_ = gx["strand"].iloc[0]
        kl = cls.get(gx["gene_id"].iloc[0], None)
        color = "#7f7f7f" if kl == "mRNA" else ("#2a9d8f" if kl in LNC_SET else "#4d4d4d")
        ax.plot([max(gs_, s), min(ge_, e)], [y, y], color=color, lw=0.8)
        merged = gx.sort_values("start")[["start", "end"]].values
        for a, b in merged:
            ax.add_patch(Rectangle((a, y - 0.25), b - a, 0.5, color=color, lw=0))
        ax.text(max(gs_, s), y + 0.35, f"{gname} ({st_})", fontsize=7, color=color, va="bottom")
        if gname == "NEAT1":
            spans = gx.groupby("tx").agg(a=("start", "min"), b=("end", "max"))
            spans["L"] = spans["b"] - spans["a"]
            short, long_ = spans.sort_values("L").iloc[0], spans.sort_values("L").iloc[-1]
            r0, r1 = (short["b"], long_["b"]) if st_ == "+" else (long_["a"], short["a"])
            for a_ in axes:
                a_.axvspan(r0, r1, color="#f2e6c9", alpha=0.35, lw=0, zorder=0)
            ax.text((r0 + r1) / 2, y - 0.6, "NEAT1_2 only", fontsize=6, ha="center", va="top", color="#8a6d1f")
    ax.set_ylim(-1.5 * len(order) + 0.6, 1.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(title, fontsize=9, loc="left")
    # sciezki pokrycia
    gid = fg["gene_id"]
    for ax, (rbp, cell, kd, kk) in zip(axes[1:1 + nt], tr_ok):
        ctrl = kk[kk["condition"] == "control"]
        kdr = kk[kk["condition"] == "kd"]
        c_cov = np.mean([coverage(ex_, g_["bio_rep"].tolist(), strand, chrom, s, e) for ex_, g_ in ctrl.groupby("sample_experiment")], axis=0)
        k_cov = coverage(kd, kdr["bio_rep"].tolist(), strand, chrom, s, e)
        ax.fill_between(xs, 0, c_cov, color="#c7c7c7", lw=0, step="mid", label="control")
        ax.plot(xs, k_cov, color=COL[cell], lw=0.9, drawstyle="steps-mid", label=f"{rbp} removed")
        lab = f"{rbp}, {cell} ({summ.loc[kd, 'method'] if kd in summ.index else ''})"
        txt = []
        lines = []
        for gname_, gid_ in [(focus, gid)] + [(x_, genes.loc[genes["gene_name"] == x_, "gene_id"].iloc[0])
                                               for x_ in EXTRA.get(fname, []) if x_ in set(genes["gene_name"])]:
            v = gene_values(kd, gid_, cell)
            txt = []
            if "raw" in v:
                txt.append(f"{gname_} log2FC {v['raw']:+.2f} (padj {fmt_p(v['padj'])})" if pd.notna(v.get("padj")) else f"{gname_} log2FC {v['raw']:+.2f}")
            if "bc" in v:
                txt.append(f"after control correction {v['bc']:+.2f} ({v['cent']:.0f}th percentile of all knockdowns in {cell})")
            lines.append("; ".join(txt))
        ax.text(0.005, 0.97, lab, transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="top")
        for li, t_ in enumerate(lines):
            ax.text(0.005, 0.74 - 0.2 * li, t_, transform=ax.transAxes, fontsize=6.5, va="top", color="0.25")
        ax.set_ylabel("signal", fontsize=7)
        ax.set_ylim(0, max(c_cov.max(), k_cov.max(), 1e-3) * (1.35 + 0.25 * (len(lines) - 1)))
        ax.legend(frameon=False, fontsize=6, loc="upper right")
    # piki eCLIP
    j = 1 + nt
    for (r, c), pk in has_ec.items():
        if pk is None:
            continue
        ax = axes[j]
        j += 1
        for _, q in pk.iterrows():
            ax.add_patch(Rectangle((q["s"], 0), max(q["e"] - q["s"], (e - s) / 600), 1, color=COL[c], lw=0))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.text(0.005, 0.5, f"eCLIP {r}, {c}: {len(pk)} peaks", transform=ax.transAxes, fontsize=6.5, va="center")
    axes[-1].set_xlim(s, e)
    axes[-1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.3f}"))
    axes[-1].set_xlabel(f"{chrom} position (Mb); RNA-seq signal (reads per million) on the {strand} strand, mean of two replicates")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"locus_{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"   zapisano locus_{fname} ({nt} sciezek, eCLIP: {n_ec})")
print(f"Zapisano: {OUT}/locus_*.png i .pdf")
