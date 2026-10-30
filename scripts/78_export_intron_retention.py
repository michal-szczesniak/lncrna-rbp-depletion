"""
Retencja intronow w PROBKACH KONTROLNYCH, na poziomie pojedynczych intronow z adnotacji GENCODE v29 (GRCh38).

Introny: miedzy kolejnymi eksonami kazdego transkryptu; unikalne wspolrzedne na gen (0-based, polotwarte).
Do sygnalu intronu licza sie tylko jego zasady niepokryte eksonami ZADNEGO genu na tej samej nici (unique_bp),
zeby odczyty z eksonow innych izoform czy genow nie udawaly retencji.
Sygnal: bigWig ENCODE "plus/minus strand signal of unique reads" probek kontrolnych pilotazu (nic sprawdzana na
silnie wyrazonych mRNA; pliki bywaja zamienione), skalibrowany do odczytow przez porownanie sumy sygnalu w eksonach
z expected_count RSEM tej samej probki. Odczyty sumowane po probkach kontrolnych danej linii.

Wyjscie: results/export/intron_retention_controls_v29.tsv.gz
  chrom, start, end, strand, gene_id, gene_name, cell_line, intron_len, unique_bp,
  intron_reads, exon_reads_gene, exon_bp_gene, ir_log2, n_ctrl
  ir_log2 = log2((intron_reads / unique_bp) / (exon_reads_gene / exon_bp_gene)); NA, gdy exon_reads_gene < 20
  lub unique_bp < 50. Wyzsze = wiecej niesplicingowanego RNA w tym intronie wzgledem dojrzalego RNA genu.
Uruchomienie: python scripts/78_export_intron_retention.py  (rownolegle na SLURM_CPUS_PER_TASK lub do 8 rdzeni)
"""
import gzip
import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

OUT = Path("results/export")
OUT.mkdir(parents=True, exist_ok=True)
GTF = Path("data/ref/gencode.v29.annotation.gtf.gz")
NPROC = int(os.environ.get("SLURM_CPUS_PER_TASK", min(8, os.cpu_count() or 1)))
MIN_EXON_READS, MIN_UNIQUE_BP = 20, 50


def merge(df, keys):
    df = df.sort_values(keys + ["start"]).reset_index(drop=True)
    same = (df[keys] == df[keys].shift()).all(axis=1)
    prev_end = df.groupby(keys)["end"].cummax().shift()
    df["blk"] = (~same | (df["start"] > prev_end)).cumsum()
    return df.groupby(keys + ["blk"], as_index=False).agg(start=("start", "min"), end=("end", "max")).drop(columns="blk")


# ---------------------------------------------------------------- 1. introny z GTF
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
keep = set(cls.loc[cls["klasa"].notna(), "gene_id"])
name_of = cls.set_index("gene_id")["gene_name"]
rows = []
with gzip.open(GTF, "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.split("\t", 8)
        if f[2] != "exon":
            continue
        gid = re.search(r'gene_id "([^"]+)"', f[8]).group(1)
        tid = re.search(r'transcript_id "([^"]+)"', f[8]).group(1)
        rows.append((gid, tid, f[0], f[6], int(f[3]) - 1, int(f[4])))
ex = pd.DataFrame(rows, columns=["gene_id", "transcript_id", "chrom", "strand", "start", "end"])
ex = ex[~ex["gene_id"].str.endswith("_PAR_Y")]                    # kopie PAR na chrY maja ten sam ID co na chrX
all_ex = merge(ex[["chrom", "strand", "start", "end"]].copy(), ["chrom", "strand"])
gene_ex = merge(ex[ex["gene_id"].isin(keep)][["gene_id", "chrom", "strand", "start", "end"]].copy(), ["gene_id", "chrom", "strand"])

e = ex[ex["gene_id"].isin(keep)].sort_values(["transcript_id", "start"])
nxt = e.groupby("transcript_id")["start"].shift(-1)
intr = pd.DataFrame({"gene_id": e["gene_id"], "chrom": e["chrom"], "strand": e["strand"], "start": e["end"], "end": nxt})
intr = intr.dropna()
intr["end"] = intr["end"].astype(int)
intr = intr[intr["end"] > intr["start"]].drop_duplicates(["gene_id", "chrom", "strand", "start", "end"]).reset_index(drop=True)
intr["iid"] = np.arange(len(intr))
print(f"1. Introny: {len(intr):,} unikalnych w {intr['gene_id'].nunique():,} genach", flush=True)

# segmenty intronu bez eksonow zadnego genu na tej samej nici
segs = []
for (ch, st), sub in intr.groupby(["chrom", "strand"]):
    u = all_ex[(all_ex["chrom"] == ch) & (all_ex["strand"] == st)].sort_values("start")
    us, ue = u["start"].values, u["end"].values
    for iid, s, t in zip(sub["iid"].values, sub["start"].values, sub["end"].values):
        i0, i1 = np.searchsorted(ue, s, "right"), np.searchsorted(us, t, "left")
        a = np.maximum(np.r_[s, ue[i0:i1]], s)
        b = np.minimum(np.r_[us[i0:i1], t], t)
        ok = b > a
        segs += [(iid, ch, st, x, y) for x, y in zip(a[ok], b[ok])]
segs = pd.DataFrame(segs, columns=["iid", "chrom", "strand", "start", "end"])
intr["unique_bp"] = segs.assign(L=segs["end"] - segs["start"]).groupby("iid")["L"].sum().reindex(intr["iid"]).fillna(0).astype(int).values
gex = gene_ex[gene_ex["gene_id"].isin(intr["gene_id"])].reset_index(drop=True)
exon_bp = gex.assign(L=gex["end"] - gex["start"]).groupby("gene_id")["L"].sum()
print(f"   segmenty intronowe bez eksonow: {len(segs):,}; eksony genow: {len(gex):,}", flush=True)


# ---------------------------------------------------------------- 2. sygnal bigWig w kontrolach
def sums(bw, df):
    chroms = bw.chroms()
    out = np.zeros(len(df))
    for i, (c, a, b) in enumerate(zip(df["chrom"].values, df["start"].values, df["end"].values)):
        if c in chroms and a < chroms[c]:
            v = bw.stats(c, int(a), int(min(b, chroms[c])), type="sum", exact=True)[0]
            out[i] = v or 0.0
    return out


def sample_job(job):
    key, plus, minus, rsem, test = job
    bwp, bwm = pyBigWig.open(plus), pyBigWig.open(minus)
    t = gex[gex["gene_id"].isin(test)]
    tp, tm = t[t["strand"] == "+"], t[t["strand"] == "-"]
    pp, mp, pm, mm = sums(bwp, tp).sum(), sums(bwm, tp).sum(), sums(bwp, tm).sum(), sums(bwm, tm).sum()
    frac = (pp + mm) / max(pp + mp + pm + mm, 1e-9)
    if frac > 0.8:
        files = {"+": bwp, "-": bwm}
    elif frac < 0.2:
        files = {"+": bwm, "-": bwp}
    else:
        raise RuntimeError(f"{key}: nie da sie ustalic nici (udzial {frac:.2f})")
    es, isg = [], []
    for st, bw in files.items():
        g = gex[gex["strand"] == st]
        es.append(pd.Series(sums(bw, g), index=g["gene_id"].values).groupby(level=0).sum())
        s = segs[segs["strand"] == st]
        isg.append(pd.Series(sums(bw, s), index=s["iid"].values).groupby(level=0).sum())
    E, I = pd.concat(es), pd.concat(isg)
    rq = pd.read_csv(rsem, sep="\t", usecols=["gene_id", "expected_count"]).set_index("gene_id")["expected_count"]
    common = E.index.intersection(rq.index)
    scale = E.loc[common].sum() / max(rq.loc[common].sum(), 1e-9)
    return key, frac, E / scale, I / scale


if __name__ == "__main__":
    man = pd.read_csv("data/raw/encode/download_manifest.tsv", sep="\t", dtype=str).drop_duplicates("file_accession", keep="last")
    bwf = pd.read_csv("data/meta/pilot_bigwig_files.tsv", sep="\t", dtype=str).merge(man[["file_accession", "path"]], on="file_accession")
    smp = pd.read_csv("data/meta/pilot_samples.tsv", sep="\t", dtype=str)
    ctrl = smp[smp["condition"] == "control"].drop_duplicates(["sample_experiment", "bio_rep"])
    q = pd.read_csv(ctrl["path"].iloc[0], sep="\t", usecols=["gene_id", "expected_count"])
    q = q[q["gene_id"].isin(set(cls.loc[cls["klasa"] == "mRNA", "gene_id"])) & q["gene_id"].isin(gex["gene_id"])]
    test = set(q.nlargest(300, "expected_count")["gene_id"])
    jobs, cell_of = [], {}
    for r in ctrl.itertuples():
        b = bwf[(bwf["experiment"] == r.sample_experiment) & (bwf["bio_rep"] == r.bio_rep)].set_index("output_type")["path"]
        if {"plus strand signal of unique reads", "minus strand signal of unique reads"} <= set(b.index):
            key = f"{r.sample_experiment}_{r.bio_rep}"
            jobs.append((key, b["plus strand signal of unique reads"], b["minus strand signal of unique reads"], r.path, test))
            cell_of[key] = r.cell
    print(f"2. Probki kontrolne z bigWig: {len(jobs)} (" + ", ".join(f"{c}: {sum(v == c for v in cell_of.values())}" for c in sorted(set(cell_of.values()))) + f"); {NPROC} rdzeni", flush=True)
    with Pool(NPROC) as pool:
        res = pool.map(sample_job, jobs)

    out = []
    for cell in sorted(set(cell_of.values())):
        rs = [x for x in res if cell_of[x[0]] == cell]
        E = pd.concat([x[2] for x in rs], axis=1).sum(axis=1)
        I = pd.concat([x[3] for x in rs], axis=1).sum(axis=1)
        d = intr.copy()
        d["cell_line"] = cell
        d["intron_len"] = d["end"] - d["start"]
        d["intron_reads"] = I.reindex(d["iid"]).fillna(0).values
        d["exon_reads_gene"] = E.reindex(d["gene_id"]).fillna(0).values
        d["exon_bp_gene"] = exon_bp.reindex(d["gene_id"]).values
        ok = (d["exon_reads_gene"] >= MIN_EXON_READS) & (d["unique_bp"] >= MIN_UNIQUE_BP)
        dens_i = d["intron_reads"] / d["unique_bp"].clip(lower=1)
        dens_e = d["exon_reads_gene"] / d["exon_bp_gene"]
        d["ir_log2"] = np.where(ok & (d["intron_reads"] > 0), np.log2(dens_i / dens_e), np.nan)
        d["n_ctrl"] = len(rs)
        print(f"   {cell}: {len(rs)} probek; nici sprawdzone (najnizszy udzial sygnalu sensownego {min(max(x[1], 1 - x[1]) for x in rs):.3f}); "
              f"introny z ir_log2: {int(d['ir_log2'].notna().sum()):,}", flush=True)
        out.append(d)
    out = pd.concat(out)
    out["gene_name"] = name_of.reindex(out["gene_id"]).values
    out["gene_id"] = out["gene_id"].str.split(".").str[0]
    cols = ["chrom", "start", "end", "strand", "gene_id", "gene_name", "cell_line", "intron_len", "unique_bp",
            "intron_reads", "exon_reads_gene", "exon_bp_gene", "ir_log2", "n_ctrl"]
    f = OUT / "intron_retention_controls_v29.tsv.gz"
    out[cols].sort_values(["cell_line", "chrom", "start"]).to_csv(f, sep="\t", index=False, float_format="%.4g")
    lnc = {"lincRNA", "antysens", "dywergentny", "sense-intronic", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
    is_lnc = out["gene_id"].isin(set(cls.loc[cls["klasa"].isin(lnc), "gene_id"].str.split(".").str[0]))
    s = out.assign(lnc=is_lnc, val=out["ir_log2"].notna()).groupby("cell_line").apply(
        lambda d: pd.Series({"introny": len(d), "z ir_log2": int(d["val"].sum()), "lncRNA introny": int(d["lnc"].sum()),
                             "lncRNA z ir_log2": int((d["lnc"] & d["val"]).sum())}))
    print(f"\nZapisano {f}\n{s.to_string()}")
    readme = OUT / "README_export.txt"
    txt = ["", "TABELA INTRONOW (retencja intronow w kontrolach)",
           f"  {Path.cwd() / f}",
           "  Kolumny: chrom, start, end (0-based, polotwarte, GRCh38), strand, gene_id (bez wersji), gene_name, cell_line;",
           "  intron_len = dlugosc intronu z adnotacji; unique_bp = zasady intronu niepokryte eksonami zadnego genu na tej nici;",
           "  intron_reads = odczyty (sygnal bigWig skalibrowany do RSEM) w unique_bp, suma po probkach kontrolnych;",
           "  exon_reads_gene / exon_bp_gene = odczyty i dlugosc sumy eksonow genu w tych samych probkach;",
           f"  ir_log2 = log2((intron_reads/unique_bp)/(exon_reads_gene/exon_bp_gene)), NA gdy exon_reads_gene < {MIN_EXON_READS} lub unique_bp < {MIN_UNIQUE_BP};",
           "  n_ctrl = liczba probek kontrolnych (replik) pilotazu z plikami bigWig.",
           "  Liczby intronow:", s.to_string()]
    with open(readme, "a", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")
    print(f"Dopisano do {readme}")
