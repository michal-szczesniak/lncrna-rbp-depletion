"""
Faza 3, krok 3.2B: EISA dla knockdownow spliceosomu i NMD/EJC.

Pytanie: czy spadek lncRNA po wyciszeniu spliceosomu zachodzi na poziomie prekursora (introny, d_intron),
czy dojrzalego transkryptu wzgledem prekursora (d_post = d_ekson - d_intron)?

Dla kazdego knockdownu: d_ekson, d_intron, d_post na gen (geny wieloeksonowe z >= 10 odczytami w eksonach i intronach
w kazdej probce), potem roznica lncRNA minus dopasowane mRNA dla kazdej skladowej:
  A = dopasowanie wg ekspresji i dlugosci; B = dodatkowo wydajnosc splicingu (kontrole z tego zestawu) i stabilnosc (BruChase).
lncRNA = lincRNA, antysens, dywergentny, inne lncRNA, gospodarze miRNA; gospodarze snoRNA osobno.
Podsumowanie: rozklad tych roznic w grupie (spliceosom, NMD/EJC) w kazdej linii oraz konsensus na gen
i jego zwiazek ze stabilnoscia bazowa.
Zastrzezenie: EISA liczona wzgledem wlasnej kontroli (bez korekty na partie); konsensus z wielu knockdownow z roznych
partii oslabia wplyw pojedynczej kontroli.

Sumy sygnalu bigWig: wykorzystuje probki policzone w pilotazu (results/phase1/eisa/sums), nowe zapisuje w
results/phase3/eisa/sums. Uruchamiac jako zadanie SLURM (8 rdzeni, 32 GB), ok. 15-30 min.
"""
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors

OUT = Path("results/phase3/eisa")
SUMS = OUT / "sums"
SUMS.mkdir(parents=True, exist_ok=True)
OLD_SUMS = Path("results/phase1/eisa/sums")
REGIONS = Path("data/ref/eisa_regions_v29.tsv.gz")
NPROC = int(os.environ.get("SLURM_CPUS_PER_TASK", min(8, os.cpu_count() or 1)))
MIN_READS, PSEUDO, K_NEIGHBOURS = 10, 8, 10
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz miRNA"}
GROUPS = {"lncRNA": LNC, "gospodarz snoRNA": {"gospodarz snoRNA"}}
VARIANTS = {"A": ["log_ekspr", "log_dl"], "B": ["log_ekspr", "log_dl", "splicing", "stab_6h"]}
classes = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
if not REGIONS.exists():
    raise SystemExit("Brak data/ref/eisa_regions_v29.tsv.gz - uruchom najpierw scripts/16_pilot_eisa.py (buduje regiony)")

regions = pd.read_csv(REGIONS, sep="\t")
gt = regions.groupby("gene_id", sort=True).agg(chr=("chr", "first"), strand=("strand", "first"),
                                               s=("start", "min"), e=("end", "max")).reset_index()
GENE_IDS, G_CHR, G_STRAND = gt["gene_id"].values, gt["chr"].values, gt["strand"].values
G_S, G_E, NG = gt["s"].values.astype(np.int64), gt["e"].values.astype(np.int64), len(gt)
_gi = pd.Series(np.arange(NG), index=GENE_IDS)
regions["g"] = _gi.loc[regions["gene_id"]].values
regions = regions.sort_values(["g", "start"])
P_START, P_END = regions["start"].values.astype(np.int64), regions["end"].values.astype(np.int64)
P_INTRON = (regions["kind"] == "intron").values
PTR = np.r_[0, np.cumsum(np.bincount(regions["g"].values, minlength=NG))]
LEN_EXON = np.bincount(regions["g"].values, weights=(P_END - P_START) * ~P_INTRON, minlength=NG)
LEN_INTRON = np.bincount(regions["g"].values, weights=(P_END - P_START) * P_INTRON, minlength=NG)
del regions
TEST = np.array([], dtype=int)


def gene_sums(bw, idx):
    chroms = bw.chroms()
    ex, it = np.zeros(len(idx)), np.zeros(len(idx))
    for j, g in enumerate(idx):
        c = G_CHR[g]
        if c not in chroms:
            continue
        s, e = int(G_S[g]), int(min(G_E[g], chroms[c]))
        if e <= s:
            continue
        cs = np.r_[0.0, np.cumsum(np.nan_to_num(bw.values(c, s, e, numpy=True)), dtype=np.float64)]
        a, b = PTR[g], PTR[g + 1]
        vals = cs[np.clip(P_END[a:b] - s, 0, e - s)] - cs[np.clip(P_START[a:b] - s, 0, e - s)]
        ex[j], it[j] = vals[~P_INTRON[a:b]].sum(), vals[P_INTRON[a:b]].sum()
    return ex, it


def sample_sums(job):
    key, plus_path, minus_path = job
    for d in (OLD_SUMS, SUMS):
        cache = d / f"{key[0]}_{key[1]}.npz"
        if cache.exists():
            return key, float(np.load(cache)["frac"]), str(cache), -1.0
    cache = SUMS / f"{key[0]}_{key[1]}.npz"
    t0 = time.time()
    bwp, bwm = pyBigWig.open(plus_path), pyBigWig.open(minus_path)
    tp, tm = TEST[G_STRAND[TEST] == "+"], TEST[G_STRAND[TEST] == "-"]
    pp, mp = gene_sums(bwp, tp)[0].sum(), gene_sums(bwm, tp)[0].sum()
    pm, mm = gene_sums(bwp, tm)[0].sum(), gene_sums(bwm, tm)[0].sum()
    frac = (pp + mm) / max(pp + mp + pm + mm, 1e-9)
    if frac > 0.8:
        files = {"+": bwp, "-": bwm}
    elif frac < 0.2:
        files = {"+": bwm, "-": bwp}
    else:
        raise RuntimeError(f"{key}: nie da sie ustalic nici (udzial sensowny {frac:.2f})")
    ex, it = np.zeros(NG), np.zeros(NG)
    for strand, bw in files.items():
        idx = np.where(G_STRAND == strand)[0]
        ex[idx], it[idx] = gene_sums(bw, idx)
    np.savez(cache, exon=ex, intron=it, frac=frac)
    return key, frac, str(cache), time.time() - t0


def class_diffs(d, feats):
    """Dla kazdej grupy klas: mediany roznic (gen klasy minus 10 najblizszych mRNA) dla d_ekson, d_intron, d_post."""
    d = d.dropna(subset=feats)
    X = d[feats].values
    m = (d["klasa"] == "mRNA").values
    if m.sum() < 100:
        return {}
    mu, sd = X[m].mean(0), X[m].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS).fit((X[m] - mu) / sd)
    out = {}
    for name, members in GROUPS.items():
        sel = d["klasa"].isin(members).values
        if sel.sum() < 15:
            continue
        _, idx = nn.kneighbors((X[sel] - mu) / sd)
        out[name] = {c: float(np.median(d[c].values[sel] - d[c].values[m][idx].mean(1))) for c in ["d_ekson", "d_intron", "d_post"]}
        out[name]["n"] = int(sel.sum())
    return out


if __name__ == "__main__":
    man = pd.read_csv("data/raw/encode/download_manifest.tsv", sep="\t", dtype=str)
    bwf = pd.read_csv("data/meta/phase3_bigwig_files.tsv", sep="\t", dtype=str).merge(
        man.drop_duplicates("file_accession", keep="last")[["file_accession", "path"]], on="file_accession")
    smp = pd.read_csv("data/meta/phase3_samples.tsv", sep="\t", dtype=str)
    rsem = smp.drop_duplicates(["sample_experiment", "bio_rep"]).set_index(["sample_experiment", "bio_rep"])["path"]
    q = pd.read_csv(rsem.iloc[0], sep="\t", usecols=["gene_id", "expected_count"])
    q = q.merge(classes[classes["klasa"] == "mRNA"][["gene_id"]], on="gene_id")
    TEST = _gi.loc[q[q["gene_id"].isin(_gi.index)].nlargest(300, "expected_count")["gene_id"]].values

    jobs = []
    for (exp, rep), g in bwf.groupby(["experiment", "bio_rep"]):
        p = g.set_index("output_type")["path"]
        jobs.append(((exp, rep), p["plus strand signal of unique reads"], p["minus strand signal of unique reads"]))
    print(f"1. Sumy sygnalu bigWig: {len(jobs)} probek na {NPROC} rdzeniach...", flush=True)
    res = {}
    with Pool(NPROC) as pool:
        for n, (key, frac, path, dt) in enumerate(pool.imap_unordered(sample_sums, jobs), 1):
            res[key] = (frac, path)
            if dt >= 0 or n % 25 == 0 or n == len(jobs):
                print(f"   [{n}/{len(jobs)}] {key[0]} replika {key[1]}: {'z dysku' if dt < 0 else f'{dt / 60:.1f} min'}", flush=True)
    print(f"   Probek z wlasciwa nicia >= 0.95: {sum(max(f, 1 - f) >= 0.95 for f, _ in res.values())} z {len(res)}")

    counts = {}
    for (exp, rep), (frac, path) in res.items():
        z = np.load(path)
        s = pd.DataFrame({"exon": z["exon"], "intron": z["intron"]}, index=GENE_IDS)
        rq = pd.read_csv(rsem[(exp, rep)], sep="\t", usecols=["gene_id", "expected_count"]).set_index("gene_id")["expected_count"]
        common = s.index.intersection(rq.index)
        counts[(exp, rep)] = s / (s.loc[common, "exon"].sum() / rq.loc[common].sum())

    meta = classes.set_index("gene_id")
    kin = pd.read_csv("results/phase1/bru/gene_kinetics.tsv", sep="\t")
    lens = pd.DataFrame({"len_ekson": LEN_EXON, "len_intron": LEN_INTRON}, index=GENE_IDS)
    # wydajnosc splicingu z kontroli tego zestawu, osobno dla linii
    splice = {}
    for cell, g in smp[smp["condition"] == "control"].drop_duplicates(["sample_experiment", "bio_rep"]).groupby("cell"):
        vals = []
        for e, r in zip(g["sample_experiment"], g["bio_rep"]):
            if (e, r) in counts:
                c = counts[(e, r)]
                vals.append(np.log2(((c["intron"] + 1) / lens["len_intron"]) / ((c["exon"] + 1) / lens["len_ekson"])))
        splice[cell] = pd.concat(vals, axis=1).median(axis=1)

    rows, gene_tabs = [], []
    for kd, s in smp.groupby("kd_experiment"):
        info = s.iloc[0]
        keys = {c: [(e, r) for e, r in zip(s.loc[s["condition"] == c, "sample_experiment"], s.loc[s["condition"] == c, "bio_rep"])
                    if (e, r) in counts] for c in ["kd", "control"]}
        if len(keys["kd"]) < 2 or len(keys["control"]) < 2:
            continue
        allk = keys["kd"] + keys["control"]
        E = pd.concat({k: counts[k]["exon"] for k in allk}, axis=1).fillna(0)
        I = pd.concat({k: counts[k]["intron"] for k in allk}, axis=1).fillna(0)
        cls = meta.reindex(E.index)["klasa"]
        ok = (E > 0).all(axis=1) & (E[keys["control"]].mean(axis=1) >= MIN_READS)
        sf = E[ok].div(np.exp(np.log(E[ok]).mean(axis=1)), axis=0).median()
        En, In = E / sf, I / sf
        use = (I >= MIN_READS).all(axis=1) & (E >= MIN_READS).all(axis=1) & cls.notna()
        ce, ci = En.loc[use, keys["control"]].mean(axis=1), In.loc[use, keys["control"]].mean(axis=1)
        d = pd.DataFrame({"klasa": cls[use], "ctrl_ekson": ce,
                          "d_ekson": np.log2(En.loc[use, keys["kd"]].mean(axis=1) + PSEUDO) - np.log2(ce + PSEUDO),
                          "d_intron": np.log2(In.loc[use, keys["kd"]].mean(axis=1) + PSEUDO) - np.log2(ci + PSEUDO)})
        d["d_post"] = d["d_ekson"] - d["d_intron"]
        d["log_ekspr"], d["log_dl"] = np.log10(d["ctrl_ekson"] + 1), np.log10(meta.reindex(d.index)["width"])
        d["splicing"] = splice.get(info["cell"], pd.Series(dtype=float)).reindex(d.index)
        d["stab_6h"] = kin[kin["cell"] == info["cell"]].drop_duplicates("gene_id").set_index("gene_id")["stab_6h"].reindex(d.index)
        d.to_csv(OUT / f"{kd}.tsv.gz", sep="\t")
        gene_tabs.append(d[["klasa", "d_ekson", "d_intron", "d_post", "stab_6h"]].assign(kd=kd, cell=info["cell"], grupa=info["grupa"]))
        for v, feats in VARIANTS.items():
            for grp, vals in class_diffs(d, feats).items():
                rows.append({"kd_experiment": kd, "target": info["target"], "cell": info["cell"], "grupa": info["grupa"],
                             "metoda": info["method"], "dopasowanie": v, "klasa": grp, **vals})
    kdres = pd.DataFrame(rows)
    kdres.to_csv(OUT / "kd_class_eisa.tsv", sep="\t", index=False)

    pd.set_option("display.width", 230)
    print("\n2. Roznica klasa minus dopasowane mRNA w knockdownach grupy: mediana z knockdownow; w nawiasie odsetek knockdownow < 0;")
    print("   p = test znakow rangowanych (czy roznice w grupie sa systematycznie rozne od zera)")
    out = []
    for (grupa, cell, v, k), g in kdres.groupby(["grupa", "cell", "dopasowanie", "klasa"]):
        row = {"grupa": grupa, "linia": cell, "dopas.": v, "klasa": k, "knockdowny": len(g)}
        for c in ["d_ekson", "d_intron", "d_post"]:
            x = g[c].dropna()
            p = wilcoxon(x).pvalue if len(x) >= 5 else np.nan
            row[c] = f"{x.median():+.3f} ({100 * (x < 0).mean():.0f}%) p={p:.0e}"
        out.append(row)
    print(pd.DataFrame(out).to_string(index=False))

    print("\n3. Konsensus na gen (mediana z knockdownow grupy, geny w >= 3 knockdownach) a stabilnosc bazowa (Spearman)")
    gt_all = pd.concat(gene_tabs)
    for (grupa, cell), g in gt_all.groupby(["grupa", "cell"]):
        cons = g.groupby(level=0).agg(klasa=("klasa", "first"), stab_6h=("stab_6h", "first"), n=("kd", "nunique"),
                                      d_ekson=("d_ekson", "median"), d_intron=("d_intron", "median"), d_post=("d_post", "median"))
        cons = cons[cons["n"] >= 3]
        cons.to_csv(OUT / f"consensus_{grupa.replace('/', '_')}_{cell}.tsv.gz", sep="\t")
        for name, sel in [("mRNA", cons["klasa"] == "mRNA"), ("lncRNA", cons["klasa"].isin(LNC))]:
            c = cons[sel].dropna(subset=["stab_6h"])
            if len(c) < 30:
                continue
            rs = {k: spearmanr(c["stab_6h"], c[k])[0] for k in ["d_ekson", "d_intron", "d_post"]}
            print(f"   {grupa:<11} {cell:<6} {name:<7} n={len(c):>5}: stabilnosc vs d_ekson {rs['d_ekson']:+.2f}, "
                  f"vs d_intron {rs['d_intron']:+.2f}, vs d_post {rs['d_post']:+.2f}")
    print(f"\nZapisano: {OUT}/kd_class_eisa.tsv, {OUT}/consensus_*.tsv.gz, {OUT}/<eksperyment>.tsv.gz")
