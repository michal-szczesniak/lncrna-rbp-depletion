"""
Faza 1, krok 5B (wersja 2): EISA (exon-intron split analysis) dla pilotazu.

Idea (Gaidatzis i wsp. 2015): zmiana odczytow z intronow przybliza zmiane transkrypcji, zmiana odczytow
z eksonow to transkrypcja + obrobka/stabilnosc; d_post = d_ekson - d_intron przybliza efekt potranskrypcyjny.
Zastrzezenie: lncRNA splicinguja sie wolniej i mniej wydajnie (Mukherjee i wsp. 2017), wiec ich odczyty
intronowe to nie tylko RNA powstajace - stad dopasowanie B ponizej.

Zmiany wzgledem wersji 1:
 - sygnal czytany raz na gen (caly odcinek genu) zamiast osobno dla kazdego regionu: kilkukrotnie szybciej;
 - procesy robocze dostaja tylko tablice numpy (mniej pamieci); wynik kazdej probki zapisywany
   w results/phase1/eisa/sums/, wiec przerwane obliczenia wznawiaja sie od miejsca przerwania;
 - porownanie klas z mRNA w dwoch wariantach dopasowania:
     A: ekspresja + dlugosc genu,
     B: A + wydajnosc splicingu (gestosc odczytow intron/ekson w kontrolach) + stabilnosc bazowa (BruChase 6 h).

Uruchomienie: python -u scripts/16_pilot_eisa.py   (rdzenie: SLURM_CPUS_PER_TASK, domyslnie do 8)
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

OUT = Path("results/phase1/eisa")
SUMS = OUT / "sums"
SUMS.mkdir(parents=True, exist_ok=True)
GTF = "data/ref/gencode.v29.annotation.gtf.gz"
REGIONS = Path("data/ref/eisa_regions_v29.tsv.gz")
NPROC = int(os.environ.get("SLURM_CPUS_PER_TASK", min(8, os.cpu_count() or 1)))
MIN_READS, PSEUDO, K_NEIGHBOURS, MIN_INTRON_BP = 10, 8, 10, 100
ORDER = ["lncRNA (wszystkie)", "lincRNA", "antysens", "dywergentny", "sense-intronic", "inne lncRNA",
         "gospodarz snoRNA", "gospodarz miRNA", "pseudogen transkr."]
LNC_ALL = {"lincRNA", "antysens", "dywergentny", "sense-intronic", "inne lncRNA",
           "gospodarz snoRNA", "gospodarz miRNA"}
VARIANTS = {"A": ["log_ekspr", "log_dl"], "B": ["log_ekspr", "log_dl", "splicing", "stab_6h"]}
classes = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")


# ----------------------------------------------------------------------------- 1. regiony
def merge_intervals(d, keys):
    d = d.sort_values(keys + ["start"]).reset_index(drop=True)
    same = (d[keys] == d[keys].shift()).all(axis=1)
    prev_end = d.groupby(keys)["end"].cummax().shift()
    d["blk"] = (~same | (d["start"] > prev_end)).cumsum()
    return d.groupby(keys + ["blk"], as_index=False).agg(start=("start", "min"), end=("end", "max")).drop(columns="blk")


def build_regions():
    print("Buduje regiony eksonow i intronow z GTF (raz, ok. 1-2 min)...", flush=True)
    g = pd.read_csv(GTF, sep="\t", comment="#", header=None, usecols=[0, 2, 3, 4, 6, 8],
                    names=["chr", "feature", "start", "end", "strand", "attr"])
    ex = g[g["feature"] == "exon"].copy()
    ex["gene_id"] = ex["attr"].str.extract(r'gene_id "([^"]+)"', expand=False)
    ex["start"] -= 1                                   # 0-based, polotwarte
    ex = ex[["gene_id", "chr", "strand", "start", "end"]]
    gene_ex = merge_intervals(ex, ["gene_id", "chr", "strand"])
    all_ex = merge_intervals(ex[["chr", "strand", "start", "end"]], ["chr", "strand"])
    keep = set(classes.loc[classes["klasa"].notna(), "gene_id"])
    gene_ex = gene_ex[gene_ex["gene_id"].isin(keep)]
    n_ex = gene_ex.groupby("gene_id").size()
    gene_ex = gene_ex[gene_ex["gene_id"].isin(n_ex[n_ex >= 2].index)]   # tylko geny wieloeksonowe
    span = gene_ex.groupby(["gene_id", "chr", "strand"], as_index=False).agg(s=("start", "min"), e=("end", "max"))
    introns = []
    for (chrom, strand), sp in span.groupby(["chr", "strand"]):
        u = all_ex[(all_ex["chr"] == chrom) & (all_ex["strand"] == strand)]
        us, ue = u["start"].values, u["end"].values
        for gid, s, e in zip(sp["gene_id"], sp["s"], sp["e"]):
            i0, i1 = np.searchsorted(ue, s, "right"), np.searchsorted(us, e, "left")
            gs = np.maximum(np.r_[s, ue[i0:i1]], s)
            ge = np.minimum(np.r_[us[i0:i1], e], e)
            ok = ge > gs
            introns += [(gid, chrom, strand, a, b) for a, b in zip(gs[ok], ge[ok])]
    intr = pd.DataFrame(introns, columns=["gene_id", "chr", "strand", "start", "end"])
    ilen = intr.assign(L=intr["end"] - intr["start"]).groupby("gene_id")["L"].sum()
    intr = intr[intr["gene_id"].isin(ilen[ilen >= MIN_INTRON_BP].index)]
    reg = pd.concat([gene_ex.assign(kind="exon"), intr.assign(kind="intron")], ignore_index=True)
    reg = reg[reg["gene_id"].isin(intr["gene_id"])]
    reg.to_csv(REGIONS, sep="\t", index=False)
    return reg


regions = pd.read_csv(REGIONS, sep="\t") if REGIONS.exists() else build_regions()

# tablice numpy wspoldzielone przez procesy robocze (bez tabel pandas = mniej pamieci)
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
print(f"1. Regiony: {NG} genow wieloeksonowych z intronami "
      f"({int((~P_INTRON).sum())} fragmentow eksonowych, {int(P_INTRON.sum())} intronowych)", flush=True)
del regions
TEST = np.array([], dtype=int)   # geny do wykrycia nici; ustawiane w czesci glownej przed startem procesow


# ----------------------------------------------------------------------------- 2. sumy z bigWig
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
    key, plus_path, minus_path, cache = job
    if os.path.exists(cache):
        return key, float(np.load(cache)["frac"]), cache, -1.0
    t0 = time.time()
    bwp, bwm = pyBigWig.open(plus_path), pyBigWig.open(minus_path)
    tp, tm = TEST[G_STRAND[TEST] == "+"], TEST[G_STRAND[TEST] == "-"]
    pp, mp = gene_sums(bwp, tp)[0].sum(), gene_sums(bwm, tp)[0].sum()   # geny z nici + w pliku plus / minus
    pm, mm = gene_sums(bwp, tm)[0].sum(), gene_sums(bwm, tm)[0].sum()   # geny z nici - w pliku plus / minus
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
    return key, frac, cache, time.time() - t0


def compare(d, feats, label, kd):
    """Klasa minus dopasowane mRNA (10 najblizszych wedlug cech feats) dla d_ekson, d_intron, d_post."""
    d = d.dropna(subset=feats)
    X = d[feats].values
    m = (d["klasa"] == "mRNA").values
    mu, sd = X[m].mean(0), X[m].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS).fit((X[m] - mu) / sd)
    rows = []
    for c in ORDER:
        sel = d["klasa"].isin(LNC_ALL).values if c == "lncRNA (wszystkie)" else (d["klasa"].values == c)
        if sel.sum() < 20:
            continue
        _, idx = nn.kneighbors((X[sel] - mu) / sd)
        row = {"kd": f"{kd['target']} {kd['cell']}", "pilot_role": kd["pilot_role"], "dopasowanie": label,
               "klasa": c, "n": int(sel.sum())}
        for col in ["d_ekson", "d_intron", "d_post", "splicing"]:
            diff = d[col].values[sel] - d[col].values[m][idx].mean(1)
            row[col], row[f"p_{col}"] = float(np.median(diff)), float(wilcoxon(diff).pvalue)
        rows.append(row)
    return rows


if __name__ == "__main__":
    man = pd.read_csv("data/raw/encode/download_manifest.tsv", sep="\t", dtype=str)
    bwf = pd.read_csv("data/meta/pilot_bigwig_files.tsv", sep="\t", dtype=str).merge(
        man[["file_accession", "path"]], on="file_accession")
    smp = pd.read_csv("data/meta/pilot_samples.tsv", sep="\t", dtype=str)
    rsem = smp.drop_duplicates(["sample_experiment", "bio_rep"]).set_index(["sample_experiment", "bio_rep"])["path"]

    # geny do wykrycia nici: 300 najsilniej wyrazonych mRNA w pierwszej probce
    q = pd.read_csv(rsem.iloc[0], sep="\t", usecols=["gene_id", "expected_count"])
    q = q.merge(classes[classes["klasa"] == "mRNA"][["gene_id"]], on="gene_id")
    q = q[q["gene_id"].isin(_gi.index)].nlargest(300, "expected_count")
    TEST = _gi.loc[q["gene_id"]].values

    jobs = []
    for (exp, rep), g in bwf.groupby(["experiment", "bio_rep"]):
        p = g.set_index("output_type")["path"]
        jobs.append(((exp, rep), p["plus strand signal of unique reads"], p["minus strand signal of unique reads"],
                     str(SUMS / f"{exp}_{rep}.npz")))
    print(f"2. Sumuje sygnal bigWig: {len(jobs)} probek na {NPROC} rdzeniach (postep ponizej)...", flush=True)
    res = []
    with Pool(NPROC) as pool:
        for n, (key, frac, path, dt) in enumerate(pool.imap_unordered(sample_sums, jobs), 1):
            res.append((key, frac, path))
            print(f"   [{n}/{len(jobs)}] {key[0]} replika {key[1]}: "
                  f"{'z dysku' if dt < 0 else f'{dt / 60:.1f} min'}", flush=True)

    # ------------------------------------------------------------------------- 3. kalibracja do odczytow
    counts, strand_rows = {}, []
    for (exp, rep), frac, path in res:
        z = np.load(path)
        s = pd.DataFrame({"exon": z["exon"], "intron": z["intron"]}, index=GENE_IDS)
        rq = pd.read_csv(rsem[(exp, rep)], sep="\t", usecols=["gene_id", "expected_count"]).set_index("gene_id")["expected_count"]
        common = s.index.intersection(rq.index)
        scale = s.loc[common, "exon"].sum() / rq.loc[common].sum()
        counts[(exp, rep)] = s / scale
        strand_rows.append({"experiment": exp, "rep": rep, "udzial_sensowny": round(max(frac, 1 - frac), 3),
                            "pliki": "zgodnie z nazwa" if frac > 0.5 else "zamienione", "skala_sygnal_na_odczyt": round(scale, 3)})
    strand = pd.DataFrame(strand_rows)
    strand.to_csv(OUT / "strand_check.tsv", sep="\t", index=False)
    print("   Nici: " + ", ".join(f"{k}: {v}" for k, v in strand["pliki"].value_counts().items()) +
          f"; najnizszy udzial sygnalu na wlasciwej nici: {strand['udzial_sensowny'].min():.3f}", flush=True)

    # ------------------------------------------------------------------------- 4-5. per knockdown
    meta = classes.set_index("gene_id")
    kin = pd.read_csv("results/phase1/bru/gene_kinetics.tsv", sep="\t")
    lens = pd.DataFrame({"len_ekson": LEN_EXON, "len_intron": LEN_INTRON}, index=GENE_IDS)
    kds = pd.read_csv("results/phase1/deseq2/summary.tsv", sep="\t")
    critB, cls_rows = [], []
    for kd in kds.to_dict("records"):
        s = smp[smp["kd_experiment"] == kd["kd_experiment"]]
        keys = {c: [(e, r) for e, r in zip(s.loc[s["condition"] == c, "sample_experiment"], s.loc[s["condition"] == c, "bio_rep"])]
                for c in ["kd", "control"]}
        allk = keys["kd"] + keys["control"]
        E = pd.concat({k: counts[k]["exon"] for k in allk}, axis=1).fillna(0)
        I = pd.concat({k: counts[k]["intron"] for k in allk}, axis=1).fillna(0)
        cls = meta.reindex(E.index)["klasa"]
        lnc, mrna = cls.isin(LNC_ALL), cls == "mRNA"
        expressed = E[keys["control"]].mean(axis=1) >= MIN_READS
        good_i = (I >= MIN_READS).all(axis=1)
        critB.append({"kd": f"{kd['target']} {kd['cell']}",
                      "lncRNA_wyrazone": int((lnc & expressed).sum()),
                      "proc_lncRNA_z_intronami": 100 * (good_i & lnc & expressed).sum() / max((lnc & expressed).sum(), 1),
                      "proc_mRNA_z_intronami": 100 * (good_i & mrna & expressed).sum() / max((mrna & expressed).sum(), 1)})
        # normalizacja: mediana ilorazow na eksonach (jak w DESeq2), ten sam czynnik dla intronow
        ok = (E > 0).all(axis=1) & expressed
        sf = E[ok].div(np.exp(np.log(E[ok]).mean(axis=1)), axis=0).median()
        En, In = E / sf, I / sf
        use = good_i & (E >= MIN_READS).all(axis=1) & cls.notna()
        ce, ci = En.loc[use, keys["control"]].mean(axis=1), In.loc[use, keys["control"]].mean(axis=1)
        d = pd.DataFrame({
            "klasa": cls[use], "ctrl_ekson": ce,
            "d_ekson": np.log2(En.loc[use, keys["kd"]].mean(axis=1) + PSEUDO) - np.log2(ce + PSEUDO),
            "d_intron": np.log2(In.loc[use, keys["kd"]].mean(axis=1) + PSEUDO) - np.log2(ci + PSEUDO)})
        d["d_post"] = d["d_ekson"] - d["d_intron"]
        d = d.join(meta[["gene_name", "width"]]).join(lens)
        # cechy metabolizmu bazowego: splicing = log2(gestosc w intronach / gestosc w eksonach) w kontrolach
        d["splicing"] = np.log2(((ci + 1) / d["len_intron"]) / ((ce + 1) / d["len_ekson"]))
        d = d.join(kin[kin["cell"] == kd["cell"]].set_index("gene_id")[["stab_6h"]])
        d["log_ekspr"], d["log_dl"] = np.log10(d["ctrl_ekson"] + 1), np.log10(d["width"])
        d.to_csv(OUT / f"{kd['kd_experiment']}.tsv.gz", sep="\t")
        dq = pd.read_csv(f"results/phase1/deseq2/{kd['kd_experiment']}.tsv.gz", sep="\t", usecols=["gene_id", "lfc_mle"]).set_index("gene_id")
        rho = spearmanr(d["d_ekson"], dq.reindex(d.index)["lfc_mle"], nan_policy="omit")[0]
        for label, feats in VARIANTS.items():
            cls_rows += [{**r, "rho_d_ekson_vs_DESeq2": rho} for r in compare(d, feats, label, kd)]

    critB = pd.DataFrame(critB)
    critB.to_csv(OUT / "criterion_B.tsv", sep="\t", index=False)
    cr = pd.DataFrame(cls_rows)
    cr.to_csv(OUT / "class_eisa.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print("\n3. Kryterium B: odsetek wyrazonych, wieloeksonowych genow z >= 10 odczytami intronowymi w kazdej probce")
    print(critB.round(1).to_string(index=False))
    print(f"   Kryterium B (>= 50% lncRNA) spelnione w {int((critB['proc_lncRNA_z_intronami'] >= 50).sum())} z {len(critB)} knockdownow")

    def fmt(v, p):
        return f"{v:+.2f}" + ("**" if p < 0.001 else ("*" if p < 0.01 else ""))
    lnc_rows = cr[cr["klasa"] == "lncRNA (wszystkie)"].copy()
    for c in ["d_ekson", "d_intron", "d_post", "splicing"]:
        lnc_rows[c] = [fmt(v, p) for v, p in zip(lnc_rows[c], lnc_rows[f"p_{c}"])]
    a = lnc_rows[lnc_rows["dopasowanie"] == "A"].set_index("kd")
    print("\n4. Metabolizm bazowy: splicing lncRNA minus dopasowane mRNA (dopasowanie A; dodatni = wiecej odczytow")
    print("   intronowych na pz, czyli wolniejszy lub mniej wydajny splicing)")
    print(a[["splicing"]].T.to_string())
    for label, opis in [("A", "ekspresja + dlugosc"), ("B", "ekspresja + dlugosc + splicing + stabilnosc bazowa")]:
        t = lnc_rows[lnc_rows["dopasowanie"] == label].sort_values(["pilot_role", "kd"])
        print(f"\n5{label}. lncRNA (wszystkie) minus mRNA dopasowane wedlug: {opis}")
        print("    d_ekson = zmiana calkowita, d_intron = transkrypcyjna, d_post = potranskrypcyjna (mediany log2)")
        print(t[["kd", "n", "d_ekson", "d_intron", "d_post", "rho_d_ekson_vs_DESeq2"]]
              .round({"rho_d_ekson_vs_DESeq2": 2}).to_string(index=False))
    print(f"\nZapisano: {OUT}/criterion_B.tsv, {OUT}/class_eisa.tsv, {OUT}/strand_check.tsv, {OUT}/<eksperyment>.tsv.gz")
