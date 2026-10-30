"""
Faza 1, krok 6C: walidacja EISA bezposrednim pomiarem obrotu RNA (TimeLapse-seq, Ietswaart i wsp. 2024).

TimeLapse-seq podaje frakcje nowego RNA (f) po 60 min znakowania. W stanie ustalonym f = 1 - exp(-k*t),
wiec tempo obrotu k = -ln(1 - f) / t. Zmiana stabilnosci po knockdownie:
    d_stab = log2(k_kontrola / k_knockdown)      (dodatnia = RNA stabilniejsze po knockdownie)

1. Knockdowny indukowane doksycykilna (EXOSC10, PABPN1, DIS3, ZFC3H1; kontrola scramble_dox):
   eksony i introny, cala komorka (tot) i jadro (nuc), takze dla lncRNA.
   Porownanie klas: lncRNA minus mRNA dopasowane wedlug liczby odczytow i dlugosci.
2. Knockdowny DDX3X i PABPC4 (kontrola scramble): geny, praktycznie tylko mRNA.
Wersja 2: geny z frakcja nowego RNA poza zakresem 0.02-0.95 odrzucane (nasycenie wzoru), a od zmiany stabilnosci
odejmowane jest globalne przesuniecie (mediana mRNA) - roznice w wydajnosci znakowania miedzy probkami przesuwaja
wszystkie geny naraz i nie sa biologia. Wypisywana jest diagnostyka rozkladow frakcji.
3. Kryterium C: czy zmiana stabilnosci z TimeLapse-seq zgadza sie z EISA z ENCODE dla tego samego RBP w K562
   (d_post i d_ekson wobec d_stab eksonow; d_intron wobec d_stab intronow), osobno dla mRNA i lncRNA.
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors

D = "data/raw/timelapse/processed_data_files"
OUT = Path("results/phase1/timelapse")
OUT.mkdir(parents=True, exist_ok=True)
T_MIN, MIN_READS, K_NEIGHBOURS, F_MIN, F_MAX = 60.0, 20, 10, 0.02, 0.95
LNC_ALL = {"lincRNA", "antysens", "dywergentny", "sense-intronic", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
DOX = ["EXOSC10", "PABPN1", "DIS3", "ZFC3H1"]
PLAIN = ["DDX3X", "PABPC4"]

cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
cls["gene"] = cls["gene_id"].str.split(".").str[0]
cls = cls.drop_duplicates("gene").set_index("gene")


def col(d, suffix):
    c = [x for x in d.columns if x.endswith(suffix)]
    return d[c[0]] if c else None


def load(pattern):
    """Srednia frakcja nowego RNA z replik (i suma odczytow, jesli plik je podaje); jeden wiersz na gen."""
    reps = []
    for p in sorted(glob.glob(f"{D}/{pattern}")):
        d = pd.read_csv(p, sep="\t")
        r = pd.DataFrame({"gene": d["Gene"].astype(str).str.split(".").str[0], "f": col(d, " Mean")})
        rc = col(d, " Readcount")
        r["reads"] = rc if rc is not None else np.nan
        if "Length" in d.columns:
            r["length"] = d["Length"]
        q05, q95 = col(d, " 0.05 quantile"), col(d, " 0.95 quantile")
        r["ci"] = (q95 - q05) if q05 is not None and q95 is not None else np.nan
        reps.append(r.sort_values("reads", ascending=False).drop_duplicates("gene").set_index("gene"))
    if len(reps) < 2:
        raise FileNotFoundError(f"brak dwoch replik dla {pattern}")
    m = pd.concat(reps, axis=1, keys=range(len(reps)))
    out = pd.DataFrame({"f": m.xs("f", axis=1, level=1).mean(axis=1),
                        "reads_min": m.xs("reads", axis=1, level=1).min(axis=1),
                        "ci_max": m.xs("ci", axis=1, level=1).max(axis=1)})
    if "length" in reps[0].columns:
        out["length"] = m.xs("length", axis=1, level=1).mean(axis=1)
    return out


DIAG = []


def d_stab(kd, ctrl, use_reads=True, label=""):
    j = kd.join(ctrl, lsuffix="_kd", rsuffix="_ctrl", how="inner")
    if use_reads:
        ok = (j["reads_min_kd"] >= MIN_READS) & (j["reads_min_ctrl"] >= MIN_READS)
    else:
        ok = (j["ci_max_kd"] < 0.3) & (j["ci_max_ctrl"] < 0.3)   # waski przedzial wiarygodnosci
    j = j[ok & j["f_kd"].notna() & j["f_ctrl"].notna()].copy()
    n0 = len(j)
    diag = {"porownanie": label, "geny": n0, "mediana_f_kd": j["f_kd"].median(), "mediana_f_ctrl": j["f_ctrl"].median(),
            "proc_f_kd_>0.95": 100 * (j["f_kd"] > F_MAX).mean(), "proc_f_ctrl_>0.95": 100 * (j["f_ctrl"] > F_MAX).mean()}
    j = j[j["f_kd"].between(F_MIN, F_MAX) & j["f_ctrl"].between(F_MIN, F_MAX)].copy()
    k = lambda f: -np.log(1 - f) / T_MIN
    j["klasa"] = cls.reindex(j.index)["klasa"].values
    j = j.dropna(subset=["klasa"])
    raw = np.log2(k(j["f_ctrl"]) / k(j["f_kd"]))
    shift = raw[j["klasa"] == "mRNA"].median()
    j["d_stab"] = raw - shift
    diag.update({"po_filtrze": len(j), "przesuniecie_globalne": shift})
    DIAG.append(diag)
    return j


def class_diff(j):
    """lncRNA (wszystkie) minus mRNA dopasowane wedlug odczytow i dlugosci."""
    X = np.c_[np.log10(j[["reads_min_kd", "reads_min_ctrl"]].mean(axis=1) + 1), np.log10(j["length_kd"])]
    m, s = (j["klasa"] == "mRNA").values, j["klasa"].isin(LNC_ALL).values
    if s.sum() < 20 or m.sum() < 100:
        return np.nan, np.nan, int(s.sum())
    mu, sd = X[m].mean(0), X[m].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    _, idx = NearestNeighbors(n_neighbors=K_NEIGHBOURS).fit((X[m] - mu) / sd).kneighbors((X[s] - mu) / sd)
    diff = j["d_stab"].values[s] - j["d_stab"].values[m][idx].mean(1)
    return float(np.median(diff)), float(wilcoxon(diff).pvalue), int(s.sum())


pd.set_option("display.width", 250)
# ---------------------------------------------------------------- 1. knockdowny z doksycykilna: klasy
rows, dox = [], {}
for rbp in DOX:
    for frac in ["tot", "nuc"]:
        for feat in ["exons", "introns"]:
            j = d_stab(load(f"K562_dox_{rbp}_{frac}_rep*_{feat}.tsv"), load(f"K562_dox_scramble_{frac}_rep*_{feat}.tsv"),
                       label=f"{rbp} {frac} {feat}")
            dox[(rbp, frac, feat)] = j
            med_m = j.loc[j["klasa"] == "mRNA", "d_stab"].median()
            med_l = j.loc[j["klasa"].isin(LNC_ALL), "d_stab"].median()
            diff, p, n = class_diff(j)
            rows.append({"RBP": rbp, "frakcja": frac, "cecha": feat, "n_mRNA": int((j["klasa"] == "mRNA").sum()),
                         "n_lncRNA": n, "mediana_mRNA": med_m, "mediana_lncRNA": med_l,
                         "lncRNA_minus_dopas_mRNA": diff, "p": p})
cl = pd.DataFrame(rows)
print("0. Diagnostyka: frakcja nowego RNA (f) w knockdownie i kontroli, nasycenie (f > 0.95) i przesuniecie globalne")
print(pd.DataFrame(DIAG).round(2).to_string(index=False))
print()
cl.to_csv(OUT / "timelapse_class_stability.tsv", sep="\t", index=False)
print("1. Zmiana stabilnosci po knockdownie (TimeLapse-seq, K562), po odjeciu przesuniecia globalnego; dodatnia = stabilniejsze")
print(cl.round({"mediana_mRNA": 2, "mediana_lncRNA": 2, "lncRNA_minus_dopas_mRNA": 2})
        .to_string(index=False, formatters={"p": "{:.1e}".format}))

# ---------------------------------------------------------------- 2-3. walidacja EISA
kds = pd.read_csv("results/phase1/deseq2/summary.tsv", sep="\t")


def eisa_for(rbp):
    hit = kds[(kds["target"] == rbp) & (kds["cell"] == "K562")]
    if hit.empty:
        return None, None
    e = pd.read_csv(f"results/phase1/eisa/{hit['kd_experiment'].iloc[0]}.tsv.gz", sep="\t", index_col=0)
    e.index = e.index.astype(str).str.split(".").str[0]
    return hit["kd_experiment"].iloc[0], e


val = []
for rbp in DOX + PLAIN:
    acc, e = eisa_for(rbp)
    if e is None:
        continue
    if rbp in DOX:
        pairs = [("eksony tot", dox[(rbp, "tot", "exons")], ["d_post", "d_ekson"]),
                 ("eksony nuc", dox[(rbp, "nuc", "exons")], ["d_post", "d_ekson"]),
                 ("introny tot", dox[(rbp, "tot", "introns")], ["d_intron"]),
                 ("introny nuc", dox[(rbp, "nuc", "introns")], ["d_intron"])]
    else:
        j = d_stab(load(f"K562_{rbp}_tot_rep*.tsv"), load("K562_scramble_tot_rep*.tsv"), use_reads=False,
                   label=f"{rbp} tot geny")
        pairs = [("geny tot", j, ["d_post", "d_ekson"])]
    for label, j, cols in pairs:
        mm = j[["d_stab", "klasa"]].join(e[["d_ekson", "d_intron", "d_post"]], how="inner")
        for grp, sel in [("mRNA", mm["klasa"] == "mRNA"), ("lncRNA", mm["klasa"].isin(LNC_ALL))]:
            if sel.sum() < 30:
                continue
            for c in cols:
                rho, p = spearmanr(mm.loc[sel, "d_stab"], mm.loc[sel, c])
                val.append({"RBP": rbp, "ENCODE": acc, "TimeLapse": label, "geny": grp, "n": int(sel.sum()),
                            "EISA": c, "rho": rho, "p": p})
val = pd.DataFrame(val)
val.to_csv(OUT / "timelapse_vs_eisa.tsv", sep="\t", index=False)
print("\n2. Kryterium C: korelacja zmiany stabilnosci z TimeLapse-seq ze skladowymi EISA z ENCODE (Spearman, K562)")
print("   oczekiwane: dodatnia dla d_post (stabilniejsze dojrzale RNA) i dla d_intron wobec intronow")
print(val.round({"rho": 2}).to_string(index=False, formatters={"p": "{:.1e}".format}))
pd.DataFrame(DIAG).to_csv(OUT / "timelapse_diagnostics.tsv", sep="\t", index=False)
print(f"\nZapisano: {OUT}/timelapse_class_stability.tsv, {OUT}/timelapse_vs_eisa.tsv, {OUT}/timelapse_diagnostics.tsv")
