"""
Tabele uzupelniajace S3 i S4 (do arkusza z opisem kolumn skladanego poza klastrem).
  S3 (results/figures_pub/tableS3_experiments.tsv): wszystkie eksperymenty wyciszen RBP z ENCODE uzyte w pracy:
     akcesja, RBP, linia, metoda, biblioteka, eksperymenty kontrolne, liczba wyciszen dzielacych kontrole,
     czy w analizie class effect i dlaczego nie.
  S3b (results/figures_pub/tableS3_other_data.tsv): pozostale dane ENCODE (Bru-seq/BruChase, frakcjonowanie, eCLIP).
  S4 (results/figures_pub/tableS4_class_effects.tsv): class effect kazdego eksperymentu: wariant glowny (p, q),
     z dopasowaniem na GC i po usunieciu trendu GC; RBP z listy 35 i z rdzenia 17.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/figures_pub")
MIN_BATCH = 4


def pick(df, names, what):
    for n in names:
        if n in df.columns:
            return n
    raise SystemExit(f"Brak kolumny {what} (szukano {names}); kolumny: {list(df.columns)}")


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


# ---------------------------------------------------------------- S3: wyciszenia i kontrole
smp = pd.read_csv("data/meta/phase2_samples.tsv", sep="\t", dtype=str)
kd = smp[smp["condition"] != "control"].drop_duplicates("kd_experiment")
ctrl = (smp[smp["condition"] == "control"].groupby("kd_experiment")["sample_experiment"]
        .apply(lambda s: "; ".join(sorted(set(s)))))
kds = pd.read_csv("data/meta/phase2_kds.tsv", sep="\t", dtype=str)
kcol = pick(kds, ["experiment", "kd_experiment"], "eksperymentu w phase2_kds")
batch_of = kds.set_index(kcol)["possible_controls"] if "possible_controls" in kds.columns else pd.Series(dtype=str)
batch_n = batch_of.map(batch_of.value_counts()) if len(batch_of) else pd.Series(dtype=float)

res = pd.read_csv("results/phase2/classes/class_effects_all.tsv.gz", sep="\t")
ecol = pick(res, ["kd_experiment", "eksperyment", "experiment", "wyciszenie"], "eksperymentu w class_effects_all")
main = res[(res["klasa"] == "lncRNA (razem)") & (res["macierz"] == "skorygowana") & (res["dopasowanie"] == "B")].copy()
in_main = set(main[ecol])

t3 = pd.DataFrame({
    "experiment": kd["kd_experiment"].values,
    "RBP": kd["target"].values,
    "cell_line": kd["cell"].values,
    "method": kd["method"].values,
    "library": kd["biblioteka"].map({"poliA": "poly(A)", "calkowite RNA": "total RNA"}).fillna(kd["biblioteka"]).values,
})
t3["control_experiments"] = t3["experiment"].map(ctrl)
t3["depletions_sharing_control"] = t3["experiment"].map(batch_n)
t3["in_class_effect_analysis"] = np.where(t3["experiment"].isin(in_main), "yes", "no")
t3["reason_if_not"] = np.select(
    [t3["in_class_effect_analysis"] == "yes", t3["library"] != "poly(A)", t3["depletions_sharing_control"] < MIN_BATCH],
    ["", "total RNA library", f"control shared by fewer than {MIN_BATCH} depletions"], default="other (see Methods)")
t3 = t3.sort_values(["cell_line", "RBP", "experiment"])
t3.to_csv(OUT / "tableS3_experiments.tsv", sep="\t", index=False)
print(f"S3: {len(t3)} wyciszen; w analizie class effect: {int((t3['in_class_effect_analysis'] == 'yes').sum())}")
print(t3.groupby(["cell_line", "library", "in_class_effect_analysis"]).size().to_string())

# ---------------------------------------------------------------- S3b: pozostale dane ENCODE
rows = []
bru = sorted(set(re.findall(r"ENCSR[0-9]{3}[A-Z]{3}", Path("scripts/14_pilot_bru_stability.py").read_text())))
rows += [{"dataset": "Bru-seq / BruChase-seq", "experiment": e, "cell_line": "", "detail": ""} for e in bru]
ff = pd.read_csv("data/meta/fraction_files.tsv", sep="\t", dtype=str)
if "experiment" in ff.columns:
    for (e, c, f, n), _ in ff.groupby(["experiment", "cell", "subcellular_fraction", "nucleic_acid"]):
        if n == "polyadenylated mRNA" and f in ("nucleus", "cytosol"):
            rows.append({"dataset": "subcellular fractionation (poly(A) RNA-seq)", "experiment": e, "cell_line": c, "detail": f})
ef = pd.read_csv("data/meta/eclip_files.tsv", sep="\t", dtype=str)
if "experiment" in ef.columns:
    for (e, c, t), _ in ef.groupby(["experiment", "cell", "target"]):
        rows.append({"dataset": "eCLIP", "experiment": e, "cell_line": c, "detail": t})
t3b = pd.DataFrame(rows).drop_duplicates()
t3b.to_csv(OUT / "tableS3_other_data.tsv", sep="\t", index=False)
print(f"S3b: {len(t3b)} eksperymentow; " + ", ".join(f"{k}: {v}" for k, v in t3b["dataset"].value_counts().items()))

# ---------------------------------------------------------------- S4: class effect kazdego eksperymentu
pcol = next((c for c in ["p", "p_wilcoxon", "pval"] if c in main.columns), None)
s4 = main[[ecol, "target", "linia", "roznica"] + ([pcol] if pcol else [])].rename(
    columns={ecol: "experiment", "target": "RBP", "linia": "cell_line", "roznica": "class_effect_main"})
if pcol:
    s4 = s4.rename(columns={pcol: "p_main"})
    s4["q_main"] = s4.groupby("cell_line")["p_main"].transform(bh)
gcf = Path("results/phase4/followups/gc_class_effects.tsv.gz")
if gcf.exists():
    gc = pd.read_csv(gcf, sep="\t")
    gce = pick(gc, ["kd_experiment", "eksperyment", "experiment"], "eksperymentu w gc_class_effects")
    for v, name in [("C", "class_effect_GC_matched"), ("B_gc", "class_effect_GC_trend_removed")]:
        g = gc[gc["wariant"] == v].set_index(gce)["roznica"]
        s4[name] = s4["experiment"].map(g)
s4["method"] = s4["experiment"].map(t3.set_index("experiment")["method"])
gc35 = Path("results/phase4/followups/gc_35rbp.tsv")
if gc35.exists():
    g = pd.read_csv(gc35, sep="\t")
    s4["RBP_clear_in_both_lines"] = s4["RBP"].isin(set(g.loc[g["w_B"].astype(str).eq("True"), "RBP"]))
    s4["RBP_robust_to_GC"] = s4["RBP"].isin(set(g.loc[g["w_B"].astype(str).eq("True") & g["w_C"].astype(str).eq("True"), "RBP"]))
s4 = s4.sort_values(["cell_line", "RBP", "experiment"])
s4.to_csv(OUT / "tableS4_class_effects.tsv", sep="\t", index=False, float_format="%.4g")
print(f"S4: {len(s4)} eksperymentow; kolumny: {list(s4.columns)}")
if "RBP_robust_to_GC" in s4.columns:
    print(f"   RBP z listy 35: {s4.loc[s4['RBP_clear_in_both_lines'], 'RBP'].nunique()}; z rdzenia 17: {s4.loc[s4['RBP_robust_to_GC'], 'RBP'].nunique()}")
