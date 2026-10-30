"""
Faza 2, krok 2.5: RBP o powtarzalnym wplywie na lncRNA i sprawdzenia odpornosci.

1. Lista RBP z replikacja (macierz skorygowana, dopasowanie B, lncRNA razem): w kazdej linii |roznica| >= 0.1
   i q < 0.05 (Benjamini-Hochberg w obrebie linii), ten sam znak w obu liniach. Oczekiwana liczba takich RBP przy
   braku zwiazku miedzy liniami: permutacje (losowe parowanie RBP z K562 i HepG2).
2. Profil klas dla RBP z listy.
3. Sprawdzenia odpornosci:
   a) dopasowanie A na tym samym zbiorze genow co B (oddziela efekt dopasowania od efektu wezszego zbioru genow);
   b) B z odniesieniem tylko do mRNA BEZ izoform z zatrzymanym intronem w GENCODE (artefakt: po zahamowaniu
      splicingu takie izoformy "lapia" odczyty intronowe i zawyzaja geny kodujace);
   c) zgodnosc metod: RBP wyciszone przez shRNA i przez CRISPR w tej samej linii.
"""
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors

C, M = Path("results/phase2/classes"), Path("results/phase2/matrix")
EFFECT, Q, N_PERM, K_NN, MIN_N, ROBUST = 0.1, 0.05, 1000, 10, 20, 0.05
POOLED = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
LNC = "lncRNA (razem)"
PROFILE = ["lincRNA", "antysens", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA", "pseudogen transkr."]
COLS_A, COLS_B = ["log_ekspr", "log_dl"], ["log_ekspr", "log_dl", "splicing", "stab_6h"]
pd.set_option("display.width", 230)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def fmt_p(df, cols):
    return {c: "{:.1e}".format for c in cols if c in df.columns}


# ---------------------------------------------------------------- 1. lista RBP z replikacja
res = pd.read_csv(C / "class_effects_all.tsv.gz", sep="\t")
main = res[(res["macierz"] == "skorygowana") & (res["dopasowanie"] == "B")]
lnc = main[main["klasa"] == LNC].copy()
lnc["q"] = lnc.groupby("linia")["p"].transform(bh)
agg = lnc.groupby(["target", "linia"]).agg(roznica=("roznica", "mean"), q=("q", "max")).unstack("linia")
both = agg.dropna(subset=[("roznica", "K562"), ("roznica", "HepG2")])
sig = {l: ((both[("roznica", l)].abs() >= EFFECT) & (both[("q", l)] < Q)).values for l in ["K562", "HepG2"]}
sgn = {l: np.sign(both[("roznica", l)].values) for l in ["K562", "HepG2"]}
is_hit = sig["K562"] & sig["HepG2"] & (sgn["K562"] == sgn["HepG2"])
rng = np.random.default_rng(0)
null = []
for _ in range(N_PERM):
    pm = rng.permutation(len(both))
    null.append(int((sig["K562"] & sig["HepG2"][pm] & (sgn["K562"] == sgn["HepG2"][pm])).sum()))
null = np.array(null)
meth = res.groupby("target")["metoda"].agg(lambda s: "/".join(sorted(set(s))))
hits = pd.DataFrame({"K562": both[("roznica", "K562")], "HepG2": both[("roznica", "HepG2")],
                     "q_K562": both[("q", "K562")], "q_HepG2": both[("q", "HepG2")]})[is_hit]
hits["srednia"] = hits[["K562", "HepG2"]].mean(axis=1)
hits["kierunek"] = np.where(hits["srednia"] > 0, "lncRNA rosna", "lncRNA spadaja")
hits["metoda"] = meth.reindex(hits.index)
hits = hits.sort_values("srednia")
print("1. RBP z powtarzalnym wplywem na lncRNA (skorygowana, B; |roznica| >= 0.1 i q < 0.05 w obu liniach, ten sam znak)")
print(f"   RBP w obu liniach: {len(both)}; istotne w K562: {int(sig['K562'].sum())}, w HepG2: {int(sig['HepG2'].sum())}")
print(f"   RBP z replikacja: {len(hits)} (lncRNA rosna: {int((hits['srednia'] > 0).sum())}, spadaja: {int((hits['srednia'] < 0).sum())})")
print(f"   Oczekiwane przy losowym parowaniu RBP: srednio {null.mean():.1f} (95. percentyl {np.percentile(null, 95):.0f}); "
      f"szacowany odsetek falszywych: {100 * null.mean() / max(len(hits), 1):.0f}%")

# ---------------------------------------------------------------- 2. profil klas
prof = main[main["target"].isin(hits.index) & main["klasa"].isin(PROFILE)]
prof = prof.groupby(["target", "klasa"])["roznica"].mean().unstack("klasa").reindex(columns=PROFILE)
tab = hits[["K562", "HepG2", "metoda"]].join(prof.round(2))
print("\n2. Profil klas dla RBP z listy (srednia z obu linii; kolumny K562/HepG2 = lncRNA razem)")
print(tab.round(2).to_string())
tab.to_csv(C / "rbp_hits.tsv", sep="\t")

# ---------------------------------------------------------------- 3a-b. przeliczenie dla wariantow odpornosci
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
kin = pd.read_csv("results/phase1/bru/gene_kinetics.tsv", sep="\t")
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")


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
                ratio = (I / lens["intron"].reindex(gene_ids)) / (E / lens["exon"].reindex(gene_ids))
                vals.append(np.log2(ratio.where((E > 0) & (I > 0))))
        out[cell] = pd.concat(vals, axis=1).median(axis=1)
    return out


def ri_genes():
    out = set()
    with gzip.open("data/ref/gencode.v29.annotation.gtf.gz", "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if f[2] == "transcript" and 'transcript_type "retained_intron"' in f[8]:
                out.add(re.search(r'gene_id "([^"]+)"', f[8]).group(1))
    return out


def lnc_neighbours(feat, match_cols, ref_mask):
    """lncRNA (razem) z zakresu cech B, dopasowane do mRNA (z ref_mask) wedlug match_cols."""
    f = feat.dropna(subset=COLS_B)
    mr_all = f[f["klasa"] == "mRNA"]
    lo, hi = mr_all[COLS_B].quantile(0.01), mr_all[COLS_B].quantile(0.99)
    inside = ((f[COLS_B] >= lo) & (f[COLS_B] <= hi)).all(axis=1)
    mr = mr_all[ref_mask.reindex(mr_all.index).fillna(False).values]
    mu, sd = mr[match_cols].mean().values, mr[match_cols].std().values
    sd = np.where(sd > 0, sd, 1.0)
    sel = f[f["klasa"].isin(POOLED) & inside]
    _, idx = NearestNeighbors(n_neighbors=K_NN).fit((mr[match_cols].values - mu) / sd).kneighbors((sel[match_cols].values - mu) / sd)
    return sel.index.values, mr.index.values[idx], len(mr)


print("\n3. Sprawdzenia odpornosci")
ri = ri_genes()
is_mrna = cls["klasa"] == "mRNA"
print(f"   Geny kodujace z izoforma 'retained_intron' w GENCODE v29: {int(cls.index[is_mrna].isin(ri).sum())} "
      f"z {int(is_mrna.sum())} ({100 * cls.index[is_mrna].isin(ri).mean():.0f}%)")
splice = splicing_by_cell()
rows = []
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    g = L.index
    feat = pd.DataFrame({"klasa": cls.reindex(g)["klasa"], "log_ekspr": np.log10(expr.reindex(g) + 1),
                         "log_dl": np.log10(cls.reindex(g)["width"]), "splicing": splice[cell].reindex(g),
                         "stab_6h": kin[kin["cell"] == cell].drop_duplicates("gene_id").set_index("gene_id")["stab_6h"].reindex(g)}, index=g)
    all_mrna = pd.Series(True, index=g)
    no_ri = pd.Series(~g.isin(ri), index=g)
    variants = {"A_na_zbiorze_B": lnc_neighbours(feat, COLS_A, all_mrna),
                "B_mRNA_bez_RI": lnc_neighbours(feat, COLS_B, no_ri)}
    print(f"   {cell}: odniesienie B_mRNA_bez_RI = {variants['B_mRNA_bez_RI'][2]} mRNA "
          f"(z {variants['A_na_zbiorze_B'][2]} w pelnym zbiorze)")
    for kd in L.columns:
        x = L[kd]
        for v, (genes, idx, _) in variants.items():
            d = x.reindex(genes).values - np.nanmean(x.reindex(idx.ravel()).values.reshape(idx.shape), axis=1)
            d = d[~np.isnan(d)]
            if len(d) >= MIN_N:
                rows.append({"linia": cell, "kd_experiment": kd, "target": summ.loc[kd, "target"],
                             "wariant": v, "roznica": float(np.median(d)), "p": float(wilcoxon(d).pvalue)})
lnc_a = res[(res["macierz"] == "skorygowana") & (res["dopasowanie"] == "A") & (res["klasa"] == LNC)]
rob = pd.concat([pd.DataFrame(rows),
                 lnc[["linia", "kd_experiment", "target", "roznica", "p"]].assign(wariant="B"),
                 lnc_a[["linia", "kd_experiment", "target", "roznica", "p"]].assign(wariant="A_pelne")], ignore_index=True)
rob.to_csv(C / "robustness_variants.tsv.gz", sep="\t", index=False)

rep = []
for v in ["A_pelne", "A_na_zbiorze_B", "B", "B_mRNA_bez_RI"]:
    w = rob[rob["wariant"] == v].groupby(["target", "linia"])["roznica"].mean().unstack("linia").dropna()
    r, p = spearmanr(w["K562"], w["HepG2"])
    rep.append({"wariant": v, "RBP": len(w), "replikacja_rho": r, "p": p,
                "mediana_|roznicy|": rob.loc[rob["wariant"] == v, "roznica"].abs().median()})
rep = pd.DataFrame(rep)
print(rep.round(3).to_string(index=False, formatters=fmt_p(rep, ["p"])))

wv = rob[rob["target"].isin(hits.index)].groupby(["target", "wariant", "linia"])["roznica"].mean().unstack(["wariant", "linia"])
chk = pd.DataFrame(index=hits.index)
for v in ["A_pelne", "A_na_zbiorze_B", "B", "B_mRNA_bez_RI"]:
    chk[v] = wv[v].mean(axis=1).reindex(hits.index)
ok = pd.Series(True, index=hits.index)
for v in ["B_mRNA_bez_RI"]:
    for l in ["K562", "HepG2"]:
        val = wv[(v, l)].reindex(hits.index)
        ok &= (np.sign(val) == np.sign(hits["srednia"])) & (val.abs() >= ROBUST)
chk["odporny"] = ok
print(f"\n   RBP z listy w wariantach (srednia z linii). odporny = w wariancie B_mRNA_bez_RI ten sam znak i |roznica| >= {ROBUST}"
      f" w obu liniach.\n   A_pelne i A_na_zbiorze_B tylko do diagnozy: jesli A_na_zbiorze_B ~ A_pelne, roznica A-B wynika z dopasowania,"
      f" jesli ~ B, z wezszego zbioru genow")
print(chk.round(2).to_string())
print(f"   Odporne: {int(ok.sum())} z {len(hits)}")
chk.to_csv(C / "rbp_hits_robustness.tsv", sep="\t")

# ---------------------------------------------------------------- 3c. zgodnosc metod
print("\n   c) RBP wyciszone obiema metodami w tej samej linii (lncRNA razem, skorygowana, B):")
pairs = lnc.groupby(["linia", "target"])["metoda"].nunique()
pairs = pairs[pairs > 1].index
if len(pairs):
    t = lnc.set_index(["linia", "target"]).loc[list(pairs)].reset_index().pivot_table(
        index=["linia", "target"], columns="metoda", values="roznica")
    print(t.round(3).to_string())
    if len(t) >= 5:
        print(f"   Spearman shRNA vs CRISPR: rho = {spearmanr(t['shRNA'], t['CRISPR'])[0]:.2f} ({len(t)} par)")
else:
    print("   brak takich par po korekcie na partie")
print(f"\nZapisano: {C}/rbp_hits.tsv, {C}/rbp_hits_robustness.tsv, {C}/robustness_variants.tsv.gz")
