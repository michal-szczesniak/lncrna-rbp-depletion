"""
Faza 3, krok 3.1: spliceosom, NMD i geny gospodarze snoRNA na poziomie pojedynczych genow.

Hipoteza: wyciszenie spliceosomu (a) obniza lncRNA wymagajace splicingu (gorsze dojrzewanie) i (b) podnosi lncRNA,
ktore normalnie usuwa NMD (bez splicingu nie ma kompleksow EJC, wiec NMD ich nie rozpoznaje). Geny gospodarze
snoRNA naleza do najsilniej usuwanych przez NMD transkryptow, co tlumaczyloby ich odwrotne zachowanie.
Przewidywania:
  1. lncRNA jednoeksonowe nie spadaja po wyciszeniu spliceosomu (kontrola wbudowana);
  2. geny rosnace po wyciszeniu spliceosomu to te same, ktore rosna po wyciszeniu NMD/EJC;
  3. gospodarze snoRNA sa najsilniej zalezni od NMD.

Odpowiedz konsensusowa genu = mediana log2FC (macierz skorygowana na kontrole) z knockdownow danej grupy w danej linii.
  spliceosom: RBP z kategoria "Spliceosome" w anotacji ENCODE;
  NMD/EJC:    UPF1, UPF2, UPF3B, SMG1, SMG5, SMG6, SMG7, SMG8, SMG9, RBM8A, MAGOH, EIF4A3, CASC3 (te, ktore sa w danych).
"""
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

M, OUT = Path("results/phase2/matrix"), Path("results/phase3/splice_nmd")
OUT.mkdir(parents=True, exist_ok=True)
ALIASES = {"MTREX": ["SKIV2L2"], "ATP5A1": ["ATP5F1A"], "ATP5C1": ["ATP5F1C"], "GNB2L1": ["RACK1"]}
NMD = ["UPF1", "UPF2", "UPF3B", "SMG1", "SMG5", "SMG6", "SMG7", "SMG8", "SMG9", "RBM8A", "MAGOH", "EIF4A3", "CASC3"]
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz miRNA"}      # bez gospodarzy snoRNA (osobno)
ORDER = ["mRNA", "lincRNA", "antysens", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA", "pseudogen transkr."]
MIN_KD = 3
pd.set_option("display.width", 230)

cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
kin = pd.read_csv("results/phase1/bru/gene_kinetics.tsv", sep="\t")

# ---------------------------------------------------------------- czlonkowie spliceosomu (anotacja ENCODE)
raw = pd.read_excel("data/ref/rbp_annotation/supp_data1.xlsx", header=None)
sub = [str(s).strip() for s in raw.iloc[1].tolist()]
ann = raw.iloc[2:].copy()
ann.columns = ["rbp", "gene"] + sub[2:]
spl_names = set(ann.loc[pd.to_numeric(ann["Spliceosome"], errors="coerce") == 1, "rbp"].astype(str))
for k, v in ALIASES.items():                      # nazwy w anotacji moga byc stare
    if any(a in spl_names for a in v):
        spl_names.add(k)

# ---------------------------------------------------------------- liczba eksonow (scalone eksony wszystkich transkryptow)
print("Licze eksony genow z GTF (ok. 1 min)...", flush=True)
ex = []
with gzip.open("data/ref/gencode.v29.annotation.gtf.gz", "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.split("\t", 9)
        if f[2] == "exon":
            ex.append((re.search(r'gene_id "([^"]+)"', f[8]).group(1), int(f[3]), int(f[4])))
ex = pd.DataFrame(ex, columns=["gene_id", "start", "end"]).sort_values(["gene_id", "start"])
prev_end = ex.groupby("gene_id")["end"].cummax().shift()
new_block = (ex["gene_id"] != ex["gene_id"].shift()) | (ex["start"] > prev_end)
n_exon = new_block.groupby(ex["gene_id"]).sum()


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


splice = splicing_by_cell()


def rank_ols(df, y, xs):
    """Regresja na rangach (standaryzowanych): wspolczynniki porownywalne miedzy cechami."""
    d = df[[y] + xs].dropna()
    Z = d.rank().apply(lambda c: (c - c.mean()) / c.std())
    X = np.c_[np.ones(len(Z)), Z[xs].values]
    beta, *_ = np.linalg.lstsq(X, Z[y].values, rcond=None)
    resid = Z[y].values - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (resid @ resid) / (len(Z) - X.shape[1]))
    return pd.DataFrame({"beta": beta[1:], "t": beta[1:] / se[1:]}, index=xs), len(Z)


cons_all = {}
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    tgt = summ.reindex(L.columns)["target"]
    spl_kd = tgt[tgt.isin(spl_names)].index
    nmd_kd = tgt[tgt.isin(NMD)].index
    print(f"\n===== {cell}: knockdowny spliceosomu: {len(spl_kd)}; NMD/EJC: {len(nmd_kd)} ({', '.join(sorted(set(tgt[nmd_kd])))})")
    if len(spl_kd) < MIN_KD or len(nmd_kd) < 1:
        print("   za malo knockdownow, pomijam linie")
        continue
    g = L.index
    d = pd.DataFrame({"klasa": cls.reindex(g)["klasa"],
                      "spl": L[spl_kd].median(axis=1, skipna=True).where(L[spl_kd].notna().sum(axis=1) >= MIN_KD),
                      "nmd": L[nmd_kd].median(axis=1, skipna=True),
                      "n_ekson": n_exon.reindex(g), "log_ekspr": np.log10(expr.reindex(g) + 1),
                      "log_dl": np.log10(cls.reindex(g)["width"]), "splicing": splice[cell].reindex(g),
                      "stab_6h": kin[kin["cell"] == cell].drop_duplicates("gene_id").set_index("gene_id")["stab_6h"].reindex(g),
                      "gen": cls.reindex(g)["gene_name"]}, index=g).dropna(subset=["klasa", "spl"])
    d["log_ekson"] = np.log2(d["n_ekson"])
    cons_all[cell] = d
    d.to_csv(OUT / f"consensus_{cell}.tsv.gz", sep="\t")

    # 1. kontrola: liczba eksonow
    d["eksony"] = pd.cut(d["n_ekson"], [0, 1, 2, 4, 1e9], labels=["1", "2", "3-4", "5+"])
    med = d.pivot_table(index="klasa", columns="eksony", values="spl", aggfunc="median", observed=False).reindex(ORDER)
    cnt = d.pivot_table(index="klasa", columns="eksony", values="spl", aggfunc="size", observed=False).reindex(ORDER)
    print("\n1. Odpowiedz na wyciszenie spliceosomu (mediana log2FC) wedlug klasy i liczby eksonow; w nawiasie liczba genow")
    print(med.round(3).astype(str).add(" (").add(cnt.fillna(0).astype(int).astype(str)).add(")").to_string())
    lnc_minus = med.loc[["lincRNA", "antysens", "inne lncRNA"]].median() - med.loc["mRNA"]
    print("   lncRNA (lincRNA, antysens, inne; mediana median) minus mRNA w tym samym przedziale eksonow:",
          ", ".join(f"{k}: {v:+.3f}" for k, v in lnc_minus.items()))

    # 2. co przewiduje odpowiedz
    print("\n2. Co przewiduje odpowiedz na wyciszenie spliceosomu? Regresja na rangach (beta standaryzowane; |t| > 3 ~ istotne)")
    for name, sel in [("mRNA", d["klasa"] == "mRNA"), ("lncRNA", d["klasa"].isin(LNC))]:
        coef, n = rank_ols(d[sel], "spl", ["log_ekson", "splicing", "stab_6h", "log_ekspr", "log_dl", "nmd"])
        print(f"   {name} (n = {n}): " + "; ".join(f"{i} {r.beta:+.2f} (t {r.t:+.1f})" for i, r in coef.iterrows()))

    # 3. spliceosom a NMD
    print("\n3. Korelacja odpowiedzi na spliceosom i na NMD/EJC (Spearman) w obrebie klas:")
    for k in ORDER:
        s = d[d["klasa"] == k][["spl", "nmd"]].dropna()
        if len(s) >= 20:
            r, p = spearmanr(s["spl"], s["nmd"])
            print(f"   {k:<20} rho = {r:+.2f} (p = {p:.1e}, n = {len(s)}); mediana NMD = {s['nmd'].median():+.3f}")

# ---------------------------------------------------------------- 4. geny gospodarze snoRNA i replikacja
if len(cons_all) == 2:
    k, h = cons_all["K562"], cons_all["HepG2"]
    print("\n4. Replikacja odpowiedzi konsensusowej miedzy liniami (Spearman, geny wspolne)")
    for k_ in ["mRNA", "lncRNA", "gospodarz snoRNA"]:
        sel = (k["klasa"].isin(LNC) if k_ == "lncRNA" else k["klasa"] == k_)
        j = k[sel][["spl", "nmd"]].join(h[["spl", "nmd"]], lsuffix="_K", rsuffix="_H").dropna()
        if len(j) >= 10:
            print(f"   {k_:<18} spliceosom rho = {spearmanr(j['spl_K'], j['spl_H'])[0]:+.2f}, "
                  f"NMD rho = {spearmanr(j['nmd_K'], j['nmd_H'])[0]:+.2f} (n = {len(j)})")
    sno = k[k["klasa"] == "gospodarz snoRNA"][["gen", "n_ekson", "spl", "nmd", "log_ekspr"]].join(
        h[["spl", "nmd"]], lsuffix="_K562", rsuffix="_HepG2", how="outer")
    sno["gen"] = sno["gen"].fillna(cls.reindex(sno.index)["gene_name"])
    sno = sno.sort_values("spl_K562", ascending=False)
    sno.to_csv(OUT / "snohost_genes.tsv", sep="\t")
    print("\n5. Geny gospodarze snoRNA: odpowiedz na spliceosom i NMD/EJC (mediany log2FC), posortowane wg spliceosomu w K562")
    print(sno.round(2).to_string())
print(f"\nZapisano: {OUT}/consensus_*.tsv.gz, {OUT}/snohost_genes.tsv")
