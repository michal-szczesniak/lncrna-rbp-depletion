"""
Czesc 1 poprawek: sprawdzenia statystyczne.

1. Istotnosc class effect w pojedynczych eksperymentach (test Wilcoxona na roznicach lncRNA - dopasowane mRNA,
   korekta Benjaminiego-Hochberga w obrebie linii), macierz skorygowana, dopasowanie B.
2. Replikacja miedzy liniami: rho Spearmana, p analityczne i p permutacyjne (losowe przypisanie RBP miedzy liniami,
   10 000 permutacji); przedzial ufnosci 95% dla rho metoda bootstrap (resampling RBP).
3. Czy wynik zostaje bez RBP zwiazanych ze splicingiem i NMD (kategorie ENCODE: Spliceosome, Splicing regulation,
   Exon Junction Complex + czynniki NMD)?
4. Spliceosom: czy slabsza kumulacja intronow lncRNA wynika z ich wolniejszego splicingu? Porownanie lncRNA i mRNA
   o tej samej wydajnosci splicingu w kontrolach (pieciu przedzialach) i regresja na rangach.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

C, OUT = Path("results/phase2/classes"), Path("results/phase4/stats")
OUT.mkdir(parents=True, exist_ok=True)
LNC_ALL = "lncRNA (razem)"
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz miRNA"}
NMD = {"UPF1", "UPF2", "UPF3B", "SMG1", "SMG5", "SMG6", "SMG7", "SMG8", "SMG9"}
ALIASES = {"SKIV2L2": "MTREX"}
N_PERM, N_BOOT = 10_000, 2_000
pd.set_option("display.width", 220)
rng = np.random.default_rng(1)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def replication(w):
    r, p = spearmanr(w["K562"], w["HepG2"])
    a, b = w["K562"].values, w["HepG2"].values
    perm = np.array([spearmanr(a, rng.permutation(b))[0] for _ in range(N_PERM)])
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(a), len(a))
        boot.append(spearmanr(a[i], b[i])[0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return r, p, (np.sum(perm >= r) + 1) / (N_PERM + 1), lo, hi


res = pd.read_csv(C / "class_effects_all.tsv.gz", sep="\t")
main = res[(res["klasa"] == LNC_ALL) & (res["macierz"] == "skorygowana") & (res["dopasowanie"] == "B")].copy()
main["q"] = main.groupby("linia")["p"].transform(bh)

# ---------------------------------------------------------------- 1
print("1. Istotnosc class effect w pojedynczych eksperymentach (Wilcoxon, korekta BH w obrebie linii)")
rows = []
for cell, g in main.groupby("linia"):
    rows.append({"linia": cell, "eksperymenty": len(g), "q<0.05": int((g["q"] < 0.05).sum()),
                 "q<0.05 i |efekt|>=0.1": int(((g["q"] < 0.05) & (g["roznica"].abs() >= 0.1)).sum()),
                 "w tym lncRNA wyzej": int(((g["q"] < 0.05) & (g["roznica"] >= 0.1)).sum()),
                 "w tym lncRNA nizej": int(((g["q"] < 0.05) & (g["roznica"] <= -0.1)).sum())})
t1 = pd.DataFrame(rows)
print(t1.to_string(index=False))
print("   Uwaga: test traktuje lncRNA jako niezalezne; geny sa skorelowane, wiec p sa raczej zbyt optymistyczne.")
print("   Dlatego glownym dowodem jest replikacja miedzy niezaleznymi eksperymentami w dwoch liniach (punkt 2).")

# ---------------------------------------------------------------- 2
w = main.groupby(["target", "linia"])["roznica"].mean().unstack("linia").dropna()
r, p, pp, lo, hi = replication(w)
print(f"\n2. Replikacja miedzy liniami: rho = {r:.3f} (95% CI {lo:.2f} do {hi:.2f}), n = {len(w)} RBP; "
      f"p analityczne = {p:.1e}; p permutacyjne = {pp:.1e} ({N_PERM} permutacji)")

# ---------------------------------------------------------------- 3
raw = pd.read_excel("data/ref/rbp_annotation/supp_data1.xlsx", header=None)
sub = [str(s).strip() for s in raw.iloc[1].tolist()]
ann = raw.iloc[2:].copy()
ann.columns = ["rbp", "gene"] + sub[2:]
ann["rbp"] = ann["rbp"].astype(str).replace(ALIASES)
splice_cols = [c for c in ["Spliceosome", "Splicing regulation", "Exon Junction Complex"] if c in ann.columns]
splice_rbp = set(ann.loc[(ann[splice_cols].apply(pd.to_numeric, errors="coerce") == 1).any(axis=1), "rbp"])
excl = splice_rbp | NMD
w2 = w[~w.index.isin(excl)]
r2, p2, pp2, lo2, hi2 = replication(w2)
w3 = w[w.index.isin(excl)]
print(f"\n3. Bez RBP zwiazanych ze splicingiem i NMD ({len(w) - len(w2)} RBP wylaczonych z {len(w)}):")
print(f"   pozostale {len(w2)} RBP: rho = {r2:.3f} (95% CI {lo2:.2f} do {hi2:.2f}); p analityczne = {p2:.1e}; p permutacyjne = {pp2:.1e}")
if len(w3) >= 10:
    r3, p3 = spearmanr(w3["K562"], w3["HepG2"])
    print(f"   same wylaczone {len(w3)} RBP: rho = {r3:.3f}, p = {p3:.1e}")
hits = pd.read_csv(C / "rbp_hits.tsv", sep="\t", index_col=0)
print(f"   z {len(hits)} RBP z listy {int(hits.index.isin(excl).sum())} to RBP splicingu/NMD, {int((~hits.index.isin(excl)).sum())} pozostale")
print(f"   typowy |class effect|: pozostale {main[~main['target'].isin(excl)]['roznica'].abs().median():.3f}, "
      f"wylaczone {main[main['target'].isin(excl)]['roznica'].abs().median():.3f}")

# ---------------------------------------------------------------- 4
print("\n4. Spliceosom: kumulacja intronow a wydajnosc splicingu w kontrolach (EISA)")
smp = pd.read_csv("data/meta/phase3_samples.tsv", sep="\t", dtype=str).drop_duplicates("kd_experiment")
smp = smp[smp["grupa"] == "spliceosom"]
out4 = []
for cell, g in smp.groupby("cell"):
    tabs = []
    for kd in g["kd_experiment"]:
        f = Path(f"results/phase3/eisa/{kd}.tsv.gz")
        if f.exists():
            tabs.append(pd.read_csv(f, sep="\t", index_col=0))
    if not tabs:
        continue
    allk = pd.concat([t[["klasa", "d_intron", "splicing", "log_ekspr"]].assign(kd=i) for i, t in enumerate(tabs)])
    cons = allk.groupby(level=0).agg(klasa=("klasa", "first"), splicing=("splicing", "first"), log_ekspr=("log_ekspr", "median"),
                                     d_intron=("d_intron", "median"), n=("kd", "nunique"))
    cons = cons[(cons["n"] >= 3) & cons["splicing"].notna()]
    cons["k"] = np.where(cons["klasa"] == "mRNA", "mRNA", np.where(cons["klasa"].isin(LNC), "lncRNA", "inne"))
    cons = cons[cons["k"] != "inne"]
    edges = np.quantile(cons.loc[cons["k"] == "mRNA", "splicing"], [0, .2, .4, .6, .8, 1])
    edges[0], edges[-1] = -np.inf, np.inf
    cons["bin"] = pd.cut(cons["splicing"], edges, labels=["1 najwydajniej", "2", "3", "4", "5 najmniej wydajnie"])
    print(f"   {cell}: {len(g)} knockdownow; mediana wydajnosci splicingu (log2 introny/eksony na pz, im wyzej tym wolniejszy splicing): "
          f"mRNA {cons.loc[cons['k'] == 'mRNA', 'splicing'].median():.2f}, lncRNA {cons.loc[cons['k'] == 'lncRNA', 'splicing'].median():.2f}")
    for b, s in cons.groupby("bin", observed=True):
        m, l = s.loc[s["k"] == "mRNA", "d_intron"], s.loc[s["k"] == "lncRNA", "d_intron"]
        p_ = mannwhitneyu(l, m).pvalue if len(l) >= 10 and len(m) >= 10 else np.nan
        out4.append({"linia": cell, "przedzial wydajnosci splicingu": b, "mRNA (n)": len(m), "lncRNA (n)": len(l),
                     "kumulacja intronow mRNA": m.median(), "kumulacja intronow lncRNA": l.median(), "p (Mann-Whitney)": p_})
    # regresja na rangach: d_intron ~ klasa + splicing + ekspresja
    d = cons.assign(lnc=(cons["k"] == "lncRNA").astype(float))[["d_intron", "lnc", "splicing", "log_ekspr"]].dropna()
    Z = d.rank().apply(lambda c: (c - c.mean()) / c.std())
    X = np.c_[np.ones(len(Z)), Z[["lnc", "splicing", "log_ekspr"]].values]
    beta, *_ = np.linalg.lstsq(X, Z["d_intron"].values, rcond=None)
    resid = Z["d_intron"].values - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (resid @ resid) / (len(Z) - X.shape[1]))
    print(f"   {cell} regresja na rangach (kumulacja intronow): klasa lncRNA beta {beta[1]:+.2f} (t {beta[1] / se[1]:+.1f}); "
          f"wydajnosc splicingu {beta[2]:+.2f} (t {beta[2] / se[2]:+.1f}); ekspresja {beta[3]:+.2f} (t {beta[3] / se[3]:+.1f})")
t4 = pd.DataFrame(out4)
t4["q"] = bh(t4["p (Mann-Whitney)"].fillna(1).values)
print(t4.round({c: 3 for c in t4.columns if c not in ("p (Mann-Whitney)", "q")}).to_string(
    index=False, formatters={"p (Mann-Whitney)": "{:.1e}".format, "q": "{:.1e}".format}))
for name, t in [("significance_per_experiment", t1), ("spliceosome_by_splicing_efficiency", t4)]:
    t.to_csv(OUT / f"{name}.tsv", sep="\t", index=False)
pd.DataFrame([{"zestaw": "wszystkie RBP", "n": len(w), "rho": r, "ci_lo": lo, "ci_hi": hi, "p": p, "p_perm": pp},
              {"zestaw": "bez splicingu i NMD", "n": len(w2), "rho": r2, "ci_lo": lo2, "ci_hi": hi2, "p": p2, "p_perm": pp2}]
             ).to_csv(OUT / "replication_tests.tsv", sep="\t", index=False)
print(f"\nZapisano: {OUT}/")
