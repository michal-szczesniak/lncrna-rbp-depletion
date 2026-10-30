"""
Analizy uzupelniajace po uwagach do raportu v2 (wpis 4.18).

A. Czy geny zwiazane przez RBP (eCLIP) sa nadreprezentowane wsrod genow reagujacych NAJSILNIEJ na jego wyciszenie?
   (czulszy test niz porownanie median; sprawdza "efekt usredniania"). Dla kazdej pary RBP-linia z eCLIP i knockdownem:
   najsilniej reagujace = gorne 5% |z| (nietypowosc zmiany na tle wszystkich knockdownow); tabela 2x2 (najsilniejsze/reszta
   x zwiazane/niezwiazane) w 5 przedzialach ekspresji; iloraz szans Mantela-Haenszela i test chi2 MH; korekta BH.
B. Dominujacy regulator w wersji top 3 i top 10: czy wsrod k najsilniejszych regulatorow genu (|z|) jest ten sam RBP
   z tym samym kierunkiem w obu liniach? Tlo: permutacja genow w HepG2 (zachowuje czestosc RBP). lncRNA vs mRNA
   o tej samej ekspresji; RBP najczesciej bedace wspolnym regulatorem lncRNA.
C. Replikacja class effect miedzy liniami osobno dla kategorii polozenia lncRNA (z kroku 5.1), w tym tylko miedzygenowe.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, fisher_exact, spearmanr

M, OUT = Path("results/phase2/matrix"), Path("results/phase4/followups")
OUT.mkdir(parents=True, exist_ok=True)
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA", "sense-intronic"}
MIN_EXPR, MIN_TESTED, MAD_FLOOR, TOP_FRAC, BIN, N_PERM = 20, 0.9, 0.05, 0.05, 100_000, 100
rng = np.random.default_rng(7)
pd.set_option("display.width", 220)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
genes = pd.read_csv("data/ref/genes_v29.tsv", sep="\t")
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
cis = pd.read_csv("data/meta/phase2_cis_genes.tsv", sep="\t")
cis50 = cis[cis["w50kb"]].groupby("kd_experiment")["gene_id"].apply(set)
rbps_both = set(summ.groupby("target")["cell"].nunique().loc[lambda s: s == 2].index)

Z, ZALL, EX = {}, {}, {}
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    for kd in L.columns.intersection(cis50.index):
        L.loc[L.index.isin(cis50[kd]), kd] = np.nan
    e = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"].reindex(L.index)
    L = L[(L.notna().mean(axis=1) >= MIN_TESTED) & (e >= MIN_EXPR)]
    med = L.median(axis=1)
    mad = (1.4826 * L.sub(med, axis=0).abs().median(axis=1)).clip(lower=MAD_FLOOR)
    z = L.sub(med, axis=0).div(mad, axis=0)
    t = summ.reindex(L.columns)["target"].values
    ZALL[cell] = z.T.groupby(t).mean().T                                    # geny x RBP (wszystkie RBP w linii)
    Z[cell] = ZALL[cell][[c for c in ZALL[cell].columns if c in rbps_both]]
    EX[cell] = e.reindex(L.index)

# ================================================================ A. eCLIP
man = pd.read_csv("data/raw/encode/download_manifest.tsv", sep="\t", dtype=str).drop_duplicates("file_accession", keep="last")
ef = pd.read_csv("data/meta/eclip_files.tsv", sep="\t", dtype=str).merge(man[["file_accession", "path"]], on="file_accession", how="left")
ef["target"] = ef["target"].replace({"SKIV2L2": "MTREX"})
g = genes[["gene_id", "chr", "start", "end", "strand"]].copy()
g["b0"], g["b1"] = g["start"] // BIN, g["end"] // BIN
g = g.loc[g.index.repeat(g["b1"] - g["b0"] + 1)].copy()
g["bin"] = g["b0"] + g.groupby(level=0).cumcount()


def bound_genes(path):
    p = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 5, 6, 7])
    p.columns = ["chr", "s", "e", "strand", "l2fc", "mlog10p"]
    p = p[(p["l2fc"] >= 3) & (p["mlog10p"] >= 3)].copy()
    p["mid"] = (p["s"] + p["e"]) // 2
    p["bin"] = p["mid"] // BIN
    m = p.merge(g, on=["chr", "bin", "strand"])
    return set(m.loc[(m["mid"] >= m["start"]) & (m["mid"] <= m["end"]), "gene_id"])


def mantel_haenszel(df):
    num = den = sa = se = sv = 0.0
    for _, s in df.groupby("bin", observed=True):
        a = ((s["top"]) & (s["bound"])).sum(); b = ((s["top"]) & (~s["bound"])).sum()
        c = ((~s["top"]) & (s["bound"])).sum(); d = ((~s["top"]) & (~s["bound"])).sum()
        n = a + b + c + d
        if n < 2:
            continue
        num += a * d / n; den += b * c / n
        sa += a; se += (a + b) * (a + c) / n
        sv += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1)) if n > 1 else 0
    or_ = num / den if den > 0 else np.nan
    stat = (abs(sa - se) - 0.5) ** 2 / sv if sv > 0 else np.nan
    return or_, (chi2.sf(stat, 1) if sv > 0 else np.nan), int(sa)


rows = []
for (cell, rbp), f in ef.groupby(["cell", "target"]):
    if rbp not in ZALL.get(cell, pd.DataFrame()).columns or pd.isna(f["path"].iloc[0]):
        continue
    bset = bound_genes(f["path"].iloc[0])
    z = ZALL[cell][rbp].dropna()
    for name, sel in [("mRNA", cls.reindex(z.index)["klasa"] == "mRNA"), ("lncRNA", cls.reindex(z.index)["klasa"].isin(LNC))]:
        zz = z[sel.values]
        if len(zz) < 100:
            continue
        df = pd.DataFrame({"top": zz.abs() >= zz.abs().quantile(1 - TOP_FRAC), "bound": zz.index.isin(bset)}, index=zz.index)
        if df["bound"].sum() < 10:
            continue
        df["bin"] = pd.qcut(np.log10(EX[cell].reindex(zz.index) + 1), 5, labels=False, duplicates="drop")
        or_, p, a = mantel_haenszel(df)
        rows.append({"linia": cell, "RBP": rbp, "geny": name, "zwiazane": int(df["bound"].sum()), "zwiazane_wsrod_top": a, "OR_MH": or_, "p": p})
tA = pd.DataFrame(rows)
tA["q"] = bh(tA["p"].fillna(1).values)
tA.to_csv(OUT / "eclip_top_responders.tsv", sep="\t", index=False)
print("A. Czy geny zwiazane (eCLIP) sa nadreprezentowane wsrod 5% najsilniej reagujacych? (iloraz szans Mantela-Haenszela w 5 przedzialach ekspresji)")
sumA = tA.groupby(["linia", "geny"]).agg(RBP=("RBP", "size"), mediana_OR=("OR_MH", "median"),
                                         proc_OR_powyzej_1=("OR_MH", lambda x: 100 * (x > 1).mean()),
                                         proc_istotnie_wzbogacone=("q", lambda x: 100 * ((x < 0.05) & (tA.loc[x.index, "OR_MH"] > 1)).mean()),
                                         proc_istotnie_zubozone=("q", lambda x: 100 * ((x < 0.05) & (tA.loc[x.index, "OR_MH"] < 1)).mean()))
print(sumA.round(2).to_string())
top_en = tA[(tA["q"] < 0.05) & (tA["OR_MH"] > 1)].sort_values("OR_MH", ascending=False)
print("   RBP z istotnym wzbogaceniem (top 15):")
print(top_en.head(15)[["linia", "RBP", "geny", "zwiazane", "zwiazane_wsrod_top", "OR_MH", "q"]].round(2).to_string(index=False,
      formatters={"q": "{:.1e}".format}))

# ================================================================ B. top 3 / top 10 regulatorow
common = Z["K562"].index.intersection(Z["HepG2"].index)
rbps = Z["K562"].columns.intersection(Z["HepG2"].columns)
ZK, ZH = Z["K562"].loc[common, rbps].fillna(0), Z["HepG2"].loc[common, rbps].fillna(0)
em = (np.log10(EX["K562"].reindex(common) + 1) + np.log10(EX["HepG2"].reindex(common) + 1)) / 2
kl = cls.reindex(common)["klasa"]
lnc = common[kl.isin(LNC).values]
pool = em[common[(kl == "mRNA").values]].sort_values()
used, mm = set(), []
for gid in lnc:
    i = int(np.searchsorted(pool.values, em[gid]))
    for j in sorted(range(max(0, i - 60), min(len(pool), i + 60)), key=lambda j: abs(pool.values[j] - em[gid])):
        if pool.index[j] not in used:
            used.add(pool.index[j]); mm.append(pool.index[j]); break
mm = pd.Index(mm)


def topk_sets(Zm, k):
    A = Zm.values
    idx = np.argsort(-np.abs(A), axis=1)[:, :k]
    sign = np.sign(np.take_along_axis(A, idx, axis=1))
    return idx, sign


def shared(idxK, sK, idxH, sH):
    out = np.zeros(len(idxK), dtype=int)
    for i in range(len(idxK)):
        dK = dict(zip(idxK[i], sK[i]))
        out[i] = sum(1 for j, s in zip(idxH[i], sH[i]) if j in dK and dK[j] == s)
    return out


print(f"\nB. Wspolny regulator wsrod k najsilniejszych w obu liniach (ten sam kierunek); {len(rbps)} RBP; {len(lnc)} lncRNA i {len(mm)} mRNA o tej samej ekspresji")
rowsB, hubs = [], {}
for k in [1, 3, 10]:
    for name, gs in [("lncRNA", lnc), ("mRNA", mm)]:
        iK, sK = topk_sets(ZK.loc[gs], k)
        iH, sH = topk_sets(ZH.loc[gs], k)
        sh = shared(iK, sK, iH, sH)
        null = []
        for _ in range(N_PERM):
            pm = rng.permutation(len(gs))
            null.append((shared(iK, sK, iH[pm], sH[pm]) > 0).mean())
        rowsB.append({"k": k, "geny": name, "n": len(gs), "% genow ze wspolnym regulatorem": 100 * (sh > 0).mean(),
                      "% w tle (permutacja)": 100 * np.mean(null), "krotnosc ponad tlo": (sh > 0).mean() / max(np.mean(null), 1e-9),
                      "n_wspolne": int((sh > 0).sum())})
        if k in (3, 10):
            cnt = {}
            for i in range(len(gs)):
                dK = dict(zip(iK[i], sK[i]))
                for j, s in zip(iH[i], sH[i]):
                    if j in dK and dK[j] == s:
                        cnt[rbps[j]] = cnt.get(rbps[j], 0) + 1
            hubs[(k, name)] = pd.Series(cnt, dtype=float)
tB = pd.DataFrame(rowsB)
print(tB.round(2).to_string(index=False))
for k in [3, 10]:
    a = tB[(tB["k"] == k) & (tB["geny"] == "lncRNA")].iloc[0]; b = tB[(tB["k"] == k) & (tB["geny"] == "mRNA")].iloc[0]
    o, p = fisher_exact([[a["n_wspolne"], a["n"] - a["n_wspolne"]], [b["n_wspolne"], b["n"] - b["n_wspolne"]]])
    print(f"   k = {k}: lncRNA vs mRNA iloraz szans {o:.2f}, p = {p:.1e} (Fisher)")
tB.to_csv(OUT / "topk_regulators.tsv", sep="\t", index=False)
for k in [3, 10]:
    h = pd.DataFrame({"lncRNA": hubs[(k, "lncRNA")], "mRNA": hubs[(k, "mRNA")]}).fillna(0).astype(int)
    tot_l, tot_m = h["lncRNA"].sum(), h["mRNA"].sum()
    h["p"] = [fisher_exact([[r.lncRNA, tot_l - r.lncRNA], [r.mRNA, tot_m - r.mRNA]])[1] for r in h.itertuples()]
    h["q"] = bh(h["p"].values)
    h = h.sort_values("lncRNA", ascending=False)
    h.to_csv(OUT / f"topk_hubs_k{k}.tsv", sep="\t")
    print(f"   k = {k}: RBP najczesciej wspolnym regulatorem lncRNA (top 12; liczba genow lncRNA / mRNA; q = wzbogacenie lncRNA vs mRNA)")
    print("   " + "; ".join(f"{r} {int(v.lncRNA)}/{int(v.mRNA)}" + (f" (q={v.q:.0e})" if v.q < 0.05 else "") for r, v in h.head(12).iterrows()))

# ================================================================ C. class effect wg kategorii polozenia
ce = pd.read_csv("results/phase4/network/category_effects_per_kd.tsv", sep="\t")
print("\nC. Replikacja class effect miedzy liniami osobno dla kategorii polozenia lncRNA (95% CI z bootstrapu po RBP)")
rowsC = []
for c in ["dywergentne", "antysensowe", "w poblizu", "miedzygenowe"]:
    w = ce.groupby(["target", "linia"])[c].mean().unstack("linia").dropna()
    r = spearmanr(w["K562"], w["HepG2"])[0]
    boot = []
    for _ in range(1000):
        i = rng.integers(0, len(w), len(w))
        boot.append(spearmanr(w["K562"].values[i], w["HepG2"].values[i])[0])
    rowsC.append({"kategoria": c, "RBP": len(w), "rho": r, "CI_lo": np.percentile(boot, 2.5), "CI_hi": np.percentile(boot, 97.5),
                  "p": spearmanr(w["K562"], w["HepG2"])[1], "typowy |efekt|": ce[c].abs().median()})
tC = pd.DataFrame(rowsC)
print(tC.round({"rho": 3, "CI_lo": 2, "CI_hi": 2, "typowy |efekt|": 3}).to_string(index=False, formatters={"p": "{:.1e}".format}))
tC.to_csv(OUT / "class_effect_by_position.tsv", sep="\t", index=False)
print(f"\nZapisano: {OUT}/")
