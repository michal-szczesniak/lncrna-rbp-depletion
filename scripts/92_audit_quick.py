"""
92_audit_quick.py - szybkie analizy z audytu (A3b, A5-A8, A10) + podglad struktury plikow do A9. Niczego nie nadpisuje.

A3b  zgodnosc miedzy liniami po usunieciu wplywu sily odpowiedzi (n_padj05 z summary_all)
A5   GC: liczba RBP z wyraznym efektem w wariancie glownym (B), z dopasowaniem GC (C) i w obu; oczekiwana liczba losowa
     (permutacja etykiet RBP miedzy liniami) dla kazdej z tych list
A6   class effect kategorii polozenia (rozbiezne, NAT, w poblizu, miedzygenowe) dla MTREX, NELFE, PPP1R10, EXOSC*:
     wartosci bezwzgledne (wzgledem mRNA) i roznica rozbiezne - miedzygenowe z pozycja w rankingu w kazdej linii
A7   rycina 5A: nadwyzka nad poziomem losowym dla lncRNA i mRNA
A8   "basic" vs "full" matching: czy roznica rho (Fig. 1E vs 2A) wynika ze zbioru lncRNA
A10  pelna tabela skutecznosci wyciszen NMD
Wynik: logs/92_audit_quick.log (tee) i results/audit/A5_A8_*.tsv
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=RuntimeWarning)
if not Path("results/phase2/matrix").is_dir():
    sys.exit("Uruchom z katalogu projektu rbp_classes.")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)
M, C, FU = Path("results/phase2/matrix"), Path("results/phase2/classes"), Path("results/phase4/followups")
CELLS = ["K562", "HepG2"]
rng = np.random.default_rng(2)
OUT = Path("results/audit")
try:
    OUT.mkdir(parents=True, exist_ok=True)
except OSError:
    OUT = Path.home() / "rbp_audit"; OUT.mkdir(parents=True, exist_ok=True)


def head(path, n=3):
    try:
        d = pd.read_csv(path, sep="\t", nrows=2000)
        print(f"--- {path}: kolumny {list(d.columns)}")
        print(d.head(n).to_string()[:1500])
        return d
    except Exception as e:  # noqa: BLE001
        print(f"--- {path}: nie mozna odczytac ({e})")


def qcol(df):
    return next((c for c in df.columns if re.fullmatch(r"(q|q_.*|.*_q|padj|q_bh|fdr|q_wilcoxon)", str(c), re.I)), None)


summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
res = pd.read_csv(C / "class_effects_all.tsv.gz", sep="\t")
ecol = next(c for c in res.columns if res[c].astype(str).str.startswith("ENCSR").any())
print("== Struktura class_effects_all:", list(res.columns))
for c in ["klasa", "macierz", "dopasowanie"]:
    if c in res.columns:
        print(f"   {c}: {res[c].value_counts().to_dict()}")
main = res[(res["klasa"] == "lncRNA (razem)") & (res["macierz"] == "skorygowana") & (res["dopasowanie"] == "B")].set_index(ecol)


def per_rbp(s):
    t = summ.reindex(s.index)
    return s.groupby([t["target"].values, t["cell"].values]).mean().unstack().dropna()


def rho_ci(w, nb=2000):
    x, y = w["K562"].to_numpy(), w["HepG2"].to_numpy()
    r = spearmanr(x, y).statistic
    bs = [spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, len(x), len(x)) for _ in range(nb))]
    return r, *np.nanpercentile(bs, [2.5, 97.5])


# ---------------------------------------------------------------- A3b
print("\n== A3b. Sila odpowiedzi (n_padj05)")
ce = main["roznica"]
x = np.log10(pd.to_numeric(summ.reindex(ce.index)["n_padj05"], errors="coerce") + 1)
resid = pd.Series(index=ce.index, dtype=float)
for cell in CELLS:
    idx = ce.index[summ.reindex(ce.index)["cell"].values == cell]
    ok = idx[x[idx].notna() & ce[idx].notna()]
    print(f"   {cell}: rho(class effect, log10 n_padj05) = {spearmanr(x[ok], ce[ok]).statistic:.3f}; "
          f"rho(|class effect|, log10 n_padj05) = {spearmanr(x[ok], ce[ok].abs()).statistic:.3f}")
    b = np.polyfit(x[ok], ce[ok], 2)
    resid[ok] = ce[ok] - np.polyval(b, x[ok])
w0, w1 = per_rbp(ce), per_rbp(resid.dropna())
print(f"   glowny: rho = {rho_ci(w0)[0]:.3f}; po usunieciu wplywu sily odpowiedzi (wielomian 2. st.): rho = %.3f (%.2f-%.2f), n = {len(w1)}" % rho_ci(w1))

# ---------------------------------------------------------------- A5: GC i oczekiwana liczba RBP
print("\n== A5. RBP z wyraznym efektem i oczekiwana liczba losowa")
gc = head(FU / "gc_class_effects.tsv.gz")
gc = pd.read_csv(FU / "gc_class_effects.tsv.gz", sep="\t")
hits = pd.read_csv(C / "rbp_hits.tsv", sep="\t", index_col=0)
print(f"   rbp_hits.tsv: {len(hits)} RBP; kolumny {list(hits.columns)}")
head(C / "rbp_hits_robustness.tsv")
qm = qcol(main)
print(f"   kolumna q w class_effects_all: {qm}")


def rbp_line_table(ce_s, q_s):
    t = summ.reindex(ce_s.index)
    d = pd.DataFrame({"rbp": t["target"].values, "cell": t["cell"].values, "ce": ce_s.values, "q": q_s.reindex(ce_s.index).values})
    g = d.groupby(["rbp", "cell"]).agg(ce=("ce", "mean"), qmax=("q", "max"), qmin=("q", "min")).unstack("cell")
    return g.dropna()


def clear(g, qrule):
    ok = pd.Series(True, index=g.index)
    for cell in CELLS:
        ok &= (g[("ce", cell)].abs() >= 0.1) & (g[(qrule, cell)] < 0.05)
    ok &= np.sign(g[("ce", "K562")]) == np.sign(g[("ce", "HepG2")])
    return ok


def null_count(gs, rules, nperm=1000):
    """gs: lista tabel (warianty); liczba RBP spelniajacych kryterium we wszystkich wariantach po permutacji etykiet HepG2"""
    idx = gs[0].index
    cnt = []
    for _ in range(nperm):
        p = rng.permutation(len(idx))
        ok = pd.Series(True, index=idx)
        for g, rule in zip(gs, rules):
            g2 = g.copy()
            for col in g.columns:
                if col[1] == "HepG2":
                    g2[col] = g[col].to_numpy()[p]
            ok &= clear(g2, rule).reindex(idx).fillna(False)
        cnt.append(int(ok.sum()))
    return np.array(cnt)


try:
    gB = rbp_line_table(main["roznica"], main[qm])
    found = None
    for rule in ["qmax", "qmin"]:
        ok = clear(gB, rule)
        same = len(set(ok[ok].index) & set(hits.index))
        print(f"   regula {rule}: {int(ok.sum())} RBP w wariancie B, wspolnych z rbp_hits: {same}")
        if found is None or abs(int(ok.sum()) - len(hits)) < abs(found[1] - len(hits)):
            found = (rule, int(ok.sum()))
    rule = found[0]
    gecol = next(c for c in gc.columns if gc[c].astype(str).str.startswith("ENCSR").any())
    vcol = next((c for c in gc.columns if c.lower() in ("wariant", "variant")), None)
    print(f"   gc_class_effects: warianty {gc[vcol].value_counts().to_dict() if vcol else 'brak kolumny wariant'}")
    gq = qcol(gc)
    vcolval = next((c for c in gc.columns if c in ("roznica", "class_effect", "ce")), None)
    gcC = gc[gc[vcol] == "C"].set_index(gecol) if vcol else gc.set_index(gecol)
    gC = rbp_line_table(gcC[vcolval], gcC[gq])
    okB, okC = clear(gB, rule), clear(gC, rule).reindex(gB.index).fillna(False)
    both = okB & okC
    print(f"   obserwowane: B {int(okB.sum())}, C {int(okC.sum())}, oba {int(both.sum())} (oczekiwane 35 / ? / 17)")
    common = gB.index.intersection(gC.index)
    nB = null_count([gB.loc[common]], [rule])
    nC = null_count([gC.loc[common]], [rule])
    nBC = null_count([gB.loc[common], gC.loc[common]], [rule, rule])
    for lab, obs, nn in [("B", okB.sum(), nB), ("C (GC)", okC.sum(), nC), ("B i C", both.sum(), nBC)]:
        print(f"   {lab}: obserwowane {int(obs)}, losowo srednio {nn.mean():.1f} (95% {np.percentile(nn, 2.5):.0f}-{np.percentile(nn, 97.5):.0f})")
    pd.DataFrame({"B": nB, "C": nC, "B_i_C": nBC}).to_csv(OUT / "A5_null_counts.tsv", sep="\t", index=False)
    rC = rho_ci(gC.xs("ce", axis=1, level=0))
    print(f"   rho miedzy liniami w wariancie C: %.3f (%.2f-%.2f)" % rC)
except Exception as e:  # noqa: BLE001
    print(f"   A5 nie powiodlo sie: {e.__class__.__name__}: {e}")

# ---------------------------------------------------------------- A6: kategorie polozenia
print("\n== A6. Kategorie polozenia: MTREX, NELFE, PPP1R10, eksosom")
kl = res["klasa"].unique()
POS = [k for k in kl if re.search(r"(dywerg|antys|pobli|miedzyg|interg|diverg|nearby)", str(k), re.I)]
print(f"   klasy polozenia w class_effects_all: {POS}")
try:
    pos = res[res["klasa"].isin(POS) & (res["macierz"] == "skorygowana")]
    for dop in sorted(pos["dopasowanie"].unique()):
        d = pos[pos["dopasowanie"] == dop]
        t = d.groupby(["target", "linia", "klasa"])["roznica"].mean().unstack("klasa")
        div = next(k for k in POS if re.search("dywerg|diverg", k, re.I))
        itg = next(k for k in POS if re.search("miedzyg|interg", k, re.I))
        t["div_minus_interg"] = t[div] - t[itg]
        t["rank_div_minus_interg"] = t.groupby(level="linia")["div_minus_interg"].rank(ascending=False)
        t["n_RBP_w_linii"] = t.groupby(level="linia")["div_minus_interg"].transform("count")
        sel = t[t.index.get_level_values(0).isin(["MTREX", "NELFE", "PPP1R10", "EXOSC2", "EXOSC5", "EXOSC9", "EXOSC10", "XPO1", "SSB"])]
        print(f"   dopasowanie {dop}:")
        print(sel.round(3).to_string())
        t.to_csv(OUT / f"A6_position_{dop}.tsv", sep="\t")
    allc = main["roznica"]
    for g_ in ["MTREX", "NELFE", "PPP1R10"]:
        e = summ.index[summ["target"] == g_]
        print(f"   {g_}: class effect wszystkich lncRNA (glowny) = {allc.reindex(e).dropna().round(3).to_dict()}")
except Exception as e:  # noqa: BLE001
    print(f"   A6 nie powiodlo sie: {e.__class__.__name__}: {e}")

# ---------------------------------------------------------------- A7: rycina 5A
print("\n== A7. Ryc. 5A: nadwyzka nad poziomem losowym")
tk = pd.read_csv(FU / "topk_regulators.tsv", sep="\t")
obs_c, bg_c = "% genow ze wspolnym regulatorem", "% w tle (permutacja)"
tk["nadwyzka_pp"] = tk[obs_c] - tk[bg_c]
tk["krotnosc"] = tk[obs_c] / tk[bg_c]
print(tk.round(3).to_string(index=False))
for k in sorted(tk["k"].unique()):
    a = tk[(tk["k"] == k) & (tk["geny"] == "lncRNA")].iloc[0]; b = tk[(tk["k"] == k) & (tk["geny"] == "mRNA")].iloc[0]
    pa, pb = a[obs_c] / 100, b[obs_c] / 100
    se = np.sqrt(pa * (1 - pa) / a["n"] + pb * (1 - pb) / b["n"])
    z = ((a["nadwyzka_pp"] - b["nadwyzka_pp"]) / 100) / se
    print(f"   top {k}: nadwyzka lncRNA {a['nadwyzka_pp']:.1f} pp vs mRNA {b['nadwyzka_pp']:.1f} pp (z = {z:.2f}); "
          f"krotnosc {a['krotnosc']:.1f} vs {b['krotnosc']:.1f}")
tk.to_csv(OUT / "A7_topk_excess.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- A8: basic vs full matching
print("\n== A8. Dopasowanie podstawowe (ekspresja, dlugosc) vs pelne")
try:
    for dop in sorted(res["dopasowanie"].unique()):
        d = res[(res["klasa"] == "lncRNA (razem)") & (res["macierz"] == "skorygowana") & (res["dopasowanie"] == dop)].set_index(ecol)["roznica"]
        w = per_rbp(d)
        print(f"   zapisane, dopasowanie {dop}: rho = %.3f (%.2f-%.2f), n = {len(w)}" % rho_ci(w))
    cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
    kin = pd.read_csv("results/phase1/bru/gene_kinetics.tsv", sep="\t")
    LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
    sys.path.insert(0, "scripts")
    reg = pd.read_csv("data/ref/eisa_regions_v29.tsv.gz", sep="\t")
    reg["L"] = reg["end"] - reg["start"]
    lens = reg.pivot_table(index="gene_id", columns="kind", values="L", aggfunc="sum")
    gene_ids = np.sort(reg["gene_id"].unique())
    smp = pd.read_csv("data/meta/pilot_samples.tsv", sep="\t", dtype=str)
    ctrl = smp[smp["condition"] == "control"].drop_duplicates(["sample_experiment", "bio_rep"])
    spl = {}
    for cell, g in ctrl.groupby("cell"):
        vals = []
        for e, r in zip(g["sample_experiment"], g["bio_rep"]):
            f = Path(f"results/phase1/eisa/sums/{e}_{r}.npz")
            if f.exists():
                z = np.load(f)
                E, I = pd.Series(z["exon"], index=gene_ids), pd.Series(z["intron"], index=gene_ids)
                vals.append(np.log2(((I / lens["intron"].reindex(gene_ids)) / (E / lens["exon"].reindex(gene_ids))).where((E > 0) & (I > 0))))
        spl[cell] = pd.concat(vals, axis=1).median(axis=1)
    out = {}
    for cell in CELLS:
        L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
        expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
        f = pd.DataFrame({"k": cls.reindex(L.index)["klasa"], "e": np.log10(expr.reindex(L.index) + 1),
                          "w": np.log10(cls.reindex(L.index)["width"]), "s": spl[cell].reindex(L.index),
                          "t": kin[kin["cell"] == cell].drop_duplicates("gene_id").set_index("gene_id")["stab_6h"].reindex(L.index)})
        for lab, cols, need in [("A_wszystkie", ["e", "w"], ["e", "w"]), ("A_zbior_B", ["e", "w"], ["e", "w", "s", "t"]), ("B", ["e", "w", "s", "t"], ["e", "w", "s", "t"])]:
            ff = f.dropna(subset=need + ["k"])
            mr = ff[ff["k"] == "mRNA"]
            lo_, hi_ = mr[cols].quantile(0.01), mr[cols].quantile(0.99)
            q = ff[ff["k"].isin(LNC) & ((ff[cols] >= lo_) & (ff[cols] <= hi_)).all(axis=1)]
            if lab == "A_zbior_B":   # te same lncRNA co w wariancie B
                lo4, hi4 = mr[["e", "w", "s", "t"]].quantile(0.01), mr[["e", "w", "s", "t"]].quantile(0.99)
                q = q[((q[["e", "w", "s", "t"]] >= lo4) & (q[["e", "w", "s", "t"]] <= hi4)).all(axis=1)]
            mu, sd = mr[cols].mean().values, mr[cols].std().values
            _, idx = NearestNeighbors(n_neighbors=10).fit((mr[cols].values - mu) / sd).kneighbors((q[cols].values - mu) / sd)
            X = L.reindex(q.index).to_numpy(float)
            R = L.reindex(mr.index.values[idx].ravel()).to_numpy(float).reshape(idx.shape + (L.shape[1],))
            ce_ = pd.Series(np.nanmedian(X - np.nanmean(R, axis=1), axis=0), index=L.columns)
            out.setdefault(lab, []).append(ce_)
            print(f"   {cell} {lab}: lncRNA {len(q)}")
    for lab, parts in out.items():
        w = per_rbp(pd.concat(parts))
        print(f"   {lab}: rho = %.3f (%.2f-%.2f), n = {len(w)}" % rho_ci(w))
except Exception as e:  # noqa: BLE001
    print(f"   A8 nie powiodlo sie: {e.__class__.__name__}: {e}")

# ---------------------------------------------------------------- A10: NMD
print("\n== A10. Skutecznosc wyciszen NMD (nmd_skuteczne.tsv)")
print(pd.read_csv(FU / "nmd_skuteczne.tsv", sep="\t").round(3).to_string(index=False))

# ---------------------------------------------------------------- struktura do A9 (wiazanie)
print("\n== Do A9: pliki eCLIP")
for p in sorted(Path("results/phase3/eclip").rglob("*"))[:40]:
    print("  ", p, p.stat().st_size if p.is_file() else "")
for p in sorted(Path("results/phase3/eclip").glob("*.tsv*"))[:6]:
    head(p, 2)
head(FU / "eclip_top_responders.tsv", 2)
head(C / "functional_categories.tsv", 2)
print("\nGotowe.")
