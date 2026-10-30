"""
91_audit_controls.py - analizy kontrolne A1-A4 z audytu (19.10). Niczego nie nadpisuje w projekcie.

A0  Odtworzenie class effect (glowny wariant) wlasna implementacja i porownanie z results/phase2/classes/class_effects_all.tsv.gz
A1  Czy korekta wspolnych kontroli (odejmowanie mediany pozostalych wyciszen z ta sama kontrola) nie zawyza zgodnosci miedzy liniami:
      a) dane surowe (bez korekty) na tych samych RBP
      b) korekta bez "wspolnych sasiadow": od kazdego wyciszenia odejmujemy mediane tylko tych wyciszen z serii,
         ktorych RBP NIE sa w serii tego samego RBP w drugiej linii
      c) "wirtualna kontrola": wyciszenie wzgledem przecietnej kontroli linii (lfc + odchylenie wlasnej kontroli od mediany
         wszystkich kontroli), bez uzycia innych wyciszen
      d) korelacja czastkowa: zgodnosc po usunieciu wplywu "skladnika sasiadow" (mediana class effect pozostalych wyciszen serii)
      e) zgodnosc w grupach RBP wg tego, jak bardzo sklad serii pokrywa sie miedzy liniami
A2  Test zerowy miary: "pseudo-lncRNA" (mRNA o cechach lncRNA) przez ten sam potok; |class effect|, rho miedzy liniami,
    odsetek eksperymentow z q < 0,05 (Wilcoxon, BH) - 20 losowan
A3  Metody: rho dla RBP wyciszonych ta sama i rozna metoda; rho po usunieciu wplywu sily odpowiedzi (jesli jest kolumna z liczba genow)
A4  UPF1 w HepG2: 12 celow NMD przed i po korekcie, ze sasiadami z serii; wszystkie wyciszenia NMD
Wyniki: results/audit/A1_A4_*.tsv (albo ~/rbp_audit/, gdy brak miejsca) + podsumowanie na ekranie.
"""
import gzip
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, rankdata
from sklearn.neighbors import NearestNeighbors

if not (Path("scripts").is_dir() and Path("results/phase2/matrix").is_dir()):
    sys.exit("Uruchom z katalogu projektu rbp_classes.")
M, C = Path("results/phase2/matrix"), Path("results/phase2/classes")
CELLS = ["K562", "HepG2"]
COLS = ["e", "w", "s", "t"]
LNC_A = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
LNC_B = LNC_A | {"sense-intronic"}
NMD_TARGETS = ["GAS5", "SNHG1", "SNHG5", "SNHG6", "SNHG8", "ZFAS1", "GADD45A", "GADD45B", "ATF4", "SMG5", "SMG7", "UPF3B"]
N_PERM, N_BOOT, N_PSEUDO = 10000, 2000, 20
rng = np.random.default_rng(1)
strip = lambda s: pd.Index(pd.Series(s).astype(str).str.split(".").str[0])

OUT = None
for d in [Path("results/audit"), Path.home() / "rbp_audit"]:
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / ".test").write_text("x"); (d / ".test").unlink()
        OUT = d
        break
    except OSError as e:
        print(f"   nie mozna pisac w {d}: {e}")
print(f"Wyniki do: {OUT}", flush=True)
REPORT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    REPORT.append(s)


def save(df, name):
    if OUT is not None:
        df.to_csv(OUT / name, sep="\t", index=True)


# ---------------------------------------------------------------- dane wspolne
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t")
say("summary_all kolumny:", list(summ.columns))
summ = summ.set_index("kd_experiment")
mcol = next((c for c in ["method", "metoda", "technika"] if c in summ.columns), None)
kds = pd.read_csv("data/meta/phase2_kds.tsv", sep="\t", dtype=str).set_index("experiment")
ctrl_of = kds["possible_controls"]
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
kin = pd.read_csv("results/phase1/bru/gene_kinetics.tsv", sep="\t")

# nazwy genow z GTF
names = {}
with gzip.open("data/ref/gencode.v29.annotation.gtf.gz", "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.split("\t", 9)
        if f[2] != "gene":
            continue
        gid = re.search(r'gene_id "([^"]+)"', f[8]).group(1).split(".")[0]
        names[gid] = re.search(r'gene_name "([^"]+)"', f[8]).group(1)
name_of = pd.Series(names)


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


spl = splicing_by_cell()


def features(cell, index):
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    return pd.DataFrame({"k": cls.reindex(index)["klasa"], "e": np.log10(expr.reindex(index) + 1),
                         "w": np.log10(cls.reindex(index)["width"]), "s": spl[cell].reindex(index),
                         "t": kin[kin["cell"] == cell].drop_duplicates("gene_id").set_index("gene_id")["stab_6h"].reindex(index)}).dropna()


def matching(f, lnc_set, k=10, pool=None, query=None):
    """zwraca (indeks genow zapytania, macierz indeksow dopasowanych mRNA w puli)"""
    mr = f[f["k"] == "mRNA"] if pool is None else f.loc[pool]
    lo_, hi_ = mr[COLS].quantile(0.01), mr[COLS].quantile(0.99)
    q = f[f["k"].isin(lnc_set)] if query is None else f.loc[query]
    q = q[((q[COLS] >= lo_) & (q[COLS] <= hi_)).all(axis=1)]
    mu, sd = mr[COLS].mean().values, mr[COLS].std().values
    sd = np.where(sd > 0, sd, 1.0)
    _, idx = NearestNeighbors(n_neighbors=k).fit((mr[COLS].values - mu) / sd).kneighbors((q[COLS].values - mu) / sd)
    return q.index, mr.index.values[idx]


def class_effects(L, q_idx, ref_idx, with_p=False):
    X = L.reindex(q_idx).to_numpy(float)
    R = L.reindex(ref_idx.ravel()).to_numpy(float).reshape(ref_idx.shape + (L.shape[1],))
    with np.errstate(all="ignore"):
        D = X - np.nanmean(R, axis=1)
    ce = pd.Series(np.nanmedian(D, axis=0), index=L.columns)
    if not with_p:
        return ce
    p = []
    for j in range(D.shape[1]):
        d = D[:, j]; d = d[np.isfinite(d)]
        p.append(wilcoxon(d).pvalue if len(d) > 10 else np.nan)
    p = np.array(p)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if ok.any():
        pv = p[ok]; o = np.argsort(pv); m = len(pv)
        qq = np.minimum.accumulate((pv[o] * m / (np.arange(m) + 1))[::-1])[::-1]
        tmp = np.empty(m); tmp[o] = np.minimum(qq, 1); q[ok] = tmp
    return ce, pd.Series(q, index=L.columns)


def per_rbp(ce_by_cell):
    """ce_by_cell: {cell: Series(exp -> ce)} -> DataFrame RBP x cell (srednia po eksperymentach)"""
    cols = {}
    for cell, s in ce_by_cell.items():
        t = summ.reindex(s.index)["target"]
        cols[cell] = s.groupby(t.values).mean()
    return pd.DataFrame(cols).dropna()


def rho_stats(w, perm=True, boot=True):
    if len(w) < 10:
        return {"rho": np.nan, "n": len(w)}
    x, y = w["K562"].to_numpy(), w["HepG2"].to_numpy()
    r = spearmanr(x, y).statistic
    out = {"rho": r, "n": len(w)}
    if boot:
        bs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(x), len(x))
            bs.append(spearmanr(x[i], y[i]).statistic)
        out["lo"], out["hi"] = np.nanpercentile(bs, [2.5, 97.5])
    if perm:
        rx, ry = rankdata(x), rankdata(y)
        rx = (rx - rx.mean()) / rx.std(); ry = (ry - ry.mean()) / ry.std()
        null = np.array([np.mean(rx * rng.permutation(ry)) for _ in range(N_PERM)])
        out["p_perm"] = (np.sum(null >= r) + 1) / (N_PERM + 1)
        out["perm_max"] = null.max()
    return out


def fmt(d):
    if not np.isfinite(d.get("rho", np.nan)):
        return f"za malo RBP (n = {d['n']})"
    s = f"rho = {d['rho']:.3f}"
    if "lo" in d:
        s += f" ({d['lo']:.2f}-{d['hi']:.2f})"
    if "p_perm" in d:
        s += f", p_perm = {d['p_perm']:.1e}"
    return s + f", n = {d['n']}"


# ---------------------------------------------------------------- macierze
Lb, Lr, F = {}, {}, {}
for cell in CELLS:
    Lb[cell] = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    Lr[cell] = pd.read_csv(M / f"lfc_{cell}.tsv.gz", sep="\t", index_col=0)[Lb[cell].columns]
    F[cell] = features(cell, Lb[cell].index)
    say(f"{cell}: {Lb[cell].shape[1]} wyciszen, {Lb[cell].shape[0]} genow; geny z cechami: {len(F[cell])}")

batch = {}  # exp -> control
for cell in CELLS:
    for e in Lb[cell].columns:
        batch[e] = str(ctrl_of.get(e, "NA"))
batch = pd.Series(batch)

# ---------------------------------------------------------------- A0: odtworzenie
say("\n== A0. Odtworzenie class effect")
res = pd.read_csv(C / "class_effects_all.tsv.gz", sep="\t")
ecol = next(c for c in res.columns if res[c].astype(str).str.startswith("ENCSR").any())
stored = res[(res["klasa"] == "lncRNA (razem)") & (res["macierz"] == "skorygowana") & (res["dopasowanie"] == "B")].set_index(ecol)["roznica"]
best = None
for tag, lset in [("bez sense-intronic", LNC_A), ("z sense-intronic", LNC_B)]:
    ce = {cell: class_effects(Lb[cell], *matching(F[cell], lset)) for cell in CELLS}
    allce = pd.concat(ce.values())
    common = allce.index.intersection(stored.index)
    r = np.corrcoef(allce[common], stored[common])[0, 1]
    mad = (allce[common] - stored[common]).abs().median()
    say(f"   {tag}: r z zapisanymi = {r:.4f}, mediana |roznicy| = {mad:.4f}, n = {len(common)}")
    if best is None or r > best[0]:
        best = (r, tag, lset)
LNC = best[2]
say(f"   dalej uzywam: {best[1]}")
MATCH = {cell: matching(F[cell], LNC) for cell in CELLS}
CE_main = {cell: class_effects(Lb[cell], *MATCH[cell]) for cell in CELLS}
w_main = per_rbp(CE_main)
s_main = rho_stats(w_main)
say(f"   glowny wariant (odtworzony): {fmt(s_main)}")
w_stored = per_rbp({cell: stored.reindex(Lb[cell].columns).dropna() for cell in CELLS})
say(f"   glowny wariant (zapisany):   {fmt(rho_stats(w_stored, perm=False, boot=False))}")
RBP342 = w_main.index

rows = [{"wariant": "glowny (odtworzony)", **s_main}]

# ---------------------------------------------------------------- A1a: dane surowe
say("\n== A1. Korekta wspolnych kontroli")
CE_raw = {cell: class_effects(Lr[cell], *MATCH[cell]) for cell in CELLS}
w_raw = per_rbp(CE_raw).reindex(RBP342).dropna()
s = rho_stats(w_raw); rows.append({"wariant": "bez korekty (te same RBP)", **s})
say(f"   a) bez korekty, te same RBP: {fmt(s)}")


# ---------------------------------------------------------------- A1b: korekta bez wspolnych sasiadow
def correct(L, cell, exclude_shared=False):
    out = {}
    other = [c for c in CELLS if c != cell][0]
    for e in L.columns:
        mates = [m for m in L.columns if m != e and batch[m] == batch[e]]
        if exclude_shared:
            rbp = summ.loc[e, "target"]
            oth_exps = [x for x in Lr[other].columns if summ.loc[x, "target"] == rbp]
            oth_batches = set(batch[x] for x in oth_exps)
            shared = set(summ.loc[[x for x in Lr[other].columns if batch[x] in oth_batches], "target"])
            mates = [m for m in mates if summ.loc[m, "target"] not in shared]
        if len(mates) >= (2 if exclude_shared else 1):
            out[e] = L[e] - L[mates].median(axis=1)
    return pd.DataFrame(out)


Lmine, Ldis = {}, {}
for cell in CELLS:
    Lmine[cell] = correct(Lr[cell], cell)
    smpl = list(Lmine[cell].columns[:30])
    rr = [np.corrcoef(*pd.concat([Lmine[cell][e], Lb[cell][e]], axis=1).dropna().to_numpy().T)[0, 1] for e in smpl]
    say(f"   {cell}: moja korekta (mediana sasiadow) vs zapisana lfc_bc: mediana r = {np.median(rr):.4f}")
    Ldis[cell] = correct(Lr[cell], cell, exclude_shared=True)
    say(f"   {cell}: korekta bez wspolnych sasiadow mozliwa dla {Ldis[cell].shape[1]} z {Lr[cell].shape[1]} wyciszen")
CE_mine = {cell: class_effects(Lmine[cell], *MATCH[cell]) for cell in CELLS}
s = rho_stats(per_rbp(CE_mine).reindex(RBP342).dropna()); rows.append({"wariant": "moja korekta (kontrola zgodnosci)", **s})
say(f"   moja korekta standardowa: {fmt(s)}")
CE_dis = {cell: class_effects(Ldis[cell], *MATCH[cell]) for cell in CELLS}
w_dis = per_rbp(CE_dis).reindex(RBP342).dropna()
s = rho_stats(w_dis); rows.append({"wariant": "korekta bez wspolnych sasiadow", **s})
say(f"   b) korekta bez wspolnych sasiadow: {fmt(s)}")
w_main_same = w_main.reindex(w_dis.index)
say(f"      dla porownania glowny wariant na tych samych RBP: {fmt(rho_stats(w_main_same, perm=False))}")

# ---------------------------------------------------------------- A1c: wirtualna kontrola
try:
    smp = pd.read_csv("data/meta/phase2_samples.tsv", sep="\t", dtype=str)
    say("   phase2_samples kolumny:", list(smp.columns))
    Lvc = {}
    for cell in CELLS:
        cs = smp[(smp["condition"] == "control") & (smp["cell"] == cell) & (smp["biblioteka"] == "poliA")].drop_duplicates(["sample_experiment", "bio_rep"])
        cs = cs[cs["path"].map(lambda p: Path(str(p)).exists())]
        cnt = {}
        for _, r in cs.iterrows():
            q = pd.read_csv(r["path"], sep="\t", usecols=["gene_id", "expected_count"])
            cnt[(r["sample_experiment"], r["bio_rep"])] = q.groupby(strip(q["gene_id"]))["expected_count"].sum()
        X = pd.DataFrame(cnt).fillna(0)
        pos = (X > 0).all(axis=1)
        lg = np.log(X[pos])
        sf = np.exp((lg.sub(lg.mean(axis=1), axis=0)).median(axis=0))
        Y = np.log2(X.div(sf, axis=1) + 1)
        ctrl_mean = Y.T.groupby(level=0).mean().T          # gen x kontrola
        dev = ctrl_mean.sub(ctrl_mean.median(axis=1), axis=0)
        L = Lr[cell].copy()
        L.index = strip(L.index)
        out = {}
        for e in L.columns:
            cc = [c for c in re.split(r"[;,\s]+", batch[e]) if c in dev.columns]
            if cc:
                out[e] = L[e] + dev[cc].mean(axis=1).reindex(L.index)
        Lvc[cell] = pd.DataFrame(out, index=L.index)
        Lvc[cell].index = Lr[cell].index
        say(f"   {cell}: wirtualna kontrola z {len(cs)} bibliotek kontrolnych ({ctrl_mean.shape[1]} kontroli); wyciszen: {Lvc[cell].shape[1]}")
    CE_vc = {cell: class_effects(Lvc[cell], *MATCH[cell]) for cell in CELLS}
    w_vc = per_rbp(CE_vc).reindex(RBP342).dropna()
    s = rho_stats(w_vc); rows.append({"wariant": "wirtualna kontrola (bez innych wyciszen)", **s})
    say(f"   c) wirtualna kontrola: {fmt(s)}")
except Exception as ex:  # noqa: BLE001
    say(f"   c) wirtualna kontrola pominieta: {ex.__class__.__name__}: {ex}")

# ---------------------------------------------------------------- A1d: korelacja czastkowa
mate_term = {}
for cell in CELLS:
    ce = CE_raw[cell]
    t = summ.reindex(ce.index)["target"]
    m = {}
    for e in ce.index:
        mates = [x for x in ce.index if x != e and batch[x] == batch[e] and t[x] != t[e]]
        m[e] = ce[mates].median() if mates else np.nan
    mate_term[cell] = pd.Series(m)
w_m = per_rbp(mate_term).reindex(RBP342)
say(f"   d) skladnik sasiadow K562 vs HepG2: rho = {spearmanr(w_m['K562'], w_m['HepG2'], nan_policy='omit').statistic:.3f}")
dfp = pd.concat([w_main.add_prefix("ce_"), w_m.add_prefix("m_")], axis=1).dropna()
Z = dfp.apply(rankdata)
Xc = np.column_stack([np.ones(len(Z)), Z["m_K562"], Z["m_HepG2"]])
res_k = Z["ce_K562"] - Xc @ np.linalg.lstsq(Xc, Z["ce_K562"], rcond=None)[0]
res_h = Z["ce_HepG2"] - Xc @ np.linalg.lstsq(Xc, Z["ce_HepG2"], rcond=None)[0]
pr = np.corrcoef(res_k, res_h)[0, 1]
say(f"      zgodnosc glowna po usunieciu skladnika sasiadow (czastkowa Spearmana): {pr:.3f} (bez usuwania: {spearmanr(dfp['ce_K562'], dfp['ce_HepG2']).statistic:.3f}; n = {len(dfp)})")
rows.append({"wariant": "glowny, czastkowa po usunieciu skladnika sasiadow", "rho": pr, "n": len(dfp)})

# ---------------------------------------------------------------- A1e: wg pokrycia skladu serii
ov = {}
for rbp in RBP342:
    sets = []
    for cell in CELLS:
        ex = [x for x in Lb[cell].columns if summ.loc[x, "target"] == rbp]
        bs = set(batch[x] for x in ex)
        sets.append(set(summ.loc[[x for x in Lb[cell].columns if batch[x] in bs], "target"]) - {rbp})
    u = sets[0] | sets[1]
    ov[rbp] = len(sets[0] & sets[1]) / len(u) if u else np.nan
ov = pd.Series(ov)
grp = pd.cut(ov, [-0.01, 0.0, 0.25, 0.5, 1.0], labels=["0", "(0-0.25]", "(0.25-0.5]", "(0.5-1]"])
for g in grp.cat.categories:
    idx = grp[grp == g].index
    if len(idx) >= 10:
        s = rho_stats(w_main.reindex(idx).dropna(), perm=False)
        rows.append({"wariant": f"glowny, pokrycie serii {g}", **s})
        say(f"   e) pokrycie skladu serii {g}: {fmt(s)}")
save(pd.DataFrame({"overlap": ov, "grupa": grp}), "A1_overlap_per_rbp.tsv")

# ---------------------------------------------------------------- A2: pseudo-lncRNA
say("\n== A2. Test zerowy: pseudo-lncRNA (mRNA o cechach lncRNA)")
real_q = {}
for cell in CELLS:
    _, real_q[cell] = class_effects(Lb[cell], *MATCH[cell], with_p=True)
qq = pd.concat(real_q.values())
say(f"   prawdziwe lncRNA: q < 0,05 w {int((qq < 0.05).sum())} z {qq.notna().sum()} eksperymentow ({(qq < 0.05).mean():.0%})")
null_rows = []
for rep in range(N_PSEUDO):
    ce_n, q_n = {}, {}
    for cell in CELLS:
        f = F[cell]
        q_idx, _ = MATCH[cell]
        mr = f[f["k"] == "mRNA"]
        mu, sd = mr[COLS].mean().values, mr[COLS].std().values
        sd = np.where(sd > 0, sd, 1.0)
        nn = NearestNeighbors(n_neighbors=5).fit((mr[COLS].values - mu) / sd)
        _, cand = nn.kneighbors((f.loc[q_idx, COLS].values - mu) / sd)
        used, pseudo = set(), []
        for row in cand[rng.permutation(len(cand))]:
            for c in rng.permutation(row):
                g = mr.index[c]
                if g not in used:
                    used.add(g); pseudo.append(g); break
        pool = mr.index.difference(pd.Index(pseudo))
        qi, ri = matching(f, None, pool=pool, query=pd.Index(pseudo))
        ce_n[cell], q_n[cell] = class_effects(Lb[cell], qi, ri, with_p=True)
    wn = per_rbp(ce_n)
    allq = pd.concat(q_n.values()); allce = pd.concat(ce_n.values())
    null_rows.append({"powtorzenie": rep, "mediana_|ce|": allce.abs().median(), "odsetek_q<0.05": (allq < 0.05).mean(),
                      "rho_miedzy_liniami": spearmanr(wn["K562"], wn["HepG2"]).statistic, "n_rbp": len(wn)})
    print(f"   pseudo {rep + 1}/{N_PSEUDO}: rho = {null_rows[-1]['rho_miedzy_liniami']:.3f}, q<0,05 = {null_rows[-1]['odsetek_q<0.05']:.0%}", flush=True)
nd = pd.DataFrame(null_rows)
save(nd, "A2_pseudo_lncRNA.tsv")
allce_real = pd.concat(CE_main.values())
say(f"   prawdziwe lncRNA: mediana |ce| = {allce_real.abs().median():.4f}, rho = {s_main['rho']:.3f}")
say(f"   pseudo-lncRNA ({N_PSEUDO}x): mediana |ce| = {nd['mediana_|ce|'].median():.4f} ({nd['mediana_|ce|'].min():.4f}-{nd['mediana_|ce|'].max():.4f}); "
    f"rho = {nd['rho_miedzy_liniami'].median():.3f} ({nd['rho_miedzy_liniami'].min():.3f}-{nd['rho_miedzy_liniami'].max():.3f}); "
    f"q < 0,05 = {nd['odsetek_q<0.05'].median():.0%} ({nd['odsetek_q<0.05'].min():.0%}-{nd['odsetek_q<0.05'].max():.0%})")

# ---------------------------------------------------------------- A3: metody
say("\n== A3. Metoda wyciszenia")
if mcol:
    meth = summ[mcol]
    mm = {}
    for cell in CELLS:
        e = Lb[cell].columns
        mm[cell] = meth.reindex(e).groupby(summ.reindex(e)["target"].values).agg(lambda x: "+".join(sorted(set(x))))
    mm = pd.DataFrame(mm).reindex(RBP342)
    same = mm["K562"] == mm["HepG2"]
    for lab, idx in [("ta sama metoda", same[same].index), ("rozne metody", same[~same].index)]:
        s = rho_stats(w_main.reindex(idx).dropna(), perm=False)
        rows.append({"wariant": f"glowny, {lab}", **s})
        say(f"   {lab}: {fmt(s)}")
else:
    say("   brak kolumny z metoda w summary_all - pomijam")
ncol = next((c for c in summ.columns if re.search(r"(n_sig|n_de|sig|istotn|liczba)", c, re.I)), None)
if ncol:
    say(f"   sila odpowiedzi: kolumna '{ncol}'")
    resid = {}
    for cell in CELLS:
        ce = CE_main[cell]
        x = np.log10(pd.to_numeric(summ.reindex(ce.index)[ncol], errors="coerce") + 1)
        ok = x.notna() & ce.notna()
        b = np.polyfit(x[ok], ce[ok], 1)
        resid[cell] = (ce - np.polyval(b, x)).where(ok)
        say(f"   {cell}: korelacja class effect z log10(liczba genow+1): rho = {spearmanr(x[ok], ce[ok]).statistic:.3f}")
    s = rho_stats(per_rbp(resid).reindex(RBP342).dropna(), perm=False)
    rows.append({"wariant": "glowny, po usunieciu sily odpowiedzi", **s})
    say(f"   po usunieciu wplywu sily odpowiedzi: {fmt(s)}")
else:
    say("   brak kolumny z liczba zmienionych genow w summary_all - pomijam")

# ---------------------------------------------------------------- A4: UPF1 i cele NMD
say("\n== A4. Cele NMD przed i po korekcie")
rows4 = []
for cell in CELLS:
    gid = strip(Lb[cell].index)
    nm = pd.Series(name_of.reindex(gid).values, index=Lb[cell].index)
    tg = nm[nm.isin(NMD_TARGETS)].index
    for e in Lb[cell].columns:
        t = summ.loc[e, "target"]
        if t in {"UPF1", "UPF2", "SMG5", "SMG6", "SMG7", "SMG8", "SMG9"} or (cell == "HepG2" and batch[e] == batch.get(
                next((x for x in Lb["HepG2"].columns if summ.loc[x, "target"] == "UPF1"), ""), "")):
            r_ = Lr[cell].loc[tg, e]; b_ = Lb[cell].loc[tg, e]
            rows4.append({"linia": cell, "RBP": t, "eksperyment": e, "kontrola": batch[e],
                          "mediana_surowa": r_.median(), "mediana_po_korekcie": b_.median(),
                          "celow_w_gore_surowe(>0.1)": int((r_ > 0.1).sum()), "celow_w_gore_po_korekcie(>0.1)": int((b_ > 0.1).sum()),
                          **{f"{nm[g]}_surowe": round(Lr[cell].loc[g, e], 2) for g in tg},
                          **{f"{nm[g]}_po": round(Lb[cell].loc[g, e], 2) for g in tg}})
a4 = pd.DataFrame(rows4)
save(a4, "A4_nmd_targets.tsv")
say(a4[["linia", "RBP", "kontrola", "mediana_surowa", "mediana_po_korekcie", "celow_w_gore_surowe(>0.1)", "celow_w_gore_po_korekcie(>0.1)"]]
    .round(3).to_string(index=False))

# ---------------------------------------------------------------- zapis
tab = pd.DataFrame(rows)
save(tab.set_index("wariant"), "A1_A4_rho.tsv")
say("\n== Podsumowanie zgodnosci miedzy liniami")
say(tab.round(3).to_string(index=False))
if OUT is not None:
    (OUT / "A1_A4_raport.txt").write_text("\n".join(REPORT))
    print(f"\nZapisano w {OUT}: A1_A4_rho.tsv, A1_overlap_per_rbp.tsv, A2_pseudo_lncRNA.tsv, A4_nmd_targets.tsv, A1_A4_raport.txt")
