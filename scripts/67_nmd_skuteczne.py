"""
Sprawdzenie twierdzenia o NMD na samych skutecznych wyciszeniach czynnikow NMD.

Skuteczne = wyciszenie podnosi znane cele NMD (z ryciny S2c): co najmniej 3 z 12 celow istotnie w gore i mediana zmiany
celow > 0,1. Dla kazdego wyciszenia czynnika NMD (UPF, SMG; dane EISA z kroku 3.2):
  - dojrzale RNA (eksony): mediana zmiany lncRNA minus mRNA w 5 przedzialach ekspresji (wyznaczonych na mRNA),
  - test w kazdym przedziale (Mann-Whitney, geny lncRNA vs mRNA), korekta BH w obrebie wyciszenia,
  - class effect z glownej analizy (lncRNA vs dopasowane mRNA, macierz skorygowana).
Wyjscie: results/phase4/followups/nmd_skuteczne.tsv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz miRNA"}
NMDF = {"UPF1", "UPF2", "UPF3B", "SMG1", "SMG5", "SMG6", "SMG7", "SMG8", "SMG9"}
OUT = Path("results/phase4/followups")
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


eff = pd.read_csv("results/figures_pub/figS2/c_NMD_wyciszenia.tsv", sep="\t").set_index("kd_experiment")
eff["skuteczne"] = (eff["celow_NMD_w_gore_padj05"] >= 3) & (eff["mediana_zmiany_celow_NMD"] > 0.1)
ce = pd.read_csv("results/phase2/classes/class_effects_all.tsv.gz", sep="\t")
ce = ce[(ce["klasa"] == "lncRNA (razem)") & (ce["macierz"] == "skorygowana") & (ce["dopasowanie"] == "B")].set_index("kd_experiment")["roznica"]
smp = pd.read_csv("data/meta/phase3_samples.tsv", sep="\t", dtype=str).drop_duplicates("kd_experiment")
smp = smp[smp["target"].isin(NMDF)]
rows = []
for _, r in smp.iterrows():
    f = Path(f"results/phase3/eisa/{r['kd_experiment']}.tsv.gz")
    if not f.exists():
        continue
    d = pd.read_csv(f, sep="\t", index_col=0)
    d["k"] = np.where(d["klasa"] == "mRNA", "mRNA", np.where(d["klasa"].isin(LNC), "lncRNA", "other"))
    d = d[d["k"] != "other"]
    x = np.log10(d["ctrl_ekson"] + 1)
    edges = np.quantile(x[d["k"] == "mRNA"], [0, .2, .4, .6, .8, 1])
    edges[0], edges[-1] = -np.inf, np.inf
    d["bin"] = pd.cut(x, edges, labels=[1, 2, 3, 4, 5])
    diffs, ps = [], []
    for b, g in d.groupby("bin", observed=True):
        lm, mm = g.loc[g["k"] == "lncRNA", "d_ekson"].dropna(), g.loc[g["k"] == "mRNA", "d_ekson"].dropna()
        if len(lm) >= 10 and len(mm) >= 10:
            diffs.append(lm.median() - mm.median())
            ps.append(mannwhitneyu(lm, mm).pvalue)
        else:
            diffs.append(np.nan); ps.append(np.nan)
    qs = bh(np.nan_to_num(ps, nan=1.0))
    e = eff.loc[r["kd_experiment"]] if r["kd_experiment"] in eff.index else None
    rows.append({"wyciszenie": f"{r['target']} {r['cell']}", "kd_experiment": r["kd_experiment"],
                 "skuteczne_NMD": bool(e["skuteczne"]) if e is not None else None,
                 "cele_NMD_w_gore": int(e["celow_NMD_w_gore_padj05"]) if e is not None else None,
                 **{f"lnc-mRNA_przedz{i + 1}": v for i, v in enumerate(diffs)},
                 "srednia_roznica": np.nanmean(diffs),
                 "przedzialy_lnc>mRNA_q<0.05": int(sum((dv > 0) and (q < 0.05) for dv, q in zip(diffs, qs) if np.isfinite(dv))),
                 "przedzialy_lnc<mRNA_q<0.05": int(sum((dv < 0) and (q < 0.05) for dv, q in zip(diffs, qs) if np.isfinite(dv))),
                 "class_effect": ce.get(r["kd_experiment"], np.nan)})
t = pd.DataFrame(rows).sort_values(["skuteczne_NMD", "wyciszenie"], ascending=[False, True])
t.to_csv(OUT / "nmd_skuteczne.tsv", sep="\t", index=False)
print("Wyciszenia czynnikow NMD: dojrzale RNA lncRNA minus mRNA (mediany) w 5 przedzialach ekspresji; class effect z glownej analizy")
print(t.round(3).to_string(index=False))
for flag, g in t.groupby("skuteczne_NMD"):
    print(f"\n   {'SKUTECZNE' if flag else 'nieskuteczne'} wyciszenia NMD ({len(g)}): srednia roznica lncRNA - mRNA "
          f"{g['srednia_roznica'].mean():+.3f}; mediana class effect {g['class_effect'].median():+.3f}; "
          f"przedzialy z lncRNA istotnie wyzej: {int(g['przedzialy_lnc>mRNA_q<0.05'].sum())} z {5 * len(g)}")
