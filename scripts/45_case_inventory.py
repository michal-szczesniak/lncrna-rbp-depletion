"""
Czesc 4A: inwentaryzacja do studiow przypadkow.

1. Ktore RBP znane z biologii lncRNA maja knockdown (linia, metoda, spadek celu, class effect, eCLIP).
2. Znane lncRNA: ekspresja w obu liniach i knockdowny, po ktorych zmieniaja sie najsilniej
   (log2FC z macierzy skorygowanej na kontrole; obok log2FC i padj z DESeq2 wzgledem wlasnej kontroli).
3. Pary RBP-lncRNA znane z literatury: zmiana lncRNA po wyciszeniu RBP, pozycja tego RBP wsrod wszystkich
   knockdownow w danej linii (centyl: 0 = najsilniejszy spadek, 100 = najsilniejszy wzrost) i liczba pikow eCLIP
   tego RBP w genie lncRNA (ta sama nic).
Wyjscie: results/phase4/cases/*.tsv
"""
from pathlib import Path

import numpy as np
import pandas as pd

M, OUT = Path("results/phase2/matrix"), Path("results/phase4/cases")
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 230)
pd.set_option("display.max_rows", 200)

RBP_LNC = ["ELAVL1", "NONO", "SFPQ", "PSPC1", "PUM1", "PUM2", "HNRNPU", "HNRNPK", "HNRNPC", "HNRNPA2B1", "HNRNPL",
           "RBM15", "YTHDC1", "SRSF1", "SRSF3", "IGF2BP1", "IGF2BP2", "IGF2BP3", "STAU1", "KHSRP", "PTBP1", "TARDBP",
           "FUS", "TAF15", "EWSR1", "MATR3", "SAFB", "SAFB2", "DDX5", "DDX17", "DHX9", "ADAR", "ILF3", "LIN28B",
           "UPF1", "UPF2", "SMG7", "MTREX", "EXOSC10", "EXOSC5", "EXOSC9", "ZC3H8", "PABPN1", "XRN2", "CPSF6", "NUDT21"]
LNC_KNOWN = ["NEAT1", "MALAT1", "XIST", "NORAD", "LINC00657", "GAS5", "ZFAS1", "SNHG1", "SNHG5", "SNHG6", "TUG1",
             "OIP5-AS1", "UGGT1-AS1", "CYTOR", "LINC00152", "PVT1", "H19", "MEG3", "DANCR", "FTX", "JPX", "KCNQ1OT1",
             "MIR17HG", "HOTAIRM1", "EPB41L4A-AS1", "LRRC75A-AS1", "SNHG14", "TERC", "RMRP", "RPPH1", "MIR4435-2HG"]
PAIRS = [("NONO", "NEAT1"), ("SFPQ", "NEAT1"), ("PSPC1", "NEAT1"), ("NONO", "MALAT1"), ("SRSF1", "MALAT1"),
         ("HNRNPU", "XIST"), ("HNRNPK", "XIST"), ("RBM15", "XIST"), ("YTHDC1", "XIST"), ("PUM1", "NORAD"), ("PUM2", "NORAD"),
         ("ELAVL1", "OIP5-AS1"), ("ELAVL1", "NORAD"), ("UPF1", "GAS5"), ("UPF1", "ZFAS1"), ("UPF1", "SNHG5"),
         ("IGF2BP1", "H19"), ("IGF2BP2", "H19"), ("IGF2BP3", "H19"), ("MTREX", "NEAT1"), ("HNRNPC", "MALAT1"),
         ("STAU1", "TUG1"), ("TARDBP", "NEAT1"), ("FUS", "NEAT1")]

summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
summ = summ[summ["biblioteka"] == "poliA"] if "biblioteka" in summ.columns else summ
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
genes = pd.read_csv("data/ref/genes_v29.tsv", sep="\t")
res = pd.read_csv("results/phase2/classes/class_effects_all.tsv.gz", sep="\t")
ce = res[(res["klasa"] == "lncRNA (razem)") & (res["macierz"] == "skorygowana") & (res["dopasowanie"] == "B")].set_index("kd_experiment")
ef = pd.read_csv("data/meta/eclip_files.tsv", sep="\t", dtype=str)
ef["target"] = ef["target"].replace({"SKIV2L2": "MTREX"})
eclip_have = set(zip(ef["cell"], ef["target"]))

# ---------------------------------------------------------------- 1. RBP
rows = []
for t in RBP_LNC:
    s = summ[summ["target"] == t]
    if s.empty:
        rows.append({"RBP": t, "linia": "-", "metoda": "brak knockdownu w ENCODE"})
        continue
    for kd, r in s.iterrows():
        rows.append({"RBP": t, "linia": r["cell"], "metoda": r["method"], "kd_experiment": kd,
                     "spadek_celu_log2": r["target_lfc"], "geny_istotne": r.get("n_padj05", np.nan),
                     "class_effect": ce["roznica"].get(kd, np.nan), "eCLIP": "tak" if (r["cell"], t) in eclip_have else "nie"})
t1 = pd.DataFrame(rows)
t1.to_csv(OUT / "rbp_inventory.tsv", sep="\t", index=False)
print("1. RBP zwiazane z biologia lncRNA: knockdowny w ENCODE (class effect: macierz skorygowana, dopasowanie B)")
print(t1.round(3).to_string(index=False))

# ---------------------------------------------------------------- 2. znane lncRNA
name2id = cls.drop_duplicates("gene_name").set_index("gene_name")["gene_id"]
lnc_ids = {n: name2id[n] for n in LNC_KNOWN if n in name2id.index}
print(f"\n2. Znane lncRNA odnalezione w GENCODE v29: {len(lnc_ids)} z {len(LNC_KNOWN)} "
      f"(brak: {', '.join(n for n in LNC_KNOWN if n not in lnc_ids) or '-'})")
print("   Czytam wyniki DESeq2 wszystkich knockdownow dla tych genow (kilka minut)...", flush=True)
want = set(lnc_ids.values())
raw = []
for kd in summ.index:
    f = Path(f"results/phase2/deseq2/{kd}.tsv.gz")
    if f.exists():
        d = pd.read_csv(f, sep="\t", usecols=["gene_id", "baseMean", "lfc_mle", "padj"])
        raw.append(d[d["gene_id"].isin(want)].assign(kd_experiment=kd))
raw = pd.concat(raw).set_index(["kd_experiment", "gene_id"])
id2name = {v: k for k, v in lnc_ids.items()}
top_rows, expr_rows = [], []
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    for n, gid in lnc_ids.items():
        expr_rows.append({"lncRNA": n, "linia": cell, "baseMean": expr.get(gid, np.nan),
                          "knockdowny_z_pomiarem": int(L.loc[gid].notna().sum()) if gid in L.index else 0})
        if gid not in L.index or L.loc[gid].notna().sum() < 20:
            continue
        x = L.loc[gid].dropna()
        z = (x - x.median()) / (1.4826 * (x - x.median()).abs().median() + 1e-9)
        for kd in list(x.sort_values().index[:4]) + list(x.sort_values().index[-4:][::-1]):
            rr = raw.loc[(kd, gid)] if (kd, gid) in raw.index else pd.Series(dtype=float)
            top_rows.append({"lncRNA": n, "linia": cell, "RBP": summ.loc[kd, "target"], "metoda": summ.loc[kd, "method"],
                             "log2FC_skoryg": x[kd], "z_wsrod_knockdownow": z[kd],
                             "log2FC_DESeq2": rr.get("lfc_mle", np.nan), "padj_DESeq2": rr.get("padj", np.nan)})
ex = pd.DataFrame(expr_rows).pivot_table(index="lncRNA", columns="linia", values="baseMean")
print("   Ekspresja (mediana baseMean w knockdownach i kontrolach):")
print(ex.round(0).to_string())
top = pd.DataFrame(top_rows)
top.to_csv(OUT / "known_lncRNA_top_knockdowns.tsv", sep="\t", index=False)
print("\n   Knockdowny, po ktorych znane lncRNA zmieniaja sie najsilniej (4 najsilniejsze spadki i 4 wzrosty na linie;")
print("   z = odleglosc od mediany wszystkich knockdownow w odchyleniach odpornych):")
for (n, cell), g in top.groupby(["lncRNA", "linia"], sort=False):
    items = "; ".join(f"{r.RBP}({r.metoda[0]}) {r.log2FC_skoryg:+.2f}" + ("*" if r.padj_DESeq2 < 0.05 else "") for r in g.itertuples())
    print(f"   {n:<13} {cell:<6} {items}")
print("   * = padj < 0.05 w DESeq2 wzgledem wlasnej kontroli")

# ---------------------------------------------------------------- 3. znane pary
g = genes.set_index("gene_id")
man = pd.read_csv("data/raw/encode/download_manifest.tsv", sep="\t", dtype=str).drop_duplicates("file_accession", keep="last")
ef = ef.merge(man[["file_accession", "path"]], on="file_accession", how="left")


def eclip_peaks(cell, rbp, gid):
    f = ef[(ef["cell"] == cell) & (ef["target"] == rbp)]
    if f.empty or gid not in g.index or pd.isna(f["path"].iloc[0]):
        return np.nan
    p = pd.read_csv(f["path"].iloc[0], sep="\t", header=None, usecols=[0, 1, 2, 5, 6, 7])
    p.columns = ["chr", "s", "e", "strand", "l2fc", "mlog10p"]
    gg = g.loc[gid]
    mid = (p["s"] + p["e"]) // 2
    return int(((p["chr"] == gg["chr"]) & (p["strand"] == gg["strand"]) & (mid >= gg["start"]) & (mid <= gg["end"])
                & (p["l2fc"] >= 3) & (p["mlog10p"] >= 3)).sum())


rows = []
for rbp, n in PAIRS:
    gid = lnc_ids.get(n) or lnc_ids.get({"NORAD": "LINC00657"}.get(n, ""), None)
    if gid is None:
        rows.append({"RBP": rbp, "lncRNA": n, "linia": "-", "uwaga": "lncRNA nie znaleziony"})
        continue
    s = summ[summ["target"] == rbp]
    if s.empty:
        rows.append({"RBP": rbp, "lncRNA": n, "linia": "-", "uwaga": "brak knockdownu"})
        continue
    for kd, r in s.iterrows():
        cell = r["cell"]
        rr = raw.loc[(kd, gid)] if (kd, gid) in raw.index else pd.Series(dtype=float)
        rows.append({"RBP": rbp, "lncRNA": n, "gene_id": gid, "linia": cell, "metoda": r["method"], "kd_experiment": kd,
                     "log2FC_DESeq2": rr.get("lfc_mle", np.nan), "padj_DESeq2": rr.get("padj", np.nan),
                     "baseMean": rr.get("baseMean", np.nan), "piki_eCLIP_w_genie": eclip_peaks(cell, rbp, gid)})
pairs = pd.DataFrame(rows)
# centyl i log2FC po korekcie
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    for i, r in pairs[pairs["linia"] == cell].iterrows():
        gid = r["gene_id"]
        if gid in L.index and r["kd_experiment"] in L.columns:
            x = L.loc[gid].dropna()
            v = L.loc[gid, r["kd_experiment"]]
            pairs.loc[i, "log2FC_skoryg"] = v
            pairs.loc[i, "centyl_wsrod_knockdownow"] = 100 * (x < v).mean() if pd.notna(v) else np.nan
pairs.to_csv(OUT / "known_pairs.tsv", sep="\t", index=False)
print("\n3. Pary RBP-lncRNA znane z literatury")
print("   centyl: pozycja zmiany lncRNA po wyciszeniu tego RBP wsrod wszystkich knockdownow w linii (0 = najsilniejszy spadek, 100 = najsilniejszy wzrost)")
cols = [c for c in ["RBP", "lncRNA", "linia", "metoda", "baseMean", "log2FC_DESeq2", "padj_DESeq2", "log2FC_skoryg",
                    "centyl_wsrod_knockdownow", "piki_eCLIP_w_genie", "uwaga"] if c in pairs.columns]
print(pairs[cols].round({"baseMean": 0, "log2FC_DESeq2": 2, "log2FC_skoryg": 2, "centyl_wsrod_knockdownow": 1}).to_string(
    index=False, formatters={"padj_DESeq2": "{:.1e}".format}))
print(f"\nZapisano: {OUT}/rbp_inventory.tsv, known_lncRNA_top_knockdowns.tsv, known_pairs.tsv")
