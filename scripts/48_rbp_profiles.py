"""
Czesc 3: co wiemy o 35 RBP z powtarzalnym class effect (krok 2.5).

Dla kazdego RBP:
  - funkcje i lokalizacja wedlug recznej anotacji ENCODE (Van Nostrand i wsp. 2020, Supplementary Data 1), gen niezbedny;
  - jak go wyciszono i jak skutecznie (spadek mRNA RBP, liczba zmienionych genow) w obu liniach; czy jest eCLIP;
  - class effect w obu liniach i profil podklas lncRNA (z kroku 2.5);
  - najbardziej swoiste lncRNA, ktore zmieniaja sie po jego usunieciu w OBU liniach: dla kazdego lncRNA liczymy, jak
    nietypowa jest jego zmiana po usunieciu tego RBP na tle wszystkich knockdownow w linii (odporny z = (x - mediana) / MAD,
    macierz skorygowana na kontrole); ranking wedlug mniejszego |z| z dwoch linii, ten sam znak w obu liniach.
Wyjscie: results/phase4/rbp_profiles/rbp_profiles.tsv, rbp_top_lncRNA.tsv
"""
from pathlib import Path

import numpy as np
import pandas as pd

M, OUT = Path("results/phase2/matrix"), Path("results/phase4/rbp_profiles")
OUT.mkdir(parents=True, exist_ok=True)
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA", "sense-intronic"}
ALIASES = {"SKIV2L2": "MTREX"}
MIN_EXPR, TOP = 20, 3
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 120)

hits = pd.read_csv("results/phase2/classes/rbp_hits.tsv", sep="\t", index_col=0)
rob = pd.read_csv("results/phase2/classes/rbp_hits_robustness.tsv", sep="\t", index_col=0)
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
summ = summ[summ["biblioteka"] == "poliA"] if "biblioteka" in summ.columns else summ
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")
ef = pd.read_csv("data/meta/eclip_files.tsv", sep="\t", dtype=str)
ef["target"] = ef["target"].replace(ALIASES)

# ---------------------------------------------------------------- anotacja ENCODE (dwa wiersze naglowka)
raw = pd.read_excel("data/ref/rbp_annotation/supp_data1.xlsx", header=None)
top_row, sub = raw.iloc[0].tolist(), [str(x).strip() for x in raw.iloc[1].tolist()]
group, cur = [], None
for t in top_row:
    cur = t if isinstance(t, str) else cur
    group.append(cur)
ann = raw.iloc[2:].copy()
ann.columns = ["rbp", "gene"] + sub[2:]
ann["rbp"] = ann["rbp"].astype(str).replace(ALIASES)
ann = ann.drop_duplicates("rbp").set_index("rbp")
FUNC = [n for n, g in zip(["rbp", "gene"] + sub[2:], group) if g == "Functions" and n != "Essential Genes"]
LOC = [n for n, g in zip(["rbp", "gene"] + sub[2:], group) if g == "Localization"]


def flags(r, cols):
    if r is None:
        return "brak anotacji ENCODE"
    on = [c for c in cols if pd.to_numeric(r.get(c), errors="coerce") == 1]
    return ", ".join(on) if on else "-"


rows = []
for t in hits.index:
    a = ann.loc[t] if t in ann.index else None
    row = {"RBP": t, "class_effect_K562": hits.loc[t, "K562"], "class_effect_HepG2": hits.loc[t, "HepG2"],
           "odporny": bool(rob["odporny"].get(t, False)) if "odporny" in rob else np.nan,
           "funkcje (ENCODE)": flags(a, FUNC), "lokalizacja (ENCODE)": flags(a, LOC),
           "niezbedny": ("tak" if pd.to_numeric(a.get("Essential Genes"), errors="coerce") == 1 else "nie") if a is not None else "?"}
    for cell in ["K562", "HepG2"]:
        s = summ[(summ["target"] == t) & (summ["cell"] == cell)]
        row[f"metoda_{cell}"] = "/".join(sorted(set(s["method"]))) if len(s) else "-"
        row[f"spadek_RBP_{cell}"] = s["target_lfc"].mean() if len(s) else np.nan
        row[f"geny_zmienione_{cell}"] = int(s["n_padj05"].mean()) if len(s) and "n_padj05" in s else np.nan
        row[f"eCLIP_{cell}"] = "tak" if ((ef["target"] == t) & (ef["cell"] == cell)).any() else "nie"
    rows.append(row)
prof = pd.DataFrame(rows).set_index("RBP")
prof["srednia"] = prof[["class_effect_K562", "class_effect_HepG2"]].mean(axis=1)
prof = prof.sort_values("srednia")

# ---------------------------------------------------------------- najbardziej swoiste lncRNA dla kazdego RBP
print("Licze odporne z dla lncRNA we wszystkich knockdownach...", flush=True)
Z, LF = {}, {}
for cell in ["K562", "HepG2"]:
    L = pd.read_csv(M / f"lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    expr = pd.read_csv(M / f"expr_{cell}.tsv", sep="\t", index_col=0)["baseMean_mediana"]
    keep = cls.reindex(L.index)["klasa"].isin(LNC) & (expr.reindex(L.index) >= MIN_EXPR)
    L = L[keep]
    med = L.median(axis=1)
    mad = 1.4826 * L.sub(med, axis=0).abs().median(axis=1)
    Z[cell] = L.sub(med, axis=0).div(mad.replace(0, np.nan), axis=0)
    LF[cell] = L
tops = []
for t in prof.index:
    zs, lf = {}, {}
    for cell in ["K562", "HepG2"]:
        kds = summ[(summ["target"] == t) & (summ["cell"] == cell)].index.intersection(Z[cell].columns)
        if len(kds):
            zs[cell] = Z[cell][kds].mean(axis=1)
            lf[cell] = LF[cell][kds].mean(axis=1)
    if len(zs) < 2:
        continue
    d = pd.DataFrame({"z_K562": zs["K562"], "z_HepG2": zs["HepG2"], "lfc_K562": lf["K562"], "lfc_HepG2": lf["HepG2"]}).dropna()
    d = d[np.sign(d["z_K562"]) == np.sign(d["z_HepG2"])]
    d["z_min"] = d[["z_K562", "z_HepG2"]].abs().min(axis=1) * np.sign(d["z_K562"])
    d["gen"] = cls.reindex(d.index)["gene_name"].values
    d["klasa"] = cls.reindex(d.index)["klasa"].values
    for direction, sel in [("wzrost", d[d["z_min"] > 0].nlargest(TOP, "z_min")), ("spadek", d[d["z_min"] < 0].nsmallest(TOP, "z_min"))]:
        for gid, r in sel.iterrows():
            tops.append({"RBP": t, "kierunek": direction, "lncRNA": r["gen"], "gene_id": gid, "klasa": r["klasa"],
                         "log2FC_K562": r["lfc_K562"], "log2FC_HepG2": r["lfc_HepG2"], "z_K562": r["z_K562"], "z_HepG2": r["z_HepG2"]})
tops = pd.DataFrame(tops)
tops.to_csv(OUT / "rbp_top_lncRNA.tsv", sep="\t", index=False)


def short(g, direction):
    s = tops[(tops["RBP"] == g) & (tops["kierunek"] == direction)]
    return "; ".join(f"{r.lncRNA} ({r.log2FC_K562:+.1f}/{r.log2FC_HepG2:+.1f})" for r in s.itertuples()) or "-"


prof["lncRNA_rosna (K562/HepG2 log2FC)"] = [short(t, "wzrost") for t in prof.index]
prof["lncRNA_spadaja (K562/HepG2 log2FC)"] = [short(t, "spadek") for t in prof.index]
prof.to_csv(OUT / "rbp_profiles.tsv", sep="\t")

print("\n1. 35 RBP: anotacja ENCODE, wyciszenie, eCLIP (posortowane od najsilniejszego spadku lncRNA)")
cols1 = ["class_effect_K562", "class_effect_HepG2", "odporny", "niezbedny", "metoda_K562", "spadek_RBP_K562", "geny_zmienione_K562",
         "metoda_HepG2", "spadek_RBP_HepG2", "geny_zmienione_HepG2", "eCLIP_K562", "eCLIP_HepG2"]
print(prof[cols1].round(2).to_string())
print("\n2. Funkcje i lokalizacja wedlug anotacji ENCODE")
print(prof[["funkcje (ENCODE)", "lokalizacja (ENCODE)"]].to_string())
print(f"\n3. Najbardziej swoiste lncRNA (top {TOP}; zmiana w tym samym kierunku w obu liniach; w nawiasie log2FC K562/HepG2 po korekcie)")
print(prof[["lncRNA_rosna (K562/HepG2 log2FC)", "lncRNA_spadaja (K562/HepG2 log2FC)"]].to_string())
print(f"\nZapisano: {OUT}/rbp_profiles.tsv, {OUT}/rbp_top_lncRNA.tsv")
