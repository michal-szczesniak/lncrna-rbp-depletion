"""
Faza 3, krok 3.6: efekty cis wyciszenia na NAT-y i sasiadow wyciszanego genu (CRISPR = nokaut Cas9 vs shRNA).

Dla kazdego knockdownu bierzemy geny z okna cis (krok 2.1; bez samego wyciszanego genu) i ich niesciagniete log2FC
z DESeq2 (bez maskowania i bez korekty na partie - kazdy knockdown wzgledem wlasnej kontroli).
Tlo: dla kazdego genu cis odejmujemy mediane log2FC genow tej samej grupy (mRNA albo lncRNA) z tego samego decyla
ekspresji w tym samym knockdownie (delta). Kategorie genow cis:
  NAT              - naklada sie na wyciszany gen, przeciwna nic
  naklad. ta nic   - naklada sie, ta sama nic
  sasiad <=10 kb, przeciwna nic (m.in. dywergentne) / ta sama nic
  sasiad 10-50 kb
1. Mediana delta i odsetek istotnych spadkow (padj < 0.05) wzgledem odsetka spadkow w calym transkryptomie.
2. Czy spadek NAT idzie w parze z sila wyciszenia celu (korelacja log2FC celu i delta NAT miedzy knockdownami).
3. Przyklady najsilniejszych zmian NAT.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

OUT = Path("results/phase3/cis_nat")
OUT.mkdir(parents=True, exist_ok=True)
LNC = {"lincRNA", "antysens", "dywergentny", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA", "sense-intronic"}
MIN_BASEMEAN = 10
pd.set_option("display.width", 220)

cis = pd.read_csv("data/meta/phase2_cis_genes.tsv", sep="\t")
cis = cis[cis["gene_id"] != cis["target_gene_id"]].copy()
summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t").set_index("kd_experiment")
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t").set_index("gene_id")["klasa"]


def category(r):
    if r["odleglosc"] == 0:
        return "NAT" if not r["ta_sama_nic"] else "naklad. ta sama nic"
    if r["odleglosc"] <= 10_000:
        return "sasiad <=10 kb, przeciwna nic" if not r["ta_sama_nic"] else "sasiad <=10 kb, ta sama nic"
    return "sasiad 10-50 kb"


cis["kategoria"] = cis.apply(category, axis=1)
rows, bg = [], []
for kd, g in cis.groupby("kd_experiment"):
    f = Path(f"results/phase2/deseq2/{kd}.tsv.gz")
    if not f.exists() or kd not in summ.index:
        continue
    d = pd.read_csv(f, sep="\t", usecols=["gene_id", "baseMean", "lfc_mle", "padj"]).set_index("gene_id")
    d = d[d["baseMean"] >= MIN_BASEMEAN].dropna(subset=["lfc_mle"])
    d["grupa"] = np.select([cls.reindex(d.index).isin(LNC), cls.reindex(d.index) == "mRNA"], ["lncRNA", "mRNA"], "inne")  # 29.10: "mRNA" = tylko geny kodujace bialka
    d["decyl"] = pd.qcut(np.log10(d["baseMean"]), 10, labels=False, duplicates="drop")
    ref = d.groupby(["grupa", "decyl"])["lfc_mle"].median()
    bg.append({"kd_experiment": kd, "metoda": summ.loc[kd, "method"],
               "tlo_spadki_proc": 100 * ((d["padj"] < 0.05) & (d["lfc_mle"] < 0)).mean()})
    gg = g[g["gene_id"].isin(d.index)]
    for r in gg.itertuples():
        x = d.loc[r.gene_id]
        rows.append({"kd_experiment": kd, "linia": summ.loc[kd, "cell"], "metoda": summ.loc[kd, "method"],
                     "target": r.target, "target_lfc": summ.loc[kd, "target_lfc"], "gen": r.gene_name,
                     "grupa": x["grupa"], "kategoria": r.kategoria, "lfc": x["lfc_mle"], "padj": x["padj"],
                     "delta": x["lfc_mle"] - ref.get((x["grupa"], x["decyl"]), np.nan)})
res = pd.DataFrame(rows)
bgd = pd.DataFrame(bg)
res.to_csv(OUT / "cis_effects.tsv", sep="\t", index=False)

print("1. Zmiana genow cis wzgledem tla (delta = log2FC minus mediana genow tej samej grupy i ekspresji)")
print(f"   Tlo: odsetek istotnych spadkow w calym transkryptomie, mediana z knockdownow: "
      + ", ".join(f"{m} {v:.1f}%" for m, v in bgd.groupby("metoda")["tlo_spadki_proc"].median().items()))
out = []
for (kat, met, grp), s in res.groupby(["kategoria", "metoda", "grupa"]):
    if len(s) < 10:
        continue
    out.append({"kategoria": kat, "metoda": met, "geny": grp, "par": len(s), "mediana_delta": s["delta"].median(),
                "p": wilcoxon(s["delta"].dropna()).pvalue,
                "istotny_spadek_%": 100 * ((s["padj"] < 0.05) & (s["lfc"] < 0)).mean(),
                "istotny_wzrost_%": 100 * ((s["padj"] < 0.05) & (s["lfc"] > 0)).mean()})
out = pd.DataFrame(out).sort_values(["kategoria", "geny", "metoda"])
print(out.round({"mediana_delta": 3, "istotny_spadek_%": 1, "istotny_wzrost_%": 1}).to_string(
    index=False, formatters={"p": "{:.1e}".format}))

print("\n2. Czy zmiana NAT idzie w parze z sila wyciszenia celu? Spearman log2FC celu vs delta NAT (miedzy knockdownami)")
for met, s in res[res["kategoria"] == "NAT"].groupby("metoda"):
    s = s.dropna(subset=["target_lfc", "delta"])
    if len(s) >= 10:
        r, p = spearmanr(s["target_lfc"], s["delta"])
        print(f"   {met}: rho = {r:+.2f} (p = {p:.1e}, {len(s)} par cel-NAT); dodatnie rho = NAT spada tym bardziej, im mocniej wyciszono cel")

print("\n3. NAT-y o najsilniejszym spadku i wzroscie (delta), z istotnoscia w DESeq2")
nat = res[res["kategoria"] == "NAT"].sort_values("delta")
cols = ["linia", "metoda", "target", "target_lfc", "gen", "grupa", "lfc", "padj", "delta"]
print("   spadki:")
print(nat.head(12)[cols].round(3).to_string(index=False))
print("   wzrosty:")
print(nat.tail(8).iloc[::-1][cols].round(3).to_string(index=False))
print(f"\nZapisano: {OUT}/cis_effects.tsv")
