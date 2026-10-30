"""
Faza 2, krok 2.1: przygotowanie analizy wszystkich knockdownow.

1. Tabela probek dla wszystkich uzytecznych knockdownow i ich kontroli (kwantyfikacje V29)
   -> data/meta/phase2_kds.tsv, data/meta/phase2_samples.tsv
2. Jaki to CRISPR: CRISPRi (wyciszenie promotora, efekty na chromatyne) czy nokaut (ciecie DNA)?
   Opisy eksperymentow z inwentaryzacji + modyfikacje genetyczne z API ENCODE dla kilku przykladow.
3. Geny cis dla kazdego knockdownu: nakladajace sie na wyciszany gen lub lezace w promieniu 10 kb (i 50 kb,
   test wrazliwosci), na obu niciach; w tym NAT-y wyciszanego genu -> data/meta/phase2_cis_genes.tsv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ALIASES = {"MTREX": ["SKIV2L2"], "ATP5A1": ["ATP5F1A"], "ATP5C1": ["ATP5F1C"], "GNB2L1": ["RACK1"]}       # etykiety celow ENCODE -> nazwy w GENCODE v29
WINDOWS = (10_000, 50_000)
BASE = "https://www.encodeproject.org"
pd.set_option("display.width", 220)

# ---------------------------------------------------------------- 1. tabela probek
kd = pd.read_csv("data/meta/kd_experiments.tsv", sep="\t", dtype=str).fillna("")
u = kd[kd["usable"] == "True"].copy()
u["metoda"] = u["assay"].str.split().str[0]
u["biblioteka"] = np.where(u["nucleic_acid"] == "RNA", "calkowite RNA", "poliA")

f = pd.read_csv("data/meta/files_to_download_v29.tsv", sep="\t", dtype=str).fillna("")
m = pd.read_csv("data/raw/encode/download_manifest.tsv", sep="\t", dtype=str).fillna("")
gq = f[(f["set"] == "kd_gene_quant") & (f["annotation"] == "V29")].merge(m[["file_accession", "path"]], on="file_accession")

rows = []
for r in u.to_dict("records"):
    for cond, accs in [("kd", [r["experiment"]]), ("control", [a for a in r["possible_controls"].split(";") if a])]:
        for fr in gq[gq["experiment"].isin(accs)].to_dict("records"):
            rows.append({"kd_experiment": r["experiment"], "target": r["target_canon"], "cell": r["cell"],
                         "method": r["metoda"], "biblioteka": r["biblioteka"], "condition": cond,
                         "sample_experiment": fr["experiment"], "bio_rep": fr["bio_rep"],
                         "file_accession": fr["file_accession"], "path": fr["path"]})
s = pd.DataFrame(rows)
s["exists"] = s["path"].map(lambda p: Path(p).exists())
u[["experiment", "target_canon", "cell", "metoda", "biblioteka", "possible_controls"]].to_csv(
    "data/meta/phase2_kds.tsv", sep="\t", index=False)
s.to_csv("data/meta/phase2_samples.tsv", sep="\t", index=False)

per = s.groupby(["kd_experiment", "condition"]).size().unstack(fill_value=0)
bad = per[(per.get("kd", 0) < 2) | (per.get("control", 0) < 2)]
print("1. Knockdowny w fazie 2 (metoda x linia x biblioteka):")
print(u.groupby(["metoda", "cell", "biblioteka"]).size().unstack(["cell"]).fillna(0).astype(int).to_string())
print(f"   Pliki probek: {len(s)}, na dysku: {int(s['exists'].sum())}; "
      f"unikalne kontrole: {s.loc[s['condition'] == 'control', 'sample_experiment'].nunique()}")
print("   Knockdowny bez kompletu (< 2 pliki KD lub kontroli):", "brak" if bad.empty else bad.to_dict("index"))

# ---------------------------------------------------------------- 2. jaki CRISPR
print("\n2. Rodzaj CRISPR")
ex = pd.read_csv("data/meta/experiments_all.tsv", sep="\t", dtype=str).fillna("")
cr = ex[ex["experiment"].isin(u.loc[u["metoda"] == "CRISPR", "experiment"])]
desc = cr["description"].str.lower()
for kw in ["interference", "crispri", "dcas9", "krab", "knockout", "knock-out", "deletion", "knockdown"]:
    print(f"   opisy zawierajace '{kw}': {int(desc.str.contains(kw, regex=False).sum())} z {len(cr)}")
print("   przyklady opisow:")
for t in cr["description"].drop_duplicates().head(3):
    print(f"     - {t[:160]}")


def walk(o, key):
    """Wszystkie wartosci pod kluczem 'key' w zagniezdzonym JSON."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                yield v
            yield from walk(v, key)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v, key)


try:
    sess = requests.Session()
    sess.headers.update({"Accept": "application/json"})
    for acc in cr["experiment"].head(3):
        e = sess.get(f"{BASE}/experiments/{acc}/", params={"format": "json"}, timeout=120).json()
        gms = [g for v in walk(e, "genetic_modifications") for g in (v if isinstance(v, list) else [v])]
        seen = set()
        for g in gms:
            if isinstance(g, str):
                if g in seen:
                    continue
                seen.add(g)
                g = sess.get(BASE + g, params={"format": "json"}, timeout=120).json()
            if isinstance(g, dict):
                print(f"   API {acc}: category={g.get('category')}, method={g.get('method')}, purpose={g.get('purpose')}, "
                      f"opis={str(g.get('description', ''))[:80]}")
except Exception as err:  # noqa: BLE001
    print(f"   [uwaga] zapytanie do API ENCODE nie powiodlo sie: {err}")

# ---------------------------------------------------------------- 3. geny cis
genes = pd.read_csv("data/ref/genes_v29.tsv", sep="\t")
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")[["gene_id", "klasa"]]
genes = genes.merge(cls, on="gene_id", how="left")
genes = genes[~genes["gene_id"].str.endswith("_PAR_Y")]
cis, unmatched = [], []
for r in u.drop_duplicates("experiment").to_dict("records"):
    names = ALIASES.get(r["target_canon"], []) + [r["target_canon"]]
    t = genes[genes["gene_name"].isin(names)]
    if t.empty:
        unmatched.append(r["target_canon"])
        continue
    t = t.iloc[0]
    g = genes[(genes["chr"] == t["chr"]) & (genes["end"] >= t["start"] - max(WINDOWS)) &
              (genes["start"] <= t["end"] + max(WINDOWS))].copy()
    g["odleglosc"] = np.maximum(0, np.maximum(t["start"] - g["end"], g["start"] - t["end"]))
    g["ta_sama_nic"] = g["strand"] == t["strand"]
    g["kd_experiment"], g["target"], g["target_gene_id"] = r["experiment"], r["target_canon"], t["gene_id"]
    cis.append(g[["kd_experiment", "target", "target_gene_id", "gene_id", "gene_name", "gene_type", "klasa",
                  "odleglosc", "ta_sama_nic"]])
cis = pd.concat(cis, ignore_index=True)
for w in WINDOWS:
    cis[f"w{w // 1000}kb"] = cis["odleglosc"] <= w
cis.to_csv("data/meta/phase2_cis_genes.tsv", sep="\t", index=False)

nk = cis["kd_experiment"].nunique()
print(f"\n3. Geny cis: dopasowano wyciszany gen dla {nk} z {u['experiment'].nunique()} knockdownow")
for w in WINDOWS:
    c = cis[cis[f"w{w // 1000}kb"]]
    per_kd = c.groupby("kd_experiment").size()
    print(f"   okno {w // 1000} kb: mediana {per_kd.median():.0f} genow na knockdown (zakres {per_kd.min()}-{per_kd.max()}), "
          f"w tym nakladajace sie: mediana {c[c['odleglosc'] == 0].groupby('kd_experiment').size().reindex(per_kd.index, fill_value=0).median():.0f}")
c10 = cis[cis["w10kb"] & (cis["gene_id"] != cis["target_gene_id"])]
print("   klasy genow cis w oknie 10 kb (bez samego wyciszanego genu, wszystkie knockdowny razem):")
print("   ", c10["klasa"].value_counts(dropna=False).to_dict())
nat = c10[(c10["odleglosc"] == 0) & ~c10["ta_sama_nic"] & c10["klasa"].isin(["antysens", "lincRNA", "inne lncRNA", "dywergentny", "sense-intronic"])]
print(f"   knockdowny, ktorych wyciszany gen ma nakladajacy sie lncRNA na przeciwnej nici (NAT): "
      f"{nat['kd_experiment'].nunique()} z {nk}")
print(f"   wyciszane geny bez dopasowania w GENCODE v29 ({len(set(unmatched))}):", ", ".join(sorted(set(unmatched))) or "brak")
print("\nZapisano: data/meta/phase2_kds.tsv, data/meta/phase2_samples.tsv, data/meta/phase2_cis_genes.tsv")
