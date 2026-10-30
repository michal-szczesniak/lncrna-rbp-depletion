import pandas as pd
import requests

f = pd.read_csv("data/meta/files_to_download_phase0.tsv", sep="\t", dtype=str).fillna("")
kd = pd.read_csv("data/meta/kd_experiments.tsv", sep="\t", dtype=str).fillna("")
u = kd[kd["usable"] == "True"].copy()

# 1. lista do pobrania: kwantyfikacje tylko V29 + wszystkie pliki Bru poza BAM
gq = f[(f["set"] == "kd_gene_quant") & (f["annotation"] == "V29")]
bru = f[f["set"] == "bru_nonbam"]
pd.concat([gq, bru]).to_csv("data/meta/files_to_download_v29.tsv", sep="\t", index=False)

# 2. czy kazdy knockdown i jego kontrole maja >= 2 repliki w V29
reps = gq.groupby("experiment")["bio_rep"].nunique()
ok = lambda acc: reps.get(acc, 0) >= 2
u["kd_ok"] = u["experiment"].map(ok)
u["ctrl_ok"] = u["possible_controls"].map(lambda v: v != "" and all(ok(a) for a in v.split(";") if a))
n_ok = int((u["kd_ok"] & u["ctrl_ok"]).sum())
print(f"1. Knockdowny z kompletem V29 (KD i kontrole): {n_ok} / {len(u)}")
bad = u[~(u["kd_ok"] & u["ctrl_ok"])]
if len(bad):
    print(bad[["experiment", "cell", "assay", "target_canon", "possible_controls"]].to_string(index=False))
gb = lambda d: pd.to_numeric(d["size"], errors="coerce").sum() / 1e9
print(f"\n2. Do pobrania: {len(gq)} plikow kwantyfikacji V29 ({gb(gq):.1f} GB) + {len(bru)} plikow Bru ({gb(bru):.1f} GB)")

print("\n3. Knockdowny na calkowitym RNA (nie poliA):")
print(u.loc[u["nucleic_acid"] == "RNA", ["experiment", "cell", "target_canon", "possible_controls"]].to_string(index=False))

print("\n4. Eksperymenty Bru (z API ENCODE):")
s = requests.Session()
for acc in sorted(bru["experiment"].unique()):
    try:
        j = s.get(f"https://www.encodeproject.org/experiments/{acc}/", params={"format": "json"},
                  headers={"Accept": "application/json"}, timeout=120).json()
        print(f"  {acc} | {j.get('assay_title', '')} | {j.get('biosample_summary', '')} | wydane {j.get('date_released', '')}")
    except Exception as e:
        print(f"  {acc} | blad zapytania: {e}")
