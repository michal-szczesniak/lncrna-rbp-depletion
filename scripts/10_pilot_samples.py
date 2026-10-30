"""Faza 1, krok 2A: wybor knockdownow do pilotazu i tabela probek (knockdown + kontrole)."""
from pathlib import Path
import pandas as pd

PILOT = [  # (gen, linia, rola w pilotazu)
    ("MTREX", "K562", "kontrola pozytywna"), ("MTREX", "HepG2", "kontrola pozytywna"),
    ("EXOSC10", "K562", "kontrola pozytywna"), ("EXOSC10", "HepG2", "kontrola pozytywna"),
    ("PABPN1", "K562", "kontrola pozytywna"),
    ("DDX3X", "K562", "walidacja TimeLapse"), ("DDX3X", "HepG2", "walidacja TimeLapse"),
    ("PABPC4", "K562", "walidacja TimeLapse"), ("PABPC4", "HepG2", "walidacja TimeLapse"),
    ("XRN1", "K562", "degradacja cytoplazmatyczna"), ("UPF1", "K562", "degradacja cytoplazmatyczna"),
]
N_RANDOM_PER_CELL, SEED = 2, 1

kd = pd.read_csv("data/meta/kd_experiments.tsv", sep="\t", dtype=str).fillna("")
u = kd[(kd["usable"] == "True") & (kd["nucleic_acid"] != "RNA")]   # bez 16 bibliotek calkowitego RNA

rows = []
for gene, cell, role in PILOT:
    hit = u[(u["target_canon"] == gene) & (u["cell"] == cell)]
    if hit.empty:
        print(f"[uwaga] brak {gene} w {cell}")
    rows += [{**r, "pilot_role": role} for r in hit.to_dict("records")]
rest = u[~u["target_canon"].isin([g for g, _, _ in PILOT])]
for cell in ["K562", "HepG2"]:
    pool = rest[rest["cell"] == cell]
    rows += [{**r, "pilot_role": "losowy"} for r in
             pool.sample(min(N_RANDOM_PER_CELL, len(pool)), random_state=SEED).to_dict("records")]
pick = pd.DataFrame(rows)

f = pd.read_csv("data/meta/files_to_download_v29.tsv", sep="\t", dtype=str).fillna("")
m = pd.read_csv("data/raw/encode/download_manifest.tsv", sep="\t", dtype=str).fillna("")
gq = f[f["set"] == "kd_gene_quant"].merge(m[["file_accession", "path"]], on="file_accession")

samples = []
for r in pick.to_dict("records"):
    groups = [("kd", [r["experiment"]]), ("control", [a for a in r["possible_controls"].split(";") if a])]
    for condition, accs in groups:
        for fr in gq[gq["experiment"].isin(accs)].to_dict("records"):
            samples.append({"kd_experiment": r["experiment"], "target": r["target_canon"], "cell": r["cell"],
                            "method": r["assay"].split()[0], "pilot_role": r["pilot_role"],
                            "condition": condition, "sample_experiment": fr["experiment"],
                            "bio_rep": fr["bio_rep"], "file_accession": fr["file_accession"], "path": fr["path"]})
s = pd.DataFrame(samples)
s["exists"] = s["path"].map(lambda p: Path(p).exists())

pick[["experiment", "target_canon", "cell", "assay", "pilot_role", "possible_controls"]] \
    .to_csv("data/meta/pilot_kds.tsv", sep="\t", index=False)
s.to_csv("data/meta/pilot_samples.tsv", sep="\t", index=False)

pd.set_option("display.width", 200)
print(f"1. Knockdowny w pilotazu: {len(pick)}")
tab = s.groupby(["kd_experiment", "target", "cell", "method", "pilot_role", "condition"]).size() \
       .unstack("condition", fill_value=0).reset_index()
print(tab.to_string(index=False))
print(f"\n2. Pliki probek: {len(s)} (istnieje na dysku: {int(s['exists'].sum())})")

first = s.loc[s["exists"], "path"]
if len(first):
    q = pd.read_csv(first.iloc[0], sep="\t", usecols=["gene_id", "expected_count"])
    def kind(g):
        if g.startswith("ENSG"):
            return "ENSG"
        if "ERCC" in g or "spike" in g.lower():
            return "ERCC/spike-in"
        return "numeryczne (tRNA?)" if g.isdigit() else "inne"
    q["typ"] = q["gene_id"].map(kind)
    print(f"\n3. Typy identyfikatorow w {first.iloc[0]}:")
    print(q.groupby("typ").agg(geny=("gene_id", "size"), odczyty=("expected_count", "sum")).round(0).to_string())
    print("   przyklady 'inne':", ", ".join(q.loc[q["typ"] == "inne", "gene_id"].head(5)))
