"""
Faza 3, krok 3.2A: pliki bigWig do EISA dla knockdownow spliceosomu i NMD/EJC (oraz ich kontroli).

Wybor knockdownow: RBP z kategoria "Spliceosome" w anotacji ENCODE oraz czynniki NMD/EJC; tylko biblioteki poliA.
Pomijane sa pliki juz pobrane (np. w pilotazu). Wyjscie:
  data/meta/phase3_samples.tsv       - probki (knockdowny + kontrole) do EISA, w formacie jak pilot_samples.tsv
  data/meta/phase3_bigwig_files.tsv  - pliki bigWig do pobrania
"""
import pandas as pd

ALIASES = {"MTREX": ["SKIV2L2"], "ATP5A1": ["ATP5F1A"], "ATP5C1": ["ATP5F1C"], "GNB2L1": ["RACK1"]}
NMD = ["UPF1", "UPF2", "UPF3B", "SMG1", "SMG5", "SMG6", "SMG7", "SMG8", "SMG9", "RBM8A", "MAGOH", "EIF4A3", "CASC3"]

raw = pd.read_excel("data/ref/rbp_annotation/supp_data1.xlsx", header=None)
sub = [str(s).strip() for s in raw.iloc[1].tolist()]
ann = raw.iloc[2:].copy()
ann.columns = ["rbp", "gene"] + sub[2:]
spl = set(ann.loc[pd.to_numeric(ann["Spliceosome"], errors="coerce") == 1, "rbp"].astype(str))
spl |= {k for k, v in ALIASES.items() if any(a in spl for a in v)}

s = pd.read_csv("data/meta/phase2_samples.tsv", sep="\t", dtype=str)
s = s[s["biblioteka"] == "poliA"]
s["grupa"] = s["target"].map(lambda t: "spliceosom" if t in spl else ("NMD/EJC" if t in NMD else None))
s = s.dropna(subset=["grupa"])
s.to_csv("data/meta/phase3_samples.tsv", sep="\t", index=False)
kds = s.drop_duplicates("kd_experiment")
print("Knockdowny:")
print(kds.groupby(["grupa", "cell"]).size().unstack(fill_value=0).to_string())

f = pd.read_csv("data/meta/files_file_level.tsv", sep="\t", dtype=str).fillna("")
types = ["plus strand signal of unique reads", "minus strand signal of unique reads"]
exps = set(s["sample_experiment"])
bw = f[f["experiment"].isin(exps) & (f["file_status"] == "released") & (f["assembly"] == "GRCh38")
       & f["output_type"].isin(types)].copy()
bw["pref"] = (bw["annotation"] == "V29").astype(int)
bw = (bw.sort_values(["experiment", "bio_rep", "output_type", "pref"], ascending=[True, True, True, False])
        .drop_duplicates(["experiment", "bio_rep", "output_type"]).drop(columns="pref"))
man = pd.read_csv("data/raw/encode/download_manifest.tsv", sep="\t", dtype=str)
have = set(man.loc[man["status"].isin(["ok", "skipped"]), "file_accession"])
bw["set"] = "phase3_bigwig"
per_exp = bw.groupby("experiment").size()
missing_exp = sorted(exps - set(per_exp.index))
odd = per_exp[per_exp != 4]
new = bw[~bw["file_accession"].isin(have)]
bw.to_csv("data/meta/phase3_bigwig_files.tsv", sep="\t", index=False)
gb = lambda d: pd.to_numeric(d["size"], errors="coerce").sum() / 1e9
print(f"\nEksperymenty (knockdowny + kontrole): {len(exps)}; z kompletem 4 plikow bigWig: {int((per_exp == 4).sum())}")
print(f"Pliki bigWig lacznie: {len(bw)} ({gb(bw):.1f} GB); juz pobrane: {len(bw) - len(new)}; "
      f"do pobrania: {len(new)} ({gb(new):.1f} GB)")
print("Eksperymenty bez plikow bigWig:", missing_exp if missing_exp else "brak")
print("Eksperymenty z liczba plikow inna niz 4:", "brak" if odd.empty else odd.to_dict())
