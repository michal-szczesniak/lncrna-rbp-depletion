"""Faza 1, krok 5A: lista plikow bigWig (unikalne odczyty, obie nici) dla knockdownow pilotazu i ich kontroli."""
import pandas as pd

s = pd.read_csv("data/meta/pilot_samples.tsv", sep="\t", dtype=str)
exps = set(s["sample_experiment"])
f = pd.read_csv("data/meta/files_file_level.tsv", sep="\t", dtype=str).fillna("")
types = ["plus strand signal of unique reads", "minus strand signal of unique reads"]
bw = f[f["experiment"].isin(exps) & (f["file_status"] == "released") & (f["assembly"] == "GRCh38")
       & f["output_type"].isin(types)].copy()
n_all = len(bw)
# gdy ENCODE ma kilka wersji tego samego pliku (ponowne przetwarzanie), bierzemy te z anotacji V29
bw["pref"] = (bw["annotation"] == "V29").astype(int)
bw = (bw.sort_values(["experiment", "bio_rep", "output_type", "pref"], ascending=[True, True, True, False])
        .drop_duplicates(["experiment", "bio_rep", "output_type"]))
bw["set"] = "pilot_bigwig"
bw.drop(columns="pref").to_csv("data/meta/pilot_bigwig_files.tsv", sep="\t", index=False)

per_exp = bw.groupby("experiment").size()
print(f"Eksperymenty (knockdowny + kontrole): {len(exps)}, z plikami bigWig: {per_exp.size}")
print(f"Pliki bigWig: {len(bw)} (przed usunieciem duplikatow wersji: {n_all}), "
      f"{pd.to_numeric(bw['size']).sum() / 1e9:.1f} GB")
odd = per_exp[per_exp != 4]
print("Eksperymenty z liczba plikow inna niz 4 (2 repliki x 2 nici):", "brak" if odd.empty else odd.to_dict())
missing = sorted(exps - set(per_exp.index))
print("Eksperymenty bez plikow bigWig:", missing if missing else "brak")
