"""
Czesc 4B-1: pliki bigWig do rycin locus (studia przypadkow).

Dla kazdego przypadku: knockdowny wskazanych RBP w danych liniach (biblioteki poliA) i ich kontrole.
Wybiera pliki "plus/minus strand signal of unique reads" (GRCh38, najlepiej V29), pomija juz pobrane.
Wyjscie: data/meta/case_samples.tsv (probki), data/meta/case_bigwig_files.tsv (lista do pobrania).
Liste przypadkow mozna zmienic w CASES (RBP, linie).
"""
import pandas as pd

CASES = {"eksosom -> NEAT1, KCNQ1OT1": [("MTREX", "K562"), ("MTREX", "HepG2"), ("EXOSC2", "HepG2"), ("EXOSC5", "HepG2"), ("EXOSC2", "K562"), ("EXOSC5", "K562"), ("EXOSC9", "K562"), ("EXOSC10", "K562")],
         "EJC -> ZFAS1, SNHG5": [("MAGOH", "K562"), ("MAGOH", "HepG2"), ("EIF4A3", "K562"), ("EIF4A3", "HepG2")],
         "NORAD": [("SRSF1", "K562"), ("SRSF1", "HepG2"), ("RCC2", "K562"), ("RCC2", "HepG2"), ("EIF3D", "K562"), ("EIF3D", "HepG2")],
         "CIZ1 -> XIST": [("CIZ1", "K562")],
         "NONO -> NEAT1": [("NONO", "K562"), ("NONO", "HepG2")],
         "HuR -> OIP5-AS1": [("ELAVL1", "K562"), ("ELAVL1", "HepG2")]}

s = pd.read_csv("data/meta/phase2_samples.tsv", sep="\t", dtype=str)
s = s[s["biblioteka"] == "poliA"]
rows = []
for case, items in CASES.items():
    for rbp, cell in items:
        sel = s[(s["target"] == rbp) & (s["cell"] == cell)]
        if sel.empty:
            print(f"   [uwaga] brak knockdownu {rbp} w {cell}")
        rows.append(sel.assign(przypadek=case))
cs = pd.concat(rows).drop_duplicates(["przypadek", "kd_experiment", "condition", "sample_experiment", "bio_rep"])
cs.to_csv("data/meta/case_samples.tsv", sep="\t", index=False)
print("Knockdowny w przypadkach:")
print(cs.drop_duplicates(["przypadek", "kd_experiment"]).groupby("przypadek")[["target", "cell", "method"]]
      .apply(lambda g: ", ".join(f"{a} {b} ({c})" for a, b, c in g.values)).to_string())

f = pd.read_csv("data/meta/files_file_level.tsv", sep="\t", dtype=str).fillna("")
types = ["plus strand signal of unique reads", "minus strand signal of unique reads"]
exps = set(cs["sample_experiment"])
bw = f[f["experiment"].isin(exps) & (f["file_status"] == "released") & (f["assembly"] == "GRCh38") & f["output_type"].isin(types)].copy()
bw["pref"] = (bw["annotation"] == "V29").astype(int)
bw = (bw.sort_values(["experiment", "bio_rep", "output_type", "pref"], ascending=[True, True, True, False])
        .drop_duplicates(["experiment", "bio_rep", "output_type"]).drop(columns="pref"))
man = pd.read_csv("data/raw/encode/download_manifest.tsv", sep="\t", dtype=str)
have = set(man.loc[man["status"].isin(["ok", "skipped"]), "file_accession"])
bw["set"] = "case_bigwig"
new = bw[~bw["file_accession"].isin(have)]
bw.to_csv("data/meta/case_bigwig_files.tsv", sep="\t", index=False)
per = bw.groupby("experiment").size()
gb = lambda d: pd.to_numeric(d["size"], errors="coerce").sum() / 1e9
print(f"\nEksperymenty (knockdowny + kontrole): {len(exps)}; z kompletem 4 plikow: {int((per == 4).sum())}; "
      f"bez plikow: {sorted(exps - set(per.index)) or 'brak'}")
print(f"Pliki bigWig: {len(bw)} ({gb(bw):.1f} GB); juz pobrane: {len(bw) - len(new)}; do pobrania: {len(new)} ({gb(new):.1f} GB)")
