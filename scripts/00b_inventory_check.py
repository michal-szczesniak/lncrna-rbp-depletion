import pandas as pd
pd.set_option("display.width", 200)
kd = pd.read_csv("data/meta/kd_experiments.tsv", sep="\t", dtype=str).fillna("")
u = kd[kd["usable"] == "True"]
print("1. Uzyteczne knockdowny wedlug metody i linii")
print(pd.crosstab(u["assay"], u["cell"], margins=True, margins_name="razem"))
print("\n2. Unikalne RBP")
for c in ["K562", "HepG2"]:
    s = u[u["cell"] == c]
    sh = set(s.loc[s["assay"] == "shRNA RNA-seq", "target_canon"])
    cr = set(s.loc[s["assay"] == "CRISPR RNA-seq", "target_canon"])
    print(f"  {c}: shRNA {len(sh)}, CRISPR {len(cr)}, razem {len(sh | cr)}, oboma sposobami {len(sh & cr)}")
both = set(u.loc[u["cell"] == "K562", "target_canon"]) & set(u.loc[u["cell"] == "HepG2", "target_canon"])
print(f"  RBP w obu liniach: {len(both)}; wszystkich unikalnych RBP: {u['target_canon'].nunique()}")
print("\n3. Typ RNA w bibliotece (poliA czy calkowite)")
print(pd.crosstab(u["assay"], u["nucleic_acid"]))
print("\n4. Wersje anotacji w kwantyfikacjach")
print(pd.crosstab(u["assay"], u["annotation"]))
print("\n5. Eksperymenty Bru")
e = pd.read_csv("data/meta/experiments_all.tsv", sep="\t", dtype=str).fillna("")
b = e[e["assay"].str.startswith("Bru")]
print(b[["experiment", "assay", "cell", "description"]].to_string(index=False, max_colwidth=90))
