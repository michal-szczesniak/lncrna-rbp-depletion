"""
Tabela S1: 35 RBP z wyraznym class effect w obu liniach (analiza glowna), z efektami po dopasowaniu na GC i oznaczeniem
odpornosci na GC (wyrazne takze w wariancie z GC = rdzen 17 RBP, rycina 1d).
Zrodla: results/phase4/rbp_profiles/rbp_profiles.tsv (35 RBP), results/phase4/followups/gc_35rbp.tsv (skrypt 73).
Wyjscie: results/figures_pub/tableS1_rbp.tsv
"""
from pathlib import Path

import pandas as pd

GROUPS = {"splicing machinery": ["WBP11", "SRSF1", "PUF60", "GPKOW", "AQR", "SF1", "RBM22", "SF3A2", "SAP30BP", "DDX39B", "AKAP8L", "PSIP1"],
          "paraspeckle proteins": ["NONO", "SFPQ"], "cytoplasmic mRNA decay and repression": ["CNOT2", "DDX6", "AGO1", "PUM2"],
          "translation": ["RPS19", "EIF3G"], "transcription (pausing/termination)": ["PPP1R10"], "non-classical RBPs": ["HSPD1", "XRCC6", "PHF6", "FKBP4"],
          "RNA decay": ["MTREX", "CNOT10", "STAU1"], "hnRNP, FET and splicing regulators": ["HNRNPK", "TAF15", "DAZAP1", "SUGP2"],
          "stress granules": ["G3BP2"], "nucleolus and mitochondria": ["NOL12", "DDX28"]}
GROUP_OF = {r: g for g, rs in GROUPS.items() for r in rs}

prof = pd.read_csv("results/phase4/rbp_profiles/rbp_profiles.tsv", sep="\t", index_col=0)
gc = pd.read_csv("results/phase4/followups/gc_35rbp.tsv", sep="\t").set_index("RBP")
t = pd.DataFrame(index=prof.index)
t["group"] = [GROUP_OF.get(r, "-") for r in t.index]
t["direction"] = ["lncRNAs higher" if (prof.loc[r, "class_effect_K562"] + prof.loc[r, "class_effect_HepG2"]) > 0 else "lncRNAs lower" for r in t.index]
t["class_effect_K562"] = prof["class_effect_K562"].round(3)
t["class_effect_HepG2"] = prof["class_effect_HepG2"].round(3)
t["class_effect_K562_GC_matched"] = gc["C_K562"].reindex(t.index).round(3)
t["class_effect_HepG2_GC_matched"] = gc["C_HepG2"].reindex(t.index).round(3)
t["robust_to_GC"] = gc["w_C"].reindex(t.index).astype(str).eq("True")
t["method_K562"] = prof["metoda_K562"]
t["method_HepG2"] = prof["metoda_HepG2"]
t = t.sort_values(["robust_to_GC", "direction", "class_effect_HepG2"], ascending=[False, True, True])
t.index.name = "RBP"
out = Path("results/figures_pub/tableS1_rbp.tsv")
t.to_csv(out, sep="\t")
print(f"Zapisano {out}: {len(t)} RBP, w tym {int(t['robust_to_GC'].sum())} odpornych na GC")
print(t[["group", "direction", "class_effect_K562", "class_effect_HepG2", "class_effect_K562_GC_matched",
         "class_effect_HepG2_GC_matched", "robust_to_GC"]].to_string())
