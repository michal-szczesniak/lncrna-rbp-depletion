"""
Rycina S5F: KCNQ1OT1 w K562 po wyciszeniu czynnikow nadzoru jadrowego (MTREX i podjednostki eksosomu),
dla porownania z HepG2 (rycina 4E). Panel: results/figures_pub/figS5/KCNQ1OT1_K562_eksosom.png
"""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from figlocus import draw_locus, gene_profile  # noqa: E402

FIG = "figS5"
WANT = [("MTREX", "K562"), ("EXOSC2", "K562"), ("EXOSC5", "K562"), ("EXOSC9", "K562"), ("EXOSC10", "K562")]
ok = []
for t in WANT:
    try:
        gene_profile("KCNQ1OT1", [t], nbins=10)
        ok.append(t)
    except Exception as e:  # noqa: BLE001
        print(f"   pomijam {t[0]} {t[1]}: {e}")
print("   tory:", ", ".join(f"{a} {b}" for a, b in ok))
draw_locus(FIG, "KCNQ1OT1_K562_eksosom", ["KCNQ1OT1"], "KCNQ1OT1", ok)
prof = gene_profile("KCNQ1OT1", ok, nbins=10)
prof.to_csv(Path("results/figures_pub") / FIG / "KCNQ1OT1_K562_eksosom_profil.tsv", sep="\t", index=False)
print(prof.round(2).to_string(index=False))
