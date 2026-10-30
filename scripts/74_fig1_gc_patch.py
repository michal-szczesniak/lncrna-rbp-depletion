"""
Rycina 1 po decyzji o rdzeniu RBP odpornym na GC (wariant 2):
  d: 17 RBP wyraznych w obu liniach i odpornych na GC - pelne punkty; pozostale 18 z listy 35 - puste; legenda;
     rho, p i 95% CI z results/phase4/stats/replication_tests.tsv (te same liczby co w tekscie).
  e: wiersz glowny z tego samego pliku; dodane warianty z GC (dopasowanie na GC; usuniety trend GC) z gc_summary.tsv.
Zmienia scripts/60_fig1.py w miejscu (kopia: scripts/60_fig1.py.bak).
"""
import shutil
from pathlib import Path

P = Path("scripts/60_fig1.py")
shutil.copy(P, P.with_suffix(".py.bak"))
s = P.read_text(encoding="utf-8")

R = [
 ('''w = per_rbp(main)
r, p, lo, hi = rho_ci(w["K562"], w["HepG2"])
fig, ax = panel_fixed(36, 36)
is_hit = w.index.isin(hits.index)
ax.scatter(w.loc[~is_hit, "K562"], w.loc[~is_hit, "HepG2"], s=4, color=PAL["grey"], lw=0)
hh = w[is_hit]
ax.scatter(hh["K562"], hh["HepG2"], s=8, lw=0, color=np.where(hh.mean(axis=1) > 0, PAL["up"], PAL["down"]))
''',
  '''w = per_rbp(main)
# rho, p i 95% CI z jednego zrodla (te same liczby co w tekscie)
rtm = pd.read_csv("results/phase4/stats/replication_tests.tsv", sep="\\t").iloc[0]
r, p, lo, hi = float(rtm["rho"]), float(rtm["p"]), float(rtm["ci_lo"]), float(rtm["ci_hi"])
if int(rtm["n"]) != len(w):
    print(f"   UWAGA: replication_tests.tsv ma n = {int(rtm['n'])}, a panel d {len(w)} RBP")
gc35 = pd.read_csv("results/phase4/followups/gc_35rbp.tsv", sep="\\t")
robust = set(gc35.loc[gc35["w_B"].astype(str).eq("True") & gc35["w_C"].astype(str).eq("True"), "RBP"])
fig, ax = panel_fixed(36, 36)
is_hit = w.index.isin(hits.index)
is_rob = is_hit & w.index.isin(robust)
ax.scatter(w.loc[~is_hit, "K562"], w.loc[~is_hit, "HepG2"], s=4, color=PAL["grey"], lw=0)
hs = w[is_hit & ~is_rob]
ax.scatter(hs["K562"], hs["HepG2"], s=8, facecolor="white", lw=0.6, zorder=3,
           edgecolor=np.where(hs.mean(axis=1) > 0, PAL["up"], PAL["down"]))
hh = w[is_rob]
ax.scatter(hh["K562"], hh["HepG2"], s=8, lw=0, zorder=4, color=np.where(hh.mean(axis=1) > 0, PAL["up"], PAL["down"]))
from matplotlib.lines import Line2D  # noqa: E402
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=2.8, color="0.35", label=f"clear in both lines, robust to GC ({len(hh)})"),
                   Line2D([], [], marker="o", ls="", ms=2.8, mfc="white", mec="0.35", mew=0.6,
                          label=f"clear only without GC matching ({len(hs)})")],
          loc="lower right", fontsize=4.5, handletextpad=0.2, borderaxespad=0.1, frameon=False)
print(f"   panel d: {len(hh)} RBP odpornych na GC, {len(hs)} wrazliwych; rho = {r:.3f} ({lo:.2f}-{hi:.2f})")
'''),
 ('''add("Correction and matching", "corrected, full matching (main)", w)
''',
  '''add("Correction and matching", "corrected, full matching (main)", stored={"rho": r, "lo": lo, "hi": hi, "n": len(w)})
gs = pd.read_csv("results/phase4/followups/gc_summary.tsv", sep="\\t").set_index("wariant")


def add_gc(group, key, label):
    if key in gs.index:
        import re
        m_ = re.match(r"\\s*(-?\\d*\\.?\\d+)\\s*[-\u2013]\\s*(-?\\d*\\.?\\d+)", str(gs.loc[key, "CI95"]))
        lo_, hi_ = float(m_.group(1)), float(m_.group(2))
        add(group, label, stored={"rho": float(gs.loc[key, "rho_miedzy_liniami"]), "lo": lo_, "hi": hi_,
                                  "n": int(gs.loc[key, "RBP_w_obu"])})


add_gc("Correction and matching", "C", "full matching + GC content")
'''),
 ('''add("Robustness", "reference mRNAs without retained-intron isoforms", per_rbp(rv[rv["wariant"] == "B_mRNA_bez_RI"]))
''',
  '''add("Robustness", "reference mRNAs without retained-intron isoforms", per_rbp(rv[rv["wariant"] == "B_mRNA_bez_RI"]))
add_gc("Robustness", "B_gc", "GC trend removed from every experiment")
'''),
]
for a, b in R:
    n = s.count(a)
    assert n == 1, f"nie znaleziono dokladnie raz ({n}): {a[:70]}"
    s = s.replace(a, b)
P.write_text(s, encoding="utf-8")
print(f"Zmieniono {P} ({len(R)} zmian); kopia: {P.with_suffix('.py.bak')}")
