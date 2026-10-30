"""
Rycina 1d, poprawka ukladu: krotsza legenda z tytulem (miesci sie w pustym prawym dolnym rogu) i podpisy obejmujace
zawsze oba RBP podnoszace lncRNA z rdzenia (MTREX, STAU1). Zmienia scripts/60_fig1.py w miejscu (kopia: .py.bak2).
"""
import shutil
from pathlib import Path

P = Path("scripts/60_fig1.py")
shutil.copy(P, P.with_suffix(".py.bak2"))
s = P.read_text(encoding="utf-8")
R = [
 ('label=f"clear in both lines, robust to GC ({len(hh)})"', 'label=f"robust to GC ({len(hh)})"'),
 ('label=f"clear only without GC matching ({len(hs)})")],', 'label=f"GC-sensitive ({len(hs)})")],'),
 ('loc="lower right", fontsize=4.5, handletextpad=0.2, borderaxespad=0.1, frameon=False)',
  'loc="lower right", fontsize=4.5, title="clear in both lines", title_fontsize=4.5, handletextpad=0.2,\n          borderaxespad=0.1, frameon=False)'),
 ('lab = hh.assign(m=hh.mean(axis=1).abs()).sort_values("m", ascending=False).head(12)',
  'up_ = hh[hh.mean(axis=1) > 0]\n'
  'dn_ = hh[hh.mean(axis=1) <= 0]\n'
  'dn_ = dn_.loc[dn_.mean(axis=1).abs().sort_values(ascending=False).index[:max(12 - len(up_), 0)]]\n'
  'lab = pd.concat([up_, dn_])'),
]
for a, b in R:
    n = s.count(a)
    assert n == 1, f"nie znaleziono dokladnie raz ({n}): {a[:70]}"
    s = s.replace(a, b)
P.write_text(s, encoding="utf-8")
print(f"Zmieniono {P} ({len(R)} zmian); kopia: {P.with_suffix('.py.bak2')}")
