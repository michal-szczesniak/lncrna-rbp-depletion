"""
Ryciny 2 i 3: poprawki po uwagach prowadzacego (6.10).
  61_fig2.py: panel A - liczby lncRNA z nazwa linii, "before:/after:" zamiast "/" i strzalki; panel C - opis osi.
  62_fig3.py: panele B i C - osobna legenda dla klasy RNA (kolor) i linii komorkowej (styl linii),
              os C: "splicing efficiency in control cells", podpisy "most/least efficient"; panel D - kolory linii w legendzie.
Kopie zapasowe: *.py.bak3
"""
from pathlib import Path


def patch(path, reps):
    p = Path(path)
    s = p.read_text()
    for a, b in reps:
        n = s.count(a)
        assert n == 1, f"{path}: znaleziono {n} razy: {a[:70]!r}"
        s = s.replace(a, b)
    p.with_suffix(".py.bak3").write_text(p.read_text())
    p.write_text(s)
    print(f"{path}: {len(reps)} zmian")


patch("scripts/61_fig2.py", [
    (r'''return f"{counts['K562'][key]:,} / {counts['HepG2'][key]:,} lncRNAs" if counts else ""''',
     r'''return f"lncRNAs: {counts['K562'][key]:,} (K562),\n{counts['HepG2'][key]:,} (HepG2)" if counts else ""'''),
    (r'''"mRNAs matched on:\nexpression, length\n→ + splicing, stability"''',
     r'''"before: expression, length\nafter: + splicing, stability"'''),
    (r'''"4 properties\n→ + nucleus/cytoplasm"''',
     r'''"before: 4 properties (left)\nafter: + nucleus/cytoplasm"'''),
    (r'''"5 properties\n→ + tissue specificity,\nCpG promoter"''',
     r'''"before: 5 properties (left)\nafter: + tissue specificity,\nCpG promoter"'''),
    (r'''"expression, length;\nall lncRNAs → only lncRNAs\nnot bound (eCLIP)"''',
     r'''"matched on expression, length\nbefore: all lncRNAs\nafter: lncRNAs not bound (eCLIP)"'''),
    (r'''ax.set_ylabel("|log2FC| bound minus\nunbound genes (per RBP)")''',
     r'''ax.set_ylabel("|log2FC|: bound minus matched\nunbound genes (per RBP)")'''),
])

HELPER = r'''LS = {"K562": "-", "HepG2": "--"}


def class_leg(ax, cells):
    """Legenda: kolor = klasa RNA, styl linii = linia komorkowa."""
    h = [plt.Line2D([], [], color=PAL["mRNA"], lw=1.2, label="mRNA"), plt.Line2D([], [], color=PAL["lncRNA"], lw=1.2, label="lncRNA")]
    h += [plt.Line2D([], [], color="k", lw=0.8, ls=LS[c], label=lab) for c, lab in cells.items()]
    ax.legend(handles=h, fontsize=4.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=2.4, columnspacing=0.8)
'''
patch("scripts/62_fig3.py", [
    ('LS = {"K562": "-", "HepG2": "--"}\n', HELPER),
    ('    fig, ax = panel_fixed(40, 36)\n    tests = []\n', '    fig, ax = panel_fixed(40, 36)\n    tests = []\n    nk = {}\n'),
    ('        nkd = s["kd"].nunique()\n', '        nkd = s["kd"].nunique()\n        nk[cell] = nkd\n'),
    ('label=f"{k}, {cell} ({nkd} exp.)")', ')'),
    ('ax.set_xlabel("expression in controls (bins of mRNAs)")', 'ax.set_xlabel("expression in control cells (mRNA quintiles)")'),
    ('    ax.legend(fontsize=4.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.6, columnspacing=0.8)\n    save(fig, FIG, name)',
     '    class_leg(ax, {c: f"{c} ({n} exp.)" for c, n in nk.items() if n})\n    save(fig, FIG, name)'),
    ('ax.legend(fontsize=4.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.6, columnspacing=0.8)\nsave(fig, FIG, "c_szybkosc")',
     'class_leg(ax, {"K562": "K562", "HepG2": "HepG2"})\nsave(fig, FIG, "c_szybkosc")'),
    (r'''ax.set_xticks([1, 2, 3, 4, 5], ["fastest", "2", "3", "4", "slowest"])''',
     r'''ax.set_xticks([1, 2, 3, 4, 5], ["most\nefficient", "2", "3", "4", "least\nefficient"])'''),
    ('ax.set_xlabel("splicing in controls (bins of mRNAs)")', r'''ax.set_xlabel("splicing efficiency in control cells\n(mRNA quintiles)")'''),
    ('plt.Line2D([], [], marker="o", ls="", color="0.65", markersize=2, label="expression bins")],',
     'plt.Line2D([], [], marker="o", ls="", color="0.65", markersize=2, label="expression bins"),\n'
     '                   plt.Line2D([], [], marker="o", ls="", color=PAL["K562"], markersize=3.5, label="K562"),\n'
     '                   plt.Line2D([], [], marker="o", ls="", color=PAL["HepG2"], markersize=3.5, label="HepG2")],'),
    ('fontsize=4.6, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1,', 'fontsize=4.6, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,'),
])
