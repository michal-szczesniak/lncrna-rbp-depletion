"""
Poprawki raportu v3 po przegladzie: nazwa NAT zamiast 'antisense' (tekst i ryciny raportu ze skryptu 55),
liczba RBP porownywanych w obu liniach, zbedna wzmianka w podpisie ryciny 5, doprecyzowanie zdania o NELFE i PPP1R10,
punkt o nazewnictwie w ramce zmian. Zmienia scripts/70_report_v3.py i scripts/55_report_figs.py w miejscu.
"""
from pathlib import Path


def patch(path, reps):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for a, b in reps:
        n = s.count(a)
        assert n == 1, f"{path}: nie znaleziono dokladnie raz ({n}): {a[:60]}"
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")
    print(f"{path}: {len(reps)} zmian")


patch("scripts/55_report_figs.py", [
    ('"antisense", "other"', '"NAT", "other"'),
    ('cat_en = ["divergent", "antisense", "nearby", "intergenic"]', 'cat_en = ["divergent", "NAT", "nearby", "intergenic"]'),
    ('"lncRNA-mRNA\\nantisense"', '"lncRNA-mRNA\\nNAT"'),
])

patch("scripts/70_report_v3.py", [
    ('"antysens": "antisense"', '"antysens": "NAT"'),
    ('"antysensowe": "antisense"', '"antysensowe": "NAT"'),
    ('<em>antisense</em> (overlapping a protein-coding gene on the other strand)',
     '<em>NAT</em> (natural antisense transcript: overlapping a protein-coding gene on the other strand)'),
    ('antisense less (+0.24)', 'NAT less (+0.24)'),
    ('<dt>Divergent, antisense, intergenic lncRNA</dt>', '<dt>Divergent, NAT (natural antisense), intergenic lncRNA</dt>'),
    ('{n_both} RBPs were removed in both cell lines',
     '{n_both} RBPs were removed in both cell lines ({int(rt.iloc[0]["n"])} of them with enough measurable lncRNAs in both to be compared)'),
    (' Brackets give corrected significance (see note under the figure).', ' Brackets give corrected significance.'),
    ('also raises divergent lncRNAs in both lines,', 'also raises divergent lncRNAs relative to intergenic ones in both lines,'),
    ('<li><strong>Literature.</strong>',
     '<li><strong>Terminology.</strong> lncRNAs overlapping a protein-coding gene on the other strand are now called NAT (natural antisense transcripts), as in the figures prepared for publication.</li>\n<li><strong>Literature.</strong>'),
])
