"""
82_compose_figures.py - assembles Figures 1-5 and S1-S6 from the 300-dpi PNG panels in results/figures_pub, with panel letters
A, B, C in the order in which the panels are cited in the text.
Layout: a list of rows; a row is a list of cells; a cell is (letter, file), ("col", [(letter, file), ...]) for a column of
panels, or ("grid", [[(letter, file), ...], [...]]) for a block of panels in several rows. A row wider than MAX_W (180 mm) is
scaled down proportionally. Output: <output_dir>/<figure>.png (300 dpi).
Figure 6 and the graphical abstract are schematic drawings and are not made by this script.
Usage (from the project directory): python scripts/82_compose_figures.py results/figures_pub results/figures_composed [Figure_1,Figure_S2,...]
"""
import sys
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out")
OUT.mkdir(parents=True, exist_ok=True)
ONLY = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
DPI = 300
MAX_W = int(180 / 25.4 * DPI)          # 180 mm
GAP, LETTER_W, LETTER_PT = 45, 55, 46
FONT = ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold")), LETTER_PT)

LAYOUT = {
    "Figure_1": [[("A", "fig1/b_kontrole"), ("B", "fig1/a_przyklad"), ("C", "fig1/c_rozklad")],
                 [("D", "fig1/d_replikacja"), ("E", "fig1/e_odpornosc")]],
    "Figure_2": [[("A", "fig2/a_dopasowania"), ("B", "fig2/c_polozenie")],
                 [("C", "fig2/b1_wiazanie"), ("D", "fig2/b2_najsilniejsze"), ("E", "fig2/d_profile")]],
    "Figure_3": [[("A", "fig3/a_kategorie"), ("col", [("B", "fig3/b_spliceosom"), ("C", "fig3/c_szybkosc")])],
                 [("D", "fig3/d_nmd"), ("grid", [[("E", "fig3/e1_lokalizacja"), ("F", "fig3/e2_tkanki")],
                                                 [("G", "fig3/e3_cpg"), ("H", "fig3/e4_nmd")]])]],
    "Figure_4": [[("A", "fig4/a1_K562"), ("B", "fig4/a2_HepG2"), ("C", "fig4/b_nadmiar")],
                 [("D", "fig4/c_regulatory"), ("E", "fig4/d_KCNQ1OT1")],
                 [("F", "fig4/e1_sasiedzi_K562"), ("G", "fig4/e2_sasiedzi_HepG2")]],
    "Figure_5": [[("A", "fig5/a_topk"), ("B", "fig5/b_mapa")],
                 [("C", "fig5/c_NORAD")]],
    # Supplementary figures: panel folders keep the names of an earlier draft (figS1 -> Figure S5, figS2 -> S1,
    # figS3 -> S4, figS4 -> S6, figS5 -> S2 and S3); panel letters follow the order of citation in the text.
    "Figure_S1": [[("A", "figS2/c_NMD"), ("B", "figS2/a_spadek"), ("C", "figS2/b_skutecznosc")]],
    "Figure_S2": [[("A", "figS5/ZFAS1_EJC")], [("B", "figS5/SNHG5_EJC")], [("C", "figS5/NEAT1_eksosom")]],
    "Figure_S3": [[("A", "figS5/KCNQ1OT1_K562_eksosom")], [("B", "figS5/OIP5-AS1_HuR")], [("C", "figS5/XIST_CIZ1")]],
    "Figure_S4": [[("A", "figS3/a_korekta"), ("B", "figS3/b_centyle")]],
    "Figure_S5": [[("", "figS1/a_sasiedzi")]],
    "Figure_S6": [[("A", "figS4/a_lnc"), ("B", "figS4/b_mrna"), ("C", "figS4/c_wlinii")]],
}


def tile(letter, path):
    """Panel z miejscem na litere po lewej u gory."""
    im = Image.open(SRC / f"{path}.png").convert("RGB")
    lw = LETTER_W if letter else 0
    t = Image.new("RGB", (im.width + lw, im.height + (12 if letter else 0)), "white")
    t.paste(im, (lw, 12 if letter else 0))
    if letter:
        ImageDraw.Draw(t).text((0, 0), letter, fill="black", font=FONT)
    return t


def column(items):
    tiles = [tile(*it) for it in items]
    w = max(t.width for t in tiles)
    h = sum(t.height for t in tiles) + GAP * (len(tiles) - 1)
    c = Image.new("RGB", (w, h), "white")
    y = 0
    for t in tiles:
        c.paste(t, (0, y))
        y += t.height + GAP
    return c


def hstack(ims):
    w = sum(i.width for i in ims) + GAP * (len(ims) - 1)
    h = max(i.height for i in ims)
    r = Image.new("RGB", (w, h), "white")
    x = 0
    for i in ims:
        r.paste(i, (x, 0))
        x += i.width + GAP
    return r


def grid(rows_):
    sub = [hstack([tile(*c) for c in r_]) for r_ in rows_]
    w = max(i.width for i in sub)
    h = sum(i.height for i in sub) + GAP * (len(sub) - 1)
    g = Image.new("RGB", (w, h), "white")
    y = 0
    for i in sub:
        g.paste(i, (0, y))
        y += i.height + GAP
    return g


def row(cells):
    ims = [column(c[1]) if c[0] == "col" else grid(c[1]) if c[0] == "grid" else tile(*c) for c in cells]
    w = sum(i.width for i in ims) + GAP * (len(ims) - 1)
    h = max(i.height for i in ims)
    r = Image.new("RGB", (w, h), "white")
    x = 0
    for i in ims:
        r.paste(i, (x, 0))
        x += i.width + GAP
    if r.width > MAX_W:
        f = MAX_W / r.width
        r = r.resize((MAX_W, round(r.height * f)), Image.LANCZOS)
        return r, f
    return r, 1.0


for name, rows in LAYOUT.items():
    if ONLY and name not in ONLY:
        continue
    built = [row(r) for r in rows]
    W = max(r.width for r, _ in built)
    H = sum(r.height for r, _ in built) + GAP * (len(built) - 1)
    fig = Image.new("RGB", (W, H), "white")
    y = 0
    for r, _ in built:
        fig.paste(r, (0, y))
        y += r.height + GAP
    fig.save(OUT / f"{name}.png", dpi=(DPI, DPI))
    scales = [f for _, f in built if f < 1]
    warn = ""
    if scales:
        warn = f"; wiersze zmniejszone: {', '.join(f'{f:.2f}' for f in scales)} (najmniejsza czcionka ok. {6 * min(scales):.1f} pt)"
    print(f"{name}: {W / DPI * 25.4:.0f} x {H / DPI * 25.4:.0f} mm{warn}")
