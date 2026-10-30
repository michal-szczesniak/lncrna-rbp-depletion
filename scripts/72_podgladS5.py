"""
Podglad ryciny S5: piec paneli locus (NEAT1, ZFAS1, SNHG5, XIST, OIP5-AS1) jeden pod drugim, z literami a-e.
Panele sa skladane z gotowych plikow PNG (skrypt 66), bez rysowania od nowa.
Wyjscie: results/figures_pub/figS5/podgladS5.png
"""
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

D = Path("results/figures_pub/figS5")
ORDER = ["NEAT1_eksosom", "ZFAS1_EJC", "SNHG5_EJC", "XIST_CIZ1", "OIP5-AS1_HuR"]
GAP, MARGIN, LEFT, LETTER_PT = 40, 30, 90, 60

files = [D / f"{n}.png" for n in ORDER if (D / f"{n}.png").exists()]
extra = sorted(p for p in D.glob("*.png") if p.stem not in ORDER and not p.stem.startswith("podglad"))
files += extra
if not files:
    raise SystemExit(f"Brak paneli PNG w {D}")
ims = [Image.open(p).convert("RGB") for p in files]
W = max(im.width for im in ims)
ims = [im if im.width == W else im.resize((W, round(im.height * W / im.width)), Image.LANCZOS) for im in ims]
H = sum(im.height for im in ims) + GAP * (len(ims) - 1) + 2 * MARGIN
out = Image.new("RGB", (W + LEFT + MARGIN, H), "white")
font = ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold")), LETTER_PT)
draw = ImageDraw.Draw(out)
y = MARGIN
for letter, im in zip("abcdefghij", ims):
    out.paste(im, (LEFT, y))
    draw.text((15, y), letter, fill="black", font=font)
    y += im.height + GAP
out.save(D / "podgladS5.png", dpi=(300, 300))
print(f"Zapisano {D / 'podgladS5.png'} ({out.width} x {out.height} px); panele: " + ", ".join(p.stem for p in files))
