"""
Wspolny modul rycin do publikacji: styl, paleta, rozmiary w mm, zapis wektorowy, etykiety bez nakladania, istotnosc.
Uzycie: from figlib import *   (skrypty rycin uruchamiane z katalogu projektu; modul w scripts/)
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from matplotlib import font_manager  # noqa: E402
for _f in sorted(Path("fonts").iterdir()) if Path("fonts").exists() else []:   # np. fonts/ARIAL.TTF skopiowany z Windows
    if _f.suffix.lower() in (".ttf", ".otf"):
        font_manager.fontManager.addfont(str(_f))
MM = 1 / 25.4
W1, W2, W3, WF = 58 * MM, 89 * MM, 120 * MM, 183 * MM            # 1/3, 1/2, 2/3 i cala szerokosc strony
PAL = {"K562": "#0072B2", "HepG2": "#D55E00", "lncRNA": "#009E73", "mRNA": "#999999", "sno": "#E69F00",
       "surv": "#CC79A7", "up": "#D55E00", "down": "#0072B2", "grey": "#BBBBBB", "dark": "#333333"}
OUTROOT = Path("results/figures_pub")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 0.8, "patch.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
})


# Minimalny rozmiar czcionki (pt) dla kazdego napisu na rycinie (runda 2, 19.10): mniejsze wartosci podane w skryptach
# (4.5-5.5 pt) sa podnoszone do MIN_FS. Dotyczy tekstow, adnotacji, legend i opisow pasków kolorow. Ryciny skladane
# bez zmniejszania (wiersz <= 180 mm) zachowuja ten rozmiar w druku.
MIN_FS = 6.0
import matplotlib.text as _mtext  # noqa: E402


def _clamp_fs(t):
    try:
        if t._fontproperties.get_size_in_points() < MIN_FS:
            t._fontproperties.set_size(MIN_FS)
    except Exception:  # noqa: BLE001
        pass


def _with_min_fs(fn):
    def wrapped(self, *a, **k):
        out = fn(self, *a, **k)
        _clamp_fs(self)
        return out
    wrapped.__name__, wrapped.__doc__ = fn.__name__, fn.__doc__
    return wrapped


if not getattr(_mtext.Text, "_min_fs_patched", False):
    _mtext.Text.set_fontsize = _with_min_fs(_mtext.Text.set_fontsize)
    _mtext.Text.set_fontproperties = _with_min_fs(_mtext.Text.set_fontproperties)
    _mtext.Text._min_fs_patched = True


def panel(w, h):
    """Nowy panel o szerokosci w i wysokosci h (w calach; uzyj W1/W2/W3/WF albo x*MM)."""
    return plt.subplots(figsize=(w, h))


def panel_fixed(ax_w_mm, ax_h_mm, left=14, bottom=10, right=4, top=4):
    """Panel o DOKLADNYM rozmiarze obszaru wykresu (w mm); marginesy na opisy osi. Po zapisie z bbox tight
    obszar wykresu zachowuje rozmiar, wiec panele w jednym rzedzie maja rowna wysokosc osi."""
    W, H = (left + ax_w_mm + right) * MM, (bottom + ax_h_mm + top) * MM
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left * MM / W, bottom * MM / H, ax_w_mm * MM / W, ax_h_mm * MM / H])
    return fig, ax


def save(fig, figure, name):
    out = OUTROOT / figure
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"   {figure}/{name}", flush=True)


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n.a."
    if p == 0 or p < 1e-300:
        return "< 1e-300"
    if p < 1e-3:
        m, e = f"{p:.0e}".split("e")
        return f"{m}×10$^{{{int(e)}}}$"
    return f"{p:.2g}"


def stars(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "n.s."


def bracket(ax, x1, x2, y, text, h=0.025, fs=6):
    lo, hi = ax.get_ylim()
    d = h * (hi - lo)
    ax.plot([x1, x1, x2, x2], [y, y + d, y + d, y], color="k", lw=0.5)
    ax.text((x1 + x2) / 2, y + 1.2 * d, text, ha="center", va="bottom", fontsize=fs)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def rho_ci(a, b, n_boot=1000, seed=1):
    """Spearman rho, p i 95% przedzial ufnosci (bootstrap po jednostkach, np. RBP)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    r, p = spearmanr(a, b)
    rng = np.random.default_rng(seed)
    bs = [spearmanr(a[i], b[i])[0] for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    return r, p, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def place_labels(ax, xs, ys, names, fontsize=5.5, min_dy=0.05, min_dx=0.18, dx=0.015, dy=0.0, color="k"):
    """Etykiety punktow bez nakladania (rozsuwanie w pionie w ukladzie osi), cienka linia do punktu."""
    pts = ax.transAxes.inverted().transform(ax.transData.transform(np.c_[xs, ys]))
    lab = pts + np.array([dx, dy])
    for _ in range(300):
        moved = False
        for i in range(len(lab)):
            for j in range(i + 1, len(lab)):
                if abs(lab[i, 0] - lab[j, 0]) < min_dx and abs(lab[i, 1] - lab[j, 1]) < min_dy:
                    d = (min_dy - abs(lab[i, 1] - lab[j, 1])) / 2 + 1e-3
                    sgn = 1 if (lab[i, 1], i) >= (lab[j, 1], j) else -1
                    lab[i, 1] += sgn * d
                    lab[j, 1] -= sgn * d
                    moved = True
        if not moved:
            break
    for (x, y), (lx, ly), n in zip(pts, lab, names):
        far = np.hypot(lx - x, ly - y) > 0.025
        ax.annotate(n, xy=(x, y), xycoords="axes fraction", xytext=(lx, ly), textcoords="axes fraction",
                    fontsize=fontsize, va="center", color=color,
                    arrowprops=dict(arrowstyle="-", lw=0.3, color="0.5", shrinkA=0, shrinkB=1) if far else None)
