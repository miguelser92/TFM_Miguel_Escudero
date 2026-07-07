"""
generador_esquemas_arquitecturas.py
===================================
Esquemas de las tres arquitecturas de imputación (DeepMLP, ResidualMLP, HexCNN)
en estilo **3D isométrico** con colores, y con los **parámetros reales** leídos
del propio modelo. La HexCNN se dibuja sobre las **teselas hexagonales reales**
del detector (posiciones de psipm.tsv).

Salida: figs_arquitecturas/  (PNG + PDF por arquitectura). Etiquetas en inglés.

Uso:
    conda activate tfm
    python generador_esquemas_arquitecturas.py

Autor: Miguel Escudero (TFM)
"""

import sys
from pathlib import Path
import numpy as np
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon, FancyArrowPatch, RegularPolygon, Circle
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from dataset import load_positions, N_ACTIVE
from hex_geometry import get_neighbor_matrix, DEFAULT_PSIPM
from model import get_model, count_parameters

OUT = Path(__file__).parent / 'figs_arquitecturas'
OUT.mkdir(exist_ok=True)

# Paleta tipo "diagrama CNN"
BLUE, TEAL, GREEN, ORANGE, PURPLE, GREY = (
    '#5b9bd5', '#4bacc6', '#70ad47', '#ed7d31', '#8064a2', '#a6a6a6')

DVX, DVY = 0.42, 0.62   # vector de profundidad isométrica (por unidad de depth)


def cp(*mods):
    return sum(p.numel() for m in mods for p in m.parameters() if p.requires_grad)


def shades(c):
    r, g, b = mcolors.to_rgb(c)
    f = lambda k: tuple(min(1, v * k) for v in (r, g, b))
    return f(1.0), f(1.22), f(0.68)          # front, top (claro), side (oscuro)


def iso_block(ax, x, y, w, h, color, d=1.0, z=2, lw=0.9):
    """Prisma isométrico: cara frontal (x,y,w,h) + tapa + lado."""
    front, top, side = shades(color)
    dx, dy = DVX * d, DVY * d
    ax.add_patch(Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
                 closed=True, fc=front, ec='#2b2b2b', lw=lw, zorder=z))
    ax.add_patch(Polygon([(x, y+h), (x+w, y+h), (x+w+dx, y+h+dy), (x+dx, y+h+dy)],
                 closed=True, fc=top, ec='#2b2b2b', lw=lw, zorder=z))
    ax.add_patch(Polygon([(x+w, y), (x+w, y+h), (x+w+dx, y+h+dy), (x+w+dx, y+dy)],
                 closed=True, fc=side, ec='#2b2b2b', lw=lw, zorder=z))


def sheet_stack(ax, x, y, h, color, n=5, d=1.0, gap=0.16, sheet_w=0.10, z=2):
    """Pila de láminas finas (evoca un bloque repetido ×n)."""
    for i in range(n):
        iso_block(ax, x + i * gap, y, sheet_w, h, color, d=d, z=z + i)
    return x + (n - 1) * gap + sheet_w      # x del borde frontal derecho


def harrow(ax, x0, x1, y, color='#5a5a5a'):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>',
                 mutation_scale=15, color=color, lw=2.0, zorder=40))


def label(ax, x, y, top='', bot='', params=''):
    # y ≈ h + 0.55 (alto de la cara frontal). El título va POR ENCIMA de la tapa isométrica;
    # la dimensión y los params van POR DEBAJO de la base, para no solapar el prisma 3D.
    if top:
        ax.text(x, y + 0.30, top, ha='center', va='bottom', fontsize=10.5, fontweight='bold')
    yb = -0.30
    if bot:
        ax.text(x, yb, bot, ha='center', va='top', fontsize=8.2, color='#555'); yb -= 0.52
    if params:
        ax.text(x, yb, params, ha='center', va='top', fontsize=8.2, color=ORANGE)


def hh(dim, dmax=512, lo=0.9, hi=3.4):
    """Altura del bloque proporcional a la dimensión (raíz para comprimir rango)."""
    return lo + (hi - lo) * (dim / dmax) ** 0.5


# ═════════════════════════════════════════════════════════════
#  1) DeepMLP
# ═════════════════════════════════════════════════════════════
def fig_deepmlp():
    m = get_model('deepmlp')
    lins = [l for l in m.net if isinstance(l, nn.Linear)]
    total = count_parameters(m)

    fig, ax = plt.subplots(figsize=(13, 5.2))
    x = 0.0
    W, D, GAP = 0.55, 1.15, 1.7

    # Input
    h = hh(122); iso_block(ax, x, 0, W, h, GREY, d=D)
    label(ax, x + W/2, h + 0.55, 'Input', 'charges+mask\n2×61 → 122')
    prev_x = x + W
    x += GAP

    stagecols = [BLUE, BLUE, TEAL, TEAL]
    for i, l in enumerate(lins[:-1]):
        h = hh(l.out_features)
        harrow(ax, prev_x + 0.12, x - 0.12, 0.9)
        iso_block(ax, x, 0, W, h, stagecols[i], d=D)
        p = l.in_features*l.out_features + l.out_features + 2*l.out_features
        label(ax, x + W/2, h + 0.55, f'Dense', f'{l.in_features}→{l.out_features}\nBN·GELU·Drop',
              f'≈{p:,}')
        prev_x = x + W; x += GAP

    # Head
    last = lins[-1]; h = hh(last.out_features)
    harrow(ax, prev_x + 0.12, x - 0.12, 0.9)
    iso_block(ax, x, 0, W, h, PURPLE, d=D)
    label(ax, x + W/2, h + 0.55, 'Head', f'{last.in_features}→61\nlinear',
          f'{last.in_features*last.out_features+last.out_features:,}')
    prev_x = x + W; x += GAP - 0.35

    # Output
    h = hh(61)
    harrow(ax, prev_x + 0.12, x - 0.12, 0.9)
    iso_block(ax, x, 0, W, h, GREEN, d=D)
    label(ax, x + W/2, h + 0.55, 'Output', '61 charges')

    ax.set_title(f'DeepMLP — plain baseline    ·    {total:,} trainable parameters',
                 fontsize=14, fontweight='bold', pad=16)
    ax.set_xlim(-0.6, x + W + 1.4); ax.set_ylim(-1.2, 5.2)
    ax.axis('off'); ax.set_aspect('equal')
    _save(fig, 'deepmlp')


# ═════════════════════════════════════════════════════════════
#  2) ResidualMLP
# ═════════════════════════════════════════════════════════════
def fig_resmlp():
    m = get_model('resmlp')
    hidden = m.stem[0].out_features
    nblk = len(m.blocks)
    total = count_parameters(m)

    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    W, D = 0.55, 1.15

    # Input
    x = 0.0; h = hh(122); iso_block(ax, x, 0, W, h, GREY, d=D)
    label(ax, x+W/2, h+0.55, 'Input', '2×61 → 122'); prev = x+W

    # Stem
    x = 1.7; h = hh(hidden); harrow(ax, prev+0.12, x-0.12, 0.9)
    iso_block(ax, x, 0, W, h, BLUE, d=D)
    label(ax, x+W/2, h+0.55, 'Stem', f'122→{hidden}', f'{cp(m.stem):,}'); prev = x+W

    # Residual blocks x4 (pila de láminas verdes) con arco de skip
    x = 3.5; h = hh(hidden)
    harrow(ax, prev+0.12, x-0.12, 0.9)
    right = sheet_stack(ax, x, 0, h, GREEN, n=nblk, d=D, gap=0.34, sheet_w=0.12)
    label(ax, x+0.55, h+0.85, f'Residual block × {nblk}', 'Linear·BN·GELU·Drop', f'{cp(m.blocks):,}')
    # arco de skip por encima
    ax.add_patch(FancyArrowPatch((x-0.05, h+0.15), (right+0.55, h+0.15),
                 connectionstyle='arc3,rad=-0.45', arrowstyle='-|>', mutation_scale=13,
                 color=ORANGE, lw=2.0, zorder=50))
    ax.text((x+right)/2+0.2, h+0.98, 'skip  x + f(x)', ha='center', fontsize=8.6,
            color=ORANGE, style='italic')
    prev = right + DVX*D

    # Attention gate
    x = 6.6; h = hh(hidden)*0.9
    harrow(ax, prev+0.12, x-0.12, 0.9)
    iso_block(ax, x, 0, W, h, ORANGE, d=D)
    label(ax, x+W/2, h+0.55, 'Attention (SE)', f'{hidden}→{hidden//4}→{hidden}\nsigmoid ⊙', f'{cp(m.attn):,}')
    prev = x+W

    # Head
    x = 8.3; h = hh(61); harrow(ax, prev+0.12, x-0.12, 0.9)
    iso_block(ax, x, 0, W, h, PURPLE, d=D)
    label(ax, x+W/2, h+0.55, 'Head', f'{hidden}→61', f'{cp(m.head):,}'); prev = x+W

    # Output
    x = 9.7; h = hh(61); harrow(ax, prev+0.12, x-0.12, 0.9)
    iso_block(ax, x, 0, W, h, GREEN, d=D)
    label(ax, x+W/2, h+0.55, 'Output', '61 charges')

    ax.set_title(f'ResidualMLP — residual blocks + attention    ·    {total:,} trainable parameters',
                 fontsize=14, fontweight='bold', pad=16)
    ax.set_xlim(-0.6, x+W+1.4); ax.set_ylim(-1.2, 5.6)
    ax.axis('off'); ax.set_aspect('equal')
    _save(fig, 'resmlp')


# ═════════════════════════════════════════════════════════════
#  3) HexCNN  (flujo 3D arriba + detector de teselas hexagonales abajo)
# ═════════════════════════════════════════════════════════════
def fig_hexcnn():
    m = get_model('hexcnn')
    hidden = m.stem.out_features
    nblk = len(m.blocks)
    total = count_parameters(m)

    nbr = get_neighbor_matrix()
    xs, ys = load_positions(DEFAULT_PSIPM)
    FOCUS = 30
    spacing = np.median(cKDTree(np.c_[xs, ys]).query(np.c_[xs, ys], k=2)[0][:, 1])
    hex_r = spacing / np.sqrt(3) * 0.97      # radio centro-vértice para teselar

    fig = plt.figure(figsize=(13.5, 9.0))
    axF = fig.add_axes([0.04, 0.62, 0.92, 0.30])   # flujo (arriba)
    axH = fig.add_axes([0.06, 0.03, 0.62, 0.55])   # detector hexagonal
    axZ = fig.add_axes([0.70, 0.06, 0.28, 0.48])   # zoom HexConv

    # ── flujo 3D ──
    W, D = 0.5, 1.0
    def blk(x, dim, color, top, sub, params=''):
        h = hh(dim, dmax=64, lo=0.8, hi=2.6)
        iso_block(axF, x, 0, W, h, color, d=D)
        axF.text(x+W/2, h+DVY*D+0.16, top, ha='center', va='bottom', fontsize=10, fontweight='bold')
        axF.text(x+W/2, -0.22, sub, ha='center', va='top', fontsize=7.8, color='#555')
        if params:
            axF.text(x+W/2, -0.60, params, ha='center', va='top', fontsize=8, color=ORANGE)
        return x+W
    x = 0.0
    p = blk(x, 2, GREY, 'Input', '(2, 61)'); x = 1.7
    axF.add_patch(FancyArrowPatch((p+0.1, 0.8), (x-0.1, 0.8), arrowstyle='-|>', mutation_scale=13, color='#5a5a5a', lw=1.8))
    p = blk(x, hidden, BLUE, 'Stem', 'Linear 2→%d' % hidden, f'{cp(m.stem, m.stem_bn):,}'); x = 3.4
    axF.add_patch(FancyArrowPatch((p+0.1, 0.8), (x-0.1, 0.8), arrowstyle='-|>', mutation_scale=13, color='#5a5a5a', lw=1.8))
    hb = hh(hidden, 64, 0.8, 2.6)
    right = sheet_stack(axF, x, 0, hb, GREEN, n=nblk, d=D, gap=0.30, sheet_w=0.11)
    axF.text(x+0.5, hb+DVY*D+0.16, f'HexResBlock × {nblk}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axF.text(x+0.5, -0.22, '2× HexConv + skip', ha='center', va='top', fontsize=7.8, color='#555')
    axF.text(x+0.5, -0.60, f'{cp(m.blocks):,}', ha='center', va='top', fontsize=8, color=ORANGE)
    x = 5.4; p = right + DVX*D
    axF.add_patch(FancyArrowPatch((p+0.1, 0.8), (x-0.1, 0.8), arrowstyle='-|>', mutation_scale=13, color='#5a5a5a', lw=1.8))
    p = blk(x, 1, PURPLE, 'Head', 'Linear %d→1' % hidden, f'{cp(m.head):,}'); x = 7.0
    axF.add_patch(FancyArrowPatch((p+0.1, 0.8), (x-0.1, 0.8), arrowstyle='-|>', mutation_scale=13, color='#5a5a5a', lw=1.8))
    blk(x, 61, GREEN, 'Output', '(61)')
    axF.set_title('Data flow', fontsize=12, fontweight='bold', loc='left')
    axF.set_xlim(-0.5, 8.4); axF.set_ylim(-1.1, 4.1); axF.axis('off'); axF.set_aspect('equal')

    # ── detector como teselas hexagonales reales ──
    nb = set(int(j) for j in nbr[FOCUS] if j >= 0)
    for i in range(N_ACTIVE):
        if i == FOCUS:
            fc, ec, lw = '#f5c542', '#b5482b', 2.2
        elif i in nb:
            fc, ec, lw = '#f3c6b3', '#c0532f', 1.6
        else:
            fc, ec, lw = '#e9eef5', '#9fb2c8', 1.0
        axH.add_patch(RegularPolygon((xs[i], ys[i]), 6, radius=hex_r,
                      orientation=np.pi/6, fc=fc, ec=ec, lw=lw, zorder=2))
    for j in nb:      # aristas foco→vecinos
        axH.plot([xs[FOCUS], xs[j]], [ys[FOCUS], ys[j]], color='#b5482b', lw=2.0, zorder=3)
    axH.set_aspect('equal'); axH.axis('off')
    axH.set_title('Neighbourhood graph on the real hexagonal SiPM layout',
                  fontsize=12, fontweight='bold')
    m_ = 1.5
    axH.set_xlim(xs.min()-m_, xs.max()+m_); axH.set_ylim(ys.min()-m_, ys.max()+m_)

    # ── zoom de la HexConv ──
    axZ.axis('off'); axZ.set_xlim(0, 1); axZ.set_ylim(0, 1)
    cx, cy, r = 0.5, 0.62, 0.11
    ang = np.deg2rad([90, 150, 210, 270, 330, 30])
    for a in ang:
        nx, ny = cx + 0.26*np.cos(a), cy + 0.26*np.sin(a)
        axZ.plot([cx, nx], [cy, ny], color='#b5482b', lw=1.6, zorder=1)
        axZ.add_patch(RegularPolygon((nx, ny), 6, radius=r*0.8, orientation=np.pi/6,
                      fc='#f3c6b3', ec='#c0532f', lw=1.3, zorder=2))
    axZ.add_patch(RegularPolygon((cx, cy), 6, radius=r, orientation=np.pi/6,
                  fc='#f5c542', ec='#b5482b', lw=2, zorder=3))
    axZ.text(cx, cy, r'$x_i$', ha='center', va='center', fontsize=11, zorder=4)
    axZ.text(0.5, 0.20,
             r'$\mathrm{out}_i = W_{self}\,x_i + W_{neigh}\,\overline{x}_{\,N(i)}$',
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.45', fc='#fbeee7', ec='#c0532f'))
    axZ.text(0.5, 0.06, 'node + mean of its neighbours\n(shared weights across all 61 nodes)',
             ha='center', va='center', fontsize=8.2, color='#555')
    axZ.set_title('HexConv', fontsize=11, fontweight='bold')

    fig.suptitle(f'HexCNN — graph convolution with a real geometric prior    ·    '
                 f'{total:,} trainable parameters', fontsize=14, fontweight='bold')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'arch_hexcnn.{ext}', dpi=200, bbox_inches='tight')
        print(f'  guardado: {OUT / f"arch_hexcnn.{ext}"}')
    plt.close(fig)


def _save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'arch_{name}.{ext}', dpi=200, bbox_inches='tight')
        print(f'  guardado: {OUT / f"arch_{name}.{ext}"}')
    plt.close(fig)


if __name__ == '__main__':
    print('Generando esquemas de arquitectura (3D)...')
    fig_deepmlp()
    fig_resmlp()
    fig_hexcnn()
    print(f'\nListo. Figuras en: {OUT}')
