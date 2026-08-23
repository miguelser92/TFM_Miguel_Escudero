# -*- coding: utf-8 -*-
"""
Figura de la INTRODUCCIÓN: el efecto de un canal muerto sobre el flood map.

Dos paneles con el MISMO módulo y los MISMOS eventos; lo único que cambia entre
ellos es que en el derecho se ha puesto a cero un canal. Ese contraste aísla el
efecto: cualquier diferencia visible es atribuible al canal que falta, y no a la
variabilidad entre detectores, que es grande (sd de 7.9 puntos de recuperación
entre módulos, medida el 09/08).

Cada panel lleva un recuadro ampliado sobre la vecindad del canal, porque a
tamaño de figura el efecto es local y en la vista completa apenas se aprecia.

Dos formatos:
  --layout wide    ancho de página (figure*), hexágono completo + recuadro ampliado
  --layout narrow  ancho de UNA columna (figure). Sin ejes y con el campo de visión
                   recortado a --vista mm: a 4 cm por panel el hexágono entero se
                   convierte en textura y las estructuras dejan de resolverse, así
                   que se muestra una ventana centrada en el canal afectado.

Uso:
    python fig_floodmap_intro.py --layout narrow
    python fig_floodmap_intro.py --layout narrow --vista 16 --tag v2
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from dataset import load_dat_to_dense, load_positions, IDX_TO_ICH

PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'
GOOD_DIR   = r'E:\Datos TFM\Good\Good'
OUT_DIR    = os.path.join('TFM FINAL', 'imagenes')

# Ich 13 está exactamente en (0,0): el zoom cae entero dentro del cristal y no
# recoge borde, que distraería. Un canal de borde tiene además solo 3 vecinos y
# el efecto se reparte peor.
ICH_DEFAULT   = 13
MODULO        = 'datas057.dat'      # módulo de test
N_EVENTOS     = 600_000
BINS_FULL     = 340
BINS_ZOOM     = 150
SEMI_ZOOM_MM  = 6.0                 # semilado del recuadro ampliado


def centro_de_gravedad(X, x_sipm, y_sipm):
    """Posición del evento por centro de gravedad con pesos Rch² (lógica de Anger)."""
    w = X.astype(np.float64) ** 2
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (w * x_sipm).sum(axis=1) / s[:, 0], (w * y_sipm).sum(axis=1) / s[:, 0]


def dibuja_panel(ax, px, py, rng, bins, vmax=None, cmap='inferno'):
    h, _, _ = np.histogram2d(px, py, bins=bins, range=rng)
    if vmax is None:
        vmax = np.percentile(h[h > 0], 99.3)
    ax.imshow(h.T, origin='lower', cmap=cmap, vmax=vmax, aspect='equal',
              extent=[rng[0][0], rng[0][1], rng[1][0], rng[1][1]],
              interpolation='nearest')
    return vmax


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ich', type=int, default=ICH_DEFAULT)
    p.add_argument('--modulo', default=MODULO)
    p.add_argument('--events', type=int, default=N_EVENTOS)
    p.add_argument('--layout', choices=('wide', 'narrow'), default='narrow')
    p.add_argument('--vista', type=float, default=0.0,
                   help='lado en mm de la ventana en layout narrow; 0 = hexágono entero')
    p.add_argument('--tag', default='')
    args = p.parse_args()

    x_sipm, y_sipm = load_positions(PSIPM_PATH)
    ich2idx = {ich: i for i, ich in IDX_TO_ICH.items()}
    if args.ich not in ich2idx:
        sys.exit(f'Ich {args.ich} no es un canal activo. Activos: {sorted(ich2idx)}')
    k = ich2idx[args.ich]
    cx, cy = float(x_sipm[k]), float(y_sipm[k])

    ruta = os.path.join(GOOD_DIR, args.modulo)
    X = load_dat_to_dense(ruta, max_events=args.events)
    print(f'{args.modulo}: {len(X):,} eventos  |  canal Ich {args.ich} en ({cx:.2f}, {cy:.2f}) mm')

    X_dead = X.copy()
    X_dead[:, k] = 0.0

    px_o, py_o = centro_de_gravedad(X, x_sipm, y_sipm)
    px_d, py_d = centro_de_gravedad(X_dead, x_sipm, y_sipm)

    # Cuánto se mueve el centroide: es la cifra que respalda el pie de figura
    d = np.hypot(px_o - px_d, py_o - py_d)
    cerca = np.hypot(px_o - cx, py_o - cy) < 4.0
    med, p90 = np.median(d[cerca]), np.percentile(d[cerca], 90)
    print(f'desplazamiento del centroide en eventos a <4 mm del canal: '
          f'mediana {med:.2f} mm, p90 {p90:.2f} mm  (n={cerca.sum():,})')
    print(f'  fuera de esa vecindad: mediana {np.median(d[~cerca]):.3f} mm  -> el efecto es LOCAL')

    RNG = [[-14.0, 14.0], [-16.0, 16.0]]
    RZ = [[cx - SEMI_ZOOM_MM, cx + SEMI_ZOOM_MM], [cy - SEMI_ZOOM_MM, cy + SEMI_ZOOM_MM]]
    datos = [(px_o, py_o), (px_d, py_d)]

    if args.layout == 'wide':
        fig, axs = plt.subplots(1, 2, figsize=(7.16, 4.05))
        # El indice fisico del canal (Ich) es nomenclatura interna: en el pie de
        # figura basta con decir que es un canal central.
        titulos = ['(a)  all channels operational',
                   '(b)  one central channel disabled']
        vmax = None
        for ax, (px, py), tit in zip(axs, datos, titulos):
            # misma escala de color en los dos paneles: si no, la comparación engaña
            vmax = dibuja_panel(ax, px, py, RNG, BINS_FULL, vmax=vmax)
            ax.set_title(tit, fontsize=9, pad=5)
            ax.set_xlabel('x [mm]', fontsize=8, labelpad=1)
            ax.tick_params(labelsize=7, length=2, pad=1)
            ax.set_xticks([-10, 0, 10])
            ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])

            axz = inset_axes(ax, width='47%', height='47%', loc='upper right',
                             borderpad=0.35)
            dibuja_panel(axz, px, py, RZ, BINS_ZOOM, vmax=vmax * 0.62)
            axz.set_xticks([]); axz.set_yticks([])
            for s in axz.spines.values():
                s.set_edgecolor('white'); s.set_linewidth(0.9)
            axz.add_patch(Circle((cx, cy), 1.5, fill=False, ec='#39ff14', lw=1.2))

            # recuadro en la vista completa que señala qué zona se amplía
            ax.add_patch(Rectangle((RZ[0][0], RZ[1][0]), 2 * SEMI_ZOOM_MM,
                                   2 * SEMI_ZOOM_MM, fill=False, ec='white',
                                   lw=0.7, ls=(0, (3, 2))))
            ax.add_patch(Circle((cx, cy), 1.5, fill=False, ec='#39ff14', lw=1.1))

        axs[0].set_ylabel('y [mm]', fontsize=8, labelpad=1)
        axs[1].set_yticklabels([])
        plt.tight_layout(pad=0.4, w_pad=0.8)

    else:
        # Una columna IEEE = 8.8 cm = 3.46 in. Sin ejes ni títulos exteriores:
        # a este tamaño cada milímetro de papel cuenta, y los ejes en mm no
        # aportan nada que el pie de figura no pueda decir.
        #
        # El hexágono completo SÍ cabe: mide ~28 mm y cada panel dispone de
        # ~4.3 cm, o sea va AMPLIADO 1.5x. Las estructuras quedan separadas
        # ~2 mm en papel, que se resuelven sin problema.
        completo = args.vista <= 0
        if completo:
            R, bins, alto = RNG, BINS_FULL, 2.10
        else:
            v = args.vista / 2.0
            R = [[cx - v, cx + v], [cy - v, cy + v]]
            bins, alto = BINS_ZOOM, 1.80

        fig, axs = plt.subplots(1, 2, figsize=(3.46, alto))
        vmax = None
        for ax, (px, py), etq in zip(axs, datos, ('(a)', '(b)')):
            vmax = dibuja_panel(ax, px, py, R, bins, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.add_patch(Circle((cx, cy), 1.7, fill=False, ec='#39ff14',
                                lw=0.9 if completo else 1.0))
            ax.text(0.02, 0.985, etq, transform=ax.transAxes, color='white',
                    fontsize=7.5, va='top', ha='left')

        # barra de escala: sustituye a los ejes y ocupa mucho menos
        L = 5.0
        if completo:
            # en la esquina inferior izquierda, sobre el fondo negro de fuera
            # del cristal, donde no tapa datos
            x0, y0 = R[0][0] + 0.6, R[1][0] + 1.4
        else:
            x0, y0 = R[0][0] + 0.9, R[1][0] + 1.0
        axs[0].plot([x0, x0 + L], [y0, y0], color='white', lw=1.3,
                    solid_capstyle='butt')
        axs[0].text(x0 + L / 2, y0 + 0.5, '5 mm', color='white', fontsize=6,
                    ha='center', va='bottom')
        plt.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004,
                            wspace=0.03)

    os.makedirs(OUT_DIR, exist_ok=True)
    suf = '' if args.layout == 'narrow' else '_wide'
    base = f'floodmap_intro{suf}{("_" + args.tag) if args.tag else ""}'
    for ext in ('png', 'pdf'):
        f = os.path.join(OUT_DIR, f'{base}.{ext}')
        plt.savefig(f, dpi=600, facecolor='white',
                    bbox_inches=None if args.layout == 'narrow' else 'tight')
        print(f'guardado: {f}')


if __name__ == '__main__':
    main()
