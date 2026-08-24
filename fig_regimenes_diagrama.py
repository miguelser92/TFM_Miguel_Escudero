# -*- coding: utf-8 -*-
"""
Diagrama de los tres regímenes de fallo sobre la retícula real de 61 SiPM.

No es una figura de resultados: ilustra CÓMO se generan las máscaras, que en
el texto cuesta tres párrafos y aquí se ve de un vistazo.

Usa las funciones reales de dataset.py (_grow_cluster, _near, _scatter), no una
reimplementación, para que lo dibujado sea exactamente lo que se entrena.

Uso:  python fig_regimenes_diagrama.py [--k 3] [--seed 7] [--tag ...]
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import SiPMImputationDataset, load_positions               # noqa: E402
from hex_geometry import get_neighbor_matrix                            # noqa: E402

PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'
SALIDA = os.path.join('TFM FINAL', 'imagenes')

# Paleta sobria: el objetivo es que se lea en blanco y negro tambien.
C_SANO   = '#e8eaed'   # sensor operativo
C_BORDE  = '#9aa0a6'
C_MUERTO = '#c0392b'   # sensor apagado
C_SEMILLA = '#7b241c'  # el que siembra el fallo


def dibuja(ax, x, y, muertos, semilla, titulo, subtitulo):
    """Un panel: la retícula completa, con los sensores apagados marcados."""
    r = 3.75 / np.sqrt(3)          # circunradio del hexagono a partir del pitch
    for i, (xi, yi) in enumerate(zip(x, y)):
        if i in muertos:
            cara, borde, gr = C_MUERTO, C_SEMILLA, 1.6
        else:
            cara, borde, gr = C_SANO, C_BORDE, 0.8
        # orientation=pi/6: con vertice arriba los hexagonos dejan huecos
        # triangulares; girados 30 grados teselan la reticula sin espacios.
        ax.add_patch(RegularPolygon((xi, yi), numVertices=6, radius=r,
                                    orientation=np.pi / 6, facecolor=cara,
                                    edgecolor=borde, linewidth=gr, zorder=2))
    # el sensor semilla lleva un punto: es desde donde crece el patron
    ax.plot(x[semilla], y[semilla], 'o', ms=3.5, color='white', zorder=3)

    ax.set_aspect('equal')
    ax.set_xlim(x.min() - 3, x.max() + 3)
    ax.set_ylim(y.min() - 3, y.max() + 3)
    ax.axis('off')
    ax.set_title(titulo, fontsize=11, fontweight='bold', pad=2)
    ax.text(0.5, -0.04, subtitulo, transform=ax.transAxes, ha='center',
            va='top', fontsize=8.5, color='#3c4043')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=3, help='canales apagados por panel')
    p.add_argument('--seed', type=int, default=23)  # con 23 los tres patrones se distinguen bien
    p.add_argument('--ich', type=int, default=None, help='indice del sensor semilla')
    p.add_argument('--tag', default='')
    a = p.parse_args()

    x, y = load_positions(PSIPM_PATH)
    nbr = get_neighbor_matrix(PSIPM_PATH)
    n = len(x)

    # Un sensor semilla interior, para que los tres patrones quepan sin recortarse
    if a.ich is None:
        d = np.hypot(x - x.mean(), y - y.mean())
        semilla = int(np.argsort(d)[6])
    else:
        semilla = a.ich

    X = np.ones((4, n), dtype=np.float32)    # dataset minimo: solo queremos las mascaras
    modos = [
        ('cluster', 'Cluster',
         f'{a.k} contiguous sensors'),
        ('near',    'Near',
         f'{a.k} sensors within two hops'),
        ('scatter', 'Scatter',
         f'{a.k} sensors anywhere in the array'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    for ax, (modo, titulo, sub) in zip(axes, modos):
        ds = SiPMImputationDataset(X, seed=a.seed, n_dead=a.k, dead_mode=modo, nbr=nbr)
        if modo == 'cluster':
            muertos = ds._grow_cluster(semilla, a.k)
        elif modo == 'near':
            muertos = ds._near(semilla, a.k)
        else:
            muertos = ds._scatter(semilla, a.k)
        dibuja(ax, x, y, set(int(m) for m in muertos), semilla, titulo, sub)
        print(f'  {modo:8}: sensores apagados = {sorted(int(m) for m in muertos)}')

    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.10, wspace=0.02)
    os.makedirs(SALIDA, exist_ok=True)
    base = os.path.join(SALIDA, f'regimenes_diagrama{a.tag}')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base}.{ext}', dpi=300, bbox_inches='tight')
        print(f'guardado: {base}.{ext}')


if __name__ == '__main__':
    main()
