# -*- coding: utf-8 -*-
"""
Figura CUALITATIVA sobre un modulo realmente defectuoso.

Todo lo demas del trabajo se mide sobre fallos que inyectamos nosotros, porque un
fallo real no deja referencia contra la que medir el error. Esta figura no mide:
ensena. Coge uno de los modulos que llegaron averiados, sustituye el canal
anotado por la prediccion de la red y compara los dos flood maps.

Que se puede afirmar y que no:
  - SI: el patron de estructuras grabadas alrededor del canal se recompone.
  - NO: cuanto error se ha corregido. No hay ground truth y no lo habra: si se
    pudiera medir el valor verdadero del canal, el canal no estaria averiado.

Un detalle que conviene saber al leerla: los canales anotados de este lote NO
estan a cero, recogen entre el 35 y el 45 % de lo normal. Son degradaciones
parciales. Lo que hace la figura es el caso de uso real: dar el canal por perdido
y reemplazarlo entero por la prediccion.

Uso:
    python fig_bad_real.py
    python fig_bad_real.py --modulo datas034 --ich 4
    python fig_bad_real.py --modulo datas042 --ich 0,25 --tag near
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import torch

from dataset import load_dat_to_dense, load_positions, ICH_TO_IDX
from imputation_eval import load_model, impute_channel

PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'
BAD_DIR = r'E:\Datos TFM\Bad\Bad'
GOOD_DIR = r'E:\Datos TFM\Good\Good'
OUT_DIR = os.path.join('TFM FINAL', 'imagenes')
LABELS = os.path.join('reports', 'bad_labels_manual.json')

# Modelo del protocolo bueno (149 modulos), no el de 40. Para varios canales
# habria que usar el multi-dead entrenado en 'near'.
CKPT_1 = os.path.join('runs', 'imputer_hexcnn_s_mse_banda11', 'best_model.pth')
CKPT_N = os.path.join('runs', 'imputer_hexcnn_s_mse_dead1-4_near_banda11',
                      'best_model.pth')

# datas024: el canal anotado (Ich 27) es el segundo mas apagado de los 61 frente
# a un modulo sano, tiene los seis vecinos y cae en el interior del cristal.
MODULO = 'datas024'
N_EVENTOS = 400_000
BINS = 320
BINS_ZOOM = 150
SEMI_ZOOM = 4.5           # semilado del recuadro ampliado, en mm
RNG = [[-14.0, 14.0], [-16.0, 16.0]]


def centro_de_gravedad(X, x_sipm, y_sipm):
    """Posicion del evento por centro de gravedad con pesos Rch^2 (Anger)."""
    w = X.astype(np.float64) ** 2
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (w * x_sipm).sum(axis=1) / s[:, 0], (w * y_sipm).sum(axis=1) / s[:, 0]


def dibuja(ax, px, py, rng, bins, vmax=None):
    h, _, _ = np.histogram2d(px, py, bins=bins, range=rng)
    if vmax is None:
        vmax = np.percentile(h[h > 0], 99.3)
    ax.imshow(h.T, origin='lower', cmap='inferno', vmax=vmax, aspect='equal',
              extent=[rng[0][0], rng[0][1], rng[1][0], rng[1][1]],
              interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    return vmax


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modulo', default=MODULO)
    p.add_argument('--ich', default=None,
                   help='canales Ich separados por coma; por defecto los anotados')
    p.add_argument('--events', type=int, default=N_EVENTOS)
    p.add_argument('--layout', choices=('narrow', 'wide'), default='narrow',
                   help='narrow = una columna; wide = doble columna con recuadro ampliado')
    p.add_argument('--vista', type=float, default=0.0,
                   help='solo en narrow: lado en mm de la ventana; 0 = hexagono entero')
    p.add_argument('--zoombrillo', type=float, default=0.50,
                   help='solo en wide: fraccion del vmax que usa el recuadro; menor = mas brillante')
    p.add_argument('--tag', default='')
    args = p.parse_args()

    # ── canales: la anotacion manual manda, salvo que se pasen a mano ──
    if args.ich:
        ich = [int(v) for v in args.ich.split(',')]
    else:
        lab = json.load(open(LABELS, encoding='utf-8'))['labels']
        clave = f'{args.modulo}.dat'
        if clave not in lab:
            sys.exit(f'{clave} no esta anotado. Anotados: '
                     f'{sorted(k[:-4] for k in lab)}')
        ich = lab[clave]['dead_ich']
    idx = [ICH_TO_IDX[i] for i in ich if i in ICH_TO_IDX]
    if not idx:
        sys.exit(f'Ninguno de los Ich {ich} es un canal activo.')

    x_sipm, y_sipm = load_positions(PSIPM_PATH)
    ruta = os.path.join(BAD_DIR, f'{args.modulo}.dat')
    X = load_dat_to_dense(ruta, max_events=args.events)
    print(f'{args.modulo}: {len(X):,} eventos  |  Ich {ich} -> idx {idx}')

    # Cuanto recoge el canal frente a un modulo sano: es lo que justifica que el
    # canal este senalado, y va en el pie de figura.
    ref = load_dat_to_dense(os.path.join(GOOD_DIR, 'datas057.dat'),
                            max_events=60_000).mean(axis=0)
    for i, c in zip(idx, ich):
        r = X[:, i].mean() / ref[i] if ref[i] > 0 else float('nan')
        print(f'   Ich {c}: recoge el {100 * r:.0f} % de la carga de un modulo sano')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if len(idx) == 1:
        model = load_model(CKPT_1, device)
        X_imp, _ = impute_channel(model, X, idx[0], device)
        print(f'   imputado con {CKPT_1}')
    else:
        # Varios canales a la vez: hace falta el modelo multi-dead. impute_set
        # vive en eval_multidead, que es donde se usa.
        from eval_multidead import impute_set
        model = load_model(CKPT_N, device)
        X_imp, _ = impute_set(model, X, np.array(idx), device)
        print(f'   imputado con {CKPT_N}')

    px_o, py_o = centro_de_gravedad(X, x_sipm, y_sipm)
    px_i, py_i = centro_de_gravedad(X_imp, x_sipm, y_sipm)

    # Cuanto se mueve el centroide al imputar. No es un error (no hay verdad),
    # es la magnitud de la correccion aplicada.
    d = np.hypot(px_o - px_i, py_o - py_i)
    cx, cy = float(np.mean([x_sipm[i] for i in idx])), float(np.mean([y_sipm[i] for i in idx]))
    cerca = np.hypot(px_o - cx, py_o - cy) < 4.0
    print(f'   correccion del centroide cerca del canal: mediana {np.median(d[cerca]):.3f} mm, '
          f'p90 {np.percentile(d[cerca], 90):.3f} mm  (n={cerca.sum():,})')
    print(f'   lejos: mediana {np.median(d[~cerca]):.4f} mm')

    datos = [(px_o, py_o), (px_i, py_i)]
    TITULOS = ('(a)  as acquired', '(b)  after imputation')

    if args.layout == 'wide':
        # Vista completa con un recuadro ampliado sobre el canal, como la figura
        # de la introduccion: el hexagono situa y el recuadro deja ver el efecto,
        # que es local y a tamano de pagina no se resolveria.
        RZ = [[cx - SEMI_ZOOM, cx + SEMI_ZOOM], [cy - SEMI_ZOOM, cy + SEMI_ZOOM]]
        # El recuadro se coloca en la esquina OPUESTA al canal, para no taparlo.
        loc = ('lower left' if cx > 0 else 'lower right') if cy > 0 else \
              ('upper left' if cx > 0 else 'upper right')

        fig, axs = plt.subplots(1, 2, figsize=(7.16, 4.05))
        vmax = None
        for ax, (px, py), tit in zip(axs, datos, TITULOS):
            vmax = dibuja(ax, px, py, RNG, BINS, vmax=vmax)   # misma escala
            ax.set_title(tit, fontsize=9, pad=5)
            ax.set_xlabel('x [mm]', fontsize=8, labelpad=1)
            ax.tick_params(labelsize=7, length=2, pad=1)
            ax.set_xticks([-10, 0, 10])
            ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])

            # El recuadro se satura algo mas que la vista completa (mismo factor
            # en los dos paneles, si no la comparacion enganaria): dentro de una
            # ventana de 9 mm el rango de cuentas es mucho mas estrecho.
            axz = inset_axes(ax, width='47%', height='47%', loc=loc, borderpad=0.35)
            dibuja(axz, px, py, RZ, BINS_ZOOM, vmax=vmax * args.zoombrillo)
            for s in axz.spines.values():
                s.set_edgecolor('white')
                s.set_linewidth(0.9)
            for i in idx:
                axz.add_patch(Circle((x_sipm[i], y_sipm[i]), 1.5, fill=False,
                                     ec='#39ff14', lw=1.2))
                ax.add_patch(Circle((x_sipm[i], y_sipm[i]), 1.5, fill=False,
                                    ec='#39ff14', lw=1.1))
            ax.add_patch(Rectangle((RZ[0][0], RZ[1][0]), 2 * SEMI_ZOOM,
                                   2 * SEMI_ZOOM, fill=False, ec='white',
                                   lw=0.7, ls=(0, (3, 2))))

        axs[0].set_ylabel('y [mm]', fontsize=8, labelpad=1)
        axs[1].set_yticklabels([])
        plt.tight_layout(pad=0.4, w_pad=0.8)

    else:
        # A una columna el hexagono entero se convierte en textura y el efecto no
        # se resuelve. Con --vista se recorta una ventana centrada en el canal.
        if args.vista > 0:
            s = args.vista / 2.0
            rng = [[cx - s, cx + s], [cy - s, cy + s]]
            bins = max(60, int(BINS * args.vista / 28.0))
        else:
            rng, bins = RNG, BINS

        fig, axs = plt.subplots(1, 2, figsize=(3.46, 1.95))
        vmax = None
        for ax, (px, py), t in zip(axs, datos, TITULOS):
            vmax = dibuja(ax, px, py, rng, bins, vmax=vmax)
            ax.set_title(t, fontsize=7, pad=2.5)
            for i in idx:
                ax.add_patch(Circle((x_sipm[i], y_sipm[i]),
                                    1.9 if args.vista == 0 else 2.4, fill=False,
                                    ec='#39ff88', lw=0.7, alpha=0.9))
        plt.tight_layout(pad=0.25, w_pad=0.6)

    os.makedirs(OUT_DIR, exist_ok=True)
    base = f'bad_real{("_" + args.tag) if args.tag else ""}'
    for ext in ('png', 'pdf'):
        f = os.path.join(OUT_DIR, f'{base}.{ext}')
        plt.savefig(f, dpi=600, facecolor='white', bbox_inches='tight')
        print(f'   guardado: {f}')


if __name__ == '__main__':
    main()
