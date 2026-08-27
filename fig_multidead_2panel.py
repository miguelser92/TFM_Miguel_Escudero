# -*- coding: utf-8 -*-
"""
Figura de multi-dead para el paper: DOS paneles, no cuatro.

La que genera eval_multidead trae cuatro (recuperación p90, recuperación media,
error absoluto y MAE) y es de UN solo run. A una columna cada panel queda en
2 cm y no se lee. El texto además solo usa dos de ellos: la recuperación P90 y
el desplazamiento residual en milímetros, que son las dos mitades del argumento
de las lecturas opuestas.

Aquí se dibujan solo esos dos, promediando las TRES réplicas y sombreando su
banda, para que la figura diga lo mismo que la tabla del texto (que también es
media de las tres).

Uso:
    python fig_multidead_2panel.py
    python fig_multidead_2panel.py --tag v2 --ancho 7.16
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = 'runs'
OUT_DIR = os.path.join('TFM FINAL', 'imagenes')
MODELOS = [f'imputer_hexcnn_s_mse_dead1-4_near_banda{r}' for r in (11, 13, 17)]
CAMPANA = 'MULTIDEAD_3MODOS'

# Mismos colores y marcadores que usa eval_multidead, para que las dos figuras
# del proyecto se lean igual.
COLOR = {'cluster': '#c0392b', 'near': '#e67e22', 'scatter': '#2471a3'}
MARCA = {'cluster': 'o', 'near': '^', 'scatter': 's'}
ETIQ = {'cluster': 'contiguous', 'near': 'near-scattered', 'scatter': 'scattered'}


def carga():
    """{(modo, k): [valor por replica]} para recuperacion y residuo."""
    rec, res = {}, {}
    for m in MODELOS:
        p = os.path.join(RUNS, m, CAMPANA, 'eval_multidead_metrics.json')
        d = json.load(open(p, encoding='utf-8'))['curve']
        for v in d.values():
            rec.setdefault((v['mode'], v['k']), []).append(v['recov_p90'])
            res.setdefault((v['mode'], v['k']), []).append(v['dR_imp_p90_macro'])
    return rec, res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='')
    p.add_argument('--ancho', type=float, default=3.46,
                   help='pulgadas; 3.46 = una columna IEEE, 7.16 = doble')
    args = p.parse_args()

    rec, res = carga()
    ks = sorted({k for _, k in rec})
    n_rep = len(rec[('near', 1)])
    print(f'  {len(MODELOS)} replicas, k = {ks}, {n_rep} valores por punto')

    alto = args.ancho * 0.42
    fig, axs = plt.subplots(1, 2, figsize=(args.ancho, alto))

    for ax, datos, ylab in ((axs[0], rec, 'Recovery at P90 (%)'),
                            (axs[1], res, r'Residual $\Delta R$ at P90 (mm)')):
        for modo in ('cluster', 'near', 'scatter'):
            m = np.array([np.mean(datos[(modo, k)]) for k in ks])
            s = np.array([np.std(datos[(modo, k)], ddof=1) for k in ks])
            ax.plot(ks, m, marker=MARCA[modo], color=COLOR[modo], lw=1.6, ms=4,
                    label=ETIQ[modo])
            # banda de +/-1 sd entre replicas: sin esto la figura sugeriria una
            # precision que las tres replicas no tienen
            ax.fill_between(ks, m - s, m + s, color=COLOR[modo], alpha=0.15, lw=0)
        ax.set_xlabel('Suppressed channels $k$', fontsize=7)
        ax.set_ylabel(ylab, fontsize=7)
        ax.set_xticks(ks)
        ax.tick_params(labelsize=6.5, length=2, pad=1.5)
        ax.grid(alpha=0.3, lw=0.5)
        for s_ in ax.spines.values():
            s_.set_linewidth(0.6)

    axs[0].legend(fontsize=6, frameon=False, loc='lower right')
    plt.tight_layout(pad=0.3, w_pad=1.0)

    os.makedirs(OUT_DIR, exist_ok=True)
    base = f'multidead_2panel{("_" + args.tag) if args.tag else ""}'
    for ext in ('png', 'pdf'):
        f = os.path.join(OUT_DIR, f'{base}.{ext}')
        plt.savefig(f, dpi=600, facecolor='white', bbox_inches='tight')
        print(f'  guardado: {f}')

    print()
    print('  valores dibujados (media de las tres replicas):')
    for modo in ('cluster', 'near', 'scatter'):
        r = ' '.join(f'{np.mean(rec[(modo, k)]):5.1f}' for k in ks)
        d = ' '.join(f'{np.mean(res[(modo, k)]):5.3f}' for k in ks)
        print(f'    {ETIQ[modo]:15} rec {r}   resid {d}')


if __name__ == '__main__':
    main()
