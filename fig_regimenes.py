"""
fig_regimenes.py — ¿con qué tipo de fallo conviene entrenar?
=============================================================
Compara dos modelos idénticos salvo en el REGIMEN de fallo con el que se
entrenaron, evaluados ambos en los TRES regímenes:

  - contiguo  : los sensores averiados se tocan (sombra, conexión en serie)
  - cercano   : separados por una o dos posiciones  <- lo que se observa en
                los módulos averiados reales al etiquetarlos a mano
  - disperso  : repartidos por todo el detector (control)

El resultado es que entrenar en el régimen INTERMEDIO generaliza hacia los dos
extremos, mientras que entrenar en un extremo no alcanza el centro.

Uso:
    python fig_regimenes.py --tag def

Salida: reports/regimenes_entrenamiento<tag>.png

Autor: Miguel Escudero (TFM)
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

RUNS_BASE = Path(r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\runs')
OUT_DIR   = Path(r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\reports')

# (carpeta, campaña, etiqueta, color)
MODELOS = [
    ('imputer_hexcnn_s_mse_dead1-4',            'MULTIDEAD_3MODOS',
     'trained on CONTIGUOUS failures',  '#c0392b'),
    ('imputer_hexcnn_s_mse_dead1-4_near_near14', 'MULTIDEAD',
     'trained on NEAR failures',        '#2471a3'),
]
# el modelo de un solo canal, como referencia de "sin entrenamiento multi-dead"
BASE = ('imputer_hexcnn_s_mse', 'MULTIDEAD', 'trained on SINGLE failures', '#7f8c8d')

MODOS = [('cluster', 'CONTIGUOUS\n(sensors touching)'),
         ('near',    'NEAR\n(1-2 positions apart, realistic)'),
         ('scatter', 'SCATTERED\n(control)')]


def curva(run, camp):
    p = RUNS_BASE / run / camp / 'eval_multidead_metrics.json'
    return json.load(open(p, encoding='utf-8'))['curve'] if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    tag = f'_{args.tag}' if args.tag else ''

    datos = {}
    for run, camp, lab, col in MODELOS + [BASE]:
        c = curva(run, camp)
        if c is None:
            print(f'  (falta {run}/{camp} -> se omite)')
            continue
        datos[lab] = (c, col)
        print(f'  OK  {lab:34} <- {run}/{camp}')

    ks = [1, 2, 3, 4]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    for ax, (modo, titulo) in zip(axes, MODOS):
        for lab, (c, col) in datos.items():
            ys = [c[f'{modo}_k{k}']['recov_mean'] for k in ks if f'{modo}_k{k}' in c]
            xs = [k for k in ks if f'{modo}_k{k}' in c]
            if not xs:
                continue
            estilo = ':' if 'SINGLE' in lab else '-'
            ax.plot(xs, ys, marker='o', color=col, lw=2.4, ms=7, ls=estilo, label=lab)
        ax.set_title(titulo, fontsize=10.5)
        ax.set_xlabel('Number of simultaneous dead sensors (k)')
        ax.set_xticks(ks)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Position recovery (mean, %)\nhigher is better')
    axes[0].legend(fontsize=8.5, loc='lower right')

    fig.suptitle('Which failure type should the model be trained on?  '
                 '(same architecture, only the training regime changes)',
                 fontsize=12.5)
    fig.tight_layout()
    p = OUT_DIR / f'regimenes_entrenamiento{tag}.png'
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  figura: {p}')


if __name__ == '__main__':
    main()
