"""
fig_multidead_models.py — figura comparativa de MODELOS en multi-dead
======================================================================
La figura de eval_multidead.py pinta, para UN modelo, las curvas de los dos
regímenes juntas — y eso induce a error: el % de recuperación del contiguo SIEMPRE
sale mayor que el del disperso porque el daño de partida es mayor (denominadores
distintos), no porque el modelo vaya mejor ahí. Comparar esas dos líneas entre sí
no significa nada.

Esta figura hace la comparación que sí responde la pregunta real ("¿qué modelo
elijo?"): fija el RÉGIMEN DE EVALUACIÓN en cada panel y compara MODELOS, en error
ABSOLUTO (mm), con el degradado como referencia.

Uso:
    python fig_multidead_models.py
    python fig_multidead_models.py --campaign MULTIDEAD_K8 --tag def

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

# (carpeta del run, etiqueta en la leyenda, color, estilo)
MODELS = [
    ('imputer_hexcnn_s_mse',               'trained single-dead (k=1)', '#7f7f7f', '-'),
    ('imputer_hexcnn_s_mse_dead1-4',       'trained multi-dead 1-4 (contiguous)', '#c0392b', '-'),
    ('imputer_hexcnn_s_mse_dead1-4_scatter', 'trained multi-dead 1-4 (scattered)', '#e67e22', '--'),
    ('imputer_hexcnn_s_mse_dead1-8',       'trained multi-dead 1-8 (contiguous)', '#8e44ad', '-.'),
]
MODE_TITLE = {'cluster': 'Evaluated on CONTIGUOUS clusters (realistic failure)',
              'scatter': 'Evaluated on SCATTERED dead sensors (control)'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', default='MULTIDEAD_K8')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    tag = f'_{args.tag}' if args.tag else ''

    curves = {}
    for run, label, *_ in MODELS:
        p = RUNS_BASE / run / args.campaign / 'eval_multidead_metrics.json'
        if p.exists():
            curves[run] = json.load(open(p, encoding='utf-8'))['curve']
        else:
            print(f"  (falta {run}/{args.campaign} → se omite)")

    if not curves:
        print("No hay datos para esa campaña."); return

    ks = sorted({int(k.split('_k')[1]) for c in curves.values() for k in c if k.startswith('cluster_')})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=False)

    for ax, mode in zip(axes, ('cluster', 'scatter')):
        # Referencia: degradado (sin red). Es igual para todos los modelos.
        ref = next(iter(curves.values()))
        deg = [ref[f'{mode}_k{k}']['dR_deg_p90_macro'] for k in ks]
        ax.plot(ks, deg, color='k', ls=':', lw=2, marker='x', ms=6,
                label='no imputation (degraded)')

        for run, label, color, style in MODELS:
            if run not in curves:
                continue
            y = [curves[run][f'{mode}_k{k}']['dR_imp_p90_macro'] for k in ks]
            ax.plot(ks, y, color=color, ls=style, lw=2, marker='o', ms=5, label=label)

        ax.set_title(MODE_TITLE[mode], fontsize=11)
        ax.set_xlabel('Number of simultaneous dead sensors (k)')
        ax.set_ylabel('Position error p90 after imputation (mm)')
        ax.set_xticks(ks)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc='upper left')

    # Misma escala Y en ambos paneles → la comparación visual entre regímenes es honesta
    ymax = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(0, ymax)

    fig.suptitle('Multi-dead: which training regime matters?  (lower is better; absolute error)',
                 fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / f'multidead_models{tag}.png'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)

    # Resumen numérico del mensaje de la figura
    print(f"\n=== dR imputado p90 (mm), evaluando en CONTIGUO ===")
    print(f"{'modelo':40} " + ' '.join(f'k={k}'.rjust(7) for k in ks))
    for run, label, *_ in MODELS:
        if run in curves:
            print(f"{label:40} " + ' '.join(
                f"{curves[run][f'cluster_k{k}']['dR_imp_p90_macro']:>7.3f}" for k in ks))
    kmax = ks[-1]
    if 'imputer_hexcnn_s_mse' in curves and 'imputer_hexcnn_s_mse_dead1-4' in curves:
        a = curves['imputer_hexcnn_s_mse'][f'cluster_k{kmax}']['dR_imp_p90_macro']
        b = curves['imputer_hexcnn_s_mse_dead1-4'][f'cluster_k{kmax}']['dR_imp_p90_macro']
        print(f"\n  single→multi en k={kmax}: {a:.3f} → {b:.3f} mm ({100*(a-b)/a:.0f}% mejor)  ← lo que importa")
    if 'imputer_hexcnn_s_mse_dead1-4_scatter' in curves:
        c = curves['imputer_hexcnn_s_mse_dead1-4_scatter'][f'cluster_k{kmax}']['dR_imp_p90_macro']
        print(f"  contiguo vs disperso:    {b:.3f} → {c:.3f} mm ({100*(c-b)/b:.0f}% dif)  ← detalle menor")
    print(f"\n  figura: {out}")


if __name__ == '__main__':
    main()
