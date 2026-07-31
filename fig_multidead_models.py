"""
fig_multidead_models.py — figura comparativa de MODELOS en multi-dead
======================================================================
La figura de eval_multidead.py pinta, para UN modelo, las curvas de los dos
regímenes juntas — y eso induce a error: el % de recuperación del contiguo SIEMPRE
sale mayor que el del disperso porque el daño de partida es mayor (denominadores
distintos), no porque el modelo vaya mejor ahí. Comparar esas dos líneas entre sí
no significa nada.

Esta figura hace las comparaciones que sí responden las preguntas reales:

  Panel 1-2 : fija el RÉGIMEN DE EVALUACIÓN y compara MODELOS, en error ABSOLUTO
              (mm), con el degradado como referencia. → "¿qué modelo elijo?"
  Panel 3   : ventaja (%) de entrenar EN EL RÉGIMEN QUE LUEGO SE EVALÚA, frente a
              entrenar en el otro. → "¿el régimen de entrenamiento importa o es
              ruido?". El punto k=1 es el CONTROL DE RUIDO: con un solo muerto,
              contiguo y disperso son el MISMO problema, así que la diferencia ahí
              es puramente variabilidad de entrenamiento. La banda gris marca ese
              nivel; lo que sobresale de ella es efecto real.

Por defecto usa la MEDIA de la distribución de ΔR (el resultado real, ponderando
todos los eventos). --metric median|p90 para los otros estadísticos.

Uso:
    python fig_multidead_models.py
    python fig_multidead_models.py --metric p90
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

# (carpeta del run, etiqueta de leyenda, color, estilo de línea)
MODELS = [
    ('imputer_hexcnn_s_mse',                'trained single-dead (k=1)',          '#7f7f7f', '-'),
    ('imputer_hexcnn_s_mse_dead1-4',        'trained multi-dead 1-4 (contiguous)', '#c0392b', '-'),
    ('imputer_hexcnn_s_mse_dead1-4_scatter', 'trained multi-dead 1-4 (scattered)', '#e67e22', '--'),
    ('imputer_hexcnn_s_mse_dead1-8',        'trained multi-dead 1-8 (contiguous)', '#8e44ad', '-.'),
]
MODE_TITLE = {'cluster': 'Evaluated on CONTIGUOUS clusters (realistic failure)',
              'scatter': 'Evaluated on SCATTERED dead sensors (control)'}

A_RUN = 'imputer_hexcnn_s_mse_dead1-4'          # entrenado contiguo
B_RUN = 'imputer_hexcnn_s_mse_dead1-4_scatter'  # entrenado disperso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', default='MULTIDEAD_K8')
    ap.add_argument('--metric', choices=['mean', 'median', 'p90'], default='mean',
                    help='estadistico de la distribucion de dR (por defecto MEDIA = resultado real)')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    tag = f'_{args.tag}' if args.tag else ''

    curves, degr = {}, {}
    for run, label, *_ in MODELS:
        p = RUNS_BASE / run / args.campaign / 'eval_multidead_metrics.json'
        if not p.exists():
            print(f"  (falta {run}/{args.campaign} -> se omite)")
            continue
        d = json.load(open(p, encoding='utf-8'))
        imp, deg = {}, {}
        for ps in d['per_set']:
            imp.setdefault((ps['mode'], ps['k']), []).append(ps['dR_imp'][args.metric])
            deg.setdefault((ps['mode'], ps['k']), []).append(ps['dR_deg'][args.metric])
        curves[run] = {kk: float(np.mean(v)) for kk, v in imp.items()}
        degr[run]   = {kk: float(np.mean(v)) for kk, v in deg.items()}

    if not curves:
        print("No hay datos para esa campana."); return

    ks = sorted({k for (m, k) in next(iter(curves.values())) if m == 'cluster'})
    MLAB = args.metric
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.2))

    # ── Paneles 1-2: comparación de modelos, un panel por régimen de evaluación ──
    for ax, mode in zip(axes[:2], ('cluster', 'scatter')):
        ref = next(iter(degr.values()))          # el degradado no depende del modelo
        ax.plot(ks, [ref[(mode, k)] for k in ks], color='k', ls=':', lw=2, marker='x', ms=6,
                label='no imputation (degraded)')
        for run, label, color, style in MODELS:
            if run not in curves:
                continue
            ax.plot(ks, [curves[run][(mode, k)] for k in ks],
                    color=color, ls=style, lw=2, marker='o', ms=5, label=label)
        ax.set_title(MODE_TITLE[mode], fontsize=11)
        ax.set_xlabel('Number of simultaneous dead sensors (k)')
        ax.set_ylabel(f'Position error ({MLAB}) after imputation (mm)')
        ax.set_xticks(ks); ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc='upper left')
    ymax = max(a.get_ylim()[1] for a in axes[:2])
    for a in axes[:2]:
        a.set_ylim(0, ymax)                       # misma escala → comparación honesta

    # ── Panel 3: ¿importa el régimen de entrenamiento, o es ruido? ──
    ax = axes[2]
    adv_c = adv_s = None
    if A_RUN in curves and B_RUN in curves:
        adv_c = [(curves[B_RUN][('cluster', k)] - curves[A_RUN][('cluster', k)])
                 / curves[B_RUN][('cluster', k)] * 100 for k in ks]
        adv_s = [(curves[A_RUN][('scatter', k)] - curves[B_RUN][('scatter', k)])
                 / curves[A_RUN][('scatter', k)] * 100 for k in ks]
        noise = max(abs(adv_c[0]), abs(adv_s[0]))     # k=1: mismo problema -> ruido puro
        ax.axhspan(-noise, noise, color='0.85', label=f'noise level (k=1: +/-{noise:.1f}%)')
        ax.axhline(0, color='k', lw=0.8)
        ax.plot(ks, adv_c, color='#c0392b', lw=2, marker='o', ms=5,
                label='contiguous-trained, on contiguous')
        ax.plot(ks, adv_s, color='#2471a3', lw=2, marker='s', ms=5,
                label='scattered-trained, on scattered')
        ax.set_title('Does the training regime matter?', fontsize=11)
        ax.set_xlabel('Number of simultaneous dead sensors (k)')
        ax.set_ylabel('Advantage of matching the regime (%)')
        ax.set_xticks(ks); ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc='upper left')

    fig.suptitle(f'Multi-dead: training regime and failure size '
                 f'({MLAB} absolute error; lower is better)', fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / f'multidead_models_{args.metric}{tag}.png'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)

    # ── Resumen numérico ──
    kmax = ks[-1]
    print(f"\n=== dR {MLAB} imputado (mm), evaluando en CONTIGUO ===")
    print(f"{'modelo':40} " + ' '.join(f'k={k}'.rjust(7) for k in ks))
    for run, label, *_ in MODELS:
        if run in curves:
            print(f"{label:40} " + ' '.join(f"{curves[run][('cluster', k)]:>7.3f}" for k in ks))
    if 'imputer_hexcnn_s_mse' in curves and A_RUN in curves:
        a = curves['imputer_hexcnn_s_mse'][('cluster', kmax)]
        b = curves[A_RUN][('cluster', kmax)]
        print(f"\n  single->multi (k={kmax}, contiguo): {a:.3f} -> {b:.3f} mm = {100*(a-b)/a:.0f}% mejor")
    if adv_c is not None:
        print(f"  regimen correcto vs incorrecto (k={kmax}): +{adv_c[-1]:.1f}% en contiguo, "
              f"+{adv_s[-1]:.1f}% en disperso   (ruido k=1: +/-{noise:.1f}%)")
    print(f"\n  figura: {out}")


if __name__ == '__main__':
    main()
