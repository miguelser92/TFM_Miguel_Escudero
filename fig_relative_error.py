"""
fig_relative_error.py — error RELATIVO de carga (métrica interpretable)
=======================================================================
Hasta ahora el error del canal imputado se reportaba en ADC (MAE absoluto) y la
métrica principal era la recuperación de POSICIÓN, que va con el centroide
ponderado por Rch² — o sea, una cantidad que amplifica los errores y cuesta de
interpretar.

Esta métrica es la pregunta directa: **¿en qué porcentaje nos equivocamos al
estimar la carga del canal muerto?**

    error_relativo(canal) = MAE(canal) / carga_media_del_canal_en_eventos_modified

Se define sobre la MEDIA (no evento a evento) a propósito: el cociente por evento
explota cuando la carga real tiende a 0, y la mayoría de eventos tienen carga
pequeña en un canal dado. Dividir el MAE agregado por la carga media agregada es
estable y tiene lectura directa.

No requiere re-evaluar nada: la carga media por canal se calcula una vez de los
archivos de test (sin modelo) y se combina con los MAE ya guardados en los JSON
de eval_total de cada campaña.

Uso:
    python fig_relative_error.py
    python fig_relative_error.py --max-events 600000 --tag def

Salida: reports/relative_error[_tag].json + tabla por consola + figura.

Autor: Miguel Escudero (TFM)
"""

import sys
import json
import argparse
import datetime
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dataset import load_dat_to_dense, load_positions, get_file_split, N_ACTIVE, IDX_TO_ICH

RUNS_BASE  = Path(r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\runs')
OUT_DIR    = Path(r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\reports')
GOOD_DIR   = r'E:\Datos TFM\Good\Good'
PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'

# (carpeta, etiqueta) — modelos a comparar
MODELS = [
    ('imputer_hexcnn_s_mse',        'HexGNN s (referencia)'),
    ('ensemble_ref3',               'Ensemble de 3 réplicas'),
    ('ensemble_mix',                'Ensemble heterogéneo'),
    ('sandbox2_pna',                'PNA'),
    ('imputer_hexcnn_s_mse_aniso',  'HexCNN anisótropa (Zhao)'),
    ('imputer_hexcnn_s_mse_chnorm', 'HexGNN + norm por canal'),
    ('sandbox2_unet2',              'U-Net de grafo'),
    ('sandbox2_xformer',            'Graphormer'),
    ('imputer_hexcnn_s_mse_hetero', 'Loss heterocedástica (mu+sigma)'),
]
CAMPAIGNS = ('TOTAL', 'ATTN', 'TOTAL_full')     # orden de preferencia


def mean_charge_per_channel(max_events):
    """Carga media de cada canal SOBRE LOS EVENTOS MODIFIED (los que tenían señal)."""
    _, _, test_files = get_file_split(GOOD_DIR)
    tot = np.zeros(N_ACTIVE); cnt = np.zeros(N_ACTIVE)
    for f in test_files:
        X = load_dat_to_dense(f, max_events=max_events)
        m = X > 0
        tot += (X * m).sum(axis=0)
        cnt += m.sum(axis=0)
    return tot / np.maximum(cnt, 1)


def load_eval(run):
    for c in CAMPAIGNS:
        p = RUNS_BASE / run / c / 'eval_total_metrics.json'
        if p.exists():
            return json.load(open(p, encoding='utf-8')), c
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-events', type=int, default=600_000)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    tag = f'_{args.tag}' if args.tag else ''

    x_sipm, y_sipm = load_positions(PSIPM_PATH)
    r_center = np.hypot(x_sipm, y_sipm)
    print(f"Calculando la carga media por canal ({args.max_events} ev/archivo de test)...")
    q = mean_charge_per_channel(args.max_events)
    print(f"  carga media: min={q.min():.2f}  max={q.max():.2f}  (ADC, eventos modified)")

    rows, out = [], {}
    for run, label in MODELS:
        d, camp = load_eval(run)
        if d is None:
            print(f"  (sin eval_total: {run})"); continue
        # error relativo por canal y macro
        rel, mae_l = [], []
        for pc in d['per_channel']:
            idx = pc['idx']
            rel.append(pc['mae_mod'] / max(q[idx], 1e-6) * 100)
            mae_l.append(pc['mae_mod'])
        rel = np.array(rel)
        edge = r_center[[pc['idx'] for pc in d['per_channel']]] >= np.percentile(r_center, 66)
        out[run] = {'label': label, 'campaign': camp,
                    'rel_error_macro_pct': round(float(rel.mean()), 2),
                    'rel_error_median_pct': round(float(np.median(rel)), 2),
                    'rel_error_core_pct': round(float(rel[~edge].mean()), 2),
                    'rel_error_edge_pct': round(float(rel[edge].mean()), 2),
                    'rel_error_worst_pct': round(float(rel.max()), 2),
                    'mae_macro_adc': round(float(np.mean(mae_l)), 4),
                    'rel_per_channel_pct': [round(float(v), 2) for v in rel]}
        rows.append((label, out[run]))

    print(f"\n=== ERROR RELATIVO DE CARGA (MAE / carga media del canal) ===")
    print(f"{'modelo':26} {'MAE (ADC)':>10} {'rel MEDIA':>10} {'rel mediana':>12} {'centro':>8} {'borde':>8} {'peor':>8}")
    for label, o in rows:
        print(f"{label:26} {o['mae_macro_adc']:>10.4f} {o['rel_error_macro_pct']:>9.1f}% "
              f"{o['rel_error_median_pct']:>11.1f}% {o['rel_error_core_pct']:>7.1f}% "
              f"{o['rel_error_edge_pct']:>7.1f}% {o['rel_error_worst_pct']:>7.1f}%")

    # ── Figura: error relativo por canal frente a la distancia al centro ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ref = out.get('imputer_hexcnn_s_mse')
    if ref:
        d, _ = load_eval('imputer_hexcnn_s_mse')
        idxs = [pc['idx'] for pc in d['per_channel']]
        ax1.scatter(r_center[idxs], ref['rel_per_channel_pct'], s=45, c='#c0392b')
        ax1.set_xlabel('Distance to detector center (mm)')
        ax1.set_ylabel('Relative charge error (%)')
        ax1.set_title('Relative error per channel (reference model)')
        ax1.grid(alpha=0.3)
    labels = [l for l, _ in rows]; vals = [o['rel_error_macro_pct'] for _, o in rows]
    colors = ['#27ae60' if 'Ensemble' in l else '#2471a3' for l in labels]
    ax2.barh(range(len(labels)), vals, color=colors)
    ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels, fontsize=9)
    ax2.invert_yaxis(); ax2.set_xlabel('Relative charge error (%), macro')
    ax2.set_title('Model comparison'); ax2.grid(alpha=0.3, axis='x')
    for i, v in enumerate(vals):
        ax2.text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p_png = OUT_DIR / f'relative_error{tag}.png'
    fig.savefig(p_png, dpi=150, bbox_inches='tight'); plt.close(fig)

    p = OUT_DIR / f'relative_error{tag}.json'
    p.write_text(json.dumps({'max_events': args.max_events,
                             'mean_charge_per_channel': [round(float(v), 4) for v in q],
                             'models': out,
                             'generated': datetime.datetime.now().isoformat(timespec='seconds')},
                            indent=2), encoding='utf-8')
    print(f"\n  JSON: {p}\n  figura: {p_png}")


if __name__ == '__main__':
    main()
