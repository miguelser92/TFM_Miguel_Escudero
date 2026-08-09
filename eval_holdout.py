"""
eval_holdout.py — ¿cuánto dependen las métricas de QUÉ módulos cayeron en test?
==============================================================================
La partición reserva solo 5 módulos para test. La estadística de EVENTOS es
holgada (millones), pero la de MÓDULOS es pobre: las métricas por canal pueden
reflejar idiosincrasias de esos 5 detectores concretos (el caso Ich 31).

La validación cruzada resolvería la duda, pero exige reentrenar N veces. Aquí se
aprovecha una circunstancia del propio pipeline que la hace innecesaria para
esta pregunta:

    el entrenamiento rota un fichero por época, así que un modelo de 40 épocas
    solo vio 40 de los 149 módulos de train. Los 109 restantes, más los 5 de
    validación y los 5 de test, son 119 módulos que el modelo NO vio nunca.

Evaluando sobre esos 119 se obtiene la distribución del efecto de módulo con una
estadística 24 veces mayor que la del test, SIN reentrenar nada. Con eso se puede
responder: ¿es 58.0 % un número del detector, o de esos cinco módulos?

Lo que se reporta:
  - recuperación p90 macro POR MÓDULO (un número por detector)
  - media y desviación ENTRE módulos = la barra de error que faltaba
  - dónde caen los 5 de test dentro de esa distribución

OJO: solo es válido para modelos que NO vieron todos los módulos. Un modelo
entrenado con --epochs 149 los vio todos y aquí no aplica; el script lo detecta
por el campo n_modules_seen del checkpoint y avisa.

Uso:
    python eval_holdout.py --run imputer_hexcnn_s_mse --modules 40 --events 50000
    python eval_holdout.py --run imputer_hexcnn_s_mse --modules 119 --events 50000
    python eval_holdout.py --run imputer_hexcnn_s_mse --channels 12   # más rápido

Autor: Miguel Escudero (TFM)
"""

import sys
import json
import time
import datetime
import argparse
import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dataset import (load_dat_to_dense, load_positions, get_file_split,
                     N_ACTIVE, IDX_TO_ICH)
from imputation_eval import load_model, impute_channel, compute_xy

RUNS_BASE  = r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\runs'
GOOD_DIR   = r'E:\Datos TFM\Good\Good'
PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'
OUT_DIR    = Path(r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\reports')


def recovery_p90(dR_deg, dR_imp):
    p_deg = np.percentile(dR_deg, 90)
    p_imp = np.percentile(dR_imp, 90)
    return (p_deg - p_imp) / p_deg * 100 if p_deg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='imputer_hexcnn_s_mse')
    ap.add_argument('--modules', type=int, default=40,
                    help='cuántos módulos held-out evaluar (máx 119)')
    ap.add_argument('--events', type=int, default=50_000, help='eventos por módulo')
    ap.add_argument('--channels', type=int, default=0,
                    help='0 = los 61; N = N canales repartidos (más rápido)')
    ap.add_argument('--seen', type=int, default=None,
                    help='módulos de train que el modelo SÍ vio (por defecto, del checkpoint)')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    tag = f'_{args.tag}' if args.tag else ''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_sipm, y_sipm = load_positions(PSIPM_PATH)
    ckpt_path = Path(RUNS_BASE) / args.run / 'best_model.pth'
    model = load_model(ckpt_path, device)

    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    seen = args.seen if args.seen is not None else ck.get('n_modules_seen', 40)
    train_files, val_files, test_files = get_file_split(GOOD_DIR)

    if seen >= len(train_files):
        print(f"AVISO: este modelo vio {seen} módulos de train (todos). No quedan módulos de\n"
              f"       train sin ver, así que el held-out se reduce a val + test (10 módulos).")
    # Los que el round-robin nunca alcanzó, mas val y test: nunca vistos por el modelo
    holdout = list(train_files[seen:]) + list(val_files) + list(test_files)
    es_test = {f.name for f in test_files}
    es_val = {f.name for f in val_files}

    n_mod = min(args.modules, len(holdout))
    # Muestreo repartido por todo el rango de numeracion, no los primeros
    idx = np.linspace(0, len(holdout) - 1, n_mod).astype(int)
    modulos = [holdout[i] for i in sorted(set(idx.tolist()))]

    canales = (list(range(N_ACTIVE)) if args.channels <= 0
               else [int(v) for v in np.linspace(0, N_ACTIVE - 1, args.channels).astype(int)])

    print(f"Modelo: {args.run}  (vio {seen} módulos de train)")
    print(f"Held-out disponible: {len(holdout)} módulos  "
          f"({len(train_files) - seen} de train no vistos + {len(val_files)} val + {len(test_files)} test)")
    print(f"Evaluando {len(modulos)} módulos x {len(canales)} canales x {args.events:,} eventos\n")

    filas = []
    t0 = time.time()
    for i, f in enumerate(modulos):
        X = load_dat_to_dense(f, max_events=args.events)
        if len(X) < 1000:
            continue
        ox, oy = compute_xy(X, x_sipm, y_sipm)
        recs = []
        for c in canales:
            mod = X[:, c] > 0
            if mod.sum() < 100:
                continue
            Xd = X.copy(); Xd[:, c] = 0.0
            dx, dy = compute_xy(Xd, x_sipm, y_sipm)
            dRd = np.sqrt((dx - ox) ** 2 + (dy - oy) ** 2)[mod]
            Xi, _ = impute_channel(model, X, c, device)
            ix, iy = compute_xy(Xi, x_sipm, y_sipm)
            dRi = np.sqrt((ix - ox) ** 2 + (iy - oy) ** 2)[mod]
            recs.append(recovery_p90(dRd, dRi))
        if not recs:
            continue
        grupo = 'test' if f.name in es_test else ('val' if f.name in es_val else 'train-no-visto')
        filas.append({'modulo': f.name, 'grupo': grupo,
                      'recov_p90_macro': float(np.mean(recs)),
                      'peor_canal': float(np.min(recs)), 'n_canales': len(recs)})
        el = time.time() - t0
        eta = el / (i + 1) * (len(modulos) - i - 1)
        print(f"  {i+1:3d}/{len(modulos)}  {f.name:16} [{grupo:14}] "
              f"recP90={filas[-1]['recov_p90_macro']:5.2f}  peor={filas[-1]['peor_canal']:6.2f}  "
              f"(ETA {eta/60:.0f} min)", flush=True)

    v = np.array([r['recov_p90_macro'] for r in filas])
    vt = np.array([r['recov_p90_macro'] for r in filas if r['grupo'] == 'test'])
    vo = np.array([r['recov_p90_macro'] for r in filas if r['grupo'] != 'test'])

    print(f"\n{'='*66}")
    print(f"EFECTO DE MÓDULO  ({len(filas)} módulos held-out)")
    print(f"{'='*66}")
    print(f"  recuperación p90 macro entre módulos: {v.mean():.2f} ± {v.std(ddof=1):.2f}")
    print(f"    rango: {v.min():.2f} — {v.max():.2f}   (recorrido {v.max()-v.min():.2f} pts)")
    if len(vt) and len(vo):
        print(f"\n  los {len(vt)} módulos de TEST : {vt.mean():.2f} ± {vt.std(ddof=1):.2f}")
        print(f"  los otros {len(vo)} held-out   : {vo.mean():.2f} ± {vo.std(ddof=1):.2f}")
        z = (vt.mean() - vo.mean()) / (vo.std(ddof=1) / np.sqrt(len(vo)) + 1e-9)
        print(f"  diferencia: {vt.mean()-vo.mean():+.2f} pts   (z = {z:+.1f})")
        print(f"\n  -> {'el test es REPRESENTATIVO' if abs(z) < 2 else '*** el test NO es representativo ***'}")
    print(f"\n  ERROR ESTÁNDAR de la media con 5 módulos:  {v.std(ddof=1)/np.sqrt(5):.2f} pts")
    print(f"  ERROR ESTÁNDAR de la media con {len(filas)} módulos: {v.std(ddof=1)/np.sqrt(len(filas)):.2f} pts")
    print("  (esta es la barra de error por efecto de módulo que faltaba)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {'run': args.run, 'modules_seen_in_training': seen,
           'n_holdout_available': len(holdout), 'n_evaluated': len(filas),
           'events_per_module': args.events, 'n_channels': len(canales),
           'generated': datetime.datetime.now().isoformat(timespec='seconds'),
           'recov_p90_between_modules_mean': round(float(v.mean()), 3),
           'recov_p90_between_modules_sd': round(float(v.std(ddof=1)), 3),
           'se_with_5_modules': round(float(v.std(ddof=1) / np.sqrt(5)), 3),
           'per_module': filas}
    p = OUT_DIR / f'holdout{tag}.json'
    p.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"\n  JSON guardado: {p}")

    # Figura: distribucion del efecto de modulo, con los de test marcados
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vo, bins=18, color='#2471a3', alpha=.75, label=f'held-out ({len(vo)})')
    for x in vt:
        ax.axvline(x, color='#c0392b', lw=2)
    ax.axvline(v.mean(), color='k', ls='--', lw=2, label=f'media {v.mean():.1f}%')
    ax.plot([], [], color='#c0392b', lw=2, label=f'módulos de test ({len(vt)})')
    ax.set_xlabel('p90 position recovery per module (%)')
    ax.set_ylabel('modules')
    ax.set_title(f'Module-to-module spread — {args.run}\n'
                 f'{v.mean():.2f} ± {v.std(ddof=1):.2f} % over {len(filas)} held-out modules')
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    pf = OUT_DIR / f'holdout{tag}.png'
    fig.savefig(pf, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f"  figura: {pf}")


if __name__ == '__main__':
    main()
