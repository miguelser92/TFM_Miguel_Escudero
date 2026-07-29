"""
eval_baselines.py  — ¿hace falta el deep learning?
===================================================
Compara la red (HexCNN) contra baselines CLÁSICOS en la misma tarea de imputación
de un canal muerto, con la misma métrica (recuperación de posición ΔR + MAE),
macro sobre los 61 canales del test.

Baselines:
  - neighbor_mean : rellenar el canal con la MEDIA de sus vecinos físicos (lo más
                    tonto: interpolación local sin entrenar).
  - linear_reg    : regresión lineal por canal (predice la carga del canal muerto
                    a partir de los otros 60, ajustada en Good). El baseline clásico
                    fuerte: si empata con la red, la no-linealidad no aporta.

Uso:
    python eval_baselines.py --max-events 100000
    python eval_baselines.py --run imputer_hexcnn_s_mse --max-events 100000

Autor: Miguel Escudero (TFM)
"""

import sys
import json
import datetime
import argparse
import numpy as np
import torch
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dataset import load_dat_to_dense, load_positions, get_file_split, N_ACTIVE, IDX_TO_ICH
from hex_geometry import get_neighbor_matrix
from imputation_eval import load_model, impute_channel, compute_xy

RUNS_BASE  = r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\runs'
GOOD_DIR   = r'E:\Datos TFM\Good\Good'
PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'
OUT_DIR    = Path(r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\reports')


def fit_linear(train_X, nbr):
    """Una regresión lineal por canal: carga_c ≈ w·(otros 60) + b. Ajuste por mínimos cuadrados."""
    N = N_ACTIVE
    models = []
    Xb = np.hstack([train_X, np.ones((len(train_X), 1), dtype=np.float32)])  # +bias
    for c in range(N):
        cols = [j for j in range(N) if j != c]
        A = Xb[:, cols + [N]]                       # otros 60 + bias
        w, *_ = np.linalg.lstsq(A, train_X[:, c], rcond=None)
        models.append((cols, w))
    return models


def impute_linear(X, models, c):
    cols, w = models[c]
    A = np.hstack([X[:, cols], np.ones((len(X), 1), dtype=np.float32)])
    return np.clip(A @ w, 0, None)


def impute_neighbor(X, nbr, c):
    vec = [int(j) for j in nbr[c] if j >= 0]
    return X[:, vec].mean(axis=1)


def recovery(dR_deg, dR_imp):
    p_deg = np.percentile(dR_deg, 90); p_imp = np.percentile(dR_imp, 90)
    return (p_deg - p_imp) / p_deg * 100 if p_deg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='imputer_hexcnn_s_mse')
    ap.add_argument('--max-events', type=int, default=100_000)
    ap.add_argument('--tag', default='', help='identificador para el archivo de salida')
    args = ap.parse_args()
    tag = f'_{args.tag}' if args.tag else ''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_sipm, y_sipm = load_positions(PSIPM_PATH)
    nbr = get_neighbor_matrix(PSIPM_PATH)
    train_files, _, test_files = get_file_split(GOOD_DIR)

    model = load_model(Path(RUNS_BASE) / args.run / 'best_model.pth', device)

    print("Ajustando la regresión lineal (2 archivos Good de train)...")
    Xtr = np.concatenate([load_dat_to_dense(f, max_events=60_000) for f in train_files[:2]])
    lin_models = fit_linear(Xtr, nbr)

    print(f"Cargando {len(test_files)} archivos de test ({args.max_events} ev/archivo)...")
    X_list = [load_dat_to_dense(f, max_events=args.max_events) for f in test_files]
    orig_xy = [compute_xy(X, x_sipm, y_sipm) for X in X_list]

    # Acumular recuperación por canal para cada método
    rec = {'network': [], 'neighbor_mean': [], 'linear_reg': []}
    mae = {'network': [], 'neighbor_mean': [], 'linear_reg': []}
    for c in range(N_ACTIVE):
        dRdeg, dRnet, dRnb, dRlin = [], [], [], []
        enet, enb, elin = [], [], []
        for X, (ox, oy) in zip(X_list, orig_xy):
            mod = X[:, c] > 0
            if mod.sum() < 100:
                continue
            true = X[mod, c]
            # degradado
            Xd = X.copy(); Xd[:, c] = 0.0
            dx, dy = compute_xy(Xd, x_sipm, y_sipm)
            dRdeg.append(np.sqrt((dx-ox)**2 + (dy-oy)**2)[mod])
            # red
            Xi, pred = impute_channel(model, X, c, device)
            ix, iy = compute_xy(Xi, x_sipm, y_sipm)
            dRnet.append(np.sqrt((ix-ox)**2 + (iy-oy)**2)[mod]); enet.append(np.abs(pred[mod]-true))
            # neighbor mean
            pnb = impute_neighbor(X, nbr, c); Xnb = X.copy(); Xnb[:, c] = pnb
            nx, ny = compute_xy(Xnb, x_sipm, y_sipm)
            dRnb.append(np.sqrt((nx-ox)**2 + (ny-oy)**2)[mod]); enb.append(np.abs(pnb[mod]-true))
            # linear reg
            pli = impute_linear(X, lin_models, c); Xli = X.copy(); Xli[:, c] = pli
            lx, ly = compute_xy(Xli, x_sipm, y_sipm)
            dRlin.append(np.sqrt((lx-ox)**2 + (ly-oy)**2)[mod]); elin.append(np.abs(pli[mod]-true))
        if not dRdeg:
            continue
        dd = np.concatenate(dRdeg)
        rec['network'].append(recovery(dd, np.concatenate(dRnet)))
        rec['neighbor_mean'].append(recovery(dd, np.concatenate(dRnb)))
        rec['linear_reg'].append(recovery(dd, np.concatenate(dRlin)))
        mae['network'].append(np.concatenate(enet).mean())
        mae['neighbor_mean'].append(np.concatenate(enb).mean())
        mae['linear_reg'].append(np.concatenate(elin).mean())
        if (c + 1) % 15 == 0:
            print(f"  {c+1}/{N_ACTIVE} canales")

    print(f"\n=== ¿HACE FALTA EL DEEP LEARNING?  (macro sobre {len(rec['network'])} canales) ===")
    print(f"{'método':>16} {'recov_p90 %':>12} {'MAE (ADC)':>11}")
    for m in ('neighbor_mean', 'linear_reg', 'network'):
        print(f"{m:>16} {np.mean(rec[m]):>12.1f} {np.mean(mae[m]):>11.3f}")
    net, lin = np.mean(rec['network']), np.mean(rec['linear_reg'])
    print(f"\n  ventaja de la red sobre regresión lineal: {net - lin:+.1f} puntos de recuperación")
    print(f"  ventaja sobre media de vecinos: {net - np.mean(rec['neighbor_mean']):+.1f} puntos")

    # ── JSON estándar (persistir, no depender de la consola) ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {'run': args.run, 'max_events': args.max_events,
           'n_channels': len(rec['network']),
           'generated': datetime.datetime.now().isoformat(timespec='seconds'),
           'methods': {m: {'recov_p90_macro': round(float(np.mean(rec[m])), 2),
                           'mae_macro': round(float(np.mean(mae[m])), 4),
                           'recov_p90_per_channel': [round(float(v), 2) for v in rec[m]]}
                       for m in rec},
           'net_vs_linear_pts': round(float(net - lin), 2),
           'net_vs_neighbor_pts': round(float(net - np.mean(rec['neighbor_mean'])), 2)}
    p = OUT_DIR / f'baselines{tag}.json'
    p.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"  JSON guardado: {p}")


if __name__ == '__main__':
    main()
