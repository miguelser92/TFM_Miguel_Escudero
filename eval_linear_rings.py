"""
eval_linear_rings.py — ¿a qué distancia se agota la información? (lineal por anillos)
=====================================================================================
Cierra el hueco del argumento de localidad. Los baselines dieron:
  media de 6 vecinos 45.4%  ≈  regresión lineal con los 60 canales 46.6%
lo que SUGIERE que los 54 canales lejanos no aportan nada (linealmente) sobre el
primer anillo — pero no lo DEMUESTRA. Este experimento lo mide directamente:

  Ajusta una regresión lineal POR CANAL usando solo los canales a distancia de
  grafo <= r (anillos), con r = 1, 2, 3 y todos (60), y mide la recuperación de
  posición de cada variante. Si la curva es plana a partir de r=1, la información
  explotable linealmente se agota en el primer anillo.

Mismo protocolo que eval_baselines.py: ajuste en archivos de train, evaluación
macro sobre los 61 canales de los archivos de test, recuperación en MEDIA (métrica
principal) y p90 (cola). Sin redes: solo mínimos cuadrados → corre en minutos.

Uso:
    python eval_linear_rings.py
    python eval_linear_rings.py --max-events 200000 --tag def

Salida: reports/linear_rings[_tag].json + tabla por consola.

Autor: Miguel Escudero (TFM)
"""

import sys
import json
import datetime
import argparse
import numpy as np
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dataset import load_dat_to_dense, load_positions, get_file_split, N_ACTIVE
from hex_geometry import get_neighbor_matrix
from imputation_eval import compute_xy

GOOD_DIR   = r'E:\Datos TFM\Good\Good'
PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'
OUT_DIR    = Path(r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\reports')

RINGS = (1, 2, 3, None)          # None = todos los canales (60)


def hop_distances(nbr, src):
    """Distancia de grafo (nº de saltos) desde 'src' a cada nodo, por BFS."""
    dist = np.full(N_ACTIVE, -1, dtype=int)
    dist[src] = 0
    frontier = [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in nbr[u]:
                v = int(v)
                if v >= 0 and dist[v] < 0:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def fit_ring_models(train_X, nbr, ring):
    """
    Una regresión lineal por canal: carga_c ≈ w·(canales a <=ring saltos) + b.
    ring=None usa los 60 canales. Devuelve [(cols, w), ...] por canal.
    """
    models = []
    ones = np.ones((len(train_X), 1), dtype=np.float32)
    for c in range(N_ACTIVE):
        if ring is None:
            cols = [j for j in range(N_ACTIVE) if j != c]
        else:
            d = hop_distances(nbr, c)
            cols = [j for j in range(N_ACTIVE) if 0 < d[j] <= ring]
        A = np.hstack([train_X[:, cols], ones])
        w, *_ = np.linalg.lstsq(A, train_X[:, c], rcond=None)
        models.append((cols, w))
    return models


def impute_ring(X, models, c):
    cols, w = models[c]
    A = np.hstack([X[:, cols], np.ones((len(X), 1), dtype=np.float32)])
    return np.clip(A @ w, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-events', type=int, default=100_000)
    ap.add_argument('--train-events', type=int, default=60_000)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    tag = f'_{args.tag}' if args.tag else ''

    x_sipm, y_sipm = load_positions(PSIPM_PATH)
    nbr = get_neighbor_matrix(PSIPM_PATH)
    train_files, _, test_files = get_file_split(GOOD_DIR)

    # Tamaño típico de cada anillo (informativo; varía con el borde)
    sizes = {r: [] for r in RINGS}
    for c in range(N_ACTIVE):
        d = hop_distances(nbr, c)
        for r in RINGS:
            sizes[r].append(int(((d > 0) & (d <= (r or 99))).sum()))
    print("Canales usados por anillo (media sobre los 61 sensores):")
    for r in RINGS:
        lab = f'<= {r} saltos' if r else 'todos (60)'
        print(f"  {lab:12} -> {np.mean(sizes[r]):5.1f} canales (min {min(sizes[r])}, max {max(sizes[r])})")

    print(f"\nAjustando regresiones ({args.train_events} ev de 2 archivos de train)...")
    Xtr = np.concatenate([load_dat_to_dense(f, max_events=args.train_events)
                          for f in train_files[:2]])
    models = {r: fit_ring_models(Xtr, nbr, r) for r in RINGS}

    print(f"Cargando {len(test_files)} archivos de test ({args.max_events} ev/archivo)...")
    X_list = [load_dat_to_dense(f, max_events=args.max_events) for f in test_files]
    orig_xy = [compute_xy(X, x_sipm, y_sipm) for X in X_list]

    rec_mean = {r: [] for r in RINGS}
    rec_p90  = {r: [] for r in RINGS}
    mae      = {r: [] for r in RINGS}
    for c in range(N_ACTIVE):
        dRdeg = []
        dRimp = {r: [] for r in RINGS}
        errs  = {r: [] for r in RINGS}
        for X, (ox, oy) in zip(X_list, orig_xy):
            mod = X[:, c] > 0
            if mod.sum() < 100:
                continue
            Xd = X.copy(); Xd[:, c] = 0.0
            dx, dy = compute_xy(Xd, x_sipm, y_sipm)
            dRdeg.append(np.sqrt((dx - ox) ** 2 + (dy - oy) ** 2)[mod])
            true = X[mod, c]
            for r in RINGS:
                pred = impute_ring(X, models[r], c)
                Xi = X.copy(); Xi[:, c] = pred
                ix, iy = compute_xy(Xi, x_sipm, y_sipm)
                dRimp[r].append(np.sqrt((ix - ox) ** 2 + (iy - oy) ** 2)[mod])
                errs[r].append(np.abs(pred[mod] - true))
        if not dRdeg:
            continue
        dd = np.concatenate(dRdeg)
        dmean, dp90 = dd.mean(), np.percentile(dd, 90)
        for r in RINGS:
            di = np.concatenate(dRimp[r])
            rec_mean[r].append((dmean - di.mean()) / dmean * 100 if dmean > 0 else 0.0)
            rec_p90[r].append((dp90 - np.percentile(di, 90)) / dp90 * 100 if dp90 > 0 else 0.0)
            mae[r].append(np.concatenate(errs[r]).mean())
        if (c + 1) % 15 == 0:
            print(f"  {c+1}/{N_ACTIVE} canales")

    n = len(rec_mean[RINGS[0]])
    print(f"\n=== REGRESIÓN LINEAL POR ANILLOS (macro {n} canales) ===")
    print(f"{'anillo':>14} {'n_canales':>10} {'recov_MEAN %':>13} {'recov_p90 %':>12} {'MAE':>7}")
    out_rings = {}
    for r in RINGS:
        lab = f'<={r} saltos' if r else 'todos (60)'
        rm, rp, mm = np.mean(rec_mean[r]), np.mean(rec_p90[r]), np.mean(mae[r])
        out_rings[str(r)] = {'ring': r, 'mean_channels': round(float(np.mean(sizes[r])), 1),
                             'recov_mean_macro': round(float(rm), 2),
                             'recov_p90_macro': round(float(rp), 2),
                             'mae_macro': round(float(mm), 4)}
        print(f"{lab:>14} {np.mean(sizes[r]):>10.1f} {rm:>13.2f} {rp:>12.2f} {mm:>7.3f}")
    print("\n  Lectura: si la curva es plana desde r=1, la información explotable")
    print("  linealmente se agota en el PRIMER anillo (la red gana por la no-linealidad).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f'linear_rings{tag}.json'
    p.write_text(json.dumps({'max_events': args.max_events, 'train_events': args.train_events,
                             'n_channels': n, 'rings': out_rings,
                             'generated': datetime.datetime.now().isoformat(timespec='seconds')},
                            indent=2), encoding='utf-8')
    print(f"  JSON: {p}")


if __name__ == '__main__':
    main()
