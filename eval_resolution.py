"""
eval_resolution.py — recuperación de RESOLUCIÓN espacial (FWHM) tras imputar
============================================================================
Métrica física complementaria al ΔR. El ΔR mide el DESPLAZAMIENTO del mapa
(sesgo); esto mide el EMBORRONAMIENTO (blur = resolución espacial).

Con ground truth por inyección en GOOD: para cada evento, posición
  - verdadera  (61 canales)
  - degradada  (canal muerto, 60 canales)
  - imputada   (60 + predicción de la red)
El ERROR = reconstruida − verdadera. Se separa en:
  - bias (mm)  = |media del error|         (desplazamiento sistemático, ~ΔR)
  - FWHM (mm)  = anchura a media altura de la distribución del error (el BLUR)

Recuperación de resolución = (FWHM_degradado − FWHM_imputado) / FWHM_degradado.

Uso:
    python eval_resolution.py --max-events 200000
    python eval_resolution.py --run imputer_hexcnn_s_mse --tag def

Autor: Miguel Escudero (TFM)
"""

import sys
import json
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

from dataset import load_dat_to_dense, load_positions, get_file_split, N_ACTIVE, IDX_TO_ICH
from imputation_eval import load_model, impute_channel, compute_xy

RUNS_BASE  = r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\runs'
GOOD_DIR   = r'E:\Datos TFM\Good\Good'
PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'
OUT_DIR    = Path(r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\reports')


def error_stats(dx, dy):
    """
    bias (mm) = |media del error| = desplazamiento sistemático del mapa.
    FWHM (mm) = anchura de la distribución del error alrededor de su media (el BLUR),
                como FWHM gaussiana-equivalente = 2.3548·σ (media de los dos ejes).
                Basada en σ robusta (MAD) para no dejarse llevar por outliers extremos.
    """
    bias = float(np.hypot(dx.mean(), dy.mean()))
    def rob_sigma(v):
        return 1.4826 * np.median(np.abs(v - np.median(v)))
    fwhm = float(2.3548 * 0.5 * (rob_sigma(dx) + rob_sigma(dy)))
    return bias, fwhm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='imputer_hexcnn_s_mse')
    ap.add_argument('--max-events', type=int, default=200_000)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    tag = f'_{args.tag}' if args.tag else ''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_sipm, y_sipm = load_positions(PSIPM_PATH)
    _, _, test_files = get_file_split(GOOD_DIR)
    model = load_model(Path(RUNS_BASE) / args.run / 'best_model.pth', device)

    print(f"Cargando {len(test_files)} archivos de test ({args.max_events} ev/archivo)...")
    X_list = [load_dat_to_dense(f, max_events=args.max_events) for f in test_files]
    orig_xy = [compute_xy(X, x_sipm, y_sipm) for X in X_list]

    # Acumular, por canal, el error de posición (deg e imp)
    fwhm_deg, fwhm_imp, bias_deg, bias_imp = [], [], [], []
    err_ref = {'deg': None, 'imp': None}          # para la figura (canal Ich30)
    for c in range(N_ACTIVE):
        dxd, dyd, dxi, dyi = [], [], [], []
        for X, (ox, oy) in zip(X_list, orig_xy):
            mod = X[:, c] > 0
            if mod.sum() < 200:
                continue
            Xd = X.copy(); Xd[:, c] = 0.0
            dx, dy = compute_xy(Xd, x_sipm, y_sipm)
            dxd.append((dx - ox)[mod]); dyd.append((dy - oy)[mod])
            Xi, _ = impute_channel(model, X, c, device)
            ix, iy = compute_xy(Xi, x_sipm, y_sipm)
            dxi.append((ix - ox)[mod]); dyi.append((iy - oy)[mod])
        if not dxd:
            continue
        dxd, dyd = np.concatenate(dxd), np.concatenate(dyd)
        dxi, dyi = np.concatenate(dxi), np.concatenate(dyi)
        bd, fd = error_stats(dxd, dyd); bi, fi = error_stats(dxi, dyi)
        bias_deg.append(bd); fwhm_deg.append(fd); bias_imp.append(bi); fwhm_imp.append(fi)
        if IDX_TO_ICH[c] == 30:
            err_ref = {'deg': (dxd, dyd), 'imp': (dxi, dyi)}
        if (c + 1) % 15 == 0:
            print(f"  {c+1}/{N_ACTIVE} canales")

    fd, fi = np.nanmean(fwhm_deg), np.nanmean(fwhm_imp)
    bd, bi = np.nanmean(bias_deg), np.nanmean(bias_imp)
    print(f"\n=== RESOLUCIÓN (macro {len(fwhm_deg)} canales) ===")
    print(f"{'':>12} {'bias (mm)':>10} {'FWHM (mm)':>10}")
    print(f"{'degradado':>12} {bd:>10.3f} {fd:>10.3f}")
    print(f"{'imputado':>12} {bi:>10.3f} {fi:>10.3f}")
    print(f"\n  recuperación de BLUR (FWHM): {(fd-fi)/fd*100:>5.1f}%   "
          f"(FWHM {fd:.2f} → {fi:.2f} mm)")
    print(f"  recuperación de SESGO (bias): {(bd-bi)/bd*100:>5.1f}%   ({bd:.2f} → {bi:.2f} mm)")

    # ── JSON estándar (persistir, no depender de la consola) ──
    out = {'run': args.run, 'max_events': args.max_events,
           'n_channels': len(fwhm_deg),
           'generated': datetime.datetime.now().isoformat(timespec='seconds'),
           'fwhm_deg_macro_mm': round(float(fd), 4), 'fwhm_imp_macro_mm': round(float(fi), 4),
           'bias_deg_macro_mm': round(float(bd), 4), 'bias_imp_macro_mm': round(float(bi), 4),
           'blur_recovery_pct': round(float((fd - fi) / fd * 100), 2),
           'bias_recovery_pct': round(float((bd - bi) / bd * 100), 2),
           'fwhm_imp_per_channel': [round(float(v), 4) for v in fwhm_imp],
           'fwhm_deg_per_channel': [round(float(v), 4) for v in fwhm_deg]}
    p = OUT_DIR / f'resolution{tag}.json'
    p.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"  JSON guardado: {p}")

    # ── Figura: nube de error 2D (canal Ich30) degradado vs imputado ──
    if err_ref['deg'] is not None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
        for a, key, ttl in [(ax[0], 'deg', 'DEGRADED'), (ax[1], 'imp', 'IMPUTED')]:
            dx, dy = err_ref[key]
            a.hist2d(dx, dy, bins=120, range=[[-4, 4], [-4, 4]], cmap='inferno')
            a.set_title(f'{ttl} — position error (Ich30)'); a.set_xlabel('dx (mm)'); a.set_ylabel('dy (mm)')
            a.set_aspect('equal')
        fig.suptitle('Point-spread of the imputation error (tighter = better resolution)')
        fig.tight_layout()
        p = OUT_DIR / f'resolution_pointspread{tag}.png'
        fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig)
        print(f"  figura: {p.name}")


if __name__ == '__main__':
    main()
