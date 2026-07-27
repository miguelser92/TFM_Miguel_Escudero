"""
eval_bad_recovery.py  (la culminación: detección + imputación sobre BAD reales)
==============================================================================
Cierra el círculo del proyecto sobre datos REALES no etiquetados:

  1. DETECTAR los canales malos de un módulo BAD (bad_detect, z-score module-norm).
  2. IMPUTAR con la red: si es 1 canal → modelo base (imputer_hexcnn_s_mse);
     si es un clúster (>1) → modelo multi-dead (imputer_hexcnn_s_mse_dead1-4).
  3. Ver cómo REVIVE el flood map: el canal muerto sesga el centro de gravedad y
     abre un "agujero" en el llenado; tras imputar, el patrón se recompone.

Evaluación:
  - CUALITATIVA: flood map ORIGINAL (con agujero) vs IMPUTADO vs referencia GOOD.
  - CUANTITATIVA (auto-consistencia, sin ground truth): se vuelve a pasar el
    detector sobre el módulo YA imputado. El canal que antes marcaba z≈−6 debería
    volver a z≈0 (deja de ser anómalo) → "la red repara lo que el detector
    diagnostica", medido en un número.

Uso:
    conda activate tfm
    python eval_bad_recovery.py                       # auto-consistencia de TODOS + figuras de una muestra
    python eval_bad_recovery.py --files datas005,datas108
    python eval_bad_recovery.py --sample 10 --max-events 150000

Autor: Miguel Escudero (TFM)
"""

import sys
import glob
import json
import argparse
import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.backends.backend_pdf import PdfPages

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dataset import load_dat_to_dense, load_positions, N_ACTIVE, IDX_TO_ICH
from hex_geometry import get_neighbor_matrix
from imputation_eval import load_model, compute_xy
from eval_multidead import impute_set
from bad_detect import flag_module

RUNS_BASE  = r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\runs'
GOOD_DIR   = r'E:\Datos TFM\Good\Good'
BAD_DIR    = r'E:\Datos TFM\Bad\Bad'
PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'
OUT_DIR    = Path(r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\reports')
BASE_NPZ   = OUT_DIR / 'good_baseline.npz'

MODEL_BASE = 'imputer_hexcnn_s_mse'          # 1 canal muerto
MODEL_MULTI = 'imputer_hexcnn_s_mse_dead1-4'  # clúster
Z_OP       = 2.0
HIST_BINS  = 140
GOOD_REF   = 'datas057.dat'                   # módulo sano de referencia para el flood map


def flood_map(X, x_sipm, y_sipm, rng):
    """Histograma 2D del centro de gravedad Rch² (el flood map)."""
    px, py = compute_xy(X, x_sipm, y_sipm)
    H, _, _ = np.histogram2d(px, py, bins=HIST_BINS, range=rng)
    return H.T   # .T para que imshow lo oriente como el detector


def draw_flood(ax, H, extent, title):
    ax.imshow(H, origin='lower', extent=extent, cmap='inferno',
              norm=PowerNorm(gamma=0.5), aspect='equal')
    ax.set_title(title, fontsize=10); ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-events', type=int, default=150_000)
    ap.add_argument('--sample', type=int, default=8, help='nº de módulos para las figuras')
    ap.add_argument('--files', type=str, default=None, help='lista datasXXX,datasYYY (sin .dat)')
    ap.add_argument('--z', type=float, default=Z_OP)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', type=str, default='', help='identificador para los archivos de salida')
    args = ap.parse_args()
    tag = f'_{args.tag}' if args.tag else ''

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_sipm, y_sipm = load_positions(PSIPM_PATH)
    nbr = get_neighbor_matrix(PSIPM_PATH)
    d = np.load(BASE_NPZ)
    base = {k: d[k] for k in ('frac_median', 'frac_spread', 'ratio_median', 'ratio_spread')}

    model_base  = load_model(Path(RUNS_BASE) / MODEL_BASE / 'best_model.pth', device)
    model_multi = load_model(Path(RUNS_BASE) / MODEL_MULTI / 'best_model.pth', device)

    # Rango del flood map y referencia GOOD (fijos para todos)
    Xg = load_dat_to_dense(str(Path(GOOD_DIR) / GOOD_REF), max_events=args.max_events)
    pgx, pgy = compute_xy(Xg, x_sipm, y_sipm)
    pad = 1.0
    rng = [[pgx.min()-pad, pgx.max()+pad], [pgy.min()-pad, pgy.max()+pad]]
    extent = [rng[0][0], rng[0][1], rng[1][0], rng[1][1]]
    Hgood = flood_map(Xg, x_sipm, y_sipm, rng)

    bad_files = sorted(glob.glob(str(Path(BAD_DIR) / '*.dat')))

    # ── Pasada CUANTITATIVA sobre todos: auto-consistencia (z antes → después) ──
    print(f"=== AUTO-CONSISTENCIA sobre {len(bad_files)} módulos BAD (z>{args.z}) ===")
    z_before_all, z_after_all = [], []
    recovered = 0
    summary = {}
    for f in bad_files:
        X = load_dat_to_dense(f, max_events=args.max_events)
        r0 = flag_module(X, nbr, base, args.z)
        if len(r0['flagged']) == 0:
            continue
        dead = np.array(r0['flagged'])
        model = model_base if len(dead) == 1 else model_multi
        X_imp, _ = impute_set(model, X, dead, device)
        r1 = flag_module(X_imp, nbr, base, args.z)
        zb = r0['score'][dead]                 # score antes de imputar (muy negativo)
        za = r1['score'][dead]                 # score después (debería subir hacia 0)
        z_before_all.extend(zb.tolist()); z_after_all.extend(za.tolist())
        recovered += int((za > -args.z).sum())  # ya no marcados = reparados
        summary[Path(f).name] = {
            'ich': [int(IDX_TO_ICH[i]) for i in dead],
            'model': 'base' if len(dead) == 1 else 'multidead',
            'z_before_mean': round(float(zb.mean()), 2),
            'z_after_mean': round(float(za.mean()), 2),
            'n_recovered': int((za > -args.z).sum()), 'n_dead': int(len(dead))}
    zb = np.array(z_before_all); za = np.array(z_after_all)
    print(f"  canales detectados: {len(zb)}")
    print(f"  score medio ANTES de imputar:   {zb.mean():+.2f}  (muy negativo = anómalo)")
    print(f"  score medio DESPUÉS de imputar: {za.mean():+.2f}  (~0 = ya no anómalo)")
    print(f"  canales que dejan de marcarse (score > -{args.z}): {recovered}/{len(zb)} "
          f"({100*recovered/max(len(zb),1):.0f}%)")
    print(f"  → la red repara lo que el detector diagnostica.")

    # ── Figuras CUALITATIVAS de una muestra ──
    if args.files:
        sel = [str(Path(BAD_DIR) / (n if n.endswith('.dat') else n+'.dat'))
               for n in args.files.split(',')]
    else:
        con = [f for f in bad_files if Path(f).name in summary]
        rngp = np.random.default_rng(args.seed)
        sel = [con[i] for i in sorted(rngp.permutation(len(con))[:args.sample])]

    out_pdf = OUT_DIR / f'bad_recovery_z{args.z}{tag}.pdf'
    print(f"\nGenerando flood maps de {len(sel)} módulos → {out_pdf.name}")
    with PdfPages(out_pdf) as pdf:
        for f in sel:
            X = load_dat_to_dense(f, max_events=args.max_events)
            r0 = flag_module(X, nbr, base, args.z)
            if len(r0['flagged']) == 0:
                continue
            dead = np.array(r0['flagged'])
            model = model_base if len(dead) == 1 else model_multi
            X_imp, _ = impute_set(model, X, dead, device)
            r1 = flag_module(X_imp, nbr, base, args.z)
            ich = [int(IDX_TO_ICH[i]) for i in dead]

            Horig = flood_map(X, x_sipm, y_sipm, rng)
            Himp  = flood_map(X_imp, x_sipm, y_sipm, rng)
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
            draw_flood(axes[0], Horig, extent,
                       f'ORIGINAL (dead Ich {", ".join(map(str, ich))})')
            draw_flood(axes[1], Himp, extent,
                       f'IMPUTED ({"base" if len(dead)==1 else "multi-dead"} net)')
            draw_flood(axes[2], Hgood, extent, f'GOOD reference ({GOOD_REF})')
            fig.suptitle(f'{Path(f).stem} — self-consistency: score {r0["score"][dead].mean():+.1f} '
                         f'→ {r1["score"][dead].mean():+.1f} after imputation', fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    (OUT_DIR / f'bad_recovery{tag}.json').write_text(
        json.dumps({'z': args.z, 'max_events': args.max_events,
                    'z_before_mean': round(float(zb.mean()), 3),
                    'z_after_mean': round(float(za.mean()), 3),
                    'n_channels': int(len(zb)), 'n_recovered': int(recovered),
                    'per_module': summary}, indent=2), encoding='utf-8')
    print(f"  guardado: {out_pdf.name} y bad_recovery.json")


if __name__ == '__main__':
    main()
