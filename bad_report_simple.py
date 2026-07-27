"""
bad_report_simple.py  (punto 2 — informe visual de la detección en los BAD)
===========================================================================
Versión simplificada del bad_report v2.0. Como ya tenemos un ESTADÍSTICO de
detección validado (bad_detect.py, z-score robusto por canal), no hace falta
enseñar histogramas ni fracciones crudas para justificar el criterio: el z-score
ES el criterio. Este informe solo pinta, por cada módulo BAD, un mapa hexagonal
del detector con los canales marcados resaltados.

Cada panel:
  - Un hexágono por SiPM, coloreado por su score = min(z_frac, z_ratio)
    (rojo = por debajo de lo esperado en su posición = sospechoso; azul = normal).
  - Los canales marcados (score < −Z) llevan borde rojo grueso y su Ich anotado.
  - Título: nombre del módulo + Ich marcados.

Usa el baseline ya calculado por bad_detect.py (reports/good_baseline.npz). Si no
existe, lo reconstruye desde los Good.

Uso:
    conda activate tfm
    python bad_report_simple.py                      # todos los BAD, Z=2.5
    python bad_report_simple.py --z 2.5 --sample 12  # 12 módulos aleatorios
    python bad_report_simple.py --cols 4             # 4 columnas por página
    python bad_report_simple.py --max-events 150000

Autor: Miguel Escudero (TFM)
"""

import sys
import glob
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dataset import load_dat_to_dense, load_positions, N_ACTIVE, IDX_TO_ICH
from hex_geometry import get_neighbor_matrix
from bad_detect import module_stats, flag_module, build_baseline

BAD_DIR    = r'E:\Datos TFM\Bad\Bad'
GOOD_DIR   = r'E:\Datos TFM\Good\Good'
PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'
OUT_DIR    = Path(r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\reports')
BASE_NPZ   = OUT_DIR / 'good_baseline.npz'

VMIN, VMAX = -6.0, 2.0     # rango de color del z-score (rojo muy negativo → azul normal)


def load_or_build_baseline(nbr, max_events):
    """Carga el baseline guardado por bad_detect.py; si no está, lo reconstruye."""
    if BASE_NPZ.exists():
        d = np.load(BASE_NPZ)
        print(f"Baseline cargado de {BASE_NPZ.name} ({int(d['n_modules'])} módulos Good)")
        return {k: d[k] for k in ('frac_median', 'frac_spread', 'ratio_median', 'ratio_spread')}
    print("No hay baseline guardado → reconstruyendo desde los Good...")
    good = sorted(glob.glob(str(Path(GOOD_DIR) / '*.dat')))
    return build_baseline(good, nbr, max_events)


def draw_module(ax, score, flagged, x_sipm, y_sipm, hex_r, norm, cmap, title):
    """Pinta un módulo: hexágonos coloreados por score, marcados con borde rojo."""
    fset = set(int(i) for i in flagged)
    for i in range(N_ACTIVE):
        is_flag = i in fset
        ax.add_patch(RegularPolygon(
            (x_sipm[i], y_sipm[i]), 6, radius=hex_r, orientation=np.pi/6,
            facecolor=cmap(norm(score[i])),
            edgecolor='red' if is_flag else '0.4',
            linewidth=2.4 if is_flag else 0.4, zorder=3 if is_flag else 2))
        if is_flag:
            ax.text(x_sipm[i], y_sipm[i], str(IDX_TO_ICH[i]), ha='center', va='center',
                    fontsize=6, fontweight='bold', zorder=4)
    m = hex_r * 1.2
    ax.set_xlim(x_sipm.min()-m, x_sipm.max()+m)
    ax.set_ylim(y_sipm.min()-m, y_sipm.max()+m)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--z', type=float, default=2.0)
    ap.add_argument('--max-events', type=int, default=150_000)
    ap.add_argument('--sample', type=int, default=None, help='nº de módulos aleatorios (por defecto TODOS)')
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', type=str, default='', help='identificador para el archivo de salida')
    args = ap.parse_args()
    tag = f'_{args.tag}' if args.tag else ''

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_sipm, y_sipm = load_positions(PSIPM_PATH)
    nbr = get_neighbor_matrix(PSIPM_PATH)
    base = load_or_build_baseline(nbr, args.max_events)

    pitch = np.median(np.sort(np.hypot(x_sipm[:, None]-x_sipm, y_sipm[:, None]-y_sipm), axis=1)[:, 1])
    hex_r = pitch / np.sqrt(3) * 0.97
    norm = TwoSlopeNorm(vmin=VMIN, vcenter=0.0, vmax=VMAX)
    cmap = plt.cm.RdBu

    bad_files = sorted(glob.glob(str(Path(BAD_DIR) / '*.dat')))
    if args.sample:
        rng = np.random.default_rng(args.seed)
        bad_files = [bad_files[i] for i in sorted(rng.permutation(len(bad_files))[:args.sample])]
    print(f"Generando informe de {len(bad_files)} módulos BAD (Z={args.z})...")

    cols = args.cols
    rows_per_page = 4
    per_page = cols * rows_per_page
    out_pdf = OUT_DIR / f'bad_detection_report_z{args.z}{tag}.pdf'

    n_flagged_total = 0
    with PdfPages(out_pdf) as pdf:
        page = []
        for k, f in enumerate(bad_files):
            X = load_dat_to_dense(f, max_events=args.max_events)
            r = flag_module(X, nbr, base, args.z)
            ich = [int(IDX_TO_ICH[i]) for i in r['flagged']]
            n_flagged_total += len(ich)
            sh = r['module_shift_frac']
            qa = f"  [shift {sh:+.1f}]" if sh < -1.0 else ""
            title = Path(f).stem + qa + (f"\nIch: {', '.join(map(str, ich))}" if ich else "\n(sin fallos)")
            page.append((r['score'], r['flagged'], title))

            if len(page) == per_page or k == len(bad_files) - 1:
                fig, axes = plt.subplots(rows_per_page, cols,
                                         figsize=(cols*3.2, rows_per_page*3.2))
                axes = np.atleast_1d(axes).ravel()
                for ax in axes:
                    ax.axis('off')
                for ax, (sc, fl, ti) in zip(axes, page):
                    draw_module(ax, sc, fl, x_sipm, y_sipm, hex_r, norm, cmap, ti)
                sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.02,
                             label='detector score  (z = std below expected)')
                fig.suptitle(f'BAD detection report — z-score per SiPM  (flagged if score < −{args.z})',
                             fontsize=12)
                pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
                page = []
            if (k+1) % 10 == 0:
                print(f"  {k+1}/{len(bad_files)} módulos")

    print(f"\n  {n_flagged_total} canales marcados en {len(bad_files)} módulos.")
    print(f"  informe guardado en {out_pdf}")


if __name__ == '__main__':
    main()
