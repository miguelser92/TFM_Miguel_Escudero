"""
bad_detect.py  (punto 2 del roadmap — estadístico base de detección de canales malos)
====================================================================================
Detecta sensores averiados en los módulos BAD (no etiquetados) con un estadístico
CONSCIENTE DE LA POSICIÓN, en vez del umbral global del bad_report v2.0.

El problema del umbral global: los sensores de BORDE reciben menos luz (adhesivo
negro) → menos actividad NATURAL. Un umbral único ("< 15% de la mediana = muerto")
confunde bordes sanos con muertos y se pierde canales de borde que rinden bajo.

La solución (este script): aprender de los ~149 módulos GOOD el comportamiento
ESPERADO de CADA canal en su posición (con su dispersión natural), y en un BAD
marcar el canal que se desvía mucho de SU propio baseline, no del global.

Dos estadísticos complementarios, ambos posición-conscientes:
  1. frac_active   : fracción de eventos en que dispara el canal. Baseline por canal
                     desde los Good (mediana + MAD robusta). z-score por canal.
  2. neighbor_ratio: carga media del canal / carga media de sus vecinos. Se
                     auto-normaliza por posición (compara local con local); un
                     muerto lee ~0 mientras sus vecinos están calientes. También
                     con baseline por canal desde los Good.

Un canal se marca MALO si cae muy por debajo de su baseline en cualquiera de los dos.

Uso:
    conda activate tfm
    python bad_detect.py                       # baseline de Good + barrido de todos los Bad
    python bad_detect.py --build-only          # solo construye y guarda el baseline
    python bad_detect.py --max-events 150000   # cap de eventos por archivo
    python bad_detect.py --z 4.0               # umbral del z-score (por defecto 4)

Autor: Miguel Escudero (TFM)
"""

import sys
import glob
import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dataset import load_dat_to_dense, load_positions, N_ACTIVE, IDX_TO_ICH
from hex_geometry import get_neighbor_matrix

GOOD_DIR   = r'E:\Datos TFM\Good\Good'
BAD_DIR    = r'E:\Datos TFM\Bad\Bad'
PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'
OUT_DIR    = Path(r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\reports')

Z_THRESH   = 4.0        # nº de MAD por debajo del baseline para marcar un canal como malo
MAX_EVENTS = 150_000    # eventos por archivo (frac_active y carga media son estables)
EPS        = 1e-9


# ════════════════════════════════════════════════════════════
#  ESTADÍSTICOS POR MÓDULO
# ════════════════════════════════════════════════════════════

def module_stats(X, nbr):
    """
    Estadísticos por canal de UN módulo (X: (N,61) cargas crudas clipeadas).

    frac_active   (61,) fracción de eventos en que el canal tiene carga > 0.
    neigh_ratio   (61,) carga media del canal / carga media de sus vecinos reales.
                  Auto-normalizado por posición: ~1 en un canal sano, ~0 en un muerto.
    """
    active      = X > 0
    frac_active = active.mean(axis=0)                       # (61,)
    mean_charge = X.mean(axis=0)                            # carga media (incl. ceros)

    neigh_ratio = np.zeros(N_ACTIVE, dtype=np.float64)
    for i in range(N_ACTIVE):
        vecinos = [int(j) for j in nbr[i] if j >= 0]
        mn = mean_charge[vecinos].mean() if vecinos else 0.0
        neigh_ratio[i] = mean_charge[i] / (mn + EPS)
    return {'frac_active': frac_active, 'neigh_ratio': neigh_ratio,
            'mean_charge': mean_charge}


# ════════════════════════════════════════════════════════════
#  BASELINE DESDE LOS GOOD
# ════════════════════════════════════════════════════════════

def robust_center_spread(mat):
    """
    Por columna (canal): mediana y MAD robusta (=1.4826·mediana de |x−mediana|,
    que estima la desviación típica sin que la arrastren los outliers). Devuelve
    (median (61,), spread (61,)) con un suelo en el spread para no dividir por ~0.
    """
    med  = np.median(mat, axis=0)
    mad  = np.median(np.abs(mat - med), axis=0) * 1.4826
    mad  = np.maximum(mad, 0.01 * np.median(med) + EPS)     # suelo → evita z infinitos
    return med, mad


def build_baseline(good_files, nbr, max_events):
    """Recorre los Good y construye el baseline por canal (mediana + spread) de cada estadístico."""
    fracs, ratios = [], []
    print(f"Construyendo baseline desde {len(good_files)} módulos GOOD...")
    for k, f in enumerate(good_files):
        X = load_dat_to_dense(f, max_events=max_events)
        s = module_stats(X, nbr)
        fracs.append(s['frac_active']); ratios.append(s['neigh_ratio'])
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(good_files)} módulos")
    fracs, ratios = np.array(fracs), np.array(ratios)       # (n_mod, 61)

    fm, fs = robust_center_spread(fracs)
    rm, rs = robust_center_spread(ratios)
    return {
        'n_modules': len(good_files),
        'frac_median': fm, 'frac_spread': fs,
        'ratio_median': rm, 'ratio_spread': rs,
    }


# ════════════════════════════════════════════════════════════
#  APLICAR A UN MÓDULO (Good o Bad)
# ════════════════════════════════════════════════════════════

def flag_module(X, nbr, base, z_thresh):
    """
    z-score robusto por canal (negativo = por debajo de lo esperado en SU posición).
    Marca MALO el canal que cae > z_thresh MADs por debajo del baseline en frac_active
    o en neighbor_ratio. Devuelve el resumen por canal + los índices marcados.
    """
    s = module_stats(X, nbr)
    z_frac  = (s['frac_active'] - base['frac_median'])  / base['frac_spread']
    z_ratio = (s['neigh_ratio'] - base['ratio_median']) / base['ratio_spread']
    # Señal de "muerto": muy por debajo (una cola, negativa) en cualquiera de los dos
    score  = np.minimum(z_frac, z_ratio)                    # el más anómalo de los dos
    flagged = np.where(score < -z_thresh)[0]
    flagged = flagged[np.argsort(score[flagged])]           # más muerto primero
    return {'z_frac': z_frac, 'z_ratio': z_ratio, 'score': score,
            'frac_active': s['frac_active'], 'neigh_ratio': s['neigh_ratio'],
            'flagged': flagged}


# ════════════════════════════════════════════════════════════
#  FIGURA DEL BASELINE (para "entender" los Good)
# ════════════════════════════════════════════════════════════

def baseline_figure(base, x_sipm, y_sipm, out_path):
    """Mapa hexagonal de la actividad esperada + correlación con la distancia al centro."""
    pitch = np.median(np.sort(np.hypot(x_sipm[:, None] - x_sipm,
                                       y_sipm[:, None] - y_sipm), axis=1)[:, 1])
    hex_r = pitch / np.sqrt(3) * 0.97
    r_center = np.hypot(x_sipm, y_sipm)                     # distancia al centro (mm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    vals = base['frac_median'] * 100
    norm = plt.Normalize(vals.min(), vals.max())
    for i in range(N_ACTIVE):
        ax1.add_patch(RegularPolygon((x_sipm[i], y_sipm[i]), 6, radius=hex_r,
                      orientation=np.pi/6, facecolor=plt.cm.viridis(norm(vals[i])),
                      edgecolor='k', lw=0.4))
    ax1.set_xlim(x_sipm.min()-hex_r, x_sipm.max()+hex_r)
    ax1.set_ylim(y_sipm.min()-hex_r, y_sipm.max()+hex_r)
    ax1.set_aspect('equal'); ax1.axis('off')
    ax1.set_title('Expected active fraction per SiPM (Good baseline)')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'),
                 ax=ax1, fraction=0.046, label='Active fraction (%)')

    ax2.scatter(r_center, vals, s=25, c='#c0392b')
    ax2.set_xlabel('Distance to detector center (mm)')
    ax2.set_ylabel('Expected active fraction (%)')
    ax2.set_title('Edge channels are naturally less active\n(why a global threshold fails)')
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return r_center, vals


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-events', type=int, default=MAX_EVENTS)
    ap.add_argument('--z', type=float, default=Z_THRESH)
    ap.add_argument('--build-only', action='store_true')
    ap.add_argument('--max-good', type=int, default=None, help='limitar nº de módulos Good (prueba)')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_sipm, y_sipm = load_positions(PSIPM_PATH)
    nbr = get_neighbor_matrix(PSIPM_PATH)

    good_files = sorted(glob.glob(str(Path(GOOD_DIR) / '*.dat')))
    if args.max_good:
        good_files = good_files[:args.max_good]

    base = build_baseline(good_files, nbr, args.max_events)

    # Guardar el baseline (reutilizable por los siguientes pasos: validación e imputador-detector)
    np.savez(OUT_DIR / 'good_baseline.npz', **{k: v for k, v in base.items()
             if isinstance(v, np.ndarray)}, n_modules=base['n_modules'])
    r_center, act = baseline_figure(base, x_sipm, y_sipm, OUT_DIR / 'good_baseline.png')

    # ── "Entender" el baseline: el borde es menos activo ──
    print("\n=== BASELINE DE LOS GOOD (entendiendo los datos) ===")
    corr = np.corrcoef(r_center, act)[0, 1]
    print(f"  actividad esperada: centro ~{act[r_center < 5].mean():.1f}%  vs  "
          f"borde ~{act[r_center > 15].mean():.1f}%")
    print(f"  correlación actividad↔distancia al centro: {corr:+.2f} "
          f"(negativa = el borde dispara menos → un umbral GLOBAL falla)")
    print(f"  frac_active esperada: min={act.min():.1f}%  max={act.max():.1f}%  "
          f"(factor {act.max()/max(act.min(),0.1):.1f}× entre el más y el menos activo)")
    print(f"  baseline guardado en {OUT_DIR/'good_baseline.npz'} y figura good_baseline.png")

    if args.build_only:
        return

    # ── Aplicar a los BAD ──
    bad_files = sorted(glob.glob(str(Path(BAD_DIR) / '*.dat')))
    print(f"\n=== DETECCIÓN EN {len(bad_files)} MÓDULOS BAD (z > {args.z}) ===")
    report = {}
    n_con, n_sin = 0, 0
    for f in bad_files:
        X = load_dat_to_dense(f, max_events=args.max_events)
        r = flag_module(X, nbr, base, args.z)
        ich = [int(IDX_TO_ICH[i]) for i in r['flagged']]
        report[Path(f).name] = [
            {'ich': int(IDX_TO_ICH[i]), 'idx': int(i),
             'z_frac': round(float(r['z_frac'][i]), 1),
             'z_ratio': round(float(r['z_ratio'][i]), 1)}
            for i in r['flagged']
        ]
        if ich:
            n_con += 1
            det = '  '.join(f"Ich{IDX_TO_ICH[i]}(z={r['score'][i]:.0f})" for i in r['flagged'])
            print(f"  {Path(f).name:16} → {det}")
        else:
            n_sin += 1
    print(f"\n  {n_con} módulos con canal(es) marcado(s), {n_sin} sin marcar.")
    (OUT_DIR / 'bad_detection.json').write_text(
        json.dumps({'z_thresh': args.z, 'max_events': args.max_events,
                    'n_good_modules': base['n_modules'], 'detections': report}, indent=2),
        encoding='utf-8')
    print(f"  informe guardado en {OUT_DIR/'bad_detection.json'}")


if __name__ == '__main__':
    main()
