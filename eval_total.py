"""
eval_total.py
=============
EVAL TOTAL: evaluación multicanal de la imputación. En lugar de un solo canal
(Ich=30, como imputation_eval.py), degrada CADA uno de los 61 sensores (uno cada
vez) sobre los 5 archivos de test y calcula el error de posición ΔR y la
recuperación por canal → un porcentaje de recuperación GLOBAL (macro: cada
sensor cuenta igual) + el mapa hexagonal por sensor.

Reutiliza las piezas de imputation_eval.py (carga de modelo, imputación,
centroides). El degradado por canal NO depende del modelo → se calcula una vez
y se comparte entre todos los modelos del barrido.

Salidas (por modelo, en runs/<run>/TOTAL/):
  - recovery_map.png        mapa hexagonal: recuperación p90 por sensor (figura estrella)
  - stats_maps.png          mapas hexagonales: n_mod y MAE_mod por sensor
  - eval_total_report.pdf   PDF multipágina con todo (mapas + distribución + tabla)
  - eval_total_metrics.json métricas por canal + agregados macro
  - subida a W&B como run nuevo '<run>_EVAL_TOTAL' (job_type='eval_total')

Uso:
    conda activate tfm
    python eval_total.py all                        # todos los runs vigentes
    python eval_total.py imputer_hexcnn_s_mse ...   # runs concretos
    python eval_total.py all --out TOTAL_full       # campaña con nombre propio
    python eval_total.py all --events 200000        # limitar eventos por archivo
    python eval_total.py imputer_hexcnn_s_mse --quick   # smoke test (rápido, sin W&B)

Las campañas NUNCA se sobrescriben: si runs/<run>/<out>/eval_total_metrics.json ya
existe, ese run se salta. Para relanzar usa --out con un nombre nuevo. El modo
--quick escribe en TOTAL_quick (no toca los resultados reales).

Autor: Miguel Escudero (TFM)
"""

import re
import sys
import json
import time
import datetime
import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import cKDTree, ConvexHull

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dataset import (load_dat_to_dense, load_positions, get_file_split,
                     N_ACTIVE, IDX_TO_ICH)
# Reutilizamos las piezas ya verificadas del evaluador de un canal
from imputation_eval import load_model, impute_channel, compute_xy, crystal_hull

# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════

RUNS_BASE  = r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\runs'
GOOD_DIR   = r'E:\Datos TFM\Good\Good'
PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'

# Eventos por archivo de test. None = TODOS los eventos del archivo (~0.6-1.3M c/u):
# la métrica "real". Se puede limitar por CLI con --events N (p.ej. --events 200000
# para reproducir las campañas antiguas). OJO: con todos los eventos cada modelo
# tarda ~4-5x más que con 200k.
TEST_MAX_EVENTS = 600_000

USE_WANDB     = True
WANDB_PROJECT = 'TFM-SiPM-imputation'

# ── LÍNEAS del roadmap (punto 5): línea RADIAL (esquina → centro) y un LADO ──
# Un sensor pertenece a la línea si su centro está a < LINE_TOL·pitch del segmento
# ideal. Ambas líneas son cadenas exactas de centros (la radial sigue una de las 6
# direcciones de vecindad de la malla), así que basta una tolerancia pequeña.
LINE_TOL = 0.35
RADIAL_CORNER_ICH = 27   # esquina de partida de la radial: Ich 27-30-2-6-13(centro)
HIST_BINS = 150          # grid de los flood maps (igual que imputation_eval)


def build_lines(x_sipm, y_sipm):
    """
    Define las dos líneas de sensores del análisis por zonas:
      'radial' — de la esquina RADIAL_CORNER_ICH al sensor central (p.ej. 27-30-2-6-13).
      'side'   — la arista completa entre dos esquinas consecutivas.
    Devuelve: {name: {'channels': [idx ordenados], 'pos': {idx: coord}, 'xlabel': str}}
    """
    pts = np.c_[x_sipm, y_sipm]
    pitch = np.median(cKDTree(pts).query(pts, k=2)[0][:, 1])
    r = np.hypot(x_sipm, y_sipm)
    # esquinas del detector = vértices del hull más lejanos, ordenadas por ángulo
    corners = sorted([i for i in ConvexHull(pts).vertices if r[i] > 0.95 * r.max()],
                     key=lambda i: np.arctan2(y_sipm[i], x_sipm[i]))
    A, B = pts[corners[0]], pts[corners[1]]     # lado entre las 2 primeras esquinas
    O = np.zeros(2)                              # centro del detector
    c_rad = next(i for i in corners if IDX_TO_ICH[i] == RADIAL_CORNER_ICH)

    def dseg(p, a, b):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        return float(np.hypot(*(p - (a + t * ab))))

    tol = pitch * LINE_TOL
    side = sorted([i for i in range(N_ACTIVE) if dseg(pts[i], A, B) < tol],
                  key=lambda i: float(np.dot(pts[i] - A, B - A)))
    rad  = sorted([i for i in range(N_ACTIVE) if dseg(pts[i], pts[c_rad], O) < tol],
                  key=lambda i: float(r[i]))
    u = (B - A) / np.linalg.norm(B - A)         # dirección del lado (para la coordenada)
    return {
        'radial': {'channels': rad, 'pos': {i: float(r[i]) for i in rad},
                   'xlabel': 'distance from center [mm]'},
        'side':   {'channels': side, 'pos': {i: float(np.dot(pts[i] - A, u)) for i in side},
                   'xlabel': 'position along side [mm]'},
    }


def discover_runs():
    """Runs vigentes: imputer_* excluyendo backups con sufijo de fecha."""
    out = []
    for d in sorted(Path(RUNS_BASE).glob('imputer_*')):
        if not d.is_dir():
            continue
        if '-06-23' in d.name or re.search(r'_\d\d_\d\d$', d.name):
            continue                      # runs antiguas conservadas como referencia
        if (d / 'best_model.pth').exists() or (d / 'ensemble.json').exists():
            out.append(d.name)
    return out


# ════════════════════════════════════════════════════════════
#  DEGRADADO POR CANAL (independiente del modelo → se calcula 1 vez)
# ════════════════════════════════════════════════════════════

def precompute_degraded(X_list, orig_xy, x_sipm, y_sipm, channels,
                        hist_channels=frozenset(), rng_hist=None):
    """
    Para cada canal: apaga el canal en todos los archivos, recalcula el centroide
    y devuelve las estadísticas del ΔR degradado POOLED sobre los eventos modified.
    Para los canales de las LÍNEAS (hist_channels) acumula además el histograma 2D
    de posición degradada (para el flood-map difference agregado por línea).

    Returns: (stats: ch -> {...cuantiles...}, Hd_map: ch -> hist 2D sumado sobre archivos)
    """
    stats, Hd_map = {}, {}
    for k, ch in enumerate(channels):
        dRs = []
        for X, (ox, oy) in zip(X_list, orig_xy):
            mod = X[:, ch] > 0
            Xd = X.copy()
            Xd[:, ch] = 0.0
            dx, dy = compute_xy(Xd, x_sipm, y_sipm)
            if ch in hist_channels:      # histograma con TODOS los eventos (como el flood map)
                H, _, _ = np.histogram2d(dx, dy, bins=HIST_BINS, range=rng_hist)
                Hd_map[ch] = Hd_map.get(ch, 0) + H
            if mod.sum() == 0:
                continue
            dRs.append(np.sqrt((dx - ox) ** 2 + (dy - oy) ** 2)[mod])
        dR = np.concatenate(dRs) if dRs else np.array([0.0])
        # Guardamos varios cuantiles (no solo p90) para poder estudiar la FORMA de la
        # distribución por canal (p.ej. si en los sensores de borde cambia) sin re-correr.
        qs = np.percentile(dR, [50, 75, 90, 95, 99])
        stats[ch] = {'median': float(qs[0]), 'mean': float(dR.mean()),
                     'p75': float(qs[1]), 'p90': float(qs[2]),
                     'p95': float(qs[3]), 'p99': float(qs[4]), 'n_mod': int(dR.size)}
        print(f"  degradado {k+1:2d}/{len(channels)}  Ich={IDX_TO_ICH[ch]:2d}  "
              f"p90={stats[ch]['p90']:.4f} mm  (n={stats[ch]['n_mod']:,})", flush=True)
    return stats, Hd_map


# ════════════════════════════════════════════════════════════
#  EVAL TOTAL DE UN MODELO
# ════════════════════════════════════════════════════════════

def eval_total_model(run_name, X_list, orig_xy, deg_stats, x_sipm, y_sipm,
                     channels, device, hist_channels=frozenset(), rng_hist=None):
    """Imputa cada canal con el modelo del run y devuelve la tabla por canal
    (+ histogramas 2D imputados de los canales de las líneas)."""
    from imputation_eval import load_ckpt_meta
    ckpt_path = Path(RUNS_BASE) / run_name / 'best_model.pth'
    model = load_model(ckpt_path, device)
    ckpt = load_ckpt_meta(ckpt_path)

    per_channel, Hi_map = [], {}
    t0 = time.time()
    for k, ch in enumerate(channels):
        dRs, errs = [], []
        for X, (ox, oy) in zip(X_list, orig_xy):
            mod = X[:, ch] > 0
            if mod.sum() == 0 and ch not in hist_channels:
                continue
            X_imp, pred = impute_channel(model, X, ch, device)
            ix, iy = compute_xy(X_imp, x_sipm, y_sipm)
            if ch in hist_channels:
                H, _, _ = np.histogram2d(ix, iy, bins=HIST_BINS, range=rng_hist)
                Hi_map[ch] = Hi_map.get(ch, 0) + H
            if mod.sum() == 0:
                continue
            dRs.append(np.sqrt((ix - ox) ** 2 + (iy - oy) ** 2)[mod])
            errs.append(pred[mod] - X[mod, ch])
        dR  = np.concatenate(dRs) if dRs else np.array([0.0])
        err = np.concatenate(errs) if errs else np.array([0.0])

        dg = deg_stats[ch]
        # Mismos cuantiles que en el degradado + recuperación en CADA cuantil, para
        # poder comparar la distribución completa por canal (no solo la cola p90).
        qs = np.percentile(dR, [50, 75, 90, 95, 99])
        imp = {'median': float(qs[0]), 'mean': float(dR.mean()),
               'p75': float(qs[1]), 'p90': float(qs[2]),
               'p95': float(qs[3]), 'p99': float(qs[4])}
        # 'mean' = recuperación GLOBAL ponderada: fracción del desplazamiento total
        # acumulado (suma de todos los ΔR) que elimina la red — cada evento cuenta.
        # Los cuantiles caracterizan puntos de la distribución (p90 = régimen malo).
        rec = {q: (dg[q] - imp[q]) / dg[q] * 100 if dg[q] > 0 else 0.0
               for q in ('median', 'p75', 'p90', 'p95', 'p99', 'mean')}

        per_channel.append({
            'idx': int(ch), 'ich': int(IDX_TO_ICH[ch]), 'n_mod': int(err.size),
            'mae_mod': float(np.abs(err).mean()), 'bias': float(err.mean()),
            'dR_imp': imp,
            'dR_deg': dg,
            'recovery_pct': rec,                       # recuperación en cada cuantil
            'recovery_p90_pct': rec['p90'],            # compatibilidad (titular)
            'recovery_median_pct': rec['median'],
        })
        el = time.time() - t0
        eta = el / (k + 1) * (len(channels) - k - 1)
        print(f"  [{run_name}] {k+1:2d}/{len(channels)}  Ich={IDX_TO_ICH[ch]:2d}  "
              f"MAE={per_channel[-1]['mae_mod']:.3f}  recov_p90={rec['p90']:5.1f}%  "
              f"(ETA {eta/60:.0f} min)", flush=True)

    # ── Agregados MACRO (cada sensor cuenta igual) ──
    recs = np.array([c['recovery_p90_pct'] for c in per_channel])
    maes = np.array([c['mae_mod'] for c in per_channel])
    biases = np.array([c['bias'] for c in per_channel])
    worst = per_channel[int(np.argmin(recs))]
    macro = {
        'recovery_p90_macro_mean':   float(recs.mean()),
        'recovery_p90_macro_median': float(np.median(recs)),
        'recovery_p90_min':          float(recs.min()),
        'recovery_p90_min_ich':      worst['ich'],
        # macro también en mediana y p95 (para comprobar si la conclusión "la cola
        # discrimina, la mediana no" se sostiene con los 61 canales)
        'recovery_median_macro_mean': float(np.mean([c['recovery_pct']['median'] for c in per_channel])),
        'recovery_p95_macro_mean':    float(np.mean([c['recovery_pct']['p95'] for c in per_channel])),
        'recovery_mean_macro_mean':   float(np.mean([c['recovery_pct']['mean'] for c in per_channel])),
        'mae_mod_macro_mean':        float(maes.mean()),
        'bias_macro_mean':           float(biases.mean()),
        'n_channels':                len(per_channel),
    }
    return per_channel, macro, ckpt, Hi_map


# ════════════════════════════════════════════════════════════
#  FIGURAS (mapas hexagonales + distribución + tabla)
# ════════════════════════════════════════════════════════════

def _hex_map(ax, values, x_sipm, y_sipm, hex_r, title, cmap, fmt='{:.0f}',
             vmin=None, vmax=None, cbar_label=''):
    """Mapa del detector: un hexágono por sensor coloreado por 'values'."""
    vals = np.asarray(values, dtype=float)
    ok = ~np.isnan(vals)                       # canales evaluados (NaN = sin evaluar → gris)
    vmin = np.nanmin(vals) if vmin is None else vmin
    vmax = np.nanmax(vals) if vmax is None else vmax
    norm = plt.Normalize(vmin, vmax)
    cm = plt.get_cmap(cmap)
    for i in range(N_ACTIVE):
        fc = cm(norm(vals[i])) if ok[i] else (0.88, 0.88, 0.88, 1.0)
        ax.add_patch(RegularPolygon((x_sipm[i], y_sipm[i]), 6, radius=hex_r,
                     orientation=np.pi / 6, fc=fc, ec='#444', lw=0.7, zorder=2))
        # valor y nº de canal dentro del hexágono (texto pequeño, contraste automático)
        lum = np.dot(fc[:3], (0.299, 0.587, 0.114))
        tc = 'white' if lum < 0.5 else 'black'
        if ok[i]:
            ax.text(x_sipm[i], y_sipm[i] + hex_r * 0.22, fmt.format(vals[i]),
                    ha='center', va='center', fontsize=6.5, color=tc, zorder=3)
        ax.text(x_sipm[i], y_sipm[i] - hex_r * 0.45, f"{IDX_TO_ICH[i]}",
                ha='center', va='center', fontsize=4.6, color=tc, alpha=0.75, zorder=3)
    m = hex_r * 2
    ax.set_xlim(x_sipm.min() - m, x_sipm.max() + m)
    ax.set_ylim(y_sipm.min() - m, y_sipm.max() + m)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm); sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.045, pad=0.03, label=cbar_label)


def make_line_figures(run_name, per_channel, lines, Ho, Hd_map, Hi_map, extent,
                      x_sipm, y_sipm, out_dir):
    """
    Figuras del análisis por LÍNEAS (roadmap punto 5):
      - line_profiles.png     perfil de recuperación (global + p90) a lo largo de cada línea
      - line_floodmap_diff.png  flood-map difference PROMEDIO por línea (degradado e imputado
                                vs original), la figura "de los dipolos" agregada por línea.
    Devuelve (figs para el PDF, rutas de los PNG).
    """
    order = {c['idx']: c for c in per_channel}

    # ── Perfiles a lo largo de la línea ──
    figP, axesP = plt.subplots(1, 2, figsize=(15, 5))
    for ax, (name, L) in zip(axesP, lines.items()):
        chs = [i for i in L['channels'] if i in order]
        xs_ = [L['pos'][i] for i in chs]
        g   = [order[i]['recovery_pct']['mean'] for i in chs]
        p   = [order[i]['recovery_pct']['p90'] for i in chs]
        ax.plot(xs_, g, 'o-',  color='seagreen',  label='global (mean)')
        ax.plot(xs_, p, 's--', color='steelblue', label='tail (p90)')
        for i, xx, yy in zip(chs, xs_, g):
            ax.annotate(f"Ich{IDX_TO_ICH[i]}", (xx, yy), textcoords='offset points',
                        xytext=(0, 9), ha='center', fontsize=7.5)
        ax.set_xlabel(L['xlabel']); ax.set_ylabel('recovery [%]')
        ax.set_title(f'Recovery along the {name}')
        ax.grid(True, alpha=0.3); ax.legend()
    figP.suptitle(run_name, fontsize=12, fontweight='bold')
    pP = out_dir / 'line_profiles.png'
    figP.savefig(pP, dpi=200, bbox_inches='tight')

    # ── Flood-map difference promedio por línea (2 líneas × antes/después) ──
    hx, hy = crystal_hull(x_sipm, y_sipm)
    figD, axesD = plt.subplots(2, 2, figsize=(15.5, 13))
    for row, (name, L) in enumerate(lines.items()):
        chs = [i for i in L['channels'] if i in Hd_map and i in Hi_map]
        n = max(len(chs), 1)
        # promedio sobre los fallos de la línea → "distorsión esperada si falla un
        # sensor de esta línea", antes (deg) y después (imp) de la imputación
        dd = sum(Hd_map[i] for i in chs) / n - Ho
        di = sum(Hi_map[i] for i in chs) / n - Ho
        vmax = float(np.percentile(np.abs(dd), 99.5)) or 1.0
        for col, (diff, ttl) in enumerate([(dd, 'degraded − original'),
                                           (di, 'imputed − original')]):
            ax = axesD[row, col]
            im = ax.imshow(diff.T, origin='lower', extent=extent, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, aspect='equal')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label='Δ counts (mean per failure)')
            ax.plot(hx, hy, color='lime', lw=1.4)
            # sensores de la línea: hexágonos sutiles (finos y discontinuos, como la
            # arquitectura del detector) que no compitan con el mapa de diferencia
            pitch = np.median(cKDTree(np.c_[x_sipm, y_sipm]).query(
                np.c_[x_sipm, y_sipm], k=2)[0][:, 1])
            for i in chs:
                ax.add_patch(RegularPolygon((x_sipm[i], y_sipm[i]), 6,
                             radius=pitch / np.sqrt(3) * 0.97, orientation=np.pi / 6,
                             facecolor='none', edgecolor='0.25', lw=0.8,
                             linestyle='--', zorder=4))
            ax.set_title(f'{name} — {ttl}')
            ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')
    figD.suptitle(f'Average flood-map distortion when a sensor of the line fails — {run_name}',
                  fontsize=12, fontweight='bold')
    pD = out_dir / 'line_floodmap_diff.png'
    figD.savefig(pD, dpi=200, bbox_inches='tight')
    print(f"  Guardado: {pP}\n  Guardado: {pD}")
    return [figP, figD], [pP, pD]


def make_figures(run_name, per_channel, macro, x_sipm, y_sipm, out_dir, extra_figs=None):
    """Genera los 2 PNG + el PDF multipágina (con las figuras extra de líneas al final).
    Devuelve las rutas de los PNG."""
    spacing = np.median(cKDTree(np.c_[x_sipm, y_sipm]).query(np.c_[x_sipm, y_sipm], k=2)[0][:, 1])
    hex_r = spacing / np.sqrt(3) * 0.97

    order = {c['idx']: c for c in per_channel}
    # NaN para canales no evaluados (p.ej. modo --quick) → salen en gris en el mapa
    recs  = [order[i]['recovery_p90_pct'] if i in order else np.nan for i in range(N_ACTIVE)]
    recsm = [order[i]['recovery_pct']['mean'] if i in order else np.nan for i in range(N_ACTIVE)]
    maes  = [order[i]['mae_mod'] if i in order else np.nan for i in range(N_ACTIVE)]
    nmods = [order[i]['n_mod'] if i in order else np.nan for i in range(N_ACTIVE)]

    # ── PNG 1 (figura estrella): recuperación GLOBAL (media) + de cola (p90) ──
    # media = fracción del desplazamiento TOTAL acumulado que elimina la red (todos
    # los eventos cuentan con su peso); p90 = cuánto se encoge el régimen malo.
    fig1, axes1 = plt.subplots(1, 2, figsize=(17.5, 7.8))
    _hex_map(axes1[0], recsm, x_sipm, y_sipm, hex_r,
             f'GLOBAL recovery (of mean ΔR) per failed sensor\n'
             f'macro mean {macro["recovery_mean_macro_mean"]:.1f}%',
             cmap='RdYlGn', fmt='{:.0f}', vmin=0, vmax=100, cbar_label='recovery of mean [%]')
    _hex_map(axes1[1], recs, x_sipm, y_sipm, hex_r,
             f'TAIL recovery (p90) per failed sensor\n'
             f'macro mean {macro["recovery_p90_macro_mean"]:.1f}%  ·  '
             f'worst Ich={macro["recovery_p90_min_ich"]} ({macro["recovery_p90_min"]:.1f}%)',
             cmap='RdYlGn', fmt='{:.0f}', vmin=0, vmax=100, cbar_label='recovery p90 [%]')
    fig1.suptitle(run_name, fontsize=13, fontweight='bold')
    p1 = out_dir / 'recovery_map.png'
    fig1.savefig(p1, dpi=200, bbox_inches='tight')

    # ── PNG 2: estadística por sensor (n_mod + MAE) ──
    fig2, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    _hex_map(axes[0], np.array(nmods, dtype=float) / 1000, x_sipm, y_sipm, hex_r,
             'Modified events per sensor (statistics)', cmap='viridis',
             fmt='{:.0f}k', cbar_label='n_mod [×1000]')
    _hex_map(axes[1], maes, x_sipm, y_sipm, hex_r,
             'Imputation MAE per sensor (ADC, modified)', cmap='magma_r',
             fmt='{:.2f}', cbar_label='MAE_mod [ADC]')
    fig2.suptitle(run_name, fontsize=12, fontweight='bold')
    p2 = out_dir / 'stats_maps.png'
    fig2.savefig(p2, dpi=200, bbox_inches='tight')

    # ── Página extra: distribución + barras por canal (global vs cola) ──
    fig3, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    axes[0].hist([c['recovery_pct']['mean'] for c in per_channel], bins=20,
                 color='seagreen', alpha=0.65, edgecolor='white', label='global (mean)')
    axes[0].hist([c['recovery_p90_pct'] for c in per_channel], bins=20,
                 color='steelblue', alpha=0.65, edgecolor='white', label='tail (p90)')
    axes[0].axvline(macro['recovery_mean_macro_mean'], color='seagreen', ls='--')
    axes[0].axvline(macro['recovery_p90_macro_mean'], color='steelblue', ls='--')
    axes[0].set_xlabel('recovery [%]'); axes[0].set_ylabel('sensors')
    axes[0].set_title('Distribution of per-sensor recovery'); axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # barras ordenadas por la recuperación GLOBAL, con el p90 superpuesto como punto
    srt = sorted(per_channel, key=lambda c: c['recovery_pct']['mean'])
    xs_ = np.arange(len(srt))
    axes[1].bar(xs_, [c['recovery_pct']['mean'] for c in srt],
                color='seagreen', alpha=0.8, label='global (mean)')
    axes[1].plot(xs_, [c['recovery_p90_pct'] for c in srt], 'o', ms=3.5,
                 color='steelblue', label='tail (p90)')
    axes[1].set_xticks(xs_); axes[1].set_xticklabels([str(c['ich']) for c in srt], fontsize=6)
    axes[1].set_xlabel('Ich (sorted by global)'); axes[1].set_ylabel('recovery [%]')
    axes[1].set_title('Per-sensor recovery, worst → best'); axes[1].legend()
    axes[1].grid(True, axis='y', alpha=0.3)
    fig3.suptitle(run_name, fontsize=12, fontweight='bold')

    # ── Página tabla ──
    fig4, ax4 = plt.subplots(figsize=(12, 14))
    ax4.axis('off')
    rows = [[c['ich'], f"{c['n_mod']:,}", f"{c['mae_mod']:.3f}", f"{c['bias']:+.3f}",
             f"{c['dR_deg']['p90']:.3f}", f"{c['dR_imp']['p90']:.3f}",
             f"{c['recovery_pct']['mean']:.1f}%", f"{c['recovery_p90_pct']:.1f}%"] for c in srt]
    tbl = ax4.table(cellText=rows,
                    colLabels=['Ich', 'n_mod', 'MAE_mod', 'bias',
                               'dR_deg p90', 'dR_imp p90', 'rec. global', 'rec. p90'],
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.15)
    ax4.set_title(f'Per-channel table (worst → best) — {run_name}', fontweight='bold')

    # ── PDF multipágina con todo (incluidas las figuras de líneas) ──
    all_figs = [fig1, fig2, fig3, fig4] + list(extra_figs or [])
    pdf_path = out_dir / 'eval_total_report.pdf'
    with PdfPages(pdf_path) as pdf:
        for f in all_figs:
            pdf.savefig(f, bbox_inches='tight')
    for f in all_figs:
        plt.close(f)
    print(f"  Guardado: {p1}\n  Guardado: {p2}\n  Guardado: {pdf_path}")
    return [p1, p2]


# ════════════════════════════════════════════════════════════
#  W&B
# ════════════════════════════════════════════════════════════

def log_to_wandb(run_name, macro, pngs, meta, line_metrics=None):
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb no instalado; no se sube el EVAL TOTAL.")
        return
    campaign = meta.get('campaign', 'TOTAL')
    suffix = '' if campaign == 'TOTAL' else f"_{campaign.replace('TOTAL_', '')}"
    run = wandb.init(project=WANDB_PROJECT, name=f'{run_name}_EVAL_TOTAL{suffix}',
                     job_type='eval_total',
                     config={'run': run_name, 'arch': meta.get('arch'),
                             'n_channels': macro['n_channels'],
                             'campaign': campaign,
                             'max_events_per_file': meta.get('max_events_per_file')})
    run.summary.update({f'total/{k}': v for k, v in macro.items()})
    for name, lm in (line_metrics or {}).items():
        run.summary.update({f'line_{name}/{k}': v for k, v in lm.items()
                            if not isinstance(v, list)})
    run.log({p.stem: wandb.Image(str(p)) for p in pngs})
    print(f"  ✓ EVAL TOTAL subido a W&B: {run.url}")
    run.finish()


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    # ── CLI: runs + flags (--quick | --no-wandb | --out NAME | --events N) ──
    argv = sys.argv[1:]
    quick = '--quick' in argv
    no_wandb = '--no-wandb' in argv or quick

    def flag_value(name, default):
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default

    # Nombre de la campaña = subcarpeta de salida dentro de cada run. NUNCA se
    # sobrescribe una campaña existente → cada relanzamiento usa su propio --out.
    out_name = flag_value('--out', 'TOTAL_quick' if quick else 'TOTAL')
    ev_arg   = flag_value('--events', None)
    max_ev_cli = int(ev_arg) if ev_arg else None

    skip = set()
    args = []
    it = iter(range(len(argv)))
    for j in it:
        a = argv[j]
        if a in ('--out', '--events'):
            skip.add(j); skip.add(j + 1)
        elif a.startswith('--'):
            skip.add(j)
    args = [a for j, a in enumerate(argv) if j not in skip]

    if not args or args[0] == 'all':
        runs = discover_runs()
    else:
        runs = args
    print(f"EVAL TOTAL sobre {len(runs)} run(s): {runs}")
    print(f"  campaña de salida: runs/<run>/{out_name}/")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_sipm, y_sipm = load_positions(PSIPM_PATH)
    _, _, test_files = get_file_split(GOOD_DIR)

    # ── Líneas del análisis por zonas (apotema + lado) ──
    lines = build_lines(x_sipm, y_sipm)
    line_channels = frozenset(i for L in lines.values() for i in L['channels'])
    for name, L in lines.items():
        print(f"  línea '{name}': {[f'Ich{IDX_TO_ICH[i]}' for i in L['channels']]}")

    max_ev = 20_000 if quick else (max_ev_cli if max_ev_cli else TEST_MAX_EVENTS)
    # en --quick evaluamos SOLO los canales de las líneas (prueba el pipeline completo)
    channels = sorted(line_channels) if quick else list(range(N_ACTIVE))
    if quick:
        print(f"MODO --quick: 20k eventos/archivo, {len(channels)} canales (líneas), sin W&B")

    # ── Cargar los archivos de test UNA vez (compartidos por todos los modelos) ──
    ev_str = f"{max_ev:,}" if max_ev else 'TODOS los'
    print(f"Cargando {len(test_files)} archivos de test ({ev_str} eventos c/u)...")
    X_list = [load_dat_to_dense(f, max_events=max_ev) for f in test_files]
    orig_xy = [compute_xy(X, x_sipm, y_sipm) for X in X_list]
    print(f"  total eventos: {sum(len(X) for X in X_list):,}")

    # ── Flood map ORIGINAL (para los mapas de diferencia por línea) ──
    rng_hist = [[x_sipm.min() - 2, x_sipm.max() + 2], [y_sipm.min() - 2, y_sipm.max() + 2]]
    Ho, xe, ye = 0, None, None
    for ox, oy in orig_xy:
        H, xe, ye = np.histogram2d(ox, oy, bins=HIST_BINS, range=rng_hist)
        Ho = Ho + H
    extent = [xe[0], xe[-1], ye[0], ye[-1]]

    # ── Degradado por canal (una vez, compartido por todos los modelos) ──
    print("\nPrecalculando el degradado por canal (independiente del modelo)...")
    deg_stats, Hd_map = precompute_degraded(X_list, orig_xy, x_sipm, y_sipm, channels,
                                            hist_channels=line_channels, rng_hist=rng_hist)

    # ── Bucle de modelos ──
    fallidos = []
    for run_name in runs:
        print(f"\n{'='*64}\nEVAL TOTAL: {run_name}\n{'='*64}")
        out_dir = Path(RUNS_BASE) / run_name / out_name

        # El run debe existir y tener pesos. En una tanda desatendida es normal que
        # un entrenamiento previo no haya terminado; ese caso se salta con un aviso
        # legible en vez de abortar la tanda entera con una traza de FileNotFound.
        run_dir = Path(RUNS_BASE) / run_name
        if not run_dir.exists():
            print(f"  NO EXISTE la carpeta {run_dir} → salto este run.")
            fallidos.append((run_name, 'carpeta inexistente')); continue
        if not (run_dir / 'best_model.pth').exists() and not (run_dir / 'ensemble.json').exists():
            hay = sorted(p.name for p in run_dir.iterdir())
            print(f"  SIN CHECKPOINT en {run_dir} → salto este run.\n"
                  f"  (el entrenamiento no llegó a guardar best_model.pth; contiene: {hay})")
            fallidos.append((run_name, 'sin best_model.pth')); continue

        # PROTECCIÓN: nunca sobrescribir una campaña ya evaluada. Para relanzar,
        # usa otro nombre: python eval_total.py all --out TOTAL_full
        if (out_dir / 'eval_total_metrics.json').exists():
            print(f"  YA EXISTE {out_dir / 'eval_total_metrics.json'} → salto este run.\n"
                  f"  (para una campaña nueva usa --out <NOMBRE>)")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            per_channel, macro, ckpt, Hi_map = eval_total_model(
                run_name, X_list, orig_xy, deg_stats, x_sipm, y_sipm, channels, device,
                hist_channels=line_channels, rng_hist=rng_hist)
        except Exception as e:
            # Un run que falle no debe tumbar los demás de la tanda.
            print(f"  ERROR evaluando {run_name}: {type(e).__name__}: {e}")
            fallidos.append((run_name, f'{type(e).__name__}: {e}'))
            continue

        print(f"\n  MACRO: recovery GLOBAL(mean)={macro['recovery_mean_macro_mean']:.1f}%  "
              f"p90={macro['recovery_p90_macro_mean']:.1f}%  "
              f"peor(p90)=Ich {macro['recovery_p90_min_ich']} ({macro['recovery_p90_min']:.1f}%)")

        # ── Métricas agregadas por LÍNEA (media de sus canales) ──
        order = {c['idx']: c for c in per_channel}
        line_metrics = {}
        for name, L in lines.items():
            chs = [i for i in L['channels'] if i in order]
            line_metrics[name] = {
                'channels_ich': [int(IDX_TO_ICH[i]) for i in chs],
                'recovery_mean_avg': float(np.mean([order[i]['recovery_pct']['mean'] for i in chs])),
                'recovery_p90_avg':  float(np.mean([order[i]['recovery_pct']['p90'] for i in chs])),
                'mae_mod_avg':       float(np.mean([order[i]['mae_mod'] for i in chs])),
            }
            print(f"  línea '{name}': recovery global={line_metrics[name]['recovery_mean_avg']:.1f}%  "
                  f"p90={line_metrics[name]['recovery_p90_avg']:.1f}%")

        metrics = {
            'run': run_name,
            'arch': ckpt.get('arch'), 'train_epoch': ckpt.get('epoch'),
            'generated': datetime.datetime.now().isoformat(timespec='seconds'),
            'test_files': [f.name for f in test_files],
            'max_events_per_file': max_ev if max_ev else 'all',
            'campaign': out_name,
            'macro': macro,
            'lines': line_metrics,
            'per_channel': per_channel,
        }
        with open(out_dir / 'eval_total_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  Métricas: {out_dir / 'eval_total_metrics.json'}")

        line_figs, line_pngs = make_line_figures(
            run_name, per_channel, lines, Ho, Hd_map, Hi_map, extent,
            x_sipm, y_sipm, out_dir)
        pngs = make_figures(run_name, per_channel, macro, x_sipm, y_sipm, out_dir,
                            extra_figs=line_figs) + line_pngs

        if USE_WANDB and not no_wandb:
            log_to_wandb(run_name, macro, pngs, metrics, line_metrics)

    if fallidos:
        print(f"\n{'!'*64}")
        print(f"{len(fallidos)} run(s) NO evaluados:")
        for r, motivo in fallidos:
            print(f"  - {r}: {motivo}")
        print('!' * 64)

    print("\n✓ EVAL TOTAL terminado.")


if __name__ == '__main__':
    main()
