"""
imputation_eval.py
==================
Evaluación del modelo de imputación entrenado.

Permite:
  - Cargar el .pth guardado por train.py.
  - Imputar un canal sobre datos Good (con ground truth) o Bad (canal muerto real).
  - Métricas en el canal imputado, estratificadas modified / non-modified (Good).
  - Comparación visual del flood map: original vs canal apagado vs canal imputado.

Flood map: posición XY por centro de gravedad Rch² con las posiciones reales del
psipm.tsv. El centroide divide por la suma de pesos, así que es invariante a la
escala por evento (da igual normalizado o crudo).

Uso:
    conda activate tfm
    python imputation_eval.py

Ajusta la sección CONFIG.

Autor: Miguel Escudero (TFM)
"""

import sys
import json
import datetime
import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dataset import (
    load_dat_to_dense, load_positions, get_file_split,
    N_ACTIVE, ICH_TO_IDX, IDX_TO_ICH,
)
from model import get_model


# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════

PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'

# Solo cambias RUN_NAME (la carpeta de la arquitectura). El checkpoint y la carpeta
# eval/ se derivan solos → no hay que tocar rutas a mano y eval va dentro de cada run.
RUNS_BASE  = r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\runs'
RUN_NAME   = 'imputer_hexcnn' #Modificar segun necesites carpeta
RUN_DIR    = Path(RUNS_BASE) / RUN_NAME
CKPT_PATH  = RUN_DIR / 'best_model.pth'
OUT_DIR    = RUN_DIR / 'eval'

# El split (train/val/test) sale de dataset.get_file_split → el evaluador usa los
# MISMOS archivos de test que train.py reservó. La demo Good se hace sobre test_files[0].
GOOD_DIR    = r'E:\Datos TFM\Good\Good'
GOOD_ICH    = 30          # canal físico (Ich) a apagar/imputar en la demo Good
MAX_EVENTS  = 400_000     # eventos para la demo Good de un solo archivo

TEST_MAX_EVENTS = 200_000  # eventos por archivo de test (son varios, agregados)

# Demo Bad (opcional): imputar el canal muerto real de un archivo Bad
BAD_FILE    = r'E:\Datos TFM\Bad\Bad\datas016.dat'
BAD_ICH     = 59          # canal muerto conocido en datas016 (del bad_report)


# ════════════════════════════════════════════════════════════
#  CARGA DEL MODELO
# ════════════════════════════════════════════════════════════

def load_model(ckpt_path, device):
    """Reconstruye la arquitectura (según ckpt['arch']) desde el checkpoint."""
    # weights_only=False: el checkpoint guarda metadatos (model_kwargs, métricas),
    # no solo tensores. Es un archivo nuestro, así que es seguro.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = get_model(ckpt['arch'], **ckpt['model_kwargs']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Modelo '{ckpt['arch']}' cargado (epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f}, MAE_mod={ckpt['val_mae_mod']:.4f})")
    return model


# ════════════════════════════════════════════════════════════
#  IMPUTACIÓN
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def impute_channel(model, X_raw, ch_idx, device, batch_size=2048):
    """
    Imputa el canal ch_idx (índice denso) sobre todos los eventos de X_raw.

    Reproduce exactamente el preprocesado del Dataset: apaga el canal, normaliza
    por el máximo post-máscara, predice, y reescala la predicción a unidades crudas.

    Returns
    -------
    X_imp : (N, 61) copia de X_raw con el canal ch_idx sustituido por la predicción
    pred_raw : (N,) valor imputado (en unidades crudas) del canal
    """
    N = len(X_raw)
    X_imp    = X_raw.copy()
    pred_raw = np.zeros(N, dtype=np.float32)

    for i in range(0, N, batch_size):
        batch = X_raw[i:i+batch_size].copy()           # (b, 61)

        # Apagar el canal y normalizar por el máximo de los canales disponibles
        x_masked = batch.copy()
        x_masked[:, ch_idx] = 0.0
        norm = x_masked.max(axis=1, keepdims=True)      # (b, 1)
        norm[norm == 0] = 1.0                           # guard división por cero
        x_input = x_masked / norm

        # Máscara binaria con 0 en el canal apagado
        mask = np.ones_like(x_masked)
        mask[:, ch_idx] = 0.0

        # Entrada (b, 2, 61)
        x_in = np.stack([x_input, mask], axis=1).astype(np.float32)
        out  = model(torch.from_numpy(x_in).to(device)).cpu().numpy()   # (b, 61) normalizado

        # Reescalar la predicción del canal a unidades crudas y clipear a >=0
        pred = np.clip(out[:, ch_idx] * norm[:, 0], 0, None)
        pred_raw[i:i+len(batch)] = pred
        X_imp[i:i+len(batch), ch_idx] = pred

    return X_imp, pred_raw


def stratified_metrics(X_raw, pred_raw, ch_idx):
    """
    Métricas en el canal imputado, separando modified (tenía señal) y
    non-modified (estaba a 0) según el valor REAL del canal.
    """
    true = X_raw[:, ch_idx]
    is_mod = true > 0

    def _mae(mask):
        if mask.sum() == 0:
            return float('nan')
        return float(np.abs(pred_raw[mask] - true[mask]).mean())

    return {
        'mae_modified':     _mae(is_mod),
        'mae_non_modified': _mae(~is_mod),
        'n_modified':       int(is_mod.sum()),
        'n_non_modified':   int((~is_mod).sum()),
    }


# ════════════════════════════════════════════════════════════
#  FLOOD MAP
# ════════════════════════════════════════════════════════════

def compute_xy(X, x_sipm, y_sipm):
    """Posición XY de cada evento por centro de gravedad Rch² (invariante a escala)."""
    w = X ** 2                                  # pesos Rch²
    wsum = w.sum(axis=1, keepdims=True)         # (N, 1)
    wsum[wsum == 0] = 1.0
    pos_x = (w * x_sipm).sum(axis=1) / wsum[:, 0]
    pos_y = (w * y_sipm).sum(axis=1) / wsum[:, 0]
    return pos_x, pos_y


def plot_flood_comparison(datasets, titles, x_sipm, y_sipm, suptitle, save_path,
                          highlight_chs=None, bins=150, marker_radius=1.8):
    """
    Dibuja N flood maps lado a lado (etiquetas en inglés), con overlay de las
    posiciones de los SiPM y el/los canal(es) objetivo resaltados.

    datasets      : lista de matrices (N, 61)
    titles        : lista de títulos (inglés)
    highlight_chs : índice denso, lista de índices, o None. Los canales imputados/muertos
                    se marcan con un círculo ROJO TRANSLÚCIDO (no tapa el flood map debajo).
    marker_radius : radio del círculo en mm (≈ apotema del área del sensor)
    """
    # Normalizamos highlight_chs a una lista de índices densos
    if highlight_chs is None:
        chs = []
    elif isinstance(highlight_chs, (int, np.integer)):
        chs = [int(highlight_chs)]
    else:
        chs = [int(c) for c in highlight_chs]

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]

    for ax, X, title in zip(axes, datasets, titles):
        pos_x, pos_y = compute_xy(X, x_sipm, y_sipm)
        h = ax.hist2d(pos_x, pos_y, bins=bins, cmap='plasma')
        plt.colorbar(h[3], ax=ax, label='Counts', fraction=0.046, pad=0.04)

        # Overlay: todas las posiciones de SiPM como anillos finos (referencia geométrica)
        ax.scatter(x_sipm, y_sipm, facecolors='none', edgecolors='white',
                   s=55, linewidths=0.5, alpha=0.4, zorder=3)

        # Canal(es) objetivo: círculo rojo TRANSLÚCIDO (deja ver el flood map debajo)
        for c in chs:
            circ = Circle((x_sipm[c], y_sipm[c]), radius=marker_radius,
                          facecolor='red', alpha=0.30, edgecolor='red',
                          linewidth=1.3, zorder=4)
            ax.add_patch(circ)

        # Leyenda con los Ich resaltados (el canal también está en el título, pero ayuda)
        if chs:
            ich_list = ', '.join(f"Ich={IDX_TO_ICH[c]}" for c in chs)
            proxy = Circle((0, 0), 1, facecolor='red', alpha=0.30, edgecolor='red')
            ax.legend([proxy], [f"Target: {ich_list}"], loc='upper right', fontsize=9)

        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')

    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # dpi alto: el flood map (imagen) se rasteriza dentro del PDF; el resto es vectorial
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Guardado: {save_path}")


def plot_error_diagnostics(true_raw, pred_raw, ich, suptitle, save_path):
    """
    Diagnóstico cuantitativo de la imputación en datos Good (con ground truth).

    Recibe arrays 1D (carga real y predicha del canal, sobre TODOS los eventos),
    así sirve igual para un archivo o para el pool de varios.

    Solo sobre muestras MODIFIED (el canal tenía señal real), que es donde hay algo
    que recuperar. Dos paneles:
      1. Histograma del residuo (predicho − real): debería centrarse en 0 y ser estrecho.
      2. Histograma 2D predicho vs real con la línea ideal y=x: mide la correlación.

    Etiquetas en inglés. 'ich' es el Ich físico (solo para el título).
    """
    is_mod = true_raw > 0             # solo donde el canal tenía señal real
    t = true_raw[is_mod]
    p = pred_raw[is_mod]
    resid = p - t

    mae  = float(np.abs(resid).mean())
    rmse = float(np.sqrt((resid ** 2).mean()))
    bias = float(resid.mean())        # sesgo medio: ¿infra o sobre-estima?

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel 1: histograma del residuo ──────────────────────
    ax = axes[0]
    # recorte a percentiles 0.5-99.5 para que las colas no aplasten el histograma
    lo, hi = np.percentile(resid, [0.5, 99.5])
    ax.hist(resid, bins=120, range=(lo, hi), color='steelblue', alpha=0.85)
    ax.axvline(0, color='red', ls='--', lw=1.5, label='Zero error')
    ax.axvline(bias, color='orange', ls='-', lw=1.5, label=f'Mean bias = {bias:.2f}')
    ax.set_title(f"Imputation residual (Ich={ich}, modified events)")
    ax.set_xlabel('Predicted − True charge [ADC]')
    ax.set_ylabel('Counts')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # anotación con las métricas
    ax.text(0.02, 0.97, f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nN = {len(t):,}",
            transform=ax.transAxes, va='top', ha='left', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ── Panel 2: predicho vs real (densidad) ─────────────────
    ax = axes[1]
    lim = float(np.percentile(t, 99.5))   # límite común para ambos ejes
    hh = ax.hist2d(t, p, bins=120, range=[[0, lim], [0, lim]], cmap='viridis')
    plt.colorbar(hh[3], ax=ax, label='Counts', fraction=0.046, pad=0.04)
    ax.plot([0, lim], [0, lim], 'r--', lw=1.5, label='Ideal (y = x)')
    ax.set_aspect('equal')
    ax.set_title('Predicted vs true charge')
    ax.set_xlabel('True charge [ADC]')
    ax.set_ylabel('Predicted charge [ADC]')
    ax.legend()

    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Guardado: {save_path}")
    print(f"  Residuo: MAE={mae:.3f}  RMSE={rmse:.3f}  bias={bias:.3f}  (ADC, modified)")
    return {'mae': mae, 'rmse': rmse, 'bias': bias, 'n_modified': int(len(t))}


# ════════════════════════════════════════════════════════════
#  ERROR A NIVEL DE FLOOD MAP / POSICIÓN
# ════════════════════════════════════════════════════════════

def plot_position_error(X_orig, X_deg, X_imp, ch_idx, x_sipm, y_sipm, suptitle, save_path, bins=150):
    """
    Error de la imputación a nivel de POSICIÓN — la métrica que de verdad importa
    para el TFM (es el proxy del FWHM): no mira la carga de un canal aislado, sino
    cuánto se desplaza la posición reconstruida del evento.

    Tres paneles:
      1. Histograma del desplazamiento ΔR respecto a la posición original, para el
         caso degradado (canal apagado) vs imputado. Anota las medianas y el % de
         recuperación. Solo sobre eventos MODIFIED (los que el fallo afecta).
      2-3. Mapa 2D de diferencia de llenado (degradado − original) e (imputado − original):
         dónde sobran/faltan cuentas. Si la imputación es buena, el panel 3 ≈ 0.
    """
    # Posición XY por centro de gravedad Rch² en los tres estados
    ox, oy = compute_xy(X_orig, x_sipm, y_sipm)
    dx, dy = compute_xy(X_deg,  x_sipm, y_sipm)
    ix, iy = compute_xy(X_imp,  x_sipm, y_sipm)

    # ΔR solo donde apagar el canal cambia algo (eventos con señal real en él)
    is_mod = X_orig[:, ch_idx] > 0
    dR_deg = np.sqrt((dx - ox) ** 2 + (dy - oy) ** 2)[is_mod]
    dR_imp = np.sqrt((ix - ox) ** 2 + (iy - oy) ** 2)[is_mod]
    med_deg,  med_imp  = float(np.median(dR_deg)),       float(np.median(dR_imp))
    mean_deg, mean_imp = float(dR_deg.mean()),           float(dR_imp.mean())
    p90_deg,  p90_imp  = float(np.percentile(dR_deg, 90)), float(np.percentile(dR_imp, 90))
    # Recuperación: con la mediana (comprime) y con el p90 (eventos donde el canal domina)
    recovery     = (med_deg - med_imp) / med_deg * 100 if med_deg > 0 else 0.0
    recovery_p90 = (p90_deg - p90_imp) / p90_deg * 100 if p90_deg > 0 else 0.0

    # Histogramas 2D en una rejilla común (mismo grid para poder restar)
    rng = [[x_sipm.min() - 2, x_sipm.max() + 2], [y_sipm.min() - 2, y_sipm.max() + 2]]
    Ho, xe, ye = np.histogram2d(ox, oy, bins=bins, range=rng)
    Hd, _,  _  = np.histogram2d(dx, dy, bins=bins, range=rng)
    Hi, _,  _  = np.histogram2d(ix, iy, bins=bins, range=rng)
    diff_deg = Hd - Ho
    diff_imp = Hi - Ho
    # escala de color simétrica común (la fija el daño del degradado)
    vmax = float(np.percentile(np.abs(diff_deg), 99.5)) or 1.0
    extent = [xe[0], xe[-1], ye[0], ye[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # ── Panel 1: histograma de ΔR ────────────────────────────
    ax = axes[0]
    hi = float(np.percentile(dR_deg, 99))
    ax.hist(dR_deg, bins=100, range=(0, hi), color='coral', alpha=0.65,
            label=f'Degraded (median {med_deg:.3f}, p90 {p90_deg:.3f} mm)')
    ax.hist(dR_imp, bins=100, range=(0, hi), color='steelblue', alpha=0.65,
            label=f'Imputed (median {med_imp:.3f}, p90 {p90_imp:.3f} mm)')
    ax.axvline(med_deg, color='coral', ls='--', lw=1.5)
    ax.axvline(med_imp, color='steelblue', ls='--', lw=1.5)
    ax.set_title(f"Position shift vs original (Ich={IDX_TO_ICH[ch_idx]}, modified)\n"
                 f"Recovery: median {recovery:.0f}%  ·  p90 {recovery_p90:.0f}%")
    ax.set_xlabel('Position shift ΔR [mm]'); ax.set_ylabel('Counts')
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── Paneles 2-3: mapas de diferencia de llenado ──────────
    for ax, diff, title in [
        (axes[1], diff_deg, 'Flood-map difference: degraded − original'),
        (axes[2], diff_imp, 'Flood-map difference: imputed − original'),
    ]:
        # cmap divergente centrado en 0: rojo = sobran cuentas, azul = faltan
        im = ax.imshow(diff.T, origin='lower', extent=extent, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, aspect='equal')
        plt.colorbar(im, ax=ax, label='Δ counts', fraction=0.046, pad=0.04)
        # marcamos la posición del canal afectado
        ax.add_patch(Circle((x_sipm[ch_idx], y_sipm[ch_idx]), radius=1.8,
                            facecolor='none', edgecolor='black', linewidth=1.2))
        ax.set_title(title); ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')

    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Guardado: {save_path}")
    print(f"  ΔR (imputado)  mediana={med_imp:.4f}  media={mean_imp:.4f}  p90={p90_imp:.4f} mm")
    print(f"  ΔR (degradado) mediana={med_deg:.4f}  media={mean_deg:.4f}  p90={p90_deg:.4f} mm")
    print(f"  recuperación: mediana={recovery:.0f}%  ·  p90={recovery_p90:.0f}%")
    return {
        'dR_imp':        {'median': med_imp,  'mean': mean_imp,  'p90': p90_imp},
        'dR_deg':        {'median': med_deg,  'mean': mean_deg,  'p90': p90_deg},
        'recovery_median_pct': recovery,
        'recovery_p90_pct':    recovery_p90,
        'n_modified':    int(is_mod.sum()),
    }


# ════════════════════════════════════════════════════════════
#  EVALUACIÓN AGREGADA SOBRE VARIOS ARCHIVOS
# ════════════════════════════════════════════════════════════

def evaluate_multifile(model, files, ch_idx, device, x_sipm, y_sipm, max_events):
    """
    Imputa el canal ch_idx en VARIOS archivos held-out y agrega las métricas.

    Pooling: las cargas y los ΔR (en mm) son comparables entre archivos (misma
    geometría), así que juntarlos da una métrica más representativa que un solo .dat.

    Returns
    -------
    true_all, pred_all : np.ndarray (N_total,) carga real y predicha del canal (todos los eventos)
    """
    true_all, pred_all   = [], []
    dR_deg_all, dR_imp_all = [], []
    per_file = []   # métricas por archivo (para guardar en el JSON)

    print(f"\n=== AGREGADO sobre {len(files)} archivos held-out (Ich={IDX_TO_ICH[ch_idx]}) ===")
    for f in files:
        X = load_dat_to_dense(f, max_events=max_events)
        X_imp, pred = impute_channel(model, X, ch_idx, device)
        true = X[:, ch_idx]
        true_all.append(true); pred_all.append(pred)

        is_mod = true > 0
        ox, oy = compute_xy(X, x_sipm, y_sipm)
        Xd = X.copy(); Xd[:, ch_idx] = 0.0
        dx, dy = compute_xy(Xd, x_sipm, y_sipm)
        ix, iy = compute_xy(X_imp, x_sipm, y_sipm)
        dR_deg_all.append(np.sqrt((dx - ox)**2 + (dy - oy)**2)[is_mod])
        dR_imp_all.append(np.sqrt((ix - ox)**2 + (iy - oy)**2)[is_mod])

        rm, pm = true[is_mod], pred[is_mod]
        fmae, fbias = float(np.abs(pm - rm).mean()), float((pm - rm).mean())
        per_file.append({'file': f.name, 'mae_mod': fmae, 'bias': fbias, 'n_mod': int(is_mod.sum())})
        print(f"  {f.name}: MAE_mod={fmae:.3f}  bias={fbias:+.3f}  (n_mod={is_mod.sum():,})")

    true_all = np.concatenate(true_all)
    pred_all = np.concatenate(pred_all)
    dR_deg   = np.concatenate(dR_deg_all)
    dR_imp   = np.concatenate(dR_imp_all)

    is_mod = true_all > 0
    rm, pm = true_all[is_mod], pred_all[is_mod]
    mae  = float(np.abs(pm - rm).mean())
    rmse = float(np.sqrt(((pm - rm) ** 2).mean()))
    bias = float((pm - rm).mean())
    print(f"  -- POOLED: MAE_mod={mae:.3f}  RMSE={rmse:.3f}  bias={bias:+.3f}  "
          f"(N_mod={is_mod.sum():,})")
    print(f"  -- POOLED ΔR imputado : mediana={np.median(dR_imp):.4f}  "
          f"media={dR_imp.mean():.4f}  p90={np.percentile(dR_imp, 90):.4f} mm")
    print(f"  -- POOLED ΔR degradado: mediana={np.median(dR_deg):.4f}  "
          f"media={dR_deg.mean():.4f}  p90={np.percentile(dR_deg, 90):.4f} mm")

    metrics = {
        'n_files':  len(files),
        'per_file': per_file,
        'pooled': {
            'mae_mod': mae, 'rmse': rmse, 'bias': bias, 'n_mod': int(is_mod.sum()),
            'dR_imp': {'median': float(np.median(dR_imp)), 'mean': float(dR_imp.mean()),
                       'p90': float(np.percentile(dR_imp, 90))},
            'dR_deg': {'median': float(np.median(dR_deg)), 'mean': float(dR_deg.mean()),
                       'p90': float(np.percentile(dR_deg, 90))},
        },
    }
    return true_all, pred_all, metrics


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def run_eval(ckpt_path, out_dir):
    """
    Evalúa un checkpoint: genera las figuras (PDF) Y guarda todas las métricas en
    eval_metrics.json (para no perder los números del run en la consola).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(ckpt_path, device)
    # Metadatos del checkpoint (arch, epoch, métricas de validación) para el JSON
    ckpt_meta = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    x_sipm, y_sipm = load_positions(PSIPM_PATH)

    # Split limpio (MISMA fuente que train.py): test held-out + demo Good sobre un test file
    _, _, test_files = get_file_split(GOOD_DIR)
    good_demo = test_files[0]   # archivo de TEST para la demo individual (held-out de verdad)

    # ── Demo Good: ground truth disponible ───────────────────
    print(f"\n=== GOOD: imputar Ich={GOOD_ICH} en {Path(good_demo).name} (test) ===")
    ch = ICH_TO_IDX[GOOD_ICH]   # Ich físico → índice denso
    X_good = load_dat_to_dense(good_demo, max_events=MAX_EVENTS)
    X_imp, pred = impute_channel(model, X_good, ch, device)

    strat = stratified_metrics(X_good, pred, ch)
    print(f"  MAE modified     = {strat['mae_modified']:.4f}  (n={strat['n_modified']:,})")
    print(f"  MAE non-modified = {strat['mae_non_modified']:.4f}  (n={strat['n_non_modified']:,})")

    # Flood map: original vs canal apagado vs imputado (con overlay del canal)
    X_deg = X_good.copy(); X_deg[:, ch] = 0.0
    plot_flood_comparison(
        [X_good, X_deg, X_imp],
        [f"Original (Ich={GOOD_ICH} active)",
         f"Degraded (Ich={GOOD_ICH} off)",
         f"Imputed (Ich={GOOD_ICH} recovered)"],
        x_sipm, y_sipm,
        suptitle=f"Good file {Path(good_demo).stem} — channel Ich={GOOD_ICH}",
        save_path=str(out_dir / f'flood_good_ich{GOOD_ICH}.pdf'),
        highlight_chs=ch,
    )

    err_single = plot_error_diagnostics(
        X_good[:, ch], pred, GOOD_ICH,
        suptitle=f"Good file {Path(good_demo).stem} — imputation error Ich={GOOD_ICH}",
        save_path=str(out_dir / f'error_good_ich{GOOD_ICH}.pdf'),
    )

    pos = plot_position_error(
        X_good, X_deg, X_imp, ch, x_sipm, y_sipm,
        suptitle=f"Good file {Path(good_demo).stem} — flood-map / position error Ich={GOOD_ICH}",
        save_path=str(out_dir / f'position_error_good_ich{GOOD_ICH}.pdf'),
    )

    # ── Métricas agregadas sobre los archivos de TEST (held-out) ────
    true_all, pred_all, pooled = evaluate_multifile(
        model, test_files, ch, device, x_sipm, y_sipm, TEST_MAX_EVENTS,
    )
    err_agg = plot_error_diagnostics(
        true_all, pred_all, GOOD_ICH,
        suptitle=f"Aggregated over {len(test_files)} held-out test files — imputation error Ich={GOOD_ICH}",
        save_path=str(out_dir / f'error_aggregated_ich{GOOD_ICH}.pdf'),
    )

    # ── Demo Bad: canal muerto real (sin ground truth) ───────
    if Path(BAD_FILE).exists():
        print(f"\n=== BAD: imputar canal muerto Ich={BAD_ICH} en {Path(BAD_FILE).name} ===")
        chb = ICH_TO_IDX[BAD_ICH]
        X_bad = load_dat_to_dense(BAD_FILE, max_events=MAX_EVENTS)
        X_bad_imp, _ = impute_channel(model, X_bad, chb, device)
        plot_flood_comparison(
            [X_bad, X_bad_imp],
            [f"Bad as-is (Ich={BAD_ICH} dead)",
             f"Imputed (Ich={BAD_ICH} recovered)"],
            x_sipm, y_sipm,
            suptitle=f"Bad file {Path(BAD_FILE).stem} — dead channel Ich={BAD_ICH}",
            save_path=str(out_dir / f'flood_bad_ich{BAD_ICH}.pdf'),
            highlight_chs=chb,
        )

    # ── Persistir TODAS las métricas a JSON ──────────────────
    metrics = {
        'timestamp':    datetime.datetime.now().isoformat(timespec='seconds'),
        'checkpoint':   str(ckpt_path),
        'arch':         ckpt_meta.get('arch'),
        'train_epoch':  ckpt_meta.get('epoch'),
        'val_loss':     ckpt_meta.get('val_loss'),
        'val_mae_mod':  ckpt_meta.get('val_mae_mod'),
        'channel_ich':  GOOD_ICH,
        'good_file':    Path(good_demo).name,
        'test_files':   [f.name for f in test_files],
        'good_single_file': {
            'mae_modified':     strat['mae_modified'],
            'mae_non_modified': strat['mae_non_modified'],
            'n_modified':       strat['n_modified'],
            'residual':         err_single,   # mae, rmse, bias, n
            'position':         pos,          # ΔR mediana/media/p90 + recuperación
        },
        'aggregated_heldout': {
            'residual': err_agg,
            **pooled,                          # per_file + pooled (con ΔR media/p90)
        },
    }
    json_path = out_dir / 'eval_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Evaluación terminada. Figuras y métricas en: {out_dir}")
    print(f"  Métricas: {json_path}")
    return metrics


def main():
    # Check tontotorrón: si el checkpoint no existe, avisa claro y lista las carpetas
    # disponibles en vez de petar con un error críptico a mitad del run.
    if not Path(CKPT_PATH).exists():
        print(f"ERROR: no encuentro el checkpoint:\n  {CKPT_PATH}")
        runs = sorted(p.name for p in Path(RUNS_BASE).glob('imputer_*') if p.is_dir())
        if runs:
            print(f"Carpetas de runs disponibles en {RUNS_BASE} (ajusta RUN_NAME):")
            for r in runs:
                tiene = (Path(RUNS_BASE) / r / 'best_model.pth').exists()
                print(f"  - {r}{'' if tiene else '   (sin best_model.pth)'}")
        else:
            print(f"No hay ninguna carpeta 'imputer_*' en {RUNS_BASE}. ¿Has entrenado ya?")
        sys.exit(1)

    print(f"Evaluando run: {RUN_NAME}")
    run_eval(CKPT_PATH, OUT_DIR)


if __name__ == '__main__':
    main()
