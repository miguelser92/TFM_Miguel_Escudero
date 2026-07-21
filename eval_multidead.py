"""
eval_multidead.py
=================
EVAL MULTI-DEAD (punto 3 del roadmap): ¿hasta qué tamaño de fallo aguanta la
imputación? En vez de degradar un solo sensor (eval_total.py), degrada CONJUNTOS
de k sensores y mide cómo cae la recuperación al crecer k.

Dos regímenes, y el contraste entre ambos es el resultado clave:
  - 'cluster' : los k muertos son CONTIGUOS (caso físico real: sombras, conexiones
                en serie, efectos térmicos). Se le quitan al sensor justo los
                vecinos de los que depende → debe degradarse rápido.
  - 'scatter' : los k muertos están DISPERSOS por el detector (control). Cada
                muerto conserva sus vecinos vivos → debe degradarse mucho menos.

Barrido: por cada tamaño k y cada régimen se construye UN conjunto por sensor
semilla (61 conjuntos, igual que eval_total recorre los 61 canales), de forma
determinista → la estadística cubre todo el detector y es reproducible.

El degradado NO depende del modelo → se calcula una vez por (k, régimen, semilla)
y se comparte entre todos los modelos del barrido.

Salidas (por modelo, en runs/<run>/<CAMPAÑA>/):
  - multidead_curve.png       LA figura: recuperación y MAE vs tamaño del clúster
  - multidead_maps.png        mapas hexagonales: recuperación p90 por semilla, un panel por k
  - eval_multidead_metrics.json  métricas por (régimen, k) y por conjunto
  - subida a W&B como run nuevo '<run>_EVAL_MULTIDEAD' (job_type='eval_multidead')

Uso:
    conda activate tfm
    python eval_multidead.py imputer_hexcnn_s_mse_dead1-4
    python eval_multidead.py run1 run2 --out MULTIDEAD_1M --events 1000000
    python eval_multidead.py imputer_hexcnn_s_mse_dead1-4 --quick   # smoke test
    python eval_multidead.py <run> --seeds 20    # submuestrea semillas (más rápido)

OJO al coste: 61 semillas x 4 tamaños x 2 regímenes = 488 imputaciones por modelo,
~4x lo que cuesta eval_total. Por eso los eventos por archivo van por defecto a
200k (las medias macro son estables); súbelo solo si necesitas los extremos.

Las campañas NUNCA se sobrescriben: si runs/<run>/<out>/eval_multidead_metrics.json
ya existe, ese run se salta. Para relanzar usa --out con un nombre nuevo.

Autor: Miguel Escudero (TFM)
"""

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

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dataset import (load_dat_to_dense, load_positions, get_file_split,
                     N_ACTIVE, IDX_TO_ICH)
from hex_geometry import get_neighbor_matrix
from imputation_eval import load_model, compute_xy
from eval_total import _hex_map          # reutilizamos el pintor de mapas hexagonales

# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════

RUNS_BASE  = r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\runs'
GOOD_DIR   = r'E:\Datos TFM\Good\Good'
PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'

# Este eval hace ~4x el trabajo de eval_total (barre k y dos regímenes), así que
# arranca con menos eventos. Las medias macro son estables a partir de ~200k.
TEST_MAX_EVENTS = 200_000

K_VALUES = (1, 2, 3, 4)              # tamaños de fallo a barrer
MODES    = ('cluster', 'scatter')    # contiguo (real) vs disperso (control)
CLUSTER_SEED = 1234                  # semilla del crecimiento de clústeres (reproducible)

USE_WANDB     = True
WANDB_PROJECT = 'TFM-SiPM-imputation'


# ════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DE LOS CONJUNTOS DE MUERTOS
# ════════════════════════════════════════════════════════════

def grow_cluster(nbr, seed_ch, k, rng):
    """
    Clúster CONTIGUO de k sensores: parte de 'seed_ch' y crece agregando, una a una,
    una vecina real elegida al azar de la frontera. Misma lógica que el dataset de
    entrenamiento, para que evaluación y entrenamiento hablen el mismo idioma.
    """
    cluster = [int(seed_ch)]
    while len(cluster) < k:
        frontera = {int(j) for c in cluster for j in nbr[c]
                    if j >= 0 and int(j) not in cluster}
        if not frontera:
            break
        cluster.append(int(rng.choice(sorted(frontera))))
    return np.array(sorted(cluster), dtype=np.int64)


def scatter_set(seed_ch, k, rng):
    """Control DISPERSO: k sensores al azar por todo el detector (sin contigüidad)."""
    otros = np.setdiff1d(np.arange(N_ACTIVE), [seed_ch])
    extra = rng.choice(otros, size=min(k - 1, len(otros)), replace=False)
    return np.array(sorted([int(seed_ch)] + [int(v) for v in extra]), dtype=np.int64)


def build_dead_sets(nbr, seeds, k, mode):
    """Un conjunto de muertos por sensor semilla. Determinista (semilla fija por k/modo)."""
    rng = np.random.default_rng(CLUSTER_SEED + 1000 * k + (0 if mode == 'cluster' else 1))
    out = {}
    for s in seeds:
        out[int(s)] = (grow_cluster(nbr, s, k, rng) if mode == 'cluster'
                       else scatter_set(s, k, rng))
    return out


# ════════════════════════════════════════════════════════════
#  IMPUTACIÓN DE UN CONJUNTO DE CANALES
# ════════════════════════════════════════════════════════════

def impute_set(model, X_raw, dead_idx, device, batch_size=2048):
    """
    Apaga TODOS los canales de 'dead_idx' a la vez y los imputa con la red.

    Reproduce el preprocesado del Dataset multi-dead: apaga el conjunto, normaliza
    por el máximo post-máscara (de los canales que quedan vivos), predice, y
    reescala las predicciones a unidades crudas.

    Returns
    -------
    X_imp : (N, 61) copia de X_raw con los canales de dead_idx sustituidos por la predicción
    pred  : (N, k)  valores imputados (unidades crudas), en el orden de dead_idx
    """
    N = len(X_raw)
    X_imp = X_raw.copy()
    pred  = np.zeros((N, len(dead_idx)), dtype=np.float32)

    for i in range(0, N, batch_size):
        batch = X_raw[i:i + batch_size]

        x_masked = batch.copy()
        x_masked[:, dead_idx] = 0.0
        norm = x_masked.max(axis=1, keepdims=True)     # (b, 1) máximo de los vivos
        norm[norm == 0] = 1.0                          # guard división por cero
        x_input = x_masked / norm

        mask = np.ones_like(x_masked)
        mask[:, dead_idx] = 0.0                        # 0 en todos los apagados

        x_in = np.stack([x_input, mask], axis=1).astype(np.float32)
        with torch.no_grad():
            out = model(torch.from_numpy(x_in).to(device)).cpu().numpy()

        p = np.clip(out[:, dead_idx] * norm, 0, None)  # (b, k) a unidades crudas
        pred[i:i + len(batch)] = p
        X_imp[i:i + len(batch)][:, dead_idx] = p

    return X_imp, pred


# ════════════════════════════════════════════════════════════
#  DEGRADADO (independiente del modelo → se calcula una vez)
# ════════════════════════════════════════════════════════════

QUANTS = ('median', 'p75', 'p90', 'p95', 'p99', 'mean')


def _stats(v):
    """Resumen de una distribución de ΔR: cuantiles + media."""
    if v.size == 0:
        return {q: 0.0 for q in QUANTS}
    qs = np.percentile(v, [50, 75, 90, 95, 99])
    return {'median': float(qs[0]), 'p75': float(qs[1]), 'p90': float(qs[2]),
            'p95': float(qs[3]), 'p99': float(qs[4]), 'mean': float(v.mean())}


def precompute_degraded(X_list, orig_xy, x_sipm, y_sipm, dead_sets_all):
    """
    ΔR del DEGRADADO (apagar sin imputar) para cada (modo, k, semilla). No depende
    del modelo, así que se calcula una sola vez y se reparte entre todos los modelos.
    Devuelve {(modo,k,semilla): {'dR': stats, 'n_mod': int}}.
    """
    deg = {}
    t0 = time.time()
    total = sum(len(d) for d in dead_sets_all.values())
    done = 0
    for (mode, k), dead_sets in dead_sets_all.items():
        for seed_ch, dead_idx in dead_sets.items():
            dRs, n_mod = [], 0
            for X, (ox, oy) in zip(X_list, orig_xy):
                mod = (X[:, dead_idx] > 0).any(axis=1)   # el fallo cambia algo
                if mod.sum() == 0:
                    continue
                X_deg = X.copy()
                X_deg[:, dead_idx] = 0.0
                dx, dy = compute_xy(X_deg, x_sipm, y_sipm)
                dRs.append(np.sqrt((dx - ox) ** 2 + (dy - oy) ** 2)[mod])
                n_mod += int(mod.sum())
            dR = np.concatenate(dRs) if dRs else np.array([0.0])
            deg[(mode, k, int(seed_ch))] = {'dR': _stats(dR), 'n_mod': n_mod}
            done += 1
        print(f"  degradado {mode} k={k}: {len(dead_sets)} conjuntos "
              f"({done}/{total}, {time.time()-t0:.0f}s)")
    return deg


# ════════════════════════════════════════════════════════════
#  EVALUACIÓN DE UN MODELO
# ════════════════════════════════════════════════════════════

def eval_model(run_name, X_list, orig_xy, deg, x_sipm, y_sipm, dead_sets_all, device):
    """Barre (modo, k, semilla) imputando el conjunto y midiendo recuperación de posición."""
    ckpt_path = Path(RUNS_BASE) / run_name / 'best_model.pth'
    model = load_model(ckpt_path, device)                     # igual que en eval_total
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    per_set = []
    t0 = time.time()

    for (mode, k), dead_sets in dead_sets_all.items():
        for seed_ch, dead_idx in dead_sets.items():
            dRs, errs, n_mod = [], [], 0
            for X, (ox, oy) in zip(X_list, orig_xy):
                mod = (X[:, dead_idx] > 0).any(axis=1)
                if mod.sum() == 0:
                    continue
                X_imp, pred = impute_set(model, X, dead_idx, device)
                ix, iy = compute_xy(X_imp, x_sipm, y_sipm)
                dRs.append(np.sqrt((ix - ox) ** 2 + (iy - oy) ** 2)[mod])
                # Error en los canales muertos de los eventos afectados (media sobre los k)
                errs.append((pred[mod] - X[mod][:, dead_idx]).ravel())
                n_mod += int(mod.sum())

            dR  = np.concatenate(dRs)  if dRs  else np.array([0.0])
            err = np.concatenate(errs) if errs else np.array([0.0])
            d   = deg[(mode, k, int(seed_ch))]
            imp = _stats(dR)
            rec = {q: (d['dR'][q] - imp[q]) / d['dR'][q] * 100 if d['dR'][q] > 0 else 0.0
                   for q in QUANTS}

            per_set.append({
                'mode': mode, 'k': int(k), 'seed_idx': int(seed_ch),
                'seed_ich': int(IDX_TO_ICH[seed_ch]),
                'dead_ich': [int(IDX_TO_ICH[i]) for i in dead_idx],
                'n_mod': n_mod,
                'mae': float(np.abs(err).mean()), 'bias': float(err.mean()),
                'dR_imp': imp, 'dR_deg': d['dR'], 'recovery_pct': rec,
            })
        print(f"  {run_name}: {mode} k={k} hecho ({time.time()-t0:.0f}s)")

    # ── Agregado macro por (modo, k): LA CURVA ──
    curve = {}
    for mode in MODES:
        for k in K_VALUES:
            sel = [p for p in per_set if p['mode'] == mode and p['k'] == k]
            if not sel:
                continue
            curve[f'{mode}_k{k}'] = {
                'mode': mode, 'k': int(k), 'n_sets': len(sel),
                'mae_macro':        float(np.mean([p['mae'] for p in sel])),
                'bias_macro':       float(np.mean([p['bias'] for p in sel])),
                'recov_median':     float(np.mean([p['recovery_pct']['median'] for p in sel])),
                'recov_p90':        float(np.mean([p['recovery_pct']['p90'] for p in sel])),
                'recov_mean':       float(np.mean([p['recovery_pct']['mean'] for p in sel])),
                'recov_p90_min':    float(np.min([p['recovery_pct']['p90'] for p in sel])),
                'dR_imp_p90_macro': float(np.mean([p['dR_imp']['p90'] for p in sel])),
                'dR_deg_p90_macro': float(np.mean([p['dR_deg']['p90'] for p in sel])),
            }
    return per_set, curve, ckpt


# ════════════════════════════════════════════════════════════
#  FIGURAS
# ════════════════════════════════════════════════════════════

def make_figures(run_name, per_set, curve, x_sipm, y_sipm, out_dir):
    """La curva (figura estrella) + mapas hexagonales de recuperación por semilla."""
    pngs = []
    C = {'cluster': '#c0392b', 'scatter': '#2471a3'}
    M = {'cluster': 'o', 'scatter': 's'}

    # ── PNG 1: recuperación RELATIVA + error ABSOLUTO vs tamaño del fallo ──
    # OJO: la recuperación % es relativa a un degradado que TAMBIÉN crece con k, así
    # que puede subir mientras el error absoluto empeora. Por eso se pinta además el
    # ΔR absoluto (imputado vs degradado): juntos cuentan la historia completa.
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    panels = [('recov_p90', 'Position recovery p90 (%)'),
              ('recov_mean', 'Position recovery, global mean (%)'),
              (None, 'Absolute position error p90 (mm)'),
              ('mae_macro', 'Imputation MAE on dead channels (ADC)')]
    for ax, (key, title) in zip(axes, panels):
        for mode in MODES:
            ks = [k for k in K_VALUES if f'{mode}_k{k}' in curve]
            if not ks:
                continue
            lbl = 'contiguous cluster' if mode == 'cluster' else 'scattered (control)'
            if key is None:
                # Panel absoluto: ΔR imputado (línea) y degradado (discontinua, referencia)
                ax.plot(ks, [curve[f'{mode}_k{k}']['dR_imp_p90_macro'] for k in ks],
                        marker=M[mode], color=C[mode], lw=2, ms=7, label=f'{lbl} — imputed')
                ax.plot(ks, [curve[f'{mode}_k{k}']['dR_deg_p90_macro'] for k in ks],
                        marker=M[mode], color=C[mode], lw=1.4, ms=5, ls='--', alpha=0.55,
                        label=f'{lbl} — degraded')
            else:
                ax.plot(ks, [curve[f'{mode}_k{k}'][key] for k in ks],
                        marker=M[mode], color=C[mode], lw=2, ms=7, label=lbl)
        ax.set_xlabel('Number of dead sensors (k)')
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(list(K_VALUES))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f'{run_name} — recovery vs failure size', fontsize=13)
    fig.tight_layout()
    p = out_dir / 'multidead_curve.png'
    fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig); pngs.append(p)

    # ── PNG 2: mapa hexagonal de recuperación p90 por semilla, un panel por k ──
    pitch = np.median(np.sort(np.hypot(x_sipm[:, None] - x_sipm,
                                       y_sipm[:, None] - y_sipm), axis=1)[:, 1])
    hex_r = pitch / np.sqrt(3) * 0.97
    for mode in MODES:
        ks = [k for k in K_VALUES if any(p['mode'] == mode and p['k'] == k for p in per_set)]
        if not ks:
            continue
        fig, axes = plt.subplots(1, len(ks), figsize=(4.4 * len(ks), 4.4))
        axes = np.atleast_1d(axes)
        for ax, k in zip(axes, ks):
            d = {p['seed_idx']: p['recovery_pct']['p90']
                 for p in per_set if p['mode'] == mode and p['k'] == k}
            vals = [d.get(i, np.nan) for i in range(N_ACTIVE)]
            _hex_map(ax, vals, x_sipm, y_sipm, hex_r,
                     f'k = {k}', cmap='viridis', fmt='{:.0f}',
                     cbar_label='Recovery p90 (%)')
        fig.suptitle(f'{run_name} — recovery p90 by cluster seed ({mode})', fontsize=12)
        fig.tight_layout()
        p = out_dir / f'multidead_maps_{mode}.png'
        fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig); pngs.append(p)

    return pngs


def log_to_wandb(run_name, curve, pngs, meta):
    try:
        import wandb
    except ImportError:
        print("  WARNING: wandb no instalado, no subo nada."); return
    r = wandb.init(project=WANDB_PROJECT, name=f'{run_name}_EVAL_MULTIDEAD',
                   job_type='eval_multidead', config=meta, reinit=True)
    flat = {f'{key}/{m}': v for key, c in curve.items()
            for m, v in c.items() if isinstance(v, (int, float))}
    r.log(flat)
    for p in pngs:
        r.log({p.stem: wandb.Image(str(p))})
    r.finish()


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    argv = sys.argv[1:]
    quick    = '--quick' in argv
    no_wandb = '--no-wandb' in argv or quick

    def flag(name, default=None):
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default

    out_name = flag('--out', 'MULTIDEAD_quick' if quick else 'MULTIDEAD')
    ev       = flag('--events')
    max_ev   = 20_000 if quick else (int(ev) if ev else TEST_MAX_EVENTS)
    n_seeds  = flag('--seeds')

    skip = set()
    for j, a in enumerate(argv):
        if a in ('--out', '--events', '--seeds', '--kmax'):
            skip.add(j); skip.add(j + 1)
        elif a.startswith('--'):
            skip.add(j)
    runs = [a for j, a in enumerate(argv) if j not in skip]
    if not runs:
        print("Uso: python eval_multidead.py <run> [<run>...] [--out NOMBRE] [--events N] [--seeds N] [--quick]")
        return

    global K_VALUES
    kmax = flag('--kmax')
    if quick:
        K_VALUES = (1, 2)
    elif kmax:
        K_VALUES = tuple(range(1, int(kmax) + 1))   # estira la curva: --kmax 8 → k=1..8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_sipm, y_sipm = load_positions(PSIPM_PATH)
    nbr = get_neighbor_matrix(PSIPM_PATH)
    _, _, test_files = get_file_split(GOOD_DIR)

    # Semillas de los conjuntos: los 61 sensores, o un submuestreo regular si --seeds
    seeds = np.arange(N_ACTIVE)
    if quick:
        seeds = seeds[::16]
    elif n_seeds:
        seeds = np.linspace(0, N_ACTIVE - 1, int(n_seeds)).astype(int)

    print(f"EVAL MULTI-DEAD sobre {len(runs)} run(s): {runs}")
    print(f"  k = {list(K_VALUES)}  |  regímenes = {list(MODES)}  |  semillas = {len(seeds)}")
    print(f"  campaña: runs/<run>/{out_name}/  |  eventos/archivo: {max_ev or 'TODOS'}")

    dead_sets_all = {(mode, k): build_dead_sets(nbr, seeds, k, mode)
                     for mode in MODES for k in K_VALUES}

    print(f"Cargando {len(test_files)} archivos de test...")
    X_list = [load_dat_to_dense(f, max_events=max_ev) for f in test_files]
    print(f"  total eventos: {sum(len(X) for X in X_list):,}")
    orig_xy = [compute_xy(X, x_sipm, y_sipm) for X in X_list]

    print("Precalculando el degradado (independiente del modelo)...")
    deg = precompute_degraded(X_list, orig_xy, x_sipm, y_sipm, dead_sets_all)

    for run_name in runs:
        out_dir = Path(RUNS_BASE) / run_name / out_name
        if (out_dir / 'eval_multidead_metrics.json').exists():
            print(f"[SALTO] {run_name}: la campaña '{out_name}' ya existe."); continue
        if not (Path(RUNS_BASE) / run_name / 'best_model.pth').exists():
            print(f"[SALTO] {run_name}: sin best_model.pth."); continue
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {run_name} ===")
        per_set, curve, ckpt = eval_model(run_name, X_list, orig_xy, deg,
                                          x_sipm, y_sipm, dead_sets_all, device)

        print(f"  {'régimen':10} {'k':>2} {'recP90':>8} {'recMean':>8} {'MAE':>7} {'bias':>7}")
        for mode in MODES:
            for k in K_VALUES:
                c = curve.get(f'{mode}_k{k}')
                if c:
                    print(f"  {mode:10} {k:2d} {c['recov_p90']:8.2f} {c['recov_mean']:8.2f} "
                          f"{c['mae_macro']:7.3f} {c['bias_macro']:+7.3f}")

        meta = {'run': run_name, 'arch': ckpt.get('arch'),
                'train_epoch': ckpt.get('epoch'), 'campaign': out_name,
                'max_events_per_file': max_ev or 'all', 'n_seeds': len(seeds),
                'k_values': list(K_VALUES), 'modes': list(MODES),
                'generated': datetime.datetime.now().isoformat(timespec='seconds')}
        (out_dir / 'eval_multidead_metrics.json').write_text(
            json.dumps({**meta, 'curve': curve, 'per_set': per_set}, indent=2),
            encoding='utf-8')

        pngs = make_figures(run_name, per_set, curve, x_sipm, y_sipm, out_dir)
        print(f"  guardado en {out_dir}")
        if USE_WANDB and not no_wandb:
            log_to_wandb(run_name, curve, pngs, meta)


if __name__ == '__main__':
    main()
