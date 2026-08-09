"""
eval_modulos.py — VARIABILIDAD ENTRE DETECTORES (efecto de módulo)
==================================================================
Responde a una pregunta que el conjunto de test no puede responder: ¿cuánto
dependen las métricas de QUÉ cinco módulos quedaron reservados para test?

La partición deja solo 5 módulos en test, así que las métricas agregadas no
tienen barra de error asociada al detector. Tenemos la variabilidad entre
ENTRENAMIENTOS (las réplicas), pero no la variabilidad entre MÓDULOS.

La idea que lo hace barato: por la rotación de ficheros, un modelo de 40 épocas
solo llega a ver 40 de los 149 módulos de entrenamiento. Los demás nunca los ha
observado, así que son held-out de facto y se pueden usar para medir la
dispersión entre detectores con MUCHA más estadística que los 5 de test, y sin
reentrenar nada.

El script lee del checkpoint cuántos módulos vio el modelo y con qué semilla de
rotación (campos que train.py persiste), reconstruye exactamente la lista de
vistos y evalúa sobre los NO vistos, uno a uno.

Uso:
    conda activate tfm
    python eval_modulos.py --run imputer_hexcnn_s_mse --tag ref
    python eval_modulos.py --run imputer_hexcnn_s_mse --n-modules 40 --tag ref40
    python eval_modulos.py --run imputer_hexcnn_s_mse --channels 61 --max-events 100000

Coste: ~n_modules x n_channels x max_events. Con los valores por defecto
(30 módulos, 12 canales, 50k eventos) son unos minutos; súbelo si quieres
apurar la precisión.

Salida: reports/module_variability_<tag>.json

Autor: Miguel Escudero (TFM)
"""

import sys
import json
import time
import argparse
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

from dataset import load_dat_to_dense, load_positions, get_file_split, N_ACTIVE, IDX_TO_ICH
from imputation_eval import load_model, impute_channel, compute_xy, resolve_ckpt_path

RUNS_BASE  = r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\runs'
GOOD_DIR   = r'E:\Datos TFM\Good\Good'
PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'
OUT_DIR    = Path(r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\reports')


def modulos_vistos(run, train_files):
    """
    Reconstruye la lista EXACTA de módulos que el modelo vio al entrenar.

    train.py rota los ficheros con round-robin, así que el nº de épocas fija la
    cobertura. Desde la auditoría el checkpoint guarda n_epochs, rot_seed y
    n_modules_seen; los checkpoints anteriores no los tienen y se asume el
    protocolo histórico (40 épocas, orden alfabético).
    """
    ck = torch.load(resolve_ckpt_path(Path(RUNS_BASE) / run / 'best_model.pth'),
                    map_location='cpu', weights_only=False)
    n_ep = ck.get('n_epochs', 40)
    rot  = ck.get('rot_seed')
    orden = list(train_files)
    if rot is not None:
        rng = np.random.default_rng(rot)
        orden = [orden[i] for i in rng.permutation(len(orden))]
    vistos = {orden[(e - 1) % len(orden)] for e in range(1, n_ep + 1)}
    return vistos, n_ep, rot


def metricas_modulo(model, X, canales, x_sipm, y_sipm, device):
    """Recuperación p90 y MAE macro de UN módulo, promediando sobre los canales."""
    ox, oy = compute_xy(X, x_sipm, y_sipm)
    recs, maes = [], []
    for c in canales:
        mod = X[:, c] > 0
        if mod.sum() < 100:
            continue
        Xd = X.copy(); Xd[:, c] = 0.0
        dx, dy = compute_xy(Xd, x_sipm, y_sipm)
        dRd = np.sqrt((dx - ox) ** 2 + (dy - oy) ** 2)[mod]
        Xi, pred = impute_channel(model, X, c, device)
        ix, iy = compute_xy(Xi, x_sipm, y_sipm)
        dRi = np.sqrt((ix - ox) ** 2 + (iy - oy) ** 2)[mod]
        pd_, pi_ = np.percentile(dRd, 90), np.percentile(dRi, 90)
        if pd_ > 0:
            recs.append((pd_ - pi_) / pd_ * 100)
        maes.append(float(np.abs(pred[mod] - X[mod, c]).mean()))
    if not recs:
        return None
    return float(np.mean(recs)), float(np.mean(maes)), float(np.min(recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='imputer_hexcnn_s_mse')
    ap.add_argument('--n-modules', type=int, default=30,
                    help='cuántos módulos NO vistos evaluar (0 = todos)')
    ap.add_argument('--channels', type=int, default=12,
                    help='cuántos canales por módulo (61 = todos)')
    ap.add_argument('--max-events', type=int, default=50_000)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    tag = f'_{args.tag}' if args.tag else ''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_sipm, y_sipm = load_positions(PSIPM_PATH)
    train_files, val_files, test_files = get_file_split(GOOD_DIR)

    vistos, n_ep, rot = modulos_vistos(args.run, train_files)
    # Held-out de verdad: modulos de train que la rotacion nunca alcanzo. Se dejan
    # fuera val y test para no mezclar con los conjuntos que ya tienen su papel.
    no_vistos = [f for f in train_files if f not in vistos]

    print(f"Modelo: {args.run}  (entrenado {n_ep} épocas, rot_seed={rot})")
    print(f"  módulos vistos en entrenamiento : {len(vistos)}")
    print(f"  módulos de train NO vistos      : {len(no_vistos)}  <- held-out de facto")
    if not no_vistos:
        print("\nEste modelo vio TODOS los módulos de entrenamiento: no hay held-out "
              "disponible por esta vía. Usa un modelo de cobertura parcial.")
        return

    rng = np.random.default_rng(0)
    sel = no_vistos if args.n_modules == 0 else \
        [no_vistos[i] for i in sorted(rng.choice(len(no_vistos),
                                                 size=min(args.n_modules, len(no_vistos)),
                                                 replace=False))]
    canales = list(range(N_ACTIVE)) if args.channels >= N_ACTIVE else \
        sorted(rng.choice(N_ACTIVE, size=args.channels, replace=False).tolist())

    print(f"  se evalúan {len(sel)} módulos x {len(canales)} canales x "
          f"{args.max_events:,} eventos\n")

    model = load_model(Path(RUNS_BASE) / args.run / 'best_model.pth', device)

    filas, t0 = [], time.time()
    for k, f in enumerate(sel):
        X = load_dat_to_dense(f, max_events=args.max_events)
        if len(X) == 0:
            continue
        r = metricas_modulo(model, X, canales, x_sipm, y_sipm, device)
        if r is None:
            continue
        rec, mae, peor = r
        filas.append({'modulo': f.name, 'recov_p90': rec, 'mae': mae, 'peor_canal': peor,
                      'n_events': int(len(X))})
        el = time.time() - t0
        eta = el / (k + 1) * (len(sel) - k - 1)
        print(f"  [{k+1:3d}/{len(sel)}] {f.name:16} recP90={rec:6.2f}  MAE={mae:.4f}  "
              f"peor={peor:6.2f}   (ETA {eta/60:.0f} min)", flush=True)

    rec = np.array([r['recov_p90'] for r in filas])
    mae = np.array([r['mae'] for r in filas])

    print(f"\n{'='*66}\nVARIABILIDAD ENTRE MÓDULOS ({len(filas)} detectores no vistos)\n{'='*66}")
    print(f"  recuperación p90 : media {rec.mean():6.2f}  sd {rec.std(ddof=1):5.2f}  "
          f"min {rec.min():6.2f}  max {rec.max():6.2f}")
    print(f"  MAE              : media {mae.mean():.4f}  sd {mae.std(ddof=1):.4f}")
    ee = rec.std(ddof=1) / np.sqrt(5)
    print(f"\n  --> Error estándar de una media sobre 5 módulos (como el test): "
          f"{ee:.2f} puntos")
    print(f"      Es la barra de error POR EFECTO DE MÓDULO que le falta a las")
    print(f"      métricas agregadas. Compárala con la de entrenamientos (±0.09 en p90):")
    print(f"      si es mayor, la incertidumbre dominante es QUÉ módulos tocaron,")
    print(f"      no la semilla de entrenamiento.")

    out = {'run': args.run, 'n_epochs_train': n_ep, 'rot_seed': rot,
           'n_modules_seen': len(vistos), 'n_modules_evaluated': len(filas),
           'n_channels': len(canales), 'max_events': args.max_events,
           'recov_p90_mean': round(float(rec.mean()), 3),
           'recov_p90_sd': round(float(rec.std(ddof=1)), 3),
           'recov_p90_min': round(float(rec.min()), 3),
           'recov_p90_max': round(float(rec.max()), 3),
           'mae_mean': round(float(mae.mean()), 5),
           'mae_sd': round(float(mae.std(ddof=1)), 5),
           'se_of_mean_over_5_modules': round(float(ee), 3),
           'per_module': filas,
           'generated': datetime.datetime.now().isoformat(timespec='seconds')}
    p = OUT_DIR / f'module_variability{tag}.json'
    p.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"\n  JSON guardado: {p}")

    # Figura: distribución entre módulos, con los de test marcados
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(rec, bins=min(20, max(5, len(rec) // 3)), color='#2471a3', alpha=0.8,
            edgecolor='white')
    ax.axvline(rec.mean(), color='#c0392b', lw=2,
               label=f'mean {rec.mean():.2f} (sd {rec.std(ddof=1):.2f})')
    ax.set_xlabel('p90 position recovery (%) — one value per detector module')
    ax.set_ylabel('modules')
    ax.set_title(f'Between-module variability — {args.run} ({len(filas)} unseen modules)')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    pf = OUT_DIR / f'module_variability{tag}.png'
    fig.savefig(pf, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f"  figura: {pf}")


if __name__ == '__main__':
    main()
