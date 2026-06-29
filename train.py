r"""
train.py
========
Bucle de entrenamiento de la imputación SiPM con generación on-the-fly.

Estrategia de datos:
  - Entrena SOLO con archivos Good (los 61 canales sanos).
  - Rotación de archivos: un .dat distinto por época (los datos no caben en RAM).
  - Validación con un conjunto FIJO de archivos reservados (para que val sea comparable
    entre épocas).

Métricas: pérdida Huber sobre los 61 canales + MAE en el canal imputado, reportado
por separado para muestras modified y non-modified (evaluación estratificada).

Guarda el mejor modelo en .pth para reutilizarlo después (ver imputation_eval.py).


Prara runeo automatico en powershell
conda activate tfm
cd "C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código"
foreach ($m in 'deepmlp','resmlp','hexcnn') {
    Write-Host "=== Entrenando $m ===" -ForegroundColor Cyan
    python train.py $m
}



Uso:
    conda activate tfm
    python train.py

Ajusta la sección CONFIG según necesites.

Autor: Miguel Escudero (TFM)
"""

import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # backend sin ventana (solo guardamos figuras a archivo)
import matplotlib.pyplot as plt

# Consola de Windows: forzar UTF-8 para poder imprimir ✓ sin UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dataset import SiPMImputationDataset, load_dat_to_dense, get_file_split, N_ACTIVE
from model import get_model, count_parameters


# ════════════════════════════════════════════════════════════
#  CONFIG — toca esto
# ════════════════════════════════════════════════════════════

GOOD_DIR    = r'E:\Datos TFM\Good\Good'
RUNS_BASE   = r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\runs'

# Arquitectura a entrenar: 'deepmlp' (baseline) | 'resmlp' | 'hexcnn'
MODEL_NAME    = 'hexcnn'
# Tamaño de la HexCNN: '' = defaults del modelo | 's' | 'm' | 'l' (presets de capacidad).
# Solo aplica a hexcnn; añade sufijo a la carpeta y al run de W&B (p.ej. imputer_hexcnn_l_mse).
MODEL_SIZE    = ''
MODEL_KWARGS  = {}          # override manual EXTRA (se fusiona ENCIMA del preset de tamaño)

# Presets de capacidad de la HexCNN (ancho 'hidden' y nº de bloques residuales).
# Params aprox: s≈38K (la actual) · m≈225K · l≈399K (~ resmlp 346K → comparación a igual presupuesto).
HEXCNN_SIZES = {
    's': dict(hidden=48,  n_blocks=4),
    'm': dict(hidden=96,  n_blocks=6),
    'l': dict(hidden=128, n_blocks=6),
}

N_EPOCHS      = 40
BATCH_SIZE    = 512
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 8            # early stopping
MAX_EVENTS    = 400_000     # tope de eventos por archivo y época (controla tiempo/RAM)
HUBER_DELTA   = 0.1         # robusto a outliers; datos ~[0,1] (algún target >1)

# Función de pérdida: 'huber' | 'mae' | 'mse'. Cada una va a SU carpeta y SU run de W&B
# (se añade el sufijo de la loss, salvo para 'huber' que es la de referencia) → no se pisan.
LOSS          = 'mse'
RUN_SUFFIX    = '' if LOSS == 'huber' else f'_{LOSS}'

# Split limpio (fuente única en dataset.get_file_split): train / val / test disjuntos
N_VAL_FILES   = 5
N_TEST_FILES  = 5           # reservado: NUNCA se toca (ni train ni validación)
SPLIT_SEED    = 42
VAL_MASK_SEED = 12345       # semilla fija de las máscaras de validación (idénticas cada época)

# Weights & Biases (logging al dashboard web). USE_WANDB=False para entrenar sin logging.
USE_WANDB     = True
WANDB_PROJECT = 'TFM-SiPM-imputation'

# RUN_TAG / OUTPUT_DIR se resuelven al inicio de main() a partir de MODEL_NAME + MODEL_SIZE
# + LOSS (para que el override por CLI se refleje sin recalcular nada a mano).


# ════════════════════════════════════════════════════════════
#  ENTRENAMIENTO
# ════════════════════════════════════════════════════════════

def evaluate(model, loader, loss_fn, device):
    """
    Evalúa el modelo en un loader y devuelve (loss, mae_modified, mae_nonmod).

    El MAE se calcula SOLO sobre el canal imputado, separando las dos clases:
      - modified:     mide la capacidad de imputar de verdad
      - non-modified: mide la tasa de falsa corrección (debería ser ~0)
    """
    model.eval()
    total_loss = 0.0
    n_total    = 0
    # acumuladores de error absoluto en el canal imputado por clase
    err_mod, n_mod = 0.0, 0
    err_non, n_non = 0.0, 0

    with torch.no_grad():   # sin grafo de gradientes: más rápido y menos memoria
        for x_in, target, ch, is_mod in loader:
            x_in   = x_in.to(device)
            target = target.to(device)
            ch     = ch.to(device)
            is_mod = is_mod.to(device)

            out  = model(x_in)                       # (B, 61)
            loss = loss_fn(out, target)              # Huber sobre los 61 canales
            bs   = len(target)
            total_loss += loss.item() * bs
            n_total    += bs

            # Error absoluto en el canal imputado de cada muestra
            # out[arange(B), ch] selecciona la predicción del canal apagado de cada fila
            rows = torch.arange(bs, device=device)
            pred_ch = out[rows, ch]
            true_ch = target[rows, ch]
            abs_err = (pred_ch - true_ch).abs()

            m = is_mod.bool()
            err_mod += abs_err[m].sum().item();  n_mod += m.sum().item()
            err_non += abs_err[~m].sum().item(); n_non += (~m).sum().item()

    loss     = total_loss / max(n_total, 1)
    mae_mod  = err_mod / max(n_mod, 1)
    mae_non  = err_non / max(n_non, 1)
    return loss, mae_mod, mae_non


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Resolver tamaño → kwargs del modelo, etiqueta de run y carpeta de salida ──
    # MODEL_SIZE elige un preset (solo hexcnn); MODEL_KWARGS lo sobrescribe por encima.
    size_kwargs = {}
    if MODEL_SIZE:
        assert MODEL_NAME == 'hexcnn', f"MODEL_SIZE='{MODEL_SIZE}' solo aplica a hexcnn"
        assert MODEL_SIZE in HEXCNN_SIZES, f"tamaño '{MODEL_SIZE}' no válido: {list(HEXCNN_SIZES)}"
        size_kwargs = dict(HEXCNN_SIZES[MODEL_SIZE])
    model_kwargs = {**size_kwargs, **MODEL_KWARGS}
    size_suffix  = f'_{MODEL_SIZE}' if MODEL_SIZE else ''
    run_tag      = f'{MODEL_NAME}{size_suffix}'                 # p.ej. hexcnn_l
    out_dir = Path(RUNS_BASE) / f'imputer_{run_tag}{RUN_SUFFIX}'   # p.ej. imputer_hexcnn_l_mse
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Split limpio train / val / test (fuente única) ───────
    # El test se reserva y NO se toca aquí (ni para entrenar ni para validar).
    train_files, val_files, test_files = get_file_split(
        GOOD_DIR, n_val=N_VAL_FILES, n_test=N_TEST_FILES, seed=SPLIT_SEED,
    )
    print(f"Dispositivo: {device}")
    print(f"Split: train={len(train_files)}  val={len(val_files)}  test={len(test_files)} (test reservado)")
    print(f"  val:  {[f.name for f in val_files]}")
    print(f"  test: {[f.name for f in test_files]}")

    # ── Validación: conjunto FIJO (se carga una vez) ─────────
    print("Cargando archivos de validación...")
    X_val = np.concatenate(
        [load_dat_to_dense(f, max_events=MAX_EVENTS // len(val_files)) for f in val_files],
        axis=0,
    )
    val_ds = SiPMImputationDataset(X_val, seed=VAL_MASK_SEED)   # seed fijo: val reproducible
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=0, pin_memory=True)   # num_workers=0 obligatorio en Windows
    print(f"  Val: {len(X_val):,} eventos")

    # ── Modelo, optimizador, scheduler, loss ─────────────────
    model = get_model(MODEL_NAME, **model_kwargs).to(device)
    n_params = count_parameters(model)
    print(f"Modelo: {run_tag}  |  kwargs: {model_kwargs}  |  parámetros: {n_params:,}")

    # ── Weights & Biases (logging al dashboard) ──────────────
    wandb_run = None
    if USE_WANDB:
        try:
            import wandb
            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                name=f'{run_tag}{RUN_SUFFIX}',
                config={
                    'arch': MODEL_NAME, 'model_size': MODEL_SIZE or 'default',
                    'run_tag': run_tag, 'model_kwargs': model_kwargs, 'n_params': n_params,
                    'loss': LOSS, 'huber_delta': HUBER_DELTA,
                    'n_epochs': N_EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
                    'weight_decay': WEIGHT_DECAY, 'patience': PATIENCE, 'max_events': MAX_EVENTS,
                    'n_val_files': N_VAL_FILES, 'n_test_files': N_TEST_FILES,
                    'split_seed': SPLIT_SEED, 'device': str(device),
                },
            )
        except ImportError:
            print("WARNING: wandb no está instalado (pip install wandb). Sigo sin logging.")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR / 100)

    # Pérdida según el flag LOSS de la config
    if LOSS == 'huber':
        loss_fn = nn.HuberLoss(delta=HUBER_DELTA)
    elif LOSS == 'mae':
        loss_fn = nn.L1Loss()       # MAE = error absoluto medio
    elif LOSS == 'mse':
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"LOSS '{LOSS}' no reconocida (usa 'huber', 'mae' o 'mse')")
    print(f"Loss: {LOSS}")

    history = {'train_loss': [], 'val_loss': [], 'val_mae_mod': [], 'val_mae_non': []}
    best_val = float('inf')
    epochs_no_improve = 0
    ckpt_path = out_dir / 'best_model.pth'

    print(f"\n{'='*64}\nEntrenando {N_EPOCHS} épocas (rotación de archivos)\n{'='*64}")

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()

        # Archivo de esta época (round-robin sobre los de train)
        f_train = train_files[(epoch - 1) % len(train_files)]
        X_train = load_dat_to_dense(f_train, max_events=MAX_EVENTS)
        # seed = epoch → el masking aleatorio cambia en cada época
        train_ds = SiPMImputationDataset(X_train, seed=epoch)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True)

        # ── Una época de entrenamiento ───────────────────────
        model.train()
        run_loss, n_seen = 0.0, 0
        for x_in, target, ch, is_mod in train_loader:
            x_in   = x_in.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            out  = model(x_in)
            loss = loss_fn(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # evita exploding gradients
            optimizer.step()

            run_loss += loss.item() * len(target)
            n_seen   += len(target)

        cur_lr = optimizer.param_groups[0]['lr']   # lr usado en esta época (antes del step)
        scheduler.step()
        train_loss = run_loss / max(n_seen, 1)

        # ── Validación ───────────────────────────────────────
        # Re-sembramos el rng del val_ds para que las máscaras sean IDÉNTICAS cada
        # época (si no, el rng con estado deriva y val se mide sobre canales distintos).
        val_ds.rng = np.random.default_rng(VAL_MASK_SEED)
        val_loss, mae_mod, mae_non = evaluate(model, val_loader, loss_fn, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae_mod'].append(mae_mod)
        history['val_mae_non'].append(mae_non)

        if wandb_run is not None:
            wandb_run.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                           'val_mae_mod': mae_mod, 'val_mae_non': mae_non, 'lr': cur_lr})

        flag = ''
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            # Guardamos pesos + metadatos para poder recargar el modelo después
            torch.save({
                'model_state':  model.state_dict(),
                'model_kwargs': model_kwargs,   # incluye hidden/n_blocks → el eval reconstruye el tamaño correcto
                'arch':         MODEL_NAME,
                'epoch':        epoch,
                'val_loss':     val_loss,
                'val_mae_mod':  mae_mod,
                'val_mae_non':  mae_non,
            }, ckpt_path)
            flag = '  ✓ best'
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:3d}/{N_EPOCHS} | "
              f"train={train_loss:.4f} val={val_loss:.4f} | "
              f"MAE(mod)={mae_mod:.4f} MAE(non)={mae_non:.4f} | "
              f"{f_train.name} | {time.time()-t0:.1f}s{flag}")

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping en epoch {epoch} (sin mejora en {PATIENCE} épocas)")
            break

    # ── Guardar historial + curvas ───────────────────────────
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    plot_curves(history, out_dir / 'training_curves.png')

    # ── Cerrar W&B: resumen + figura de curvas ───────────────
    if wandb_run is not None:
        wandb_run.summary['best_val_loss'] = best_val
        wandb_run.log({'training_curves': wandb.Image(str(out_dir / 'training_curves.png'))})
        wandb_run.finish()

    print(f"\n✓ Entrenamiento terminado. Mejor val_loss: {best_val:.4f}")
    print(f"  Checkpoint: {ckpt_path}")


def plot_curves(history: dict, save_path):
    """Dibuja las curvas de entrenamiento (etiquetas en inglés)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history['train_loss'], label='Train', color='steelblue')
    axes[0].plot(history['val_loss'],   label='Validation', color='coral')
    axes[0].set_title('Training loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['val_mae_mod'], label='Modified',     color='seagreen')
    axes[1].plot(history['val_mae_non'], label='Non-modified', color='gray')
    axes[1].set_title('Imputed-channel MAE (validation)')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle('Training Curves', fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    # Override opcional por línea de comandos:
    #   python train.py hexcnn          → arquitectura (tamaño = default)
    #   python train.py hexcnn l        → arquitectura + preset de tamaño (s|m|l)
    # Sin argumentos usa MODEL_NAME/MODEL_SIZE de la config. main() resuelve carpeta y run_tag.
    if len(sys.argv) > 1:
        MODEL_NAME = sys.argv[1]
    if len(sys.argv) > 2:
        MODEL_SIZE = sys.argv[2]
    main()
