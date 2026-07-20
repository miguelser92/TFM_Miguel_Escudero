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
import torch.nn.functional as F
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

from dataset import (SiPMImputationDataset, load_dat_to_dense, get_file_split,
                     load_positions, N_ACTIVE)
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
PATIENCE      = N_EPOCHS     # = sin corte temprano: entrena el presupuesto completo y
                            # guarda el mejor checkpoint por val_mae_mod (ver selección abajo).
                            # Datos infinitos on-the-fly → sin sobreajuste por entrenar las 40.
MAX_EVENTS    = 400_000     # tope de eventos por archivo y época (controla tiempo/RAM)
HUBER_DELTA   = 0.1         # robusto a outliers; datos ~[0,1] (algún target >1)

# Función de pérdida: 'huber' | 'mae' | 'mse', y variantes PHYSICS-INFORMED con
# sufijo '_dr' ('mse_dr', 'huber_dr') que AÑADEN a la loss base un término de error
# de POSICIÓN ΔR² (desplazamiento del centroide del evento al imputar). Cada una va a
# SU carpeta/run (sufijo de loss, salvo 'huber' que no lleva) → no se pisan.
LOSS          = 'mse'
RUN_SUFFIX    = '' if LOSS == 'huber' else f'_{LOSS}'

# Loss physics-informed (solo si LOSS acaba en '_dr'):  loss = base + LAMBDA_DR · ΔR².
# El ΔR² usa el centroide ponderado por Rch² (el mismo de la posición real). LAMBDA_DR
# se calibra para que ambos términos tengan magnitud comparable al inicio ("equilibrada").
PSIPM_PATH    = r'E:\Datos TFM\psipm.tsv'
LAMBDA_DR     = 0.01     # calibrado (smoke test): equilibra λ·ΔR² ≈ MSE al inicio
LAMBDA_EN     = 0.7      # peso del término de energía (|corrimiento medio de RchT| = |bias|); calibrado 50/50 al arranque (smoke test: MSE/|Δenergía|~0.69)

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

def _centroid(X, xs_t, ys_t, eps=1e-6):
    """Centroide XY ponderado por Rch² (diferenciable). X:(B,61) → (px, py):(B,)."""
    w    = X ** 2
    wsum = w.sum(dim=1) + eps
    return (w * xs_t).sum(dim=1) / wsum, (w * ys_t).sum(dim=1) / wsum


def delta_r(out, target, ch, xs_t, ys_t):
    """
    Desplazamiento ΔR (mm) por muestra del centroide al imputar el canal 'ch' con la
    predicción de la red, respecto al centroide del vector original. Imita el eval:
    los 60 canales sanos se dejan reales (target); solo el canal apagado toma out[ch].
    Diferenciable → el gradiente fluye por out[ch]. Devuelve ΔR (B,).
    """
    mask  = F.one_hot(ch, num_classes=target.shape[1]).to(target.dtype)   # (B,61)
    x_imp = target * (1 - mask) + out * mask            # canal apagado ← predicción
    px_o, py_o = _centroid(target, xs_t, ys_t)          # posición original
    px_i, py_i = _centroid(x_imp,  xs_t, ys_t)          # posición imputada
    return torch.sqrt((px_i - px_o) ** 2 + (py_i - py_o) ** 2 + 1e-12)


def energy_shift(out, target, ch):
    """
    |corrimiento medio del espectro| por batch: |media(out[ch] − target[ch])|. Como los
    60 canales sanos no cambian, RchT_imp − RchT_orig = out[ch] − target[ch]; su media es
    el BIAS de energía. Penaliza el SESGO sistemático (lo que ni el MSE ni el ΔR ven →
    ortogonal), forzando la conservación del espectro a primer orden. Diferenciable, por batch.
    """
    rows = torch.arange(out.shape[0], device=out.device)
    return (out[rows, ch] - target[rows, ch]).mean().abs()


def evaluate(model, loader, loss_fn, device, xs_t, ys_t):
    """
    Evalúa el modelo y devuelve (loss, mae_modified, mae_nonmod, dr_modified).

    El MAE se calcula SOLO sobre el canal imputado, separando modified (capacidad real
    de imputar) y non-modified (falsa corrección, ~0). dr_modified = ΔR medio de posición
    en los eventos modified (la métrica física, para monitorear la loss physics-informed).
    """
    model.eval()
    total_loss, n_total = 0.0, 0
    err_mod, n_mod = 0.0, 0
    err_non, n_non = 0.0, 0
    err_dr = 0.0                                  # ΔR acumulado sobre modified
    signed_mod = 0.0                              # error CON signo (para el bias) sobre modified

    with torch.no_grad():   # sin grafo de gradientes: más rápido y menos memoria
        for x_in, target, ch, is_mod in loader:
            x_in   = x_in.to(device)
            target = target.to(device)
            ch     = ch.to(device)
            is_mod = is_mod.to(device)

            out  = model(x_in)                       # (B, 61)
            loss = loss_fn(out, target)              # loss base sobre los 61 canales
            bs   = len(target)
            total_loss += loss.item() * bs
            n_total    += bs

            rows = torch.arange(bs, device=device)
            signed  = out[rows, ch] - target[rows, ch]     # error con signo del canal
            dr = delta_r(out, target, ch, xs_t, ys_t)

            m = is_mod.bool()
            err_mod += signed[m].abs().sum().item();  n_mod += m.sum().item()
            err_non += signed[~m].abs().sum().item(); n_non += (~m).sum().item()
            err_dr  += dr[m].sum().item()
            signed_mod += signed[m].sum().item()

    loss     = total_loss / max(n_total, 1)
    mae_mod  = err_mod / max(n_mod, 1)
    mae_non  = err_non / max(n_non, 1)
    dr_mod   = err_dr / max(n_mod, 1)
    bias_mod = signed_mod / max(n_mod, 1)
    return loss, mae_mod, mae_non, dr_mod, bias_mod


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
                    'loss': LOSS, 'physics_term': LOSS.split('_')[1] if '_' in LOSS else None,
                    'lambda_dr': LAMBDA_DR, 'lambda_en': LAMBDA_EN, 'huber_delta': HUBER_DELTA,
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

    # Pérdida base + término physics-informed opcional. LOSS = '<base>[_<term>]':
    #   base:  'huber' | 'mae' | 'mse'
    #   term:  'dr' (ΔR², posición)  |  'en' (|Δenergía|, sesgo del espectro)  |  ninguno
    parts     = LOSS.split('_')
    base_name = parts[0]
    phys_term = parts[1] if len(parts) > 1 else None
    assert phys_term in (None, 'dr', 'en'), f"término physics '{phys_term}' no válido ('dr'|'en')"
    if base_name == 'huber':
        loss_fn = nn.HuberLoss(delta=HUBER_DELTA)
    elif base_name == 'mae':
        loss_fn = nn.L1Loss()       # MAE = error absoluto medio
    elif base_name == 'mse':
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"LOSS '{LOSS}' no reconocida (base: 'huber'|'mae'|'mse', term: '_dr'|'_en')")
    lam  = {'dr': LAMBDA_DR, 'en': LAMBDA_EN}.get(phys_term)
    desc = {'dr': f'{LAMBDA_DR}·ΔR²', 'en': f'{LAMBDA_EN}·|Δenergía|'}.get(phys_term, '')
    print(f"Loss: {base_name}" + (f"  +  {desc}  (physics-informed)" if phys_term else ""))

    # Posiciones de los SiPM (para el término ΔR), como tensores en el device
    xs_np, ys_np = load_positions(PSIPM_PATH)
    xs_t = torch.tensor(xs_np, dtype=torch.float32, device=device)
    ys_t = torch.tensor(ys_np, dtype=torch.float32, device=device)

    history = {'train_loss': [], 'train_phys': [], 'val_loss': [],
               'val_mae_mod': [], 'val_mae_non': [], 'val_dr': [], 'val_bias': []}
    best_mae_mod = float('inf')   # seleccionamos el mejor checkpoint por val_mae_mod, NO por val_loss
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
        run_loss, run_phys, n_seen = 0.0, 0.0, 0
        for x_in, target, ch, is_mod in train_loader:
            x_in   = x_in.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            out  = model(x_in)
            loss = loss_fn(out, target)                       # término base (MSE/Huber/MAE)
            if phys_term == 'dr':
                extra = (delta_r(out, target, ch.to(device), xs_t, ys_t) ** 2).mean()   # ΔR²
            elif phys_term == 'en':
                extra = energy_shift(out, target, ch.to(device))                        # |Δenergía|
            if phys_term:
                loss = loss + lam * extra
                run_phys += extra.item() * len(target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # evita exploding gradients
            optimizer.step()

            run_loss += loss.item() * len(target)
            n_seen   += len(target)

        cur_lr = optimizer.param_groups[0]['lr']   # lr usado en esta época (antes del step)
        scheduler.step()
        train_loss = run_loss / max(n_seen, 1)
        train_phys = run_phys / max(n_seen, 1)

        # ── Validación ───────────────────────────────────────
        # Re-sembramos el rng del val_ds para que las máscaras sean IDÉNTICAS cada
        # época (si no, el rng con estado deriva y val se mide sobre canales distintos).
        val_ds.rng = np.random.default_rng(VAL_MASK_SEED)
        val_loss, mae_mod, mae_non, val_dr, val_bias = evaluate(model, val_loader, loss_fn, device, xs_t, ys_t)

        history['train_loss'].append(train_loss)
        history['train_phys'].append(train_phys)
        history['val_loss'].append(val_loss)
        history['val_mae_mod'].append(mae_mod)
        history['val_mae_non'].append(mae_non)
        history['val_dr'].append(val_dr)
        history['val_bias'].append(val_bias)

        if wandb_run is not None:
            wandb_run.log({'epoch': epoch, 'train_loss': train_loss, 'train_phys': train_phys,
                           'val_loss': val_loss, 'val_mae_mod': mae_mod, 'val_mae_non': mae_non,
                           'val_dr': val_dr, 'val_bias': val_bias, 'lr': cur_lr})

        flag = ''
        # Selección del mejor modelo y early-stopping por val_mae_mod (error en el canal
        # imputado), NO por val_loss: ésta está dominada por los ~60 canales sanos, satura y
        # elige la época por ruido (ver bitácora 30/06). Lo que queremos es imputar bien el
        # canal apagado → medimos por eso.
        if mae_mod < best_mae_mod:
            best_mae_mod = mae_mod
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
                'val_dr':       val_dr,
                'val_bias':     val_bias,
            }, ckpt_path)
            flag = '  ✓ best'
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:3d}/{N_EPOCHS} | "
              f"train={train_loss:.4f} val={val_loss:.4f} | "
              f"MAE(mod)={mae_mod:.4f} MAE(non)={mae_non:.4f} dR={val_dr:.4f} bias={val_bias:+.4f} | "
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
        wandb_run.summary['best_val_mae_mod'] = best_mae_mod
        wandb_run.log({'training_curves': wandb.Image(str(out_dir / 'training_curves.png'))})
        wandb_run.finish()

    print(f"\n✓ Entrenamiento terminado. Mejor val_mae_mod: {best_mae_mod:.4f}")
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
    #   python train.py hexcnn            → arquitectura (tamaño y loss = config)
    #   python train.py hexcnn l          → + preset de tamaño (s|m|l)
    #   python train.py hexcnn l huber    → + loss (huber|mae|mse; '_dr' = +ΔR², p.ej. mse_dr)
    # Cada combinación va a SU carpeta/run (sufijo de loss salvo huber, + sufijo de tamaño)
    # → no se pisan. Sin argumentos usa la config. main() resuelve carpeta y run_tag.
    #   python train.py hexcnn s mse --shuffle     → ablación: grafo barajado (control de geometría)
    #   python train.py hexcnn s mse --shuffle 7   → idem con otra seed de barajado
    #   python train.py hexcnn s mse --geom vec    → HexConv con geometría métrica (Δx,Δy, anisótropo)
    #   python train.py hexcnn s mse --geom dist   → idem pero solo distancia (isótropo)
    argv = sys.argv[1:]
    shuffle_seed = None
    if '--shuffle' in argv:
        i = argv.index('--shuffle')
        if i + 1 < len(argv) and argv[i + 1].isdigit():
            shuffle_seed = int(argv[i + 1]); del argv[i:i + 2]
        else:
            shuffle_seed = 0; del argv[i]
        MODEL_KWARGS = {**MODEL_KWARGS, 'shuffle_seed': shuffle_seed}

    geom = None
    if '--geom' in argv:
        i = argv.index('--geom')
        assert i + 1 < len(argv) and argv[i + 1] in ('vec', 'dist'), "uso: --geom vec|dist"
        geom = argv[i + 1]; del argv[i:i + 2]
        MODEL_KWARGS = {**MODEL_KWARGS, 'geom': geom}

    # --attn N [--heads H]: añade N bloques de auto-atención global tras las HexConv.
    attn = 0
    if '--attn' in argv:
        i = argv.index('--attn')
        assert i + 1 < len(argv) and argv[i + 1].isdigit(), "uso: --attn N"
        attn = int(argv[i + 1]); del argv[i:i + 2]
        MODEL_KWARGS = {**MODEL_KWARGS, 'attn': attn}
        if '--heads' in argv:
            j = argv.index('--heads')
            assert j + 1 < len(argv) and argv[j + 1].isdigit(), "uso: --heads H"
            MODEL_KWARGS = {**MODEL_KWARGS, 'n_heads': int(argv[j + 1])}; del argv[j:j + 2]

    # --tag STR: etiqueta libre al final del nombre de carpeta (para réplicas sin pisar).
    tag = None
    if '--tag' in argv:
        i = argv.index('--tag')
        assert i + 1 < len(argv), "uso: --tag NOMBRE"
        tag = argv[i + 1]; del argv[i:i + 2]

    if len(argv) > 0:
        MODEL_NAME = argv[0]
    if len(argv) > 1:
        MODEL_SIZE = argv[1]
    if len(argv) > 2:
        LOSS = argv[2]
        RUN_SUFFIX = '' if LOSS == 'huber' else f'_{LOSS}'
    if shuffle_seed is not None:
        RUN_SUFFIX = f'{RUN_SUFFIX}_shuf{shuffle_seed}'   # carpeta/run propios → no pisa el real
    if geom is not None:
        RUN_SUFFIX = f'{RUN_SUFFIX}_g{geom}'              # p.ej. _gvec / _gdist → carpeta propia
    if attn > 0:
        RUN_SUFFIX = f'{RUN_SUFFIX}_attn{attn}'           # p.ej. _attn2 → carpeta propia
    if tag is not None:
        RUN_SUFFIX = f'{RUN_SUFFIX}_{tag}'                # réplica u otra variante → carpeta propia
    main()
