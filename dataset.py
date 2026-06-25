"""
dataset.py
==========
Generación de datos ON-THE-FLY para la imputación de señales SiPM.

Por cada evento (vector de 61 cargas sanas) se genera una muestra de entrenamiento:
  - Se elige UN canal a apagar según la clase deseada (balanceo 50/50):
      * modified     → se apaga un canal que SÍ tenía señal (la red debe imputar)
      * non-modified → se apaga un canal que YA estaba a 0 (la red NO debe corregir)
  - Entrada a la red: matriz 2×61 = [cargas con el canal apagado] + [máscara binaria]
  - Target: el vector original completo de 61 canales (regresión 61→61)

Decisiones físicas (acordadas con los tutores):
  - Clip de negativos a 0 (ruido de calibración del ADC). Nunca abs().
  - Normalización POR EVENTO, por el máximo, DESPUÉS de apagar el canal.
    Input y target comparten el MISMO factor (máx post-máscara) → el target en el
    canal imputado puede superar 1.0 si ese canal era el más brillante. Por eso la
    red NO debe llevar sigmoid/clamp en la salida.
  - Sin filtrado de energía (requeriría calibración, fuera de scope).

Autor: Miguel Escudero (TFM)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# CONSTANTES DEL DETECTOR
# ─────────────────────────────────────────────────────────────

INACTIVE_CHANNELS = {1, 16, 18}                                  # canales sin SiPM físico
ACTIVE_CH         = sorted(set(range(64)) - INACTIVE_CHANNELS)   # 61 canales activos
ICH_TO_IDX        = {ich: i for i, ich in enumerate(ACTIVE_CH)}  # Ich físico → índice denso [0,60]
IDX_TO_ICH        = {i: ich for i, ich in enumerate(ACTIVE_CH)}  # índice denso → Ich físico
N_ACTIVE          = len(ACTIVE_CH)                               # 61

# dtype para parsear cada par (Rch, Ich) de un evento de golpe con np.frombuffer
# '<f4' = float32 little-endian (carga), '<i4' = int32 little-endian (canal)
REC_DTYPE = np.dtype([('rch', '<f4'), ('ich', '<i4')])


# ─────────────────────────────────────────────────────────────
# LECTURA BINARIA → MATRIZ DENSA (cargas crudas, sin normalizar)
# ─────────────────────────────────────────────────────────────

def load_dat_to_dense(filepath, max_events: int | None = None) -> np.ndarray:
    """
    Lee un archivo datas#.dat y devuelve la matriz densa de cargas (N, 61).

    IMPORTANTE: NO normaliza. La normalización por evento se hace en el Dataset,
    DESPUÉS de apagar el canal. Aquí solo se clipean los negativos a 0.

    Parameters
    ----------
    filepath : str | Path
        Ruta al .dat (raw string en Windows: r'E:\\Datos TFM\\Good\\Good\\datas002.dat').
    max_events : int, optional
        Tope de eventos a leer (para pruebas rápidas). None = todos.

    Returns
    -------
    np.ndarray (N, 61) float32 — cargas crudas clipeadas a [0, +inf), eventos no-vacíos.
    """
    data = Path(filepath).read_bytes()   # leemos todo el archivo de golpe (es <200 MB)
    buf  = memoryview(data)              # vista sin copia sobre los bytes
    n    = len(data)
    pos  = 0                             # puntero de byte

    rows = []   # lista de vectores densos (uno por evento)
    while pos < n:
        if max_events and len(rows) >= max_events:
            break

        nint = data[pos]   # Nint: nº de SiPMs disparados (uint8)
        pos += 1
        if nint == 0:
            continue

        block = nint * 8   # cada par (Rch, Ich) ocupa 4+4 bytes
        if pos + block > n:
            break          # archivo truncado

        rec = np.frombuffer(buf, dtype=REC_DTYPE, count=nint, offset=pos)
        pos += block

        ich = rec['ich']
        rch = rec['rch']

        # Construimos el vector denso de 61 posiciones a 0
        row = np.zeros(N_ACTIVE, dtype=np.float32)
        for r, c in zip(rch, ich):
            idx = ICH_TO_IDX.get(int(c))   # None si el canal es inactivo (1,16,18)
            if idx is not None:
                row[idx] = r
        rows.append(row)

    X = np.asarray(rows, dtype=np.float32)
    X = np.clip(X, 0, None)              # negativos (ruido ADC) → 0; nunca abs()

    # Descartamos eventos totalmente vacíos (no aportan nada y rompen la normalización)
    if len(X) > 0:
        X = X[X.sum(axis=1) > 0]
    return X


def load_positions(psipm_path) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga psipm.tsv y devuelve las posiciones (x, y) en orden denso [0,60].

    OJO: el archivo NO tiene cabecera (la primera fila es el canal 37). Lo leemos
    a mano para no arrastrar el bug de pandas que se comería esa fila.

    Returns
    -------
    x_sipm, y_sipm : np.ndarray (61,) en mm, alineados con el índice denso.
    """
    x_sipm = np.zeros(N_ACTIVE, dtype=np.float64)
    y_sipm = np.zeros(N_ACTIVE, dtype=np.float64)
    with open(psipm_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            ich = int(parts[0])
            idx = ICH_TO_IDX.get(ich)
            if idx is not None:
                x_sipm[idx] = float(parts[1])
                y_sipm[idx] = float(parts[2])
    return x_sipm, y_sipm


# ─────────────────────────────────────────────────────────────
# SPLIT TRAIN / VAL / TEST (por archivo, reproducible)
# ─────────────────────────────────────────────────────────────

def get_file_split(good_dir, n_val: int = 5, n_test: int = 5, seed: int = 42):
    """
    Reparte los archivos Good en train / val / test DISJUNTOS.

    FUENTE ÚNICA DE VERDAD: la usan train.py y imputation_eval.py, así que el split
    nunca discrepa entre entrenamiento y evaluación.

    - Split por ARCHIVO (cada .dat es un módulo físico distinto): mide la
      generalización a detectores NO vistos, no a eventos memorizados.
    - Barajado con seed fijo → reproducible y representativo (no coge los primeros/
      últimos archivos, que podrían ser de adquisición similar).

    Returns
    -------
    train_files, val_files, test_files : listas de Path (ordenadas dentro de cada grupo)
    """
    files = sorted(Path(good_dir).glob('datas*.dat'))
    assert len(files) > n_val + n_test, (
        f"Solo hay {len(files)} archivos, no caben {n_val} val + {n_test} test"
    )
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(files))     # orden barajado, fijo por el seed

    test_idx  = perm[:n_test]
    val_idx   = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    pick = lambda idx: sorted(files[i] for i in idx)   # ordenamos dentro del grupo (legible)
    return pick(train_idx), pick(val_idx), pick(test_idx)


# ─────────────────────────────────────────────────────────────
# DATASET ON-THE-FLY
# ─────────────────────────────────────────────────────────────

class SiPMImputationDataset(Dataset):
    """
    Dataset de imputación con generación on-the-fly y balanceo 50/50.

    Cada __getitem__ genera una muestra a partir de un evento sano:
      - El balanceo se fuerza por la PARIDAD del índice: idx par → modified,
        idx impar → non-modified. Con shuffle=True en el DataLoader, cada batch
        sale ~50/50, y globalmente es exactamente 50/50.

    Parameters
    ----------
    X_raw : np.ndarray (N, 61)
        Cargas crudas clipeadas (salida de load_dat_to_dense). NO normalizadas.
    seed : int
        Semilla del generador aleatorio (cambiarla por época para variar el masking).
    """

    def __init__(self, X_raw: np.ndarray, seed: int = 0):
        self.X = np.ascontiguousarray(X_raw, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x_raw = self.X[idx].copy()   # .copy() para no tocar el array compartido

        # Índices de canales con señal y a cero en ESTE evento
        active = np.flatnonzero(x_raw > 0)   # flatnonzero = where(cond)[0], más directo
        zeros  = np.flatnonzero(x_raw == 0)

        # ── Elegir canal a apagar según la clase (paridad del índice) ──
        if idx % 2 == 0 and len(active) > 0:
            ch = int(self.rng.choice(active))   # MODIFIED: apagamos un canal con señal
            is_modified = 1
        elif len(zeros) > 0:
            ch = int(self.rng.choice(zeros))    # NON-MODIFIED: apagamos un canal ya a 0
            is_modified = 0
        else:
            # Fallback raro (evento sin ceros): tratamos como modified
            ch = int(self.rng.choice(active))
            is_modified = 1

        # ── Apagar el canal y normalizar POR EVENTO (después de apagar) ──
        x_masked = x_raw.copy()
        x_masked[ch] = 0.0

        norm = x_masked.max()
        if norm == 0:
            norm = 1.0   # guard: evento con un único canal que era el apagado

        # Input y target comparten el MISMO factor de normalización (máx post-máscara).
        # → target[ch] puede ser > 1 si el canal apagado era el más brillante. Es correcto.
        x_input = x_masked / norm        # (61,) cargas normalizadas con el canal a 0
        target  = x_raw / norm           # (61,) vector original completo, misma escala

        # Máscara binaria: 1 = canal presente, 0 = canal apagado
        mask = np.ones(N_ACTIVE, dtype=np.float32)
        mask[ch] = 0.0

        # Entrada como matriz 2×61 (channels-first para PyTorch):
        # fila 0 = cargas normalizadas, fila 1 = máscara
        x_in = np.stack([x_input, mask], axis=0)   # (2, 61)

        return (
            torch.from_numpy(x_in),                          # (2, 61) float32
            torch.from_numpy(target),                        # (61,)   float32
            torch.tensor(ch, dtype=torch.long),              # canal apagado (índice denso)
            torch.tensor(is_modified, dtype=torch.long),     # 1=modified, 0=non-modified
        )


if __name__ == '__main__':
    # Smoke test: cargar un archivo Good y mirar una muestra
    PATH = r'E:\Datos TFM\Good\Good\datas002.dat'
    X = load_dat_to_dense(PATH, max_events=50_000)
    print(f"Eventos cargados: {X.shape}  (canales activos medios: {(X>0).sum(1).mean():.1f})")

    ds = SiPMImputationDataset(X, seed=0)
    x_in, target, ch, is_mod = ds[0]   # idx=0 → modified
    print(f"x_in shape   : {tuple(x_in.shape)}  (fila0=cargas, fila1=máscara)")
    print(f"target shape : {tuple(target.shape)}  max={target.max():.3f}  (puede ser >1)")
    print(f"canal apagado: idx={ch.item()} (Ich={IDX_TO_ICH[ch.item()]})  modified={is_mod.item()}")
    x_in2, _, _, is_mod2 = ds[1]        # idx=1 → non-modified
    print(f"muestra idx=1: modified={is_mod2.item()} (esperado 0)")
