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
    # Cargamos TODO el archivo a memoria de una vez (un .dat es <200 MB). Mucho más
    # rápido que ir pidiéndole bytes al disco dentro del bucle.
    data = Path(filepath).read_bytes()   # bytes crudos del archivo entero
    buf  = memoryview(data)              # vista sobre esos bytes SIN copiarlos (la usa np.frombuffer)
    n    = len(data)                     # tamaño total del archivo en bytes
    pos  = 0                             # puntero: por qué byte vamos leyendo

    rows = []   # aquí acumulamos un vector denso (61,) por cada evento
    while pos < n:                       # recorremos el archivo evento a evento hasta el final
        # Tope opcional de eventos (para pruebas rápidas): paramos al alcanzarlo
        if max_events and len(rows) >= max_events:
            break

        # ── Cabecera del evento: 1 byte = Nint (cuántos SiPMs dispararon) ──
        nint = data[pos]   # indexar 1 byte en Python da directamente el int (rango 0-255)
        pos += 1           # avanzamos el puntero: ya hemos consumido el byte de Nint
        if nint == 0:
            continue       # evento vacío (raro): no hay pares que leer → al siguiente

        # ── Cuerpo del evento: Nint pares (Rch float32 + Ich int32) = 8 bytes cada uno ──
        block = nint * 8   # nº de bytes que ocupa el cuerpo de este evento
        if pos + block > n:
            break          # no quedan bytes suficientes → archivo truncado, paramos

        # np.frombuffer reinterpreta 'block' bytes como un array de 'nint' registros
        # estructurados (rch, ich), SIN copiar. count = nº de registros; offset = desde dónde.
        rec = np.frombuffer(buf, dtype=REC_DTYPE, count=nint, offset=pos)
        pos += block       # avanzamos el puntero al final de este evento

        ich = rec['ich']   # array (nint,) con los IDs de canal que dispararon
        rch = rec['rch']   # array (nint,) con sus cargas, en el mismo orden

        # ── Densificación: del formato disperso (nint pares) al vector fijo de 61 ──
        row = np.zeros(N_ACTIVE, dtype=np.float32)   # vector denso: todo a 0 por defecto
        for r, c in zip(rch, ich):                   # recorremos los pares (carga, canal) del evento
            idx = ICH_TO_IDX.get(int(c))             # Ich físico → índice denso [0,60]; None si inactivo (1,16,18)
            if idx is not None:                      # ignoramos canales inactivos o IDs basura
                row[idx] = r                         # colocamos la carga en su posición del vector denso
        rows.append(row)                             # guardamos el evento ya densificado

    # ── Post-proceso de toda la matriz ──
    X = np.asarray(rows, dtype=np.float32)   # lista de N vectores → matriz (N, 61)
    X = np.clip(X, 0, None)                  # cargas negativas (ruido del ADC) → 0. NUNCA abs() (inventaría señal)

    # Descartamos eventos totalmente vacíos: no aportan nada y romperían la
    # normalización por evento (dividir por un máximo de 0).
    if len(X) > 0:
        X = X[X.sum(axis=1) > 0]   # máscara booleana: nos quedamos solo con filas con algo de señal
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

    def __init__(self, X_raw: np.ndarray, seed: int = 0,
                 n_dead=1, dead_mode: str = 'cluster', nbr=None,
                 norm_mode: str = 'max', channel_norm: bool = False):
        # ascontiguousarray: garantiza memoria contigua y dtype float32 (acceso rápido;
        # evita sorpresas si X_raw venía de un slicing no contiguo de otro array).
        self.X = np.ascontiguousarray(X_raw, dtype=np.float32)
        # Generador aleatorio PROPIO del dataset (no el global de numpy) → reproducible
        # y aislado. train.py le pasa una semilla distinta por época para variar el masking.
        self.rng = np.random.default_rng(seed)

        # ── MULTI-DEAD ──
        # n_dead: int (k fijo) o (kmin, kmax) → k variable por muestra (curriculum mezclado).
        # dead_mode: 'cluster' = los k muertos son CONTIGUOS (caso físico real: sombras,
        #            conexiones en serie, efectos térmicos) | 'scatter' = k al azar por el
        #            detector (CONTROL: mismo nº de muertos, sin contigüidad).
        self.n_dead    = n_dead
        self.dead_mode = dead_mode
        # ── NORMALIZACIÓN POR EVENTO ──
        # 'max' (histórico): divide por el máximo post-máscara. Depende de UN canal:
        #        si el muerto era el más brillante, el factor cambia mucho y target>1.
        # 'sum' (RchT): divide por la SUMA de los canales visibles. Mucho más estable
        #        (promedia 60 canales), no la secuestra un canal ruidoso, y es el
        #        proxy de energía del evento. Escala distinta → la salida sigue lineal.
        assert norm_mode in ('max', 'sum'), "norm_mode: 'max' | 'sum'"
        self.norm_mode = norm_mode
        # ── NORMALIZACION POR CANAL (opcional) ──
        # El detector tiene un gradiente fijo: el canal central recoge ~4.3x mas carga
        # que el de borde (corr -0.88 con la distancia al centro). Dividir cada canal
        # por su escala tipica EQUALIZA los 61 -> la red no gasta capacidad aprendiendo
        # esa estructura conocida. La prediccion se des-equaliza al evaluar.
        self.channel_norm = channel_norm
        if channel_norm:
            from hex_geometry import get_channel_scale
            self.chan_scale = get_channel_scale()
        else:
            self.chan_scale = None
        if nbr is None and dead_mode == 'cluster':
            # Import PEREZOSO: hex_geometry importa de dataset, así que a nivel de módulo
            # habría ciclo. Aquí dataset ya está cargado del todo → seguro.
            from hex_geometry import get_neighbor_matrix
            nbr = get_neighbor_matrix()
        self.nbr = nbr

    def __len__(self) -> int:
        # PyTorch usa esto para saber cuántas muestras hay (= nº de eventos).
        return len(self.X)

    def _sample_k(self) -> int:
        """Nº de canales a apagar en esta muestra (fijo o sorteado en el rango)."""
        if isinstance(self.n_dead, int):
            return self.n_dead
        kmin, kmax = self.n_dead
        return int(self.rng.integers(kmin, kmax + 1))

    def _grow_cluster(self, seed_ch: int, k: int) -> np.ndarray:
        """
        Clúster CONTIGUO de k sensores: parte del semilla y crece agregando, una a una,
        una vecina real elegida al azar de la frontera actual. Reproduce el fallo físico
        realista (una zona del detector se apaga junta).
        """
        cluster = [int(seed_ch)]
        while len(cluster) < k:
            frontera = {int(j) for c in cluster for j in self.nbr[c]
                        if j >= 0 and int(j) not in cluster}
            if not frontera:
                break                      # clúster aislado (no ocurre en esta malla)
            cluster.append(int(self.rng.choice(sorted(frontera))))
        return np.array(cluster, dtype=np.int64)

    def _scatter(self, seed_ch: int, k: int) -> np.ndarray:
        """Control DISPERSO: k sensores al azar por todo el detector (sin contigüidad)."""
        otros = np.setdiff1d(np.arange(N_ACTIVE), [seed_ch])
        extra = self.rng.choice(otros, size=min(k - 1, len(otros)), replace=False)
        return np.concatenate([[seed_ch], extra]).astype(np.int64)

    def __getitem__(self, idx: int):
        # PyTorch llama a esto para obtener la muestra 'idx'. Aquí fabricamos la muestra
        # de imputación AL VUELO a partir del evento sano X[idx].
        x_raw = self.X[idx].copy()   # .copy(): trabajamos sobre una copia, no tocamos el array compartido
        # Espacio de trabajo: crudo, o equalizado por canal si channel_norm
        x_work = x_raw / self.chan_scale if self.channel_norm else x_raw

        # ── Qué canales tienen señal y cuáles están a 0 en ESTE evento ──
        active = np.flatnonzero(x_raw > 0)    # índices con carga > 0 (flatnonzero = where(cond)[0])
        zeros  = np.flatnonzero(x_raw == 0)   # índices a 0 (no dispararon)

        # ── Elegir la SEMILLA del fallo, forzando la clase por la PARIDAD del índice ──
        # (con shuffle=True en el DataLoader, esto da ~50/50 modified/non-modified por batch)
        if idx % 2 == 0 and len(active) > 0:
            seed_ch = int(self.rng.choice(active))   # MODIFIED: semilla en un canal con señal
        elif len(zeros) > 0:
            seed_ch = int(self.rng.choice(zeros))    # NON-MODIFIED: semilla en un canal ya a 0
        else:
            seed_ch = int(self.rng.choice(active))   # fallback raro: evento sin ceros

        # ── Extender la semilla a k canales muertos (contiguos o dispersos) ──
        k = self._sample_k()
        if k <= 1:
            dead_idx = np.array([seed_ch], dtype=np.int64)
        elif self.dead_mode == 'cluster':
            dead_idx = self._grow_cluster(seed_ch, k)
        else:
            dead_idx = self._scatter(seed_ch, k)

        # Con k>1 la etiqueta se generaliza: 'modified' = ALGUNO de los muertos tenía señal
        # (para k=1 coincide exactamente con el criterio de siempre). Ojo: con k>1 el
        # balanceo 50/50 pasa a ser aproximado, porque un clúster sembrado en un cero
        # puede arrastrar vecinos con señal.
        is_modified = int((x_raw[dead_idx] > 0).any())

        # ── Apagar los canales elegidos y normalizar POR EVENTO (DESPUÉS de apagar) ──
        x_masked = x_work.copy()  # copia del evento donde simularemos el fallo
        x_masked[dead_idx] = 0.0  # "matamos" los canales: ponemos su carga a 0

        # factor de escala sobre los canales VISIBLES (post-máscara)
        norm = x_masked.max() if self.norm_mode == 'max' else x_masked.sum() / N_ACTIVE
        if norm == 0:
            norm = 1.0   # guard anti división por cero: evento cuyo único canal con señal era el apagado

        # Input y target comparten el MISMO factor 'norm' (el máx post-máscara). Por eso
        # target[ch] puede salir > 1 si el canal apagado era el más brillante → es CORRECTO,
        # y obliga a que la salida de la red sea lineal (sin sigmoide/clamp).
        x_input = x_masked / norm        # (61,) entrada: cargas normalizadas con el canal a 0
        target  = x_work   / norm        # (61,) objetivo: vector original COMPLETO, misma escala

        # Máscara binaria que le dice a la red qué canales están apagados
        mask = np.ones(N_ACTIVE, dtype=np.float32)   # 1 = canal presente
        mask[dead_idx] = 0.0                         # 0 = canal apagado (a imputar)

        # Vector de muertos (complementario de la máscara): 1 donde hay que imputar.
        # Sustituye al índice escalar 'ch' de la versión de un solo canal; para k=1
        # tiene un único 1, así que las métricas dan exactamente lo mismo que antes.
        dead = np.zeros(N_ACTIVE, dtype=np.float32)
        dead[dead_idx] = 1.0

        # Apilamos en una matriz 2×61 (channels-first, como espera PyTorch):
        #   fila 0 = cargas normalizadas,  fila 1 = máscara
        x_in = np.stack([x_input, mask], axis=0)   # (2, 61)

        # Devolvemos la tupla que el DataLoader agrupará en batches:
        return (
            torch.from_numpy(x_in),                          # entrada  (2, 61) float32
            torch.from_numpy(target),                        # objetivo (61,)   float32 (los 61 canales)
            torch.from_numpy(dead),                          # (61,) 1 = canal apagado a imputar
            torch.tensor(is_modified, dtype=torch.long),     # etiqueta: 1=modified, 0=non-modified (no la ve la red)
        )


if __name__ == '__main__':
    # Smoke test: cargar un archivo Good y mirar una muestra
    PATH = r'E:\Datos TFM\Good\Good\datas002.dat'
    X = load_dat_to_dense(PATH, max_events=50_000)
    print(f"Eventos cargados: {X.shape}  (canales activos medios: {(X>0).sum(1).mean():.1f})")

    ds = SiPMImputationDataset(X, seed=0)
    x_in, target, dead, is_mod = ds[0]   # idx=0 → modified
    print(f"x_in shape   : {tuple(x_in.shape)}  (fila0=cargas, fila1=máscara)")
    print(f"target shape : {tuple(target.shape)}  max={target.max():.3f}  (puede ser >1)")
    idxs = dead.nonzero().flatten().tolist()
    print(f"canal apagado: {[IDX_TO_ICH[i] for i in idxs]} (Ich)  modified={is_mod.item()}")
    _, _, _, is_mod2 = ds[1]             # idx=1 → non-modified
    print(f"muestra idx=1: modified={is_mod2.item()} (esperado 0)")

    # Multi-dead: clúster contiguo vs disperso
    for mode in ('cluster', 'scatter'):
        d = SiPMImputationDataset(X, seed=0, n_dead=4, dead_mode=mode)
        ich = [IDX_TO_ICH[i] for i in d[0][2].nonzero().flatten().tolist()]
        print(f"k=4 modo {mode:8s}: Ich apagados = {sorted(ich)}")
    d = SiPMImputationDataset(X, seed=0, n_dead=(1, 4), dead_mode='cluster')
    ks = [int(d[i][2].sum()) for i in range(0, 12)]
    print(f"k variable (1,4): tamaños muestreados = {ks}")
