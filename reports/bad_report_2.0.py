"""
bad_report.py  (v2.0)
=====================
Genera un informe PDF de diagnóstico para TODOS los archivos de la carpeta Bad\\Bad.

Para cada archivo datas#.dat detecta el/los SiPM averiados y produce DOS páginas:

  Página 1 (diagnóstico básico, 2x2):
    1. Mapa hexagonal de fracción de actividad (posición física real) — el sensor muerto resalta
    2. Mapa hexagonal de carga media por sensor
    3. Flood map XY (centro de gravedad Rch²)
    4. Barras de fracción de actividad por canal

  Página 2 (reagrupación del flood map por área de sensor, AÑADIDA EN v2.0):
    5. Mapa hexagonal donde cada sensor se colorea según la señal del flood map
       acumulada dentro de su área circular
    6. Barras 1D de esa misma señal acumulada, un valor por canal

Además, una página resumen al final con la tabla archivo → sensor(es) muerto(s).

Novedades v2.0:
  - Segunda página por archivo con los paneles 5 y 6 (señal del flood map acumulada
    por área de sensor). Para cada sensor se suma la señal del flood map que cae
    dentro de un círculo centrado en su posición física, y se normaliza por el nº de
    eventos. El radio (apotema del hexágono) se estima de la distancia mediana entre
    sensores vecinos con scipy.spatial.cKDTree.

Uso:
    python bad_report.py
    python bad_report.py --limit 3              # solo los 3 primeros (prueba rápida)
    python bad_report.py --max-events 300000    # cap de eventos por archivo
    python bad_report.py --bad-dir "E:\\Datos TFM\\Bad\\Bad" --out informe.pdf

Autor: Miguel Escudero (TFM)
"""

import argparse
import sys
import time
import numpy as np
import matplotlib

# La consola de Windows usa cp1252 por defecto y no sabe imprimir ✓, —, etc.
# Forzamos UTF-8 en stdout para evitar UnicodeEncodeError al imprimir el progreso.
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
matplotlib.use('Agg')   # backend sin ventana: solo escribimos a archivo (más rápido y estable)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import cKDTree
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN POR DEFECTO
# ─────────────────────────────────────────────────────────────

DEFAULT_BAD_DIR = r'E:\Datos TFM\Bad\Bad'
DEFAULT_PSIPM   = r'E:\Datos TFM\psipm.tsv'
DEFAULT_OUT     = r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\reports\bad_report20.pdf'

# Constantes del detector
INACTIVE   = {1, 16, 18}
ACTIVE_CH  = sorted(set(range(64)) - INACTIVE)          # 61 canales activos
ICH_TO_IDX = {ich: i for i, ich in enumerate(ACTIVE_CH)} # Ich físico → índice denso [0,60]
IDX_TO_ICH = {i: ich for i, ich in enumerate(ACTIVE_CH)} # índice denso → Ich físico
N_ACTIVE   = len(ACTIVE_CH)

# dtype estructurado para parsear cada par (Rch, Ich) de un evento de golpe
# '<f4' = float32 little-endian (carga), '<i4' = int32 little-endian (canal)
REC_DTYPE = np.dtype([('rch', '<f4'), ('ich', '<i4')])


# ─────────────────────────────────────────────────────────────
# CARGA DE POSICIONES
# ─────────────────────────────────────────────────────────────

def load_positions(psipm_path: str):
    """
    Carga psipm.tsv y devuelve diccionarios x_pos, y_pos {Ich: mm}.

    El archivo no tiene cabecera: cada línea es 'Ich  x  y'.
    """
    x_pos, y_pos = {}, {}
    with open(psipm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()        # divide por cualquier espacio/tab
            ich   = int(parts[0])
            x_pos[ich] = float(parts[1])
            y_pos[ich] = float(parts[2])
    return x_pos, y_pos


def build_position_luts(x_pos: dict, y_pos: dict):
    """
    Construye look-up tables indexadas por Ich (0-63) para vectorizar el cálculo.

    Devuelve arrays de tamaño 64 donde el índice es el Ich físico:
      x_lut[ich], y_lut[ich] → posición en mm (0 si el canal no existe)
      idx_lut[ich]           → índice denso [0,60], o -1 si el canal es inactivo
    """
    x_lut   = np.zeros(64, dtype=np.float64)
    y_lut   = np.zeros(64, dtype=np.float64)
    idx_lut = np.full(64, -1, dtype=np.int64)   # -1 = canal inactivo/desconocido

    for ich in ACTIVE_CH:
        if ich in x_pos:
            x_lut[ich]   = x_pos[ich]
            y_lut[ich]   = y_pos[ich]
            idx_lut[ich] = ICH_TO_IDX[ich]

    return x_lut, y_lut, idx_lut


# ─────────────────────────────────────────────────────────────
# LECTURA BINARIA VECTORIZADA + ACUMULACIÓN DE ESTADÍSTICAS
# ─────────────────────────────────────────────────────────────

def process_file(
    filepath: Path,
    x_lut: np.ndarray,
    y_lut: np.ndarray,
    idx_lut: np.ndarray,
    max_events: int | None = None,
):
    """
    Lee un archivo .dat en una sola pasada y acumula las estadísticas necesarias
    para el informe, SIN almacenar todos los eventos en memoria.

    Estrategia de eficiencia:
      - Cargamos el archivo entero en memoria (son <200 MB) de una vez
      - Parseamos cada evento con np.frombuffer (vectorizado), no byte a byte
      - Calculamos el centroide XY al vuelo y solo guardamos pos_x, pos_y (2 floats/evento)

    Returns
    -------
    dict con:
      count_per_ch  (61,) — nº de disparos por sensor
      charge_per_ch (61,) — suma de carga por sensor
      mean_charge   (61,) — carga media por sensor (solo eventos donde disparó)
      frac_active   (61,) — fracción de eventos en que disparó cada sensor
      pos_x, pos_y  (M,)  — posiciones reconstruidas (centro de gravedad Rch²)
      n_events      int
    """
    # Leemos todo el archivo a memoria de una vez: mucho más rápido que f.read(4) repetido
    data = Path(filepath).read_bytes()
    buf  = memoryview(data)   # vista sin copia sobre los bytes
    n    = len(data)
    pos  = 0                  # puntero de byte actual

    # Pre-alojamos pos_x/pos_y con tamaño generoso y recortamos al final
    # (más rápido que list.append en millones de iteraciones)
    cap = max_events if max_events else 4_000_000
    pos_x = np.zeros(cap, dtype=np.float32)
    pos_y = np.zeros(cap, dtype=np.float32)

    # En vez de acumular por sensor con np.add.at en cada evento (muy lento),
    # vamos guardando los índices densos y cargas de los disparos en listas,
    # y al final hacemos UN solo np.bincount. Es mucho más rápido.
    didx_chunks = []   # lista de arrays de índices densos (disparos rch>0)
    rch_chunks  = []   # lista de arrays de cargas correspondientes

    n_events = 0
    while pos < n:
        if max_events and n_events >= max_events:
            break

        # ── Paso 1: leer Nint (1 byte uint8) ───────────────────
        nint = data[pos]   # indexar bytes en Python da directamente el int (0-255)
        pos += 1
        if nint == 0:
            continue

        # ── Paso 2: parsear los Nint pares (Rch, Ich) de golpe ─
        block_bytes = nint * 8   # cada par = 4+4 bytes
        if pos + block_bytes > n:
            break                # archivo truncado: paramos

        # np.frombuffer interpreta los bytes como array estructurado sin copiarlos
        # count=nint registros, offset=pos bytes desde el inicio del buffer
        rec = np.frombuffer(buf, dtype=REC_DTYPE, count=nint, offset=pos)
        pos += block_bytes

        rch = rec['rch']   # (nint,) cargas
        ich = rec['ich']   # (nint,) canales (0-63)

        # Máscara de seguridad: canal en rango físico [0,63] (datos ruidosos pueden traer basura)
        in_range = (ich >= 0) & (ich < 64)
        ich = ich[in_range]
        rch = rch[in_range]

        didx  = idx_lut[ich]   # índice denso de cada canal (-1 si inactivo: 1,16,18)
        valid = didx >= 0
        ich   = ich[valid]
        rch   = rch[valid]
        didx  = didx[valid]

        # ── Centro de gravedad Rch² (posición XY del evento) ──
        # clip a 0: las cargas negativas (ruido ADC) no deben contribuir al peso
        w = np.clip(rch, 0, None) ** 2     # pesos Rch²
        wsum = w.sum()
        if wsum > 0:
            pos_x[n_events] = (w * x_lut[ich]).sum() / wsum
            pos_y[n_events] = (w * y_lut[ich]).sum() / wsum

        # ── Guardamos los disparos reales (rch>0) para acumular al final ─
        fired = rch > 0
        didx_chunks.append(didx[fired])
        rch_chunks.append(rch[fired])

        n_events += 1

    # Recortamos los arrays de posición al número real de eventos leídos
    pos_x = pos_x[:n_events]
    pos_y = pos_y[:n_events]

    # ── Acumulación final por sensor con un único bincount ────
    if didx_chunks:
        # np.concatenate une todos los chunks en dos arrays planos
        didx_all = np.concatenate(didx_chunks)
        rch_all  = np.concatenate(rch_chunks).astype(np.float64)
        # np.bincount(idx, minlength=N): cuenta ocurrencias de cada índice → count por sensor
        count_per_ch  = np.bincount(didx_all, minlength=N_ACTIVE).astype(np.int64)
        # con weights: suma los pesos (cargas) de cada índice → carga total por sensor
        charge_per_ch = np.bincount(didx_all, weights=rch_all, minlength=N_ACTIVE)
    else:
        count_per_ch  = np.zeros(N_ACTIVE, dtype=np.int64)
        charge_per_ch = np.zeros(N_ACTIVE, dtype=np.float64)

    # Carga media por sensor (evitando división por cero con where)
    mean_charge = np.divide(
        charge_per_ch, count_per_ch,
        out=np.zeros_like(charge_per_ch),
        where=count_per_ch > 0,
    )
    frac_active = count_per_ch / max(n_events, 1)

    return {
        'count_per_ch':  count_per_ch,
        'charge_per_ch': charge_per_ch,
        'mean_charge':   mean_charge,
        'frac_active':   frac_active,
        'pos_x':         pos_x,
        'pos_y':         pos_y,
        'n_events':      n_events,
    }


# ─────────────────────────────────────────────────────────────
# DETECCIÓN DE SENSORES MUERTOS
# ─────────────────────────────────────────────────────────────

def detect_dead_sensors(frac_active: np.ndarray):
    """
    Marca como sospechosos los sensores con actividad anormalmente baja.

    Criterio adaptativo: por debajo del 15% de la mediana de actividad,
    o por debajo del 1% absoluto. Adaptarse a cada archivo evita falsos
    positivos en archivos globalmente menos activos.

    Returns
    -------
    np.ndarray de índices densos sospechosos (ordenados de menor a mayor actividad).
    """
    median_frac = np.median(frac_active)
    threshold   = max(0.01, 0.15 * median_frac)
    suspects    = np.where(frac_active < threshold)[0]
    # Ordenamos por actividad ascendente: el más muerto primero
    return suspects[np.argsort(frac_active[suspects])]


# ─────────────────────────────────────────────────────────────
# GENERACIÓN DE LA PÁGINA DE UN ARCHIVO
# ─────────────────────────────────────────────────────────────

def make_file_page(
    pdf: PdfPages,
    file_id: str,
    stats: dict,
    dead_idx: np.ndarray,
    x_lut: np.ndarray,
    y_lut: np.ndarray,
):
    """Dibuja la página de diagnóstico de un archivo y la añade al PDF."""
    frac_active = stats['frac_active']
    mean_charge = stats['mean_charge']
    pos_x, pos_y = stats['pos_x'], stats['pos_y']
    n_events = stats['n_events']

    # Posiciones físicas en orden denso [0,60] para los scatter hexagonales
    x_sipm = np.array([x_lut[IDX_TO_ICH[i]] for i in range(N_ACTIVE)])
    y_sipm = np.array([y_lut[IDX_TO_ICH[i]] for i in range(N_ACTIVE)])

    dead_set = set(dead_idx.tolist())

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── Panel 1: mapa hexagonal de fracción de actividad ──────
    ax = axes[0, 0]
    vals = frac_active * 100
    norm = Normalize(vmin=vals.min(), vmax=vals.max())
    cmap = plt.get_cmap('viridis')
    for idx in range(N_ACTIVE):
        ich = IDX_TO_ICH[idx]
        is_dead = idx in dead_set
        ax.scatter(x_sipm[idx], y_sipm[idx], s=420,
                   c=[cmap(norm(vals[idx]))],
                   edgecolors='red' if is_dead else 'black',
                   linewidths=2.5 if is_dead else 0.5, zorder=3)
        ax.text(x_sipm[idx], y_sipm[idx], str(ich), ha='center', va='center',
                fontsize=6, color='white')
    ax.set_aspect('equal')
    ax.set_title('Active fraction (%) per SiPM')
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    # ── Panel 2: mapa hexagonal de carga media ────────────────
    ax = axes[0, 1]
    vals = mean_charge
    norm = Normalize(vmin=vals.min(), vmax=vals.max())
    for idx in range(N_ACTIVE):
        ich = IDX_TO_ICH[idx]
        is_dead = idx in dead_set
        ax.scatter(x_sipm[idx], y_sipm[idx], s=420,
                   c=[cmap(norm(vals[idx]))],
                   edgecolors='red' if is_dead else 'black',
                   linewidths=2.5 if is_dead else 0.5, zorder=3)
        ax.text(x_sipm[idx], y_sipm[idx], str(ich), ha='center', va='center',
                fontsize=6, color='white')
    ax.set_aspect('equal')
    ax.set_title('Mean charge per SiPM')
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    # ── Panel 3: flood map XY ─────────────────────────────────
    ax = axes[1, 0]
    h = ax.hist2d(pos_x, pos_y, bins=64, cmap='plasma')
    plt.colorbar(h[3], ax=ax, label='Counts', fraction=0.046, pad=0.04)
    # Superponemos las posiciones de los SiPMs como referencia
    ax.scatter(x_sipm, y_sipm, facecolors='none', edgecolors='cyan',
               s=80, linewidths=0.8, zorder=3)
    ax.set_aspect('equal')
    ax.set_title('Flood map')
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')

    # ── Panel 4: barras de fracción activa por canal ──────────
    ax = axes[1, 1]
    xb = np.arange(N_ACTIVE)
    colors = ['#d62728' if i in dead_set else '#1f77b4' for i in range(N_ACTIVE)]
    ax.bar(xb, frac_active * 100, color=colors)
    ax.set_title('Active fraction (%) per channel')
    ax.set_xlabel('SiPM channel (Ich)')
    ax.set_ylabel('Active fraction (%)')
    ax.set_xticks(xb[::2])
    ax.set_xticklabels([IDX_TO_ICH[i] for i in xb[::2]], rotation=90, fontsize=6)

    # ── Título de la página con el resumen del diagnóstico ────
    if len(dead_idx) > 0:
        dead_str = ', '.join(
            f"Ich={IDX_TO_ICH[i]} ({frac_active[i]*100:.2f}%)" for i in dead_idx
        )
    else:
        dead_str = "no dead sensor detected"
    fig.suptitle(
        f"Bad file {file_id}   |   {n_events:,} events   |   Dead sensor: {dead_str}",
        fontsize=14, fontweight='bold',
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])   # deja hueco arriba para el suptitle
    pdf.savefig(fig)
    plt.close(fig)   # libera memoria: clave al generar decenas de páginas


# ─────────────────────────────────────────────────────────────
# REAGRUPACIÓN DEL FLOOD MAP POR ÁREA DE SENSOR (v2.0)
# ─────────────────────────────────────────────────────────────

def compute_accumulated_signal(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    n_events: int,
    x_lut: np.ndarray,
    y_lut: np.ndarray,
    bins: int = 250,
    pad: float = 0.5,
):
    """
    Reagrupa el flood map en el área física de cada sensor (v2.0).

    Idea: en vez de mirar qué sensor disparó, miramos cuántos eventos se
    reconstruyen DENTRO del área de cada sensor. Para ello hacemos un
    histograma 2D fino de las posiciones (pos_x, pos_y) y, por cada sensor,
    sumamos las cuentas de los bins cuyo centro cae dentro de un círculo
    centrado en su posición física.

    El radio del círculo es la APOTEMA del hexágono (distancia centro→lado),
    estimada como la mitad de la distancia mediana entre sensores vecinos.

    Parameters
    ----------
    pos_x, pos_y : (M,) posiciones reconstruidas de los eventos
    n_events     : nº de eventos (para normalizar)
    x_lut, y_lut : LUTs de posición indexadas por Ich
    bins         : resolución del histograma fino (250 = muy fino)
    pad          : margen en mm alrededor del array de sensores

    Returns
    -------
    acc_norm : (61,) señal acumulada por sensor, normalizada por n_events
    radius   : float, apotema del hexágono en mm (radio del círculo de acumulación)
    """
    # Posiciones de los sensores en orden denso [0,60]
    x_sipm = np.array([x_lut[IDX_TO_ICH[i]] for i in range(N_ACTIVE)])
    y_sipm = np.array([y_lut[IDX_TO_ICH[i]] for i in range(N_ACTIVE)])

    # ── Radio = apotema = mitad de la distancia mediana entre vecinos ──
    # cKDTree.query(k=2): el vecino más cercano de cada punto.
    # El índice 0 es el propio punto (distancia 0), el índice 1 es el vecino real.
    pts = np.column_stack([x_sipm, y_sipm])
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    nearest_dist = dists[:, 1]                       # distancia al vecino real
    radius = np.median(nearest_dist) / 2 * 0.96      # *0.96 para que no se solapen

    # ── Histograma 2D fino de las posiciones reconstruidas ──
    xr = [x_sipm.min() - pad, x_sipm.max() + pad]
    yr = [y_sipm.min() - pad, y_sipm.max() + pad]
    h, xe, ye = np.histogram2d(pos_x, pos_y, bins=bins, range=[xr, yr])

    # Centros de los bins (no los bordes), para medir distancias correctamente
    xc = (xe[:-1] + xe[1:]) / 2
    yc = (ye[:-1] + ye[1:]) / 2
    # indexing='ij' → XX[i,j]=xc[i], YY[i,j]=yc[j]: la rejilla coincide con h[i,j]
    XX, YY = np.meshgrid(xc, yc, indexing='ij')

    # ── Para cada sensor, sumar las cuentas dentro de su círculo ──
    acc = np.zeros(N_ACTIVE)
    for k in range(N_ACTIVE):
        sx, sy = x_sipm[k], y_sipm[k]
        # distancia de cada bin al centro del sensor k
        dist = np.sqrt((XX - sx) ** 2 + (YY - sy) ** 2)
        mask = dist <= radius                        # bins dentro del círculo
        acc[k] = h[mask].sum()

    acc_norm = acc / max(n_events, 1)                # normalizar por nº de eventos
    return acc_norm, radius


def make_accumulated_page(
    pdf: PdfPages,
    file_id: str,
    stats: dict,
    dead_idx: np.ndarray,
    x_lut: np.ndarray,
    y_lut: np.ndarray,
):
    """
    Dibuja la SEGUNDA página de un archivo (v2.0) y la añade al PDF.

    Contiene los dos plots nuevos de reagrupación del flood map por área:
      - Mapa hexagonal coloreado por señal acumulada por sensor
      - Barras 1D de la misma señal, un valor por canal
    """
    pos_x, pos_y = stats['pos_x'], stats['pos_y']
    n_events = stats['n_events']

    # Señal acumulada por área de sensor + radio (apotema)
    acc_norm, radius = compute_accumulated_signal(
        pos_x, pos_y, n_events, x_lut, y_lut,
    )

    # Posiciones de los sensores en orden denso [0,60]
    x_sipm = np.array([x_lut[IDX_TO_ICH[i]] for i in range(N_ACTIVE)])
    y_sipm = np.array([y_lut[IDX_TO_ICH[i]] for i in range(N_ACTIVE)])

    dead_set = set(dead_idx.tolist())

    # Página vertical: mapa hexagonal arriba, barras abajo
    fig, (ax_hex, ax_bar) = plt.subplots(2, 1, figsize=(14, 15))

    # ── Panel 5: mapa hexagonal de señal acumulada ────────────
    ax_hex.set_facecolor('white')
    ax_hex.axis('off')
    ax_hex.set_title('Accumulated flood-map signal per sensor area')
    norm = Normalize(vmin=acc_norm.min(), vmax=acc_norm.max())
    cmap_obj = plt.get_cmap('YlOrRd')

    for idx in range(N_ACTIVE):
        ich = IDX_TO_ICH[idx]
        cx, cy = x_sipm[idx], y_sipm[idx]
        color = cmap_obj(norm(acc_norm[idx]))
        # Resaltamos en rojo los sensores ya marcados como muertos (igual que página 1)
        is_dead = idx in dead_set
        edge = '#ff4444' if is_dead else '#111'
        lw   = 2.5 if is_dead else 0.4
        # OJO geometría: radius es la apotema (centro→lado). RegularPolygon necesita
        # el radio centro→vértice = apotema / cos(30º). orientation=pi/6 = pointy-top.
        ax_hex.add_patch(mpatches.RegularPolygon(
            (cx, cy), numVertices=6, radius=radius / np.cos(np.pi / 6),
            orientation=np.pi / 6,
            facecolor=color, edgecolor=edge, linewidth=lw))
        ax_hex.text(cx, cy + 0.3, str(ich), ha='center', va='center',
                    fontsize=5.5, color='black')
        ax_hex.text(cx, cy - 0.3, f'{acc_norm[idx]:.3f}', ha='center', va='center',
                    fontsize=4, color='#333')

    ax_hex.set_xlim(x_sipm.min() - 3, x_sipm.max() + 3)
    ax_hex.set_ylim(y_sipm.min() - 3, y_sipm.max() + 3)
    ax_hex.set_aspect('equal')
    sm = ScalarMappable(cmap=cmap_obj, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax_hex, label='Accumulated flood-map signal (normalised)',
                 fraction=0.04, pad=0.02)

    # ── Panel 6: barras de señal acumulada por canal ──────────
    ax_bar.set_facecolor('white')
    xb = np.arange(N_ACTIVE)
    colors = ['#d62728' if i in dead_set else '#1f77b4' for i in range(N_ACTIVE)]
    ax_bar.bar(xb, acc_norm, color=colors)
    ax_bar.set_title('Accumulated flood-map signal per channel')
    ax_bar.set_xlabel('SiPM channel (Ich)')
    ax_bar.set_ylabel('Accumulated flood-map signal (normalised)')
    ax_bar.set_xticks(xb[::2])
    ax_bar.set_xticklabels([IDX_TO_ICH[i] for i in xb[::2]], rotation=90, fontsize=7)
    ax_bar.grid(axis='y', linewidth=0.5, alpha=0.5)

    # ── Título de la página ───────────────────────────────────
    fig.suptitle(
        f"Bad file {file_id}   |   flood-map signal regrouped per sensor area (v2.0)",
        fontsize=14, fontweight='bold',
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])   # deja hueco arriba para el suptitle
    pdf.savefig(fig)
    plt.close(fig)   # libera memoria: clave al generar decenas de páginas


# ─────────────────────────────────────────────────────────────
# PÁGINA RESUMEN
# ─────────────────────────────────────────────────────────────

def make_summary_pages(pdf: PdfPages, summary: list):
    """
    Crea una o varias páginas con la tabla archivo → sensores muertos.

    summary : lista de tuplas (file_id, n_events, [(ich, frac_pct), ...])
    """
    rows_per_page = 28   # límite para que la tabla quepa en una página
    n = len(summary)

    for start in range(0, n, rows_per_page):
        chunk = summary[start:start + rows_per_page]

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title('Bad files — dead sensor summary', fontsize=15, fontweight='bold', pad=20)

        table_data = []
        for file_id, n_events, deads in chunk:
            if deads:
                dead_str = ', '.join(f"Ich={ich} ({pct:.2f}%)" for ich, pct in deads)
            else:
                dead_str = '—'
            table_data.append([file_id, f"{n_events:,}", dead_str])

        table = ax.table(
            cellText=table_data,
            colLabels=['File', 'Events', 'Dead sensor(s)'],
            cellLoc='left', colLoc='left', loc='upper center',
            colWidths=[0.18, 0.18, 0.64],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)

        # Cabecera en negrita con fondo gris
        for col in range(3):
            cell = table[0, col]
            cell.set_facecolor('#cccccc')
            cell.set_text_props(fontweight='bold')

        pdf.savefig(fig)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Informe PDF de diagnóstico de archivos Bad')
    parser.add_argument('--bad-dir',    type=str, default=DEFAULT_BAD_DIR)
    parser.add_argument('--psipm',      type=str, default=DEFAULT_PSIPM)
    parser.add_argument('--out',        type=str, default=DEFAULT_OUT)
    parser.add_argument('--limit',      type=int, default=None,
                        help='Procesar solo los N primeros archivos (prueba rápida)')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Limitar eventos leídos por archivo')
    args = parser.parse_args()

    bad_dir = Path(args.bad_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)   # crea reports/ si no existe

    # Cargamos posiciones una sola vez
    x_pos, y_pos = load_positions(args.psipm)
    x_lut, y_lut, idx_lut = build_position_luts(x_pos, y_pos)
    print(f"SiPMs con posición: {len(x_pos)}")

    files = sorted(bad_dir.glob('datas*.dat'))
    if args.limit:
        files = files[:args.limit]
    print(f"Archivos a procesar: {len(files)}\n")

    summary = []   # para la página resumen

    t_global = time.time()
    with PdfPages(out_path) as pdf:
        # Procesamos archivo a archivo (memoria baja: una página, se cierra, siguiente).
        # El resumen va como última página del documento.
        for k, f in enumerate(files, 1):
            file_id = f.stem   # 'datas016' sin extensión
            t0 = time.time()
            stats = process_file(f, x_lut, y_lut, idx_lut, max_events=args.max_events)
            dead_idx = detect_dead_sensors(stats['frac_active'])

            deads = [(IDX_TO_ICH[i], stats['frac_active'][i] * 100) for i in dead_idx]
            summary.append((file_id, stats['n_events'], deads))

            make_file_page(pdf, file_id, stats, dead_idx, x_lut, y_lut)
            # Segunda página (v2.0): reagrupación del flood map por área de sensor
            make_accumulated_page(pdf, file_id, stats, dead_idx, x_lut, y_lut)

            dead_txt = ', '.join(f"Ich={ich}" for ich, _ in deads) or '—'
            print(f"[{k:2d}/{len(files)}] {file_id}  "
                  f"{stats['n_events']:>9,} ev  "
                  f"dead: {dead_txt:20s}  ({time.time()-t0:.1f}s)")

        # Página(s) resumen al final del documento
        make_summary_pages(pdf, summary)

        # Metadatos del PDF
        d = pdf.infodict()
        d['Title']  = 'Bad files diagnostic report'
        d['Author'] = 'Miguel Escudero (TFM)'

    print(f"\n✓ Informe generado: {out_path}")
    print(f"  Total: {len(files)} archivos en {time.time()-t_global:.1f}s")


if __name__ == '__main__':
    main()
