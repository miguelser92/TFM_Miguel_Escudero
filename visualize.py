"""
visualize.py
============
Visualización robusta y completa de los datos del detector PET hexagonal.

Genera un informe visual con 8 paneles que cubren:
  1. Geometría del detector (mapa de SiPMs con Ich)
  2. Histograma de energía (distribución de RchT)
  3. Llenado del detector (histograma 2D XY)
  4. Distribución de carga por canal (box plot)
  5. Mapa de actividad media por SiPM
  6. Distribución de Nint (SiPMs disparados por evento)
  7. Correlación entre canales vecinos
  8. Comparación llenado bueno vs. degradado (canal apagado)

Uso:
    # Con datos reales:
    python visualize.py --data Good/datas1.dat --psipm psipm.tsv

    # Modo demo (sin datos, genera sintéticos):
    python visualize.py --demo

    # Guardar reporte completo:
    python visualize.py --data Good/datas1.dat --save report.png

Autor: Miguel Escudero (TFM)
"""

import argparse
import struct
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
import pandas as pd

# ─────────────────────────────────────────────────────────────
# CONSTANTES DEL DETECTOR
# ─────────────────────────────────────────────────────────────

N_CH_TOTAL        = 64
INACTIVE_CH       = {1, 16, 18}
ACTIVE_CH         = sorted(set(range(N_CH_TOTAL)) - INACTIVE_CH)   # 61 canales
N_ACTIVE          = len(ACTIVE_CH)                                  # 61
ICH_TO_IDX        = {ich: idx for idx, ich in enumerate(ACTIVE_CH)}
IDX_TO_ICH        = {idx: ich for ich, idx in ICH_TO_IDX.items()}

# Parámetros hexágono axial k=5 → 61 celdas
HEX_K      = 5
HEX_RADIUS = HEX_K - 1   # 4

# Tema de colores
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
ACCENT     = "#58a6ff"
ACCENT2    = "#f78166"
ACCENT3    = "#3fb950"
ACCENT4    = "#d2a8ff"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#30363d"

# ─────────────────────────────────────────────────────────────
# GEOMETRÍA HEXAGONAL
# ─────────────────────────────────────────────────────────────

def generate_hex_grid(k: int = HEX_K) -> List[Tuple[int, int]]:
    """Genera coordenadas axiales (q, r) del hexágono de lado k."""
    r_max = k - 1
    coords = []
    for q in range(-r_max, r_max + 1):
        for r in range(max(-r_max, -q - r_max), min(r_max, -q + r_max) + 1):
            coords.append((q, r))
    coords.sort(key=lambda c: (c[0], c[1]))
    return coords


HEX_COORDS   = generate_hex_grid()
AXIAL_TO_IDX = {c: i for i, c in enumerate(HEX_COORDS)}
IDX_TO_AXIAL = {i: c for i, c in enumerate(HEX_COORDS)}
# Asignación por defecto: orden ascendente de ACTIVE_CH → coordenadas axiales
ICH_TO_AXIAL = {ich: HEX_COORDS[i] for i, ich in enumerate(ACTIVE_CH)}
AXIAL_TO_ICH = {v: k for k, v in ICH_TO_AXIAL.items()}

# Vecinos hexagonales
HEX_DIR = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]


def axial_to_cartesian(q: int, r: int, size: float = 1.0) -> Tuple[float, float]:
    """Coordenadas axiales → cartesianas (pointy-top)."""
    x = size * (np.sqrt(3) * q + np.sqrt(3) / 2 * r)
    y = size * (3 / 2 * r)
    return x, y


# Posiciones cartesianas de cada SiPM
SIPM_XY: Dict[int, Tuple[float, float]] = {
    ich: axial_to_cartesian(*ICH_TO_AXIAL[ich]) for ich in ACTIVE_CH
}

HEX_NEIGHBOR_MATRIX = np.full((N_ACTIVE, 6), -1, dtype=int)
for hi, (q, r) in IDX_TO_AXIAL.items():
    for d, (dq, dr) in enumerate(HEX_DIR):
        nb = (q + dq, r + dr)
        if nb in AXIAL_TO_IDX:
            HEX_NEIGHBOR_MATRIX[hi, d] = AXIAL_TO_IDX[nb]


# ─────────────────────────────────────────────────────────────
# LECTURA DEL BINARIO
# ─────────────────────────────────────────────────────────────

def read_binary(path: str, max_events: int = 500_000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lee el archivo binario datas#.dat.

    Returns
    -------
    X      : np.ndarray (N, 61) — cargas por canal (sin normalizar)
    n_ints : np.ndarray (N,)    — Nint por evento
    """
    X       = []
    n_ints  = []

    with open(path, 'rb') as f:
        while len(X) < max_events:
            raw = f.read(1)
            if not raw:
                break
            nint = struct.unpack('B', raw)[0]
            if nint == 0:
                continue

            row = np.zeros(N_ACTIVE, dtype=np.float32)
            ok  = True
            for _ in range(nint):
                rch_b = f.read(4)
                ich_b = f.read(4)
                if len(rch_b) < 4 or len(ich_b) < 4:
                    ok = False
                    break
                rch = struct.unpack('<f', rch_b)[0]
                ich = struct.unpack('<i', ich_b)[0]
                if ich in ICH_TO_IDX:
                    row[ICH_TO_IDX[ich]] = rch

            if ok:
                X.append(row)
                n_ints.append(nint)

    return np.array(X, dtype=np.float32), np.array(n_ints, dtype=np.int32)


# ─────────────────────────────────────────────────────────────
# GENERADOR DE DATOS SINTÉTICOS (modo demo)
# ─────────────────────────────────────────────────────────────

def generate_demo_data(n_events: int = 20_000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera datos sintéticos realistas que imitan el detector PET:
    - La señal decae exponencialmente con la distancia al punto de interacción
    - Se aplica la máscara probabilística P = exp(-Rch/120)
    - Posiciones de interacción distribuidas uniformemente en el hexágono
    """
    rng = np.random.default_rng(seed)
    X      = np.zeros((n_events, N_ACTIVE), dtype=np.float32)
    n_ints = np.zeros(n_events, dtype=np.int32)

    # Posiciones XY de cada SiPM
    xy = np.array([SIPM_XY[ich] for ich in ACTIVE_CH])   # (61, 2)

    for i in range(n_events):
        # Punto de interacción aleatorio dentro del hexágono
        r_max = 4.0
        while True:
            px = rng.uniform(-r_max, r_max)
            py = rng.uniform(-r_max, r_max)
            # Comprobar si está dentro del hexágono (aprox.)
            if abs(px) + abs(py) / np.sqrt(3) < r_max * 1.15:
                break

        # Carga inversamente proporcional a dist² (modelo sencillo de luz)
        dist2 = (xy[:, 0] - px)**2 + (xy[:, 1] - py)**2 + 0.5
        charge = rng.exponential(1.0) * 5000.0 / dist2    # escala física
        charge = charge.astype(np.float32)

        # Máscara probabilística P = exp(-Rch/120)
        mask_prob = np.exp(-charge / 120.0)
        fired = rng.random(N_ACTIVE) > mask_prob
        charge[~fired] = 0.0

        X[i] = charge
        n_ints[i] = int(fired.sum())

    return X, n_ints


# ─────────────────────────────────────────────────────────────
# POSICIONAMIENTO XY
# ─────────────────────────────────────────────────────────────

def compute_positions(
    X: np.ndarray,
    psipm: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula coordenadas XY de interacción.
    Usa psipm.tsv si se proporciona, o la geometría axial por defecto.
    """
    if psipm is not None:
        x_arr = np.array([psipm.loc[ich, 'xsipm'] if ich in psipm.index
                          else SIPM_XY[ich][0] for ich in ACTIVE_CH])
        y_arr = np.array([psipm.loc[ich, 'ysipm'] if ich in psipm.index
                          else SIPM_XY[ich][1] for ich in ACTIVE_CH])
    else:
        x_arr = np.array([SIPM_XY[ich][0] for ich in ACTIVE_CH])
        y_arr = np.array([SIPM_XY[ich][1] for ich in ACTIVE_CH])

    rch2   = X ** 2
    rcht2  = rch2.sum(axis=1, keepdims=True)
    rcht2[rcht2 == 0] = 1.0
    pos_x = (rch2 * x_arr).sum(axis=1) / rcht2[:, 0]
    pos_y = (rch2 * y_arr).sum(axis=1) / rcht2[:, 0]
    return pos_x, pos_y


# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL DEL ESTILO
# ─────────────────────────────────────────────────────────────

def setup_style():
    matplotlib.rcParams.update({
        'figure.facecolor':     DARK_BG,
        'axes.facecolor':       PANEL_BG,
        'axes.edgecolor':       GRID_COLOR,
        'axes.labelcolor':      TEXT_COLOR,
        'axes.titlecolor':      TEXT_COLOR,
        'axes.titlesize':       11,
        'axes.labelsize':       9,
        'axes.grid':            True,
        'grid.color':           GRID_COLOR,
        'grid.linewidth':       0.5,
        'xtick.color':          TEXT_COLOR,
        'ytick.color':          TEXT_COLOR,
        'xtick.labelsize':      8,
        'ytick.labelsize':      8,
        'text.color':           TEXT_COLOR,
        'legend.facecolor':     PANEL_BG,
        'legend.edgecolor':     GRID_COLOR,
        'legend.labelcolor':    TEXT_COLOR,
        'legend.fontsize':      8,
        'figure.dpi':           130,
        'font.family':          'monospace',
    })


# ─────────────────────────────────────────────────────────────
# PANELES DE VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────

def panel_detector_geometry(ax: plt.Axes):
    """Panel 1: geometría del detector con etiquetas Ich."""
    ax.set_facecolor(DARK_BG)
    hex_size = 0.85

    for ich in ACTIVE_CH:
        q, r = ICH_TO_AXIAL[ich]
        cx, cy = axial_to_cartesian(q, r)
        color = "#1c2d40"
        edge  = ACCENT
        lw    = 0.8

        patch = mpatches.RegularPolygon(
            (cx, cy), numVertices=6,
            radius=hex_size * 2 / np.sqrt(3),
            orientation=0,
            facecolor=color, edgecolor=edge, linewidth=lw,
        )
        ax.add_patch(patch)
        ax.text(cx, cy + 0.05, f"{ich}", ha='center', va='center',
                fontsize=5.5, color=TEXT_COLOR, fontweight='bold')

    # Canales inactivos (fantasma)
    for ich in INACTIVE_CH:
        # Posición estimada: asignar coordenadas al centro
        ax.text(0, -5.5 + list(INACTIVE_CH).index(ich) * 0.5,
                f"Ich={ich} (inactivo)", ha='center', va='center',
                fontsize=6, color=ACCENT2, alpha=0.7)

    ax.set_xlim(-9, 9); ax.set_ylim(-9, 9)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title("Geometría del Detector\n(61 SiPMs activos · Ich)", pad=8)
    ax.text(0, 8.2, "▼ Vértice superior", ha='center', va='center',
            fontsize=6, color=ACCENT, alpha=0.8)


def panel_energy_spectrum(ax: plt.Axes, X: np.ndarray):
    """Panel 2: espectro de energía (distribución de RchT)."""
    rcht = X.sum(axis=1)
    rcht = rcht[rcht > 0]

    n, bins, patches = ax.hist(
        rcht, bins=150, color=ACCENT, alpha=0.85,
        edgecolor='none', density=False,
    )

    # Colorear la región de ventana de energía
    peak_bin = bins[np.argmax(n)]
    e_min = np.percentile(rcht, 5)
    e_max = np.percentile(rcht, 99)
    for patch, left in zip(patches, bins[:-1]):
        if e_min <= left <= e_max:
            patch.set_facecolor(ACCENT3)
            patch.set_alpha(0.9)

    ax.axvline(np.median(rcht), color=ACCENT2, lw=1.5, ls='--',
               label=f"Mediana: {np.median(rcht):.0f}")
    ax.axvline(np.percentile(rcht, 5),  color=ACCENT4, lw=1, ls=':',
               label=f"p5 = {np.percentile(rcht,5):.0f}")
    ax.axvline(np.percentile(rcht, 99), color=ACCENT4, lw=1, ls=':',
               label=f"p99 = {np.percentile(rcht,99):.0f}")

    ax.set_title("Espectro de Energía (RchT)")
    ax.set_xlabel("Carga Total (RchT)")
    ax.set_ylabel("Eventos")
    ax.legend(fontsize=7)
    ax.yaxis.set_major_locator(MaxNLocator(5))

    # Anotación
    ax.text(0.97, 0.95, f"N = {len(rcht):,}", transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color=TEXT_COLOR)


def panel_detector_filling(ax: plt.Axes, X: np.ndarray,
                           psipm: Optional[pd.DataFrame] = None,
                           title: str = "Llenado del Detector"):
    """Panel 3: histograma 2D del llenado XY."""
    pos_x, pos_y = compute_positions(X, psipm)

    # Filtrar outliers extremos
    px = np.percentile(np.abs(pos_x), 99)
    py = np.percentile(np.abs(pos_y), 99)
    mask = (np.abs(pos_x) < px * 1.5) & (np.abs(pos_y) < py * 1.5)
    pos_x, pos_y = pos_x[mask], pos_y[mask]

    h, xe, ye = np.histogram2d(pos_x, pos_y, bins=200)
    h_smooth = h.copy()
    # Suavizado gaussiano ligero
    from scipy.ndimage import gaussian_filter
    h_smooth = gaussian_filter(h_smooth, sigma=1.2)

    im = ax.imshow(
        h_smooth.T, origin='lower',
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
        cmap='plasma', aspect='equal',
        norm=LogNorm(vmin=max(1, h_smooth.min()+0.1), vmax=h_smooth.max()),
    )
    plt.colorbar(im, ax=ax, label='Cuentas (log)', fraction=0.046, pad=0.02)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.text(0.02, 0.97, f"N={len(pos_x):,}", transform=ax.transAxes,
            va='top', fontsize=7, color=TEXT_COLOR)


def panel_nint_distribution(ax: plt.Axes, n_ints: np.ndarray):
    """Panel 4: distribución del número de SiPMs disparados (Nint)."""
    counts = np.bincount(n_ints, minlength=N_ACTIVE + 1)
    nz = np.where(counts > 0)[0]

    ax.bar(nz, counts[nz], color=ACCENT4, alpha=0.85, edgecolor='none', width=0.8)
    ax.axvline(np.mean(n_ints),   color=ACCENT2, lw=1.5, ls='--',
               label=f"Media: {np.mean(n_ints):.1f}")
    ax.axvline(np.median(n_ints), color=ACCENT3, lw=1.5, ls=':',
               label=f"Mediana: {np.median(n_ints):.0f}")

    ax.set_title("Distribución de Nint\n(SiPMs disparados/evento)")
    ax.set_xlabel("Nint")
    ax.set_ylabel("Eventos")
    ax.legend(fontsize=7)
    ax.set_xlim(0, N_ACTIVE + 1)


def panel_mean_activity(ax: plt.Axes, X: np.ndarray):
    """Panel 5: mapa hexagonal de actividad media por SiPM."""
    ax.set_facecolor(DARK_BG)

    mean_charge = X.mean(axis=0)          # (61,)
    frac_active = (X > 0).mean(axis=0)    # fracción de eventos donde disparó

    norm_c = Normalize(vmin=mean_charge.min(), vmax=mean_charge.max())
    cmap   = plt.get_cmap('YlOrRd')
    hex_sz = 0.85

    for dense_idx in range(N_ACTIVE):
        ich    = IDX_TO_ICH[dense_idx]
        q, r   = ICH_TO_AXIAL[ich]
        cx, cy = axial_to_cartesian(q, r)
        color  = cmap(norm_c(mean_charge[dense_idx]))

        patch = mpatches.RegularPolygon(
            (cx, cy), numVertices=6,
            radius=hex_sz * 2 / np.sqrt(3),
            orientation=0,            facecolor=color, edgecolor=DARK_BG, linewidth=0.5,
        )
        ax.add_patch(patch)
        ax.text(cx, cy + 0.05, f"{ich}", ha='center', va='center',
                fontsize=4.8, color='black', alpha=0.8)

    sm = ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Carga media', fraction=0.046, pad=0.02)

    ax.set_xlim(-9, 9); ax.set_ylim(-9, 9)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title("Actividad Media por SiPM\n(Carga promedio)")


def panel_charge_distribution(ax: plt.Axes, X: np.ndarray, n_show: int = 15):
    """Panel 6: distribución de carga para los canales más activos."""
    mean_act = (X > 0).mean(axis=0)
    top_idx  = np.argsort(mean_act)[-n_show:][::-1]

    data_plot = []
    labels    = []
    for idx in top_idx:
        vals = X[:, idx]
        vals = vals[vals > 0]
        if len(vals) > 10:
            data_plot.append(vals)
            labels.append(f"Ich={IDX_TO_ICH[idx]}")

    if not data_plot:
        ax.text(0.5, 0.5, "Sin datos", ha='center', va='center', transform=ax.transAxes)
        return

    bp = ax.boxplot(
        data_plot, labels=labels,
        patch_artist=True, notch=False,
        medianprops={'color': ACCENT2,  'lw': 1.5},
        boxprops    ={'facecolor': '#1c3a5e', 'edgecolor': ACCENT, 'lw': 0.8},
        whiskerprops={'color': ACCENT,  'lw': 0.8},
        capprops    ={'color': ACCENT,  'lw': 0.8},
        flierprops  ={'marker': '.', 'color': ACCENT4, 'markersize': 2, 'alpha': 0.3},
    )

    ax.set_title(f"Distribución de Carga\n(Top {len(data_plot)} canales más activos)")
    ax.set_ylabel("Rch")
    ax.set_yscale('log')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)


def panel_channel_correlation(ax: plt.Axes, X: np.ndarray, center_ich: int = None):
    """Panel 7: correlación entre un canal central y sus vecinos."""
    # Elegir el canal con mayor actividad media como referencia
    if center_ich is None:
        center_idx = int(np.argmax((X > 0).mean(axis=0)))
    else:
        center_idx = ICH_TO_IDX.get(center_ich, 0)

    center_ich_val = IDX_TO_ICH[center_idx]
    q0, r0 = ICH_TO_AXIAL[center_ich_val]

    # Vecinos directos + radio 2
    neighbors_1 = []
    neighbors_2 = []
    for dq, dr in HEX_DIR:
        nb1 = (q0 + dq, r0 + dr)
        if nb1 in AXIAL_TO_IDX and AXIAL_TO_ICH.get(nb1) is not None:
            nb_ich = AXIAL_TO_ICH[nb1]
            neighbors_1.append(ICH_TO_IDX[nb_ich])
        for dq2, dr2 in HEX_DIR:
            nb2 = (q0 + dq + dq2, r0 + dr + dr2)
            if (nb2 in AXIAL_TO_IDX and AXIAL_TO_ICH.get(nb2) is not None
                    and nb2 != (q0, r0)):
                nb_ich = AXIAL_TO_ICH[nb2]
                nb2_idx = ICH_TO_IDX[nb_ich]
                if nb2_idx not in neighbors_1:
                    neighbors_2.append(nb2_idx)

    center_vals = X[:, center_idx]
    mask        = center_vals > 0

    colors_1 = plt.cm.Blues(np.linspace(0.5, 0.9, len(neighbors_1)))
    colors_2 = plt.cm.Greens(np.linspace(0.4, 0.8, len(neighbors_2)))

    for nb_idx, col in zip(neighbors_1[:6], colors_1):
        nb_vals = X[mask, nb_idx]
        ctr_sub = center_vals[mask]
        ax.scatter(ctr_sub[::5], nb_vals[::5],  # subsample
                   s=1.5, alpha=0.4, color=col)

    for nb_idx, col in zip(neighbors_2[:6], colors_2):
        nb_vals = X[mask, nb_idx]
        ctr_sub = center_vals[mask]
        ax.scatter(ctr_sub[::10], nb_vals[::10],
                   s=1.0, alpha=0.2, color=col)

    ax.set_title(f"Correlación Vecinos\n(Canal central: Ich={center_ich_val})")
    ax.set_xlabel(f"Rch (Ich={center_ich_val})")
    ax.set_ylabel("Rch vecinos")

    leg_elements = [
        mpatches.Patch(color=plt.cm.Blues(0.7), label='Radio 1 (vecinos directos)'),
        mpatches.Patch(color=plt.cm.Greens(0.6), label='Radio 2'),
    ]
    ax.legend(handles=leg_elements, fontsize=7)


def panel_degraded_vs_restored(ax: plt.Axes, X: np.ndarray,
                                channel_to_kill: int = None,
                                psipm: Optional[pd.DataFrame] = None):
    """Panel 8: llenado degradado (canal apagado) vs original."""
    if channel_to_kill is None:
        # Elegir el canal más central (el más activo)
        channel_to_kill = int(np.argmax((X > 0).mean(axis=0)))

    X_deg = X.copy()
    X_deg[:, channel_to_kill] = 0.0

    pos_x_ok,  pos_y_ok  = compute_positions(X,     psipm)
    pos_x_bad, pos_y_bad = compute_positions(X_deg, psipm)

    # Usar el mismo extent para ambos
    all_x = np.concatenate([pos_x_ok, pos_x_bad])
    all_y = np.concatenate([pos_y_ok, pos_y_bad])
    px99  = np.percentile(np.abs(all_x), 99) * 1.2
    py99  = np.percentile(np.abs(all_y), 99) * 1.2
    ext   = [-px99, px99, -py99, py99]
    bins  = 120

    h_ok,  *_ = np.histogram2d(pos_x_ok,  pos_y_ok,  bins=bins, range=[ext[:2], ext[2:]])
    h_bad, *_ = np.histogram2d(pos_x_bad, pos_y_bad, bins=bins, range=[ext[:2], ext[2:]])

    # Diferencia normalizada
    h_diff = (h_ok - h_bad) / (h_ok.max() + 1e-8)
    im = ax.imshow(
        h_diff.T, origin='lower', extent=ext,
        cmap='RdBu_r', aspect='equal',
        vmin=-h_diff.std() * 3, vmax=h_diff.std() * 3,
    )
    plt.colorbar(im, ax=ax, label='Δ normalizado', fraction=0.046, pad=0.02)

    ich_killed = IDX_TO_ICH[channel_to_kill]
    ax.set_title(f"Impacto del Fallo\n(Ich={ich_killed} apagado vs. original)")
    ax.set_xlabel("X"); ax.set_ylabel("Y")

    # Anotación de la posición del canal apagado
    cx, cy = SIPM_XY[ich_killed]
    scale  = px99 / 8.0  # escalar a mm reales
    ax.plot(cx * scale, cy * scale, 'x', color=ACCENT2,
            markersize=10, markeredgewidth=2, label=f"Ich={ich_killed}")
    ax.legend(fontsize=7)


# ─────────────────────────────────────────────────────────────
# FIGURA PRINCIPAL: REPORTE COMPLETO
# ─────────────────────────────────────────────────────────────

def plot_full_report(
    X:         np.ndarray,
    n_ints:    np.ndarray,
    psipm:     Optional[pd.DataFrame] = None,
    title:     str = "Análisis del Detector PET Hexagonal",
    save_path: Optional[str] = None,
    show:      bool = True,
) -> plt.Figure:
    """
    Genera el informe visual completo con 8 paneles.

    Parameters
    ----------
    X       : (N, 61) carga bruta sin normalizar
    n_ints  : (N,) número de SiPMs por evento
    psipm   : DataFrame con xsipm, ysipm por ich (opcional)
    title   : título del informe
    save_path: si se especifica, guarda la figura
    show    : si True, llama a plt.show()
    """
    setup_style()

    fig = plt.figure(figsize=(22, 16), facecolor=DARK_BG)
    fig.suptitle(
        title,
        fontsize=16, color=TEXT_COLOR, fontweight='bold',
        y=0.98,
    )

    # Subtítulo con estadísticas globales
    rcht = X.sum(axis=1)
    subtitle = (
        f"N eventos: {len(X):,}  ·  "
        f"RchT mediana: {np.median(rcht[rcht>0]):.0f}  ·  "
        f"Nint media: {n_ints.mean():.1f}  ·  "
        f"Canales activos: {N_ACTIVE}"
    )
    fig.text(0.5, 0.955, subtitle, ha='center', va='top',
             fontsize=9, color=ACCENT, alpha=0.85)

    # Layout: 3 filas × 3 columnas (última celda vacía para ajuste visual)
    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.42,
        wspace=0.35,
        left=0.05, right=0.97,
        top=0.93, bottom=0.04,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])   # panel extra: estadísticas

    # Panel 1: Geometría
    panel_detector_geometry(ax1)

    # Panel 2: Espectro de energía
    panel_energy_spectrum(ax2, X)

    # Panel 3: Llenado
    panel_detector_filling(ax3, X, psipm)

    # Panel 4: Distribución Nint
    panel_nint_distribution(ax4, n_ints)

    # Panel 5: Actividad media
    panel_mean_activity(ax5, X)

    # Panel 6: Distribución de carga (box plot)
    panel_charge_distribution(ax6, X)

    # Panel 7: Correlación de vecinos
    panel_channel_correlation(ax7, X)

    # Panel 8: Degradación vs original
    panel_degraded_vs_restored(ax8, X, psipm=psipm)

    # Panel 9: Tabla de estadísticas por canal
    _panel_stats_table(ax9, X, n_ints)

    # Marca de agua / pie
    fig.text(0.99, 0.01, "TFM · Miguel Escudero · Detector PET Hexagonal",
             ha='right', va='bottom', fontsize=7, color=GRID_COLOR)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        print(f"✓ Informe guardado en: {save_path}")

    if show:
        plt.show()

    return fig


def _panel_stats_table(ax: plt.Axes, X: np.ndarray, n_ints: np.ndarray):
    """Panel 9: tabla de estadísticas globales."""
    ax.axis('off')

    rcht = X.sum(axis=1)
    rcht_valid = rcht[rcht > 0]

    stats = [
        ("Eventos totales",          f"{len(X):,}"),
        ("Eventos con señal",         f"{(rcht>0).sum():,}"),
        ("RchT — media",              f"{rcht_valid.mean():.1f}"),
        ("RchT — mediana",            f"{np.median(rcht_valid):.1f}"),
        ("RchT — std",                f"{rcht_valid.std():.1f}"),
        ("Nint — media",              f"{n_ints.mean():.1f}"),
        ("Nint — mediana",            f"{np.median(n_ints):.0f}"),
        ("Nint — máx",                f"{n_ints.max()}"),
        ("Canales activos (total)",   f"{N_ACTIVE}"),
        ("Canales siempre a 0",
         f"{(X == 0).all(axis=0).sum()}"),
        ("Fracción media de fuego",
         f"{(X > 0).mean():.3f}"),
        ("Canal más activo (Ich)",
         f"{IDX_TO_ICH[int(np.argmax((X>0).mean(axis=0)))]}"),
        ("Canal menos activo (Ich)",
         f"{IDX_TO_ICH[int(np.argmin((X>0).mean(axis=0)))]}"),
    ]

    ax.set_title("Estadísticas Globales", pad=8)

    y_start = 0.96
    dy = 0.072
    for i, (label, value) in enumerate(stats):
        y = y_start - i * dy
        color_row = PANEL_BG if i % 2 == 0 else "#1a2332"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.01, y - 0.045), 0.98, dy * 0.95,
            boxstyle="round,pad=0.005",
            facecolor=color_row, edgecolor='none',
            transform=ax.transAxes, clip_on=False,
        ))
        ax.text(0.05, y - 0.01, label, transform=ax.transAxes,
                fontsize=7.5, color=TEXT_COLOR, va='center')
        ax.text(0.95, y - 0.01, value, transform=ax.transAxes,
                fontsize=7.5, color=ACCENT3, va='center', ha='right',
                fontweight='bold')


# ─────────────────────────────────────────────────────────────
# VISUALIZACIÓN INDIVIDUAL INTERACTIVA
# ─────────────────────────────────────────────────────────────

def plot_single_event(
    X:          np.ndarray,
    event_idx:  int,
    psipm:      Optional[pd.DataFrame] = None,
    save_path:  Optional[str] = None,
    show:       bool = True,
) -> plt.Figure:
    """
    Visualiza en detalle un único evento: mapa hexagonal de carga + posición XY.
    """
    setup_style()
    event = X[event_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
    fig.suptitle(f"Evento #{event_idx} — {(event > 0).sum()} SiPMs activos",
                 color=TEXT_COLOR, fontsize=13, fontweight='bold')

    # — Mapa hexagonal de carga —
    ax = axes[0]
    ax.set_facecolor(DARK_BG)
    ax.axis('off')
    ax.set_title("Mapa de Carga (Rch)", pad=6)

    cmap = plt.get_cmap('hot')
    norm = Normalize(vmin=0, vmax=event.max() if event.max() > 0 else 1)

    for dense_idx in range(N_ACTIVE):
        ich    = IDX_TO_ICH[dense_idx]
        q, r   = ICH_TO_AXIAL[ich]
        cx, cy = axial_to_cartesian(q, r)
        val    = event[dense_idx]
        color  = cmap(norm(val))
        edge   = ACCENT if val > 0 else GRID_COLOR

        patch = mpatches.RegularPolygon(
            (cx, cy), numVertices=6,
            radius=0.85 * 2 / np.sqrt(3),
            orientation=0,
            facecolor=color, edgecolor=edge, linewidth=0.6,
        )
        ax.add_patch(patch)
        if val > 0:
            ax.text(cx, cy + 0.05, f"{val:.1f}", ha='center', va='center',
                    fontsize=4.5, color='white', fontweight='bold')
        else:
            ax.text(cx, cy + 0.05, f"{ich}", ha='center', va='center',
                    fontsize=4.5, color='#444', alpha=0.6)

    ax.set_xlim(-9, 9); ax.set_ylim(-9, 9)
    ax.set_aspect('equal')

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Rch', fraction=0.046, pad=0.02)

    # — Posición estimada —
    ax2 = axes[1]
    ax2.set_title("Posición de Interacción", pad=6)

    # Dibujar silueta del detector
    theta = np.linspace(0, 2 * np.pi, 7)
    xs    = [SIPM_XY[ich][0] for ich in ACTIVE_CH]
    ys    = [SIPM_XY[ich][1] for ich in ACTIVE_CH]
    r_max = max(np.hypot(xs, ys)) * 1.05
    hex_outline_x = r_max * np.cos(theta + np.pi / 6)
    hex_outline_y = r_max * np.sin(theta + np.pi / 6)
    ax2.fill(hex_outline_x, hex_outline_y, color="#111827", alpha=0.7, zorder=0)
    ax2.plot(hex_outline_x, hex_outline_y, color=GRID_COLOR, lw=1, zorder=1)

    # SiPMs como puntos
    for dense_idx in range(N_ACTIVE):
        ich = IDX_TO_ICH[dense_idx]
        cx, cy = SIPM_XY[ich]
        val = event[dense_idx]
        sz  = 20 + 200 * norm(val)
        col = cmap(norm(val))
        ax2.scatter(cx, cy, s=sz, color=col, zorder=2, edgecolors=ACCENT if val > 0 else 'none', lw=0.5)

    # Posición estimada
    pos_x, pos_y = compute_positions(X[event_idx:event_idx+1], psipm)
    ax2.scatter(pos_x[0], pos_y[0], s=200, marker='*', color=ACCENT2,
                zorder=5, label=f"Posición\n({pos_x[0]:.2f}, {pos_y[0]:.2f})")
    ax2.scatter(pos_x[0], pos_y[0], s=400, marker='o', color='none',
                edgecolors=ACCENT2, linewidths=1.5, zorder=4)

    ax2.set_xlim(-r_max * 1.2, r_max * 1.2)
    ax2.set_ylim(-r_max * 1.2, r_max * 1.2)
    ax2.set_aspect('equal')
    ax2.set_xlabel("X (u. axiales)"); ax2.set_ylabel("Y (u. axiales)")
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        print(f"✓ Guardado: {save_path}")
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualización de datos del detector PET hexagonal'
    )
    parser.add_argument('--data',       type=str, default=None,
                        help='Ruta al archivo datas#.dat')
    parser.add_argument('--psipm',      type=str, default=None,
                        help='Ruta al archivo psipm.tsv')
    parser.add_argument('--save',       type=str, default=None,
                        help='Guardar informe completo (ej: report.png)')
    parser.add_argument('--event',      type=int, default=None,
                        help='Visualizar evento individual (índice)')
    parser.add_argument('--max_events', type=int, default=100_000,
                        help='Máximo de eventos a cargar (default: 100000)')
    parser.add_argument('--demo',       action='store_true',
                        help='Modo demo con datos sintéticos')
    parser.add_argument('--no_show',    action='store_true',
                        help='No abrir ventana interactiva (solo guardar)')
    args = parser.parse_args()

    # ── Cargar datos ─────────────────────────────────────────
    if args.demo or args.data is None:
        print("🔬 Modo demo: generando datos sintéticos...")
        X, n_ints = generate_demo_data(n_events=20_000)
        title = "Análisis del Detector PET Hexagonal [DATOS SINTÉTICOS]"
    else:
        path = Path(args.data)
        if not path.exists():
            print(f"❌ Error: no se encuentra el archivo {path}", file=sys.stderr)
            sys.exit(1)
        print(f"📂 Cargando {path.name}...")
        X, n_ints = read_binary(str(path), max_events=args.max_events)
        print(f"   → {len(X):,} eventos leídos")
        title = f"Análisis del Detector PET Hexagonal · {path.name}"

    # ── Cargar psipm.tsv ──────────────────────────────────────
    psipm = None
    if args.psipm:
        psipm_path = Path(args.psipm)
        if psipm_path.exists():
            psipm = pd.read_csv(psipm_path, sep='\t')
            psipm.columns = [c.strip().lower() for c in psipm.columns]
            # Normalizar nombres de columna
            rename = {}
            for col in psipm.columns:
                if 'ich' in col or col == 'channel': rename[col] = 'ich'
                elif col.startswith('x'):             rename[col] = 'xsipm'
                elif col.startswith('y'):             rename[col] = 'ysipm'
            psipm = psipm.rename(columns=rename)
            psipm = psipm[psipm['ich'].isin(ACTIVE_CH)].set_index('ich')
            print(f"   → psipm.tsv cargado: {len(psipm)} SiPMs")
        else:
            print(f"⚠️  psipm.tsv no encontrado, usando geometría axial por defecto")

    # ── Visualización ─────────────────────────────────────────
    show = not args.no_show

    if args.event is not None:
        if args.event >= len(X):
            print(f"❌ Error: evento {args.event} fuera de rango (N={len(X)})", file=sys.stderr)
            sys.exit(1)
        print(f"🔍 Visualizando evento #{args.event}...")
        save = args.save or f"evento_{args.event}.png"
        plot_single_event(X, args.event, psipm=psipm, save_path=save, show=show)
    else:
        print("📊 Generando informe completo...")
        save = args.save or ("report_demo.png" if args.demo else
                             f"report_{Path(args.data).stem}.png")
        plot_full_report(X, n_ints, psipm=psipm, title=title,
                         save_path=save, show=show)


if __name__ == '__main__':
    main()
