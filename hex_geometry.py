"""
hex_geometry.py
===============
Construye el grafo de vecindad REAL de los 61 SiPM a partir de psipm.tsv.


Autor: Miguel Escudero (TFM)
"""

import numpy as np
from scipy.spatial import cKDTree

from dataset import load_positions, N_ACTIVE, IDX_TO_ICH


DEFAULT_PSIPM = r'E:\Datos TFM\psipm.tsv'

# Caché por ruta para no recalcular el grafo en cada construcción de modelo
_CACHE: dict = {}


def build_neighbor_matrix(x_sipm, y_sipm, max_neighbors: int = 6, tol: float = 1.3) -> np.ndarray:
    """
    Construye la matriz de vecindad [61, 6] desde las posiciones físicas.

    neighbor_matrix[i, k] = j  → el k-ésimo vecino del sensor i es el sensor j
    neighbor_matrix[i, k] = -1 → no hay más vecinos (sensores de borde)

    Criterio: vecino = sensor dentro de un radio = tol × (distancia mediana al vecino
    más cercano). En una malla hexagonal el primer anillo está a distancia d y el
    siguiente a ~1.73·d, así que tol=1.3 captura los 6 vecinos directos sin colarse
    al segundo anillo.
    """
    pts  = np.column_stack([x_sipm, y_sipm])
    tree = cKDTree(pts)

    # Distancia al vecino más cercano de cada sensor (k=2: [0]=él mismo, [1]=vecino real)
    d, _ = tree.query(pts, k=2)
    nn_dist = np.median(d[:, 1])
    radius  = nn_dist * tol

    nbr = np.full((N_ACTIVE, max_neighbors), -1, dtype=np.int64)
    for i in range(N_ACTIVE):
        # query_ball_point: índices de todos los puntos dentro del radio
        cand = tree.query_ball_point(pts[i], radius)
        cand = [j for j in cand if j != i]          # excluir el propio sensor
        # ordenar por distancia y quedarnos con los max_neighbors más cercanos
        cand.sort(key=lambda j: (pts[j, 0] - pts[i, 0])**2 + (pts[j, 1] - pts[i, 1])**2)
        cand = cand[:max_neighbors]
        nbr[i, :len(cand)] = cand

    return nbr


def get_neighbor_matrix(psipm_path: str = DEFAULT_PSIPM) -> np.ndarray:
    """Devuelve (con caché) la matriz de vecindad construida desde psipm.tsv."""
    if psipm_path not in _CACHE:
        x_sipm, y_sipm = load_positions(psipm_path)
        _CACHE[psipm_path] = build_neighbor_matrix(x_sipm, y_sipm)
    return _CACHE[psipm_path]


def get_positions(psipm_path: str = DEFAULT_PSIPM):
    """Posiciones (x, y) de los 61 SiPM desde psipm.tsv (atajo con la ruta por defecto)."""
    return load_positions(psipm_path)


def build_edge_vectors(x_sipm, y_sipm, nbr: np.ndarray) -> np.ndarray:
    """
    Vectores de arista (geometría MÉTRICA) para la HexConv ponderada.

    edge[i, k] = (x_j - x_i, y_j - y_i) del k-ésimo vecino j del sensor i,
    NORMALIZADO por la distancia mediana entre vecinos (escala ~1 → entrenamiento
    estable e independiente de las unidades). Las ranuras de padding (nbr = -1)
    quedan a (0, 0) — se enmascaran igual que en la agregación.

    A diferencia de la matriz de vecindad (solo topología: quién toca con quién),
    esto le da a la red la DISTANCIA y la DIRECCIÓN reales de cada vecino.
    """
    pts = np.column_stack([x_sipm, y_sipm])
    d, _ = cKDTree(pts).query(pts, k=2)
    scale = np.median(d[:, 1])                      # distancia típica entre vecinos (mm)
    N, K = nbr.shape
    edge = np.zeros((N, K, 2), dtype=np.float32)
    for i in range(N):
        for k in range(K):
            j = nbr[i, k]
            if j >= 0:
                edge[i, k, 0] = (pts[j, 0] - pts[i, 0]) / scale
                edge[i, k, 1] = (pts[j, 1] - pts[i, 1]) / scale
    return edge


if __name__ == '__main__':
    nbr = get_neighbor_matrix()
    n_neigh = (nbr >= 0).sum(axis=1)
    print(f"Matriz de vecindad: {nbr.shape}")
    print(f"Vecinos por sensor: min={n_neigh.min()}  max={n_neigh.max()}  media={n_neigh.mean():.2f}")
    print(f"Sensores con 6 vecinos (interiores): {(n_neigh == 6).sum()}")
    print(f"Sensores con <6 (borde):             {(n_neigh < 6).sum()}")
    # Ejemplo: vecinos del sensor 30 (Ich físico)
    print(f"\nVecinos del idx=30 (Ich={IDX_TO_ICH[30]}): "
          f"{[IDX_TO_ICH[j] for j in nbr[30] if j >= 0]}")


# ─────────────────────────────────────────────────────────────
#  ESCALA POR CANAL (para la normalización por canal)
# ─────────────────────────────────────────────────────────────

_SCALE_CACHE = {}
DEFAULT_SCALE_NPZ = r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\reports\channel_scale.npz'


def get_channel_scale(good_dir=r'E:\Datos TFM\Good\Good', n_files=8,
                      max_events=60_000, cache_npz=DEFAULT_SCALE_NPZ):
    """
    Escala típica de CADA canal: su carga media sobre varios módulos Good,
    renormalizada a media 1 (para no cambiar la escala global del problema).

    Motivación: el detector tiene un gradiente estructural fijo — el canal central
    recoge ~4.3x más carga que el de borde (corr −0.88 con la distancia al centro,
    por el adhesivo negro). Dividir cada canal por su escala EQUALIZA los 61 canales,
    de modo que la red no gasta capacidad aprendiendo esa estructura ya conocida y
    puede centrarse en la física del evento.

    Se cachea en disco (npz): es una constante del detector.

    OJO (auditoría): la escala se estima SOLO con módulos de TRAIN. La versión
    inicial cogía los n_files primeros del directorio, entre los que caía
    datas013 (validación) → fuga leve val→preprocesado. Afectaba únicamente a los
    runs con --chnorm. El npz ya cacheado se generó con aquella versión: bórralo
    si quieres regenerar la escala limpia (y reentrenar los runs que la usen).
    """
    from pathlib import Path as _Path
    key = (good_dir, n_files, max_events)
    if key in _SCALE_CACHE:
        return _SCALE_CACHE[key]
    cache = _Path(cache_npz) if cache_npz else None
    if cache is not None and cache.exists():
        scale = np.load(cache)['scale']
    else:
        from dataset import load_dat_to_dense, get_file_split
        train_files, _, _ = get_file_split(good_dir)      # nunca val ni test
        acc = [load_dat_to_dense(f, max_events=max_events).mean(axis=0)
               for f in train_files[:n_files]]
        q = np.mean(acc, axis=0)
        scale = (q / q.mean()).astype(np.float32)          # media 1
        scale = np.maximum(scale, 1e-3)                    # guard anti división por ~0
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, scale=scale)
    _SCALE_CACHE[key] = scale
    return scale


# ─────────────────────────────────────────────────────────────
#  MATRIZ DE VECINDAD DIRECCIONAL (para la HexConv ANISÓTROPA)
# ─────────────────────────────────────────────────────────────

def build_directional_neighbor_matrix(x_sipm, y_sipm, nbr: np.ndarray) -> np.ndarray:
    """
    Reordena la matriz de vecindad para que la ranura d SIGNIFIQUE la dirección d.

    nbr_dir[i, d] = vecino de i en la dirección canónica d (d = 0..5, separadas
    60°), o -1 si en esa dirección no hay sensor (borde). Es el requisito de una
    convolución hexagonal REAL (estilo Zhao et al.): un peso por dirección solo
    tiene sentido si "ranura" = "dirección", cosa que la matriz original no
    garantiza (ordena por distancia).

    Las 6 direcciones canónicas se estiman de los propios datos: se toman los
    ángulos de todas las aristas y se calcula su fase media módulo 60° (media
    circular, robusta). Cada vecino se asigna al bin más cercano; se verifica que
    ningún nodo tenga dos vecinos en el mismo bin.
    """
    N = nbr.shape[0]
    # Ángulos de todas las aristas del grafo
    angs = []
    for i in range(N):
        for j in nbr[i]:
            if j >= 0:
                angs.append(np.arctan2(y_sipm[int(j)] - y_sipm[i],
                                       x_sipm[int(j)] - x_sipm[i]))
    angs = np.asarray(angs)
    # Fase de referencia módulo 60°, por media circular (robusta al ruido numérico)
    ref = np.angle(np.mean(np.exp(1j * 6.0 * angs))) / 6.0

    nbr_dir = np.full((N, 6), -1, dtype=np.int64)
    step = np.pi / 3.0                                   # 60°
    for i in range(N):
        for j in nbr[i]:
            if j < 0:
                continue
            ang = np.arctan2(y_sipm[int(j)] - y_sipm[i], x_sipm[int(j)] - x_sipm[i])
            d = int(np.round((ang - ref) / step)) % 6
            # Sanidad: el ángulo debe caer cerca del canónico (malla regular)
            resid = (ang - ref) - np.round((ang - ref) / step) * step
            assert abs(resid) < 0.15, (
                f"arista {i}->{int(j)} a {np.degrees(ang):.1f}° no encaja en la rejilla "
                f"de 60° (residuo {np.degrees(resid):.1f}°)")
            assert nbr_dir[i, d] == -1, (
                f"nodo {i}: dos vecinos en la dirección {d} ({nbr_dir[i, d]} y {int(j)})")
            nbr_dir[i, d] = int(j)
    # Sanidad global: no se pierde ninguna arista al reordenar
    assert (nbr_dir >= 0).sum() == (nbr >= 0).sum(), "se perdieron aristas al reordenar"
    return nbr_dir
