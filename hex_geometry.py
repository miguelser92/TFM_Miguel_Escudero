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
