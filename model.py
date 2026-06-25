"""
model.py
========
Arquitectura base para la imputación de señales SiPM.

ResidualMLPImputer: MLP con bloques residuales + una puerta de atención
(estilo squeeze-and-excitation) sobre el vector latente. Es la arquitectura
"sencilla" de partida; más adelante compararemos con una CNN hexagonal que
explote la geometría del detector.

Entrada : (B, 2, 61) — fila 0 = cargas normalizadas (canal apagado a 0),
                       fila 1 = máscara binaria
Salida  : (B, 61)    — vector de cargas reconstruido (regresión 61→61)

NOTA: la salida es lineal (sin sigmoid/clamp) a propósito, porque el target
normalizado puede superar 1.0 en el canal imputado (ver dataset.py).

Autor: Miguel Escudero (TFM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import N_ACTIVE
from hex_geometry import get_neighbor_matrix


class ResidualBlock(nn.Module):
    """
    Bloque residual de un MLP: Linear → BN → GELU → Dropout, con skip connection.

    La conexión residual (x + f(x)) deja pasar el gradiente directamente hacia
    atrás, facilitando entrenar redes algo más profundas sin que se degraden.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc   = nn.Linear(dim, dim)
        self.bn   = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        h = self.bn(h)
        h = F.gelu(h)          # GELU: activación suave, suele ir algo mejor que ReLU en MLPs
        h = self.drop(h)
        return x + h           # skip connection


class ResidualMLPImputer(nn.Module):
    """
    MLP residual con atención para imputación 61→61.

    Parameters
    ----------
    hidden : int      — dimensión del espacio latente
    n_blocks : int    — nº de bloques residuales
    dropout : float
    """

    def __init__(self, hidden: int = 256, n_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        in_dim = 2 * N_ACTIVE   # aplanamos la matriz 2×61 → 122

        # Proyección de entrada al espacio latente
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )

        # Pila de bloques residuales
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden, dropout) for _ in range(n_blocks)
        ])

        # Puerta de atención (squeeze-and-excitation): re-pondera el vector latente.
        # Aprende a dar más o menos peso a cada dimensión latente según el contexto.
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, hidden),
            nn.Sigmoid(),            # factor multiplicativo en [0,1] por dimensión
        )

        # Cabeza de salida: latente → 61 cargas (lineal, sin acotar)
        self.head = nn.Linear(hidden, N_ACTIVE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 2, 61) → flatten → (B, 122)
        """
        B = x.shape[0]
        h = x.reshape(B, -1)            # (B, 122): aplana cargas + máscara
        h = self.stem(h)               # (B, hidden)

        for block in self.blocks:
            h = block(h)               # (B, hidden)

        # Atención: multiplicamos el latente por su puerta aprendida
        h = h * self.attn(h)           # (B, hidden)

        return self.head(h)            # (B, 61)


# ─────────────────────────────────────────────────────────────
# BASELINE: MLP PROFUNDO PLANO (sin residual ni atención)
# ─────────────────────────────────────────────────────────────

class DeepMLPImputer(nn.Module):
    """
    MLP profundo plano: pila de Linear → BN → GELU → Dropout, sin trucos.

    Es el baseline limpio para comparar contra el residual+atención y la HexCNN.
    Misma entrada (2×61 aplanada a 122) y salida lineal (61, sin acotar).
    """

    def __init__(self, hidden=(512, 512, 256, 128), dropout: float = 0.1):
        super().__init__()
        in_dim = 2 * N_ACTIVE   # aplanamos la matriz 2×61 → 122

        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, N_ACTIVE))   # salida lineal, sin acotar
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.net(x.reshape(B, -1))        # (B, 2, 61) → (B, 122) → (B, 61)


# ─────────────────────────────────────────────────────────────
# CNN HEXAGONAL (prior geométrico: agrega vecinos físicos reales)
# ─────────────────────────────────────────────────────────────

class HexConv(nn.Module):
    """
    Convolución sobre el grafo hexagonal: para cada sensor combina sus propias
    features con la MEDIA de las features de sus vecinos físicos reales.

    Es una graph-convolution restringida al grafo de vecindad del detector.
    """

    def __init__(self, in_f: int, out_f: int, neighbor_matrix):
        super().__init__()
        # Dos transformaciones lineales con pesos COMPARTIDOS por los 61 nodos (esto es
        # lo que la hace "convolución": el mismo kernel en todas partes):
        self.lin_self  = nn.Linear(in_f, out_f)              # aplica al propio nodo (con bias)
        self.lin_neigh = nn.Linear(in_f, out_f, bias=False)  # aplica a la MEDIA de sus vecinos (sin bias, para no duplicarlo)
        self.bn = nn.BatchNorm1d(out_f)                      # normaliza la salida (estabiliza el entrenamiento)
        # Grafo de vecindad (61,6) como BUFFER: tensor NO entrenable que se mueve con
        # .to(device) y se guarda en el checkpoint → el grafo viaja con el modelo.
        self.register_buffer('nbr', torch.as_tensor(neighbor_matrix, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, N, Fin) → (B, N, out_f)"""
        # OJO: no usar 'F' como nombre local — taparía a torch.nn.functional as F
        B, N, Fin = x.shape   # B=batch, N=61 nodos, Fin=features por nodo

        # ── Truco del nodo de padding (para los sensores de borde con <6 vecinos) ──
        # Añadimos un nodo nº N extra lleno de CEROS al que apuntarán las ranuras vacías.
        x_pad = torch.cat([x, x.new_zeros(B, 1, Fin)], dim=1)   # (B, N+1, Fin): nodos reales + 1 de padding
        # En la matriz de vecinos, los huecos valen -1. Los redirigimos al índice N (el de ceros);
        # el resto se queda con el índice de su vecino real.
        gather_idx = torch.where(self.nbr >= 0, self.nbr, torch.full_like(self.nbr, N))  # (N, 6) índices ya válidos

        # ── Recoger los 6 vecinos de los 61 nodos de golpe (sin bucle) ──
        # Indexación avanzada sobre la dimensión de nodos: para cada nodo i y ranura d,
        # trae las features del vecino gather_idx[i,d] (ceros si era padding).
        nb = x_pad[:, gather_idx, :]                          # (B, N, 6, Fin): features de los 6 vecinos
        # Máscara de qué ranuras son vecinos REALES (1) vs padding (0), lista para broadcasting.
        valid = (self.nbr >= 0).to(x.dtype).view(1, N, -1, 1) # (1, N, 6, 1)
        # Media SOLO sobre los vecinos reales: anulamos el padding (nb*valid) y dividimos por
        # cuántos vecinos reales hay (3-6 según el nodo). clamp(min=1) evita dividir por 0.
        nb_mean = (nb * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)  # (B, N, Fin)

        # ── La convolución: nodo central + media de vecinos, con sus pesos compartidos ──
        out = self.lin_self(x) + self.lin_neigh(nb_mean)     # (B, N, out_f)
        # BatchNorm1d espera (muestras, features): aplanamos batch y nodos, normalizamos, deshacemos.
        out = self.bn(out.reshape(B * N, -1)).reshape(B, N, -1)
        return F.gelu(out)   # no-linealidad suave


class HexResBlock(nn.Module):
    """Bloque residual de dos HexConv sobre la malla hexagonal."""

    def __init__(self, dim: int, neighbor_matrix, dropout: float = 0.1):
        super().__init__()
        self.c1   = HexConv(dim, dim, neighbor_matrix)   # 1ª convolución hexagonal (mantiene la dimensión)
        self.c2   = HexConv(dim, dim, neighbor_matrix)   # 2ª convolución hexagonal
        self.drop = nn.Dropout(dropout)                  # regularización entre las dos convs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(x)     # propaga info a 1 anillo de vecinos
        h = self.drop(h)   # apaga aleatoriamente parte de las features (solo en train)
        h = self.c2(h)     # propaga otro anillo más
        # Conexión residual: sumamos la entrada original. Deja pasar el gradiente directo
        # hacia atrás (entrena mejor en profundidad) y no obliga al bloque a recalcularlo todo.
        return x + h


class HexCNNImputer(nn.Module):
    """
    CNN hexagonal para imputación, con prior geométrico de las posiciones reales.

    Cada sensor es un nodo del grafo; la red propaga información entre vecinos
    físicos y reconstruye un escalar por nodo (la carga). Salida lineal (61).
    """

    def __init__(self, hidden: int = 48, n_blocks: int = 4, dropout: float = 0.1,
                 psipm_path: str | None = None):
        super().__init__()
        # Grafo de vecindad real (61,6), calculado desde psipm.tsv (cacheado). Es el
        # "cableado" que comparten todas las HexConv de abajo.
        nbr = get_neighbor_matrix(psipm_path) if psipm_path else get_neighbor_matrix()

        # Stem: proyecta las 2 features de cada nodo (carga, máscara) → 'hidden' dimensiones
        self.stem    = nn.Linear(2, hidden)
        self.stem_bn = nn.BatchNorm1d(hidden)

        # Pila de bloques residuales hexagonales (cada uno = 2 HexConv). Más bloques =
        # mayor campo receptivo (cada nodo "ve" más lejos en el detector).
        self.blocks = nn.ModuleList([
            HexResBlock(hidden, nbr, dropout) for _ in range(n_blocks)
        ])

        self.head = nn.Linear(hidden, 1)   # cabeza: de 'hidden' features → 1 escalar por nodo (la carga)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 2, 61) → (B, 61)"""
        B = x.shape[0]
        # permute: pasamos de (B, 2_features, 61_nodos) a (B, 61_nodos, 2_features),
        # que es el formato "por nodo" que esperan el stem y las HexConv.
        h = x.permute(0, 2, 1)                               # (B, 61, 2)
        h = self.stem(h)                                     # (B, 61, hidden): proyección por nodo
        # BatchNorm1d necesita (muestras, features): aplanamos batch×nodos, normalizamos,
        # deshacemos la forma, y aplicamos GELU.
        h = F.gelu(self.stem_bn(h.reshape(B * N_ACTIVE, -1)).reshape(B, N_ACTIVE, -1))

        for blk in self.blocks:
            h = blk(h)                                       # (B, 61, hidden): propaga info entre vecinos

        # head → (B, 61, 1); squeeze(-1) quita la última dimensión → (B, 61) cargas reconstruidas
        return self.head(h).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────

# Alias → clase. Incluimos los nombres de clase para poder recargar checkpoints viejos.
_MODELS = {
    'deepmlp':              DeepMLPImputer,
    'resmlp':               ResidualMLPImputer,
    'hexcnn':               HexCNNImputer,
    'DeepMLPImputer':       DeepMLPImputer,
    'ResidualMLPImputer':   ResidualMLPImputer,
    'HexCNNImputer':        HexCNNImputer,
}


def get_model(name: str, **kwargs) -> nn.Module:
    """Devuelve una instancia del modelo. name: 'deepmlp' | 'resmlp' | 'hexcnn'."""
    assert name in _MODELS, f"Modelo '{name}' no reconocido. Opciones: {sorted(_MODELS)}"
    return _MODELS[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Cuenta los parámetros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Smoke test de las tres arquitecturas
    x = torch.randn(8, 2, N_ACTIVE)
    for name in ('deepmlp', 'resmlp', 'hexcnn'):
        model = get_model(name)
        out = model(x)
        assert out.shape == (8, N_ACTIVE), f"{name}: shape {out.shape}"
        print(f"{name:9s} | salida {tuple(out.shape)} | parámetros: {count_parameters(model):,}")
