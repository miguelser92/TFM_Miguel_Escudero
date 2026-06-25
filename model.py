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
        self.lin_self  = nn.Linear(in_f, out_f)            # transforma el nodo central
        self.lin_neigh = nn.Linear(in_f, out_f, bias=False)  # transforma la media de vecinos
        self.bn = nn.BatchNorm1d(out_f)
        # Buffer (no entrenable): se guarda en el checkpoint y viaja con el modelo
        self.register_buffer('nbr', torch.as_tensor(neighbor_matrix, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, N, Fin) → (B, N, out_f)"""
        # OJO: no usar 'F' como nombre local — taparía a torch.nn.functional as F
        B, N, Fin = x.shape

        # Nodo de padding en la posición N (todo ceros) para los vecinos inexistentes (-1)
        x_pad = torch.cat([x, x.new_zeros(B, 1, Fin)], dim=1)   # (B, N+1, Fin)
        # torch.where: los -1 apuntan al índice N (padding); el resto a su vecino real
        gather_idx = torch.where(self.nbr >= 0, self.nbr, torch.full_like(self.nbr, N))  # (N, 6)

        # Indexación avanzada: recoge las features de los 6 vecinos de cada nodo
        nb = x_pad[:, gather_idx, :]                          # (B, N, 6, F)
        valid = (self.nbr >= 0).to(x.dtype).view(1, N, -1, 1) # (1, N, 6, 1)
        # Media solo sobre los vecinos válidos (ignora el padding)
        nb_mean = (nb * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)  # (B, N, F)

        out = self.lin_self(x) + self.lin_neigh(nb_mean)     # (B, N, out_f)
        out = self.bn(out.reshape(B * N, -1)).reshape(B, N, -1)
        return F.gelu(out)


class HexResBlock(nn.Module):
    """Bloque residual de dos HexConv sobre la malla hexagonal."""

    def __init__(self, dim: int, neighbor_matrix, dropout: float = 0.1):
        super().__init__()
        self.c1   = HexConv(dim, dim, neighbor_matrix)
        self.c2   = HexConv(dim, dim, neighbor_matrix)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(x)
        h = self.drop(h)
        h = self.c2(h)
        return x + h           # skip connection


class HexCNNImputer(nn.Module):
    """
    CNN hexagonal para imputación, con prior geométrico de las posiciones reales.

    Cada sensor es un nodo del grafo; la red propaga información entre vecinos
    físicos y reconstruye un escalar por nodo (la carga). Salida lineal (61).
    """

    def __init__(self, hidden: int = 48, n_blocks: int = 4, dropout: float = 0.1,
                 psipm_path: str | None = None):
        super().__init__()
        nbr = get_neighbor_matrix(psipm_path) if psipm_path else get_neighbor_matrix()

        # Stem por nodo: 2 features (carga, máscara) → hidden
        self.stem    = nn.Linear(2, hidden)
        self.stem_bn = nn.BatchNorm1d(hidden)

        self.blocks = nn.ModuleList([
            HexResBlock(hidden, nbr, dropout) for _ in range(n_blocks)
        ])

        self.head = nn.Linear(hidden, 1)   # un escalar por nodo (carga reconstruida)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 2, 61) → (B, 61)"""
        B = x.shape[0]
        h = x.permute(0, 2, 1)                               # (B, 61, 2): nodos × features
        h = self.stem(h)                                     # (B, 61, hidden)
        h = F.gelu(self.stem_bn(h.reshape(B * N_ACTIVE, -1)).reshape(B, N_ACTIVE, -1))

        for blk in self.blocks:
            h = blk(h)                                       # (B, 61, hidden)

        return self.head(h).squeeze(-1)                      # (B, 61)


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
