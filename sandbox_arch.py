"""
sandbox_arch.py — BANCO DE PRUEBAS de arquitecturas drásticamente distintas
============================================================================
NO es código de producción. Aquí probamos ideas SoA para ver si ALGO bate el techo
del HexCNN de referencia (val_mae_mod ~0.189). Reutiliza el dataset y un esquema de
entrenamiento equivalente al de train.py (ciclo de archivos por época, val fijo,
selección por val_mae_mod, grad clip).

Arquitecturas:
  transformer : Graph Transformer PURO (Graphormer-style). Auto-atención global con
                SESGO ESPACIAL por distancia entre sensores. Sin convolución local:
                puramente atención + geometría. Es lo opuesto a nuestra HexConv.
  unet        : U-Net de grafo. Encoder de message-passing local → CUELLO DE BOTELLA
                global (latente) → decoder con skip connections. Multi-escala real.

Uso (los lanzas tú; yo no entreno):
    python sandbox_arch.py transformer
    python sandbox_arch.py unet --epochs 40
    python sandbox_arch.py transformer --dim 128 --layers 6

Compara el val_mae_mod que imprime contra 0.189 (HexCNN). Si baja de ahí de forma
clara y estable → hemos encontrado algo. Si no → confirma el techo (también vale).

Autor: Miguel Escudero (TFM)
"""

import sys
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dataset import (SiPMImputationDataset, load_dat_to_dense, get_file_split,
                     load_positions, N_ACTIVE)
from hex_geometry import get_neighbor_matrix

GOOD_DIR   = r'E:\Datos TFM\Good\Good'
PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'
RUNS_BASE  = r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\runs'

BATCH_SIZE  = 512
LR          = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EVENTS  = 300_000
N_VAL, N_TEST, SPLIT_SEED = 5, 5, 42
VAL_MASK_SEED = 12345


# ─────────────────────────────────────────────────────────────
#  PIEZAS
# ─────────────────────────────────────────────────────────────

class MP(nn.Module):
    """Message-passing local (media de vecinos), estilo HexConv pero con LayerNorm."""
    def __init__(self, dim, neighbor_matrix):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim)
        self.lin_nb   = nn.Linear(dim, dim, bias=False)
        self.norm     = nn.LayerNorm(dim)
        self.register_buffer('nbr', torch.as_tensor(neighbor_matrix, dtype=torch.long))

    def forward(self, x):                                  # (B, N, F)
        B, N, Fdim = x.shape
        xpad = torch.cat([x, x.new_zeros(B, 1, Fdim)], dim=1)
        gi = torch.where(self.nbr >= 0, self.nbr, torch.full_like(self.nbr, N))
        nb = xpad[:, gi, :]                                # (B, N, 6, F)
        valid = (self.nbr >= 0).to(x.dtype).view(1, N, -1, 1)
        nbmean = (nb * valid).sum(2) / valid.sum(2).clamp(min=1)
        return self.norm(F.gelu(self.lin_self(x) + self.lin_nb(nbmean)))


class SpatialMHA(nn.Module):
    """Auto-atención multi-cabeza con sesgo por DISTANCIA entre sensores (Graphormer)."""
    def __init__(self, dim, heads, dist):
        super().__init__()
        self.h, self.dk = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o   = nn.Linear(dim, dim)
        self.bias = nn.Sequential(nn.Linear(1, heads))     # distancia → sesgo por cabeza
        self.register_buffer('dist', torch.as_tensor(dist, dtype=torch.float32).unsqueeze(-1))

    def forward(self, x):                                  # (B, N, dim)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                   # (B, h, N, dk)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)          # (B, h, N, N)
        bias = self.bias(self.dist).permute(2, 0, 1)                  # (h, N, N)
        att = torch.softmax(att + bias.unsqueeze(0), dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, N, D)
        return self.o(out)


class TBlock(nn.Module):
    def __init__(self, dim, heads, dist, drop=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.att = SpatialMHA(dim, heads, dist)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                nn.Dropout(drop), nn.Linear(2 * dim, dim))

    def forward(self, x):
        x = x + self.att(self.n1(x))
        return x + self.ff(self.n2(x))


# ─────────────────────────────────────────────────────────────
#  ARQUITECTURAS
# ─────────────────────────────────────────────────────────────

class GraphTransformer(nn.Module):
    """Transformer de grafo puro: atención global + sesgo espacial. Sin conv local."""
    def __init__(self, dim=96, heads=4, layers=4, dist=None, dropout=0.1):
        super().__init__()
        self.stem = nn.Linear(2, dim)
        self.pos  = nn.Parameter(torch.zeros(1, N_ACTIVE, dim))     # embedding posicional aprendido
        self.blocks = nn.ModuleList([TBlock(dim, heads, dist, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):                                  # (B, 2, 61)
        h = self.stem(x.permute(0, 2, 1)) + self.pos
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h)).squeeze(-1)


class GraphUNet(nn.Module):
    """U-Net de grafo: encoder local → cuello de botella GLOBAL → decoder con skip."""
    def __init__(self, dim=64, nbr=None, dropout=0.1):
        super().__init__()
        self.stem = nn.Linear(2, dim)
        self.enc  = nn.ModuleList([MP(dim, nbr), MP(dim, nbr)])
        self.to_global = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.from_global = nn.Linear(dim, dim)
        self.merge = nn.Linear(2 * dim, dim)               # fusiona local + global
        self.dec  = nn.ModuleList([MP(dim, nbr), MP(dim, nbr)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):                                  # (B, 2, 61)
        h0 = F.gelu(self.stem(x.permute(0, 2, 1)))
        h = h0
        for b in self.enc:
            h = b(h)                                       # encoder local
        g = self.to_global(h.mean(dim=1))                  # CUELLO DE BOTELLA global (B, dim)
        h = self.merge(torch.cat([h, self.from_global(g).unsqueeze(1).expand_as(h)], dim=-1))
        h = self.drop(h)
        for b in self.dec:
            h = b(h)                                       # decoder local
        return self.head(h + h0).squeeze(-1)               # skip connection


# ─────────────────────────────────────────────────────────────
#  ENTRENAMIENTO (equivalente a train.py, compacto)
# ─────────────────────────────────────────────────────────────

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    err_mod, n_mod, tot_loss, n = 0.0, 0, 0.0, 0
    for x_in, target, dead, is_mod in loader:
        x_in, target, dead = x_in.to(device), target.to(device), dead.to(device)
        out = model(x_in)
        tot_loss += F.mse_loss(out, target).item() * len(target); n += len(target)
        err = (((out - target) * dead).sum(1)).abs()       # error del canal muerto (dead one-hot)
        m = is_mod.to(device).bool()
        err_mod += err[m].sum().item(); n_mod += m.sum().item()
    return tot_loss / max(n, 1), err_mod / max(n_mod, 1)


def eval_real(arch, out_dir, device, nbr, dist, events):
    """
    Métrica REAL sobre los archivos de TEST: recuperación p90 macro y MAE macro,
    con el mismo protocolo que eval_total.py.

    Persiste el resultado en un JSON dentro de la carpeta del run. La ronda 1 de
    este banco de pruebas solo dejaba los números por consola, así que no había
    forma de recuperarlos ni de auditarlos después; esto lo corrige.
    """
    import json, datetime
    from imputation_eval import impute_channel, compute_xy

    ck = torch.load(out_dir / 'best.pth', map_location=device, weights_only=False)
    dim = ck.get('dim')
    if arch == 'transformer':
        model = GraphTransformer(dim=dim, heads=ck.get('heads', 4),
                                 layers=ck.get('layers', 4), dist=dist)
    else:
        model = GraphUNet(dim=dim, nbr=nbr)
    model.load_state_dict(ck['model_state'])
    model.to(device).eval()
    # El preprocesado del eval DEBE coincidir con el del entrenamiento. Este banco
    # siempre entrenó con normalización por el máximo y sin normalización por canal.
    model._norm_mode = ck.get('norm_mode', 'max')
    model._chan_norm = ck.get('channel_norm', False)

    xs, ys = load_positions(PSIPM_PATH)
    _, _, test_files = get_file_split(GOOD_DIR, n_val=N_VAL, n_test=N_TEST, seed=SPLIT_SEED)
    X_list = [load_dat_to_dense(f, max_events=events) for f in test_files]
    oxy = [compute_xy(X, xs, ys) for X in X_list]

    recs, maes = [], []
    for c in range(N_ACTIVE):
        dRdeg, dRimp, errs = [], [], []
        for X, (ox, oy) in zip(X_list, oxy):
            mod = X[:, c] > 0
            if mod.sum() < 100:
                continue
            Xd = X.copy(); Xd[:, c] = 0.0
            dx, dy = compute_xy(Xd, xs, ys)
            dRdeg.append(np.sqrt((dx - ox) ** 2 + (dy - oy) ** 2)[mod])
            Xi, pred = impute_channel(model, X, c, device)
            ix, iy = compute_xy(Xi, xs, ys)
            dRimp.append(np.sqrt((ix - ox) ** 2 + (iy - oy) ** 2)[mod])
            errs.append(np.abs(pred[mod] - X[mod, c]))
        if not dRdeg:
            continue
        pd_ = np.percentile(np.concatenate(dRdeg), 90)
        pi_ = np.percentile(np.concatenate(dRimp), 90)
        recs.append((pd_ - pi_) / pd_ * 100 if pd_ > 0 else 0.0)
        maes.append(float(np.concatenate(errs).mean()))
        if (c + 1) % 15 == 0:
            print(f"  {c+1}/{N_ACTIVE} canales")

    rec, mae = float(np.mean(recs)), float(np.mean(maes))
    print(f"\n=== MÉTRICA REAL ({arch}, {events} ev/archivo, macro {len(recs)} canales) ===")
    print(f"  recuperación p90 macro : {rec:.2f} %")
    print(f"  MAE macro              : {mae:.4f} ADC")

    out = {'arch': arch, 'events': events, 'recov_p90_macro': round(rec, 2),
           'mae_macro': round(mae, 4), 'train_epoch': ck.get('epoch'),
           'val_mae_mod': ck.get('val_mae_mod'), 'dim': dim,
           'n_channels': len(recs),
           'recov_p90_per_channel': [round(float(v), 2) for v in recs],
           'generated': datetime.datetime.now().isoformat(timespec='seconds')}
    p = out_dir / f'eval_real_{events}.json'
    p.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"  JSON guardado: {p}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arch', choices=['transformer', 'unet'])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--dim', type=int, default=None)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--eval', action='store_true',
                    help='NO entrena: evalúa best.pth sobre test y persiste el JSON')
    ap.add_argument('--events', type=int, default=60_000,
                    help='eventos por archivo de test en --eval')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xs, ys = load_positions(PSIPM_PATH)
    nbr = get_neighbor_matrix(PSIPM_PATH)
    # matriz de distancias normalizada (para el sesgo espacial del transformer)
    dxy = np.hypot(xs[:, None] - xs, ys[:, None] - ys)
    dist = dxy / np.median(dxy[dxy > 0])

    out_dir = Path(RUNS_BASE) / f'sandbox_{args.arch}'
    if args.eval:
        assert (out_dir / 'best.pth').exists(), f"no hay best.pth en {out_dir}"
        eval_real(args.arch, out_dir, device, nbr, dist, args.events)
        return

    if args.arch == 'transformer':
        dim = args.dim or 96
        model = GraphTransformer(dim=dim, heads=args.heads, layers=args.layers, dist=dist).to(device)
    else:
        dim = args.dim or 64
        model = GraphUNet(dim=dim, nbr=nbr).to(device)
    print(f"Arquitectura: {args.arch}  |  dim={dim}  |  parámetros: {count_params(model):,}")
    print(f"Referencia a batir: HexCNN val_mae_mod = 0.189 (38.305 params)")

    train_files, val_files, _ = get_file_split(GOOD_DIR, n_val=N_VAL, n_test=N_TEST, seed=SPLIT_SEED)
    X_val = np.concatenate([load_dat_to_dense(f, max_events=MAX_EVENTS // len(val_files))
                            for f in val_files])
    val_loader = DataLoader(SiPMImputationDataset(X_val, seed=VAL_MASK_SEED),
                            batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=LR / 100)

    out_dir.mkdir(parents=True, exist_ok=True)
    best = 1e9
    for epoch in range(1, args.epochs + 1):
        f = train_files[(epoch - 1) % len(train_files)]
        X = load_dat_to_dense(f, max_events=MAX_EVENTS)
        loader = DataLoader(SiPMImputationDataset(X, seed=epoch),
                            batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        model.train()
        for x_in, target, dead, is_mod in loader:
            x_in, target = x_in.to(device), target.to(device)
            opt.zero_grad()
            loss = F.mse_loss(model(x_in), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        vloss, vmae = evaluate(model, val_loader, device)
        flag = ''
        if vmae < best:
            best = vmae
            torch.save({'arch': args.arch, 'model_state': model.state_dict(),
                        'epoch': epoch, 'val_mae_mod': vmae, 'dim': dim,
                        'heads': args.heads, 'layers': args.layers,
                        'norm_mode': 'max', 'channel_norm': False},
                       out_dir / 'best.pth')
            flag = '  ✓ best'
        print(f"Epoch {epoch:3d}/{args.epochs} | val_loss={vloss:.4f} "
              f"val_mae_mod={vmae:.4f}{flag}")
    print(f"\nMEJOR val_mae_mod = {best:.4f}   (HexCNN referencia: 0.189)")
    print("  < 0.189 de forma estable → algo hemos encontrado.  ~0.189 o peor → techo confirmado.")


if __name__ == '__main__':
    main()
