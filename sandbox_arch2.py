"""
sandbox_arch2.py — RONDA 2 del banco de pruebas de arquitecturas
================================================================
La ronda 1 (sandbox_arch.py) fue pequeña (U-Net de 1 nivel dim 64, transformer
4 capas dim 96, 40 épocas sin tuning) y aun así EMPATÓ con el HexCNN en test →
no cierra el techo. Esta ronda hace el intento serio:

  pna     : HexCNN pero con agregación PNA (Principal Neighbourhood Aggregation,
            Corso 2020): en vez de solo la MEDIA de vecinos, concatena
            mean/max/min/std → la agregación deja de tirar información.
            Cambio quirúrgico sobre el esqueleto que ya sabemos que funciona.
  unet2   : U-Net de grafo PROFUNDA: dim 128, 8 capas de message-passing en
            encoder/decoder, DOS skips + cuello de botella global.
  xformer : Graphormer serio: 8 capas, dim 256, 8 cabezas, sesgo espacial por
            distancia, warmup+coseno. ~4M params.

Mejoras de receta (aprendidas de la ronda 1):
  - SELECCIÓN POR MÉTRICA ALINEADA: el checkpoint se elige por val_dR_p90 (el
    p90 del error de posición en validación), NO por val_mae_mod (que en la
    ronda 1 mejoró sin que mejorara el test → proxy engañoso).
  - Warmup lineal (5 épocas) + coseno. --lr y --epochs configurables.
  - --tag para réplicas. Modo --eval integrado: evalúa el best.pth en la métrica
    REAL (recuperación p90 macro sobre los 61 canales del test) y guarda JSON.

Uso (entrenar → evaluar):
    python sandbox_arch2.py pna
    python sandbox_arch2.py unet2 --epochs 50
    python sandbox_arch2.py xformer --lr 5e-4
    python sandbox_arch2.py pna --eval               # métrica real sobre test
    python sandbox_arch2.py pna --eval --events 100000
    python sandbox_arch2.py pna --tag r2             # réplica (carpeta propia)

Referencia a batir (misma métrica, mismo test):
  - val_dR_p90: la imprime el propio script para el HexCNN si existe (informativo).
  - recov_p90 macro en --eval: HexCNN = 56.4 @60k ev / 58.0 @700k ev.

Autor: Miguel Escudero (TFM)
"""

import sys
import json
import math
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dataset import (SiPMImputationDataset, load_dat_to_dense, get_file_split,
                     load_positions, N_ACTIVE)
from hex_geometry import get_neighbor_matrix
from sandbox_arch import MP, TBlock                      # piezas de la ronda 1

GOOD_DIR   = r'E:\Datos TFM\Good\Good'
PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'
RUNS_BASE  = r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\runs'

BATCH_SIZE   = 512
WEIGHT_DECAY = 1e-4
MAX_EVENTS   = 300_000
WARMUP       = 5
VAL_MASK_SEED = 12345


# ─────────────────────────────────────────────────────────────
#  PNA: agregación multi-estadístico (mean / max / min / std)
# ─────────────────────────────────────────────────────────────

class PNAConv(nn.Module):
    """
    Convolución de grafo con agregación PNA: en vez de resumir los vecinos solo
    con la media (que tira información), concatena mean/max/min/std y proyecta.
    Mantiene el esqueleto HexConv (self + vecinos, pesos compartidos, BN).

    scalers=True añade los DEGREE SCALERS del paper original (Corso 2020): además
    de los agregadores, se escala por el grado del nodo con
        S(d, α) = (log(d+1)/δ)^α,   α ∈ {+1, 0, −1}   (amplifica / neutro / atenúa)
    y se concatenan las tres versiones (→ 12 bloques en vez de 4). Aquí el grado
    va de 3 (borde) a 6 (interior): los scalers permiten a la red tratar distinto
    a un nodo con pocos vecinos, que es justo el caso problemático del borde.
    """

    def __init__(self, in_f, out_f, neighbor_matrix, scalers=False):
        super().__init__()
        self.scalers = scalers
        n_agg = 12 if scalers else 4                        # 4 agregadores × 3 escalas
        self.lin_self = nn.Linear(in_f, out_f)
        self.lin_agg  = nn.Linear(n_agg * in_f, out_f, bias=False)
        self.bn = nn.BatchNorm1d(out_f)
        nbr_t = torch.as_tensor(neighbor_matrix, dtype=torch.long)
        self.register_buffer('nbr', nbr_t)
        # Escalas por grado, precalculadas (constantes del detector): (N,1)
        deg = (nbr_t >= 0).sum(1).to(torch.float32)
        logd = torch.log(deg + 1.0)
        delta = logd.mean()                                 # δ = media de log(d+1)
        self.register_buffer('s_amp', (logd / delta).unsqueeze(-1))          # α=+1
        self.register_buffer('s_att', (delta / logd).unsqueeze(-1))          # α=−1

    def forward(self, x):                                  # (B, N, F)
        B, N, Fin = x.shape
        xpad = torch.cat([x, x.new_zeros(B, 1, Fin)], dim=1)
        gi = torch.where(self.nbr >= 0, self.nbr, torch.full_like(self.nbr, N))
        nb = xpad[:, gi, :]                                # (B, N, 6, F)
        valid = (self.nbr >= 0).to(x.dtype).view(1, N, -1, 1)
        cnt = valid.sum(2).clamp(min=1)

        mean = (nb * valid).sum(2) / cnt
        var  = (((nb - mean.unsqueeze(2)) ** 2) * valid).sum(2) / cnt
        std  = torch.sqrt(var + 1e-8)
        big  = torch.finfo(x.dtype).max / 4
        mx = nb.masked_fill(valid == 0, -big).max(2).values
        mn = nb.masked_fill(valid == 0,  big).min(2).values

        agg = torch.cat([mean, mx, mn, std], dim=-1)       # (B, N, 4F)
        if self.scalers:                                   # identidad | amplificada | atenuada
            agg = torch.cat([agg, agg * self.s_amp, agg * self.s_att], dim=-1)
        out = self.lin_self(x) + self.lin_agg(agg)
        out = self.bn(out.reshape(B * N, -1)).reshape(B, N, -1)
        return F.gelu(out)


class PNARes(nn.Module):
    def __init__(self, dim, nbr, drop=0.1, scalers=False):
        super().__init__()
        self.c1 = PNAConv(dim, dim, nbr, scalers)
        self.c2 = PNAConv(dim, dim, nbr, scalers)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.c2(self.drop(self.c1(x)))


class PNAImputer(nn.Module):
    """HexCNN con agregación PNA: mismo esqueleto (stem + 4 bloques res + head)."""
    def __init__(self, hidden=64, n_blocks=4, nbr=None, dropout=0.1, scalers=False):
        super().__init__()
        self.stem = nn.Linear(2, hidden); self.stem_bn = nn.BatchNorm1d(hidden)
        self.blocks = nn.ModuleList([PNARes(hidden, nbr, dropout, scalers)
                                     for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        B = x.shape[0]
        h = self.stem(x.permute(0, 2, 1))
        h = F.gelu(self.stem_bn(h.reshape(B * N_ACTIVE, -1)).reshape(B, N_ACTIVE, -1))
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(-1)


class GraphUNet2(nn.Module):
    """U-Net de grafo PROFUNDA: 4+4 capas MP dim 128, dos skips + bottleneck global."""
    def __init__(self, dim=128, nbr=None, dropout=0.1):
        super().__init__()
        self.stem = nn.Linear(2, dim)
        self.e1 = nn.ModuleList([MP(dim, nbr), MP(dim, nbr)])
        self.e2 = nn.ModuleList([MP(dim, nbr), MP(dim, nbr)])
        self.to_g = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.merge = nn.Linear(2 * dim, dim)
        self.d2 = nn.ModuleList([MP(dim, nbr), MP(dim, nbr)])
        self.d1 = nn.ModuleList([MP(dim, nbr), MP(dim, nbr)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        h = F.gelu(self.stem(x.permute(0, 2, 1)))
        s1 = h
        for b in self.e1: h = b(h)
        s2 = h
        for b in self.e2: h = b(h)
        g = self.to_g(h.mean(dim=1))                       # bottleneck global
        h = self.merge(torch.cat([h, g.unsqueeze(1).expand_as(h)], dim=-1))
        h = self.drop(h)
        for b in self.d2: h = b(h)
        h = h + s2                                         # skip nivel 2
        for b in self.d1: h = b(h)
        h = h + s1                                         # skip nivel 1
        return self.head(h).squeeze(-1)


class Graphormer2(nn.Module):
    """Graphormer serio: 8 capas, dim 256, sesgo espacial por distancia."""
    def __init__(self, dim=256, heads=8, layers=8, dist=None, dropout=0.1):
        super().__init__()
        self.stem = nn.Linear(2, dim)
        self.pos  = nn.Parameter(torch.zeros(1, N_ACTIVE, dim))
        self.blocks = nn.ModuleList([TBlock(dim, heads, dist, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        h = self.stem(x.permute(0, 2, 1)) + self.pos
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h)).squeeze(-1)


# ─────────────────────────────────────────────────────────────
#  MÉTRICA DE SELECCIÓN: dR p90 en validación (alineada con el test)
# ─────────────────────────────────────────────────────────────

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, device, xs_t, ys_t):
    """val_loss, val_mae_mod y val_dR_p90 (p90 del error de posición en modified)."""
    model.eval()
    err_mod, n_mod, tot, n = 0.0, 0, 0.0, 0
    dRs = []
    for x_in, target, dead, is_mod in loader:
        x_in, target, dead = x_in.to(device), target.to(device), dead.to(device)
        out = model(x_in)
        tot += F.mse_loss(out, target).item() * len(target); n += len(target)
        err = (((out - target) * dead).sum(1)).abs()
        m = is_mod.to(device).bool()
        err_mod += err[m].sum().item(); n_mod += m.sum().item()
        # dR del centroide con el canal imputado (igual que delta_r de train.py)
        x_imp = target * (1 - dead) + out * dead
        w_o, w_i = target ** 2, x_imp ** 2
        so = w_o.sum(1) + 1e-6; si = w_i.sum(1) + 1e-6
        px_o = (w_o * xs_t).sum(1) / so; py_o = (w_o * ys_t).sum(1) / so
        px_i = (w_i * xs_t).sum(1) / si; py_i = (w_i * ys_t).sum(1) / si
        dR = torch.sqrt((px_i - px_o) ** 2 + (py_i - py_o) ** 2 + 1e-12)
        dRs.append(dR[m].cpu())
    dR_all = torch.cat(dRs).numpy() if dRs else np.array([9.9])
    return tot / max(n, 1), err_mod / max(n_mod, 1), float(np.percentile(dR_all, 90))


def build_model(arch, nbr, dist, dim=None, layers=8, heads=8, scalers=False):
    if arch == 'pna':
        return PNAImputer(hidden=dim or 64, nbr=nbr, scalers=scalers)
    if arch == 'unet2':
        return GraphUNet2(dim=dim or 128, nbr=nbr)
    if arch == 'xformer':
        return Graphormer2(dim=dim or 256, heads=heads, layers=layers, dist=dist)
    raise ValueError(arch)


# ─────────────────────────────────────────────────────────────
#  EVAL EN LA MÉTRICA REAL (recov p90 macro sobre el test)
# ─────────────────────────────────────────────────────────────

def eval_real(arch, out_dir, device, nbr, dist, events):
    from imputation_eval import impute_channel, compute_xy
    ck = torch.load(out_dir / 'best.pth', map_location=device, weights_only=False)
    model = build_model(arch, nbr, dist, dim=ck.get('dim'),
                        layers=ck.get('layers', 8), heads=ck.get('heads', 8),
                        scalers=ck.get('scalers', False))
    model.load_state_dict(ck['model_state']); model.to(device).eval()
    # El preprocesado del eval DEBE coincidir con el del entrenamiento (si no, el
    # resultado se desploma sin que el modelo tenga la culpa).
    model._norm_mode = ck.get('norm_mode', 'max')
    model._chan_norm = ck.get('channel_norm', False)
    if model._chan_norm:
        from hex_geometry import get_channel_scale
        model._chan_scale = get_channel_scale()
    xs, ys = load_positions(PSIPM_PATH)
    _, _, test_files = get_file_split(GOOD_DIR)
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
        pd = np.percentile(np.concatenate(dRdeg), 90)
        pi = np.percentile(np.concatenate(dRimp), 90)
        recs.append((pd - pi) / pd * 100 if pd > 0 else 0.0)
        maes.append(np.concatenate(errs).mean())
        if (c + 1) % 15 == 0:
            print(f"  {c+1}/{N_ACTIVE} canales")

    rec, mae = float(np.mean(recs)), float(np.mean(maes))
    print(f"\n=== MÉTRICA REAL ({arch}, {events} ev/archivo, macro {len(recs)} canales) ===")
    print(f"  recov_p90 macro = {rec:.1f}%   MAE = {mae:.3f}")
    print(f"  referencia HexCNN: 56.4 @60k ev  |  58.0 @700k ev  (comparar a MISMOS eventos)")
    out = {'arch': arch, 'events': events, 'recov_p90_macro': round(rec, 2),
           'mae_macro': round(mae, 4), 'train_epoch': ck.get('epoch'),
           'val_dr_p90': ck.get('val_dr_p90'), 'params': count_params(model),
           'generated': datetime.datetime.now().isoformat(timespec='seconds'),
           'recov_per_channel': [round(float(v), 2) for v in recs]}
    (out_dir / f'eval_real_{events}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"  JSON: {out_dir / f'eval_real_{events}.json'}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arch', choices=['pna', 'unet2', 'xformer'])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dim', type=int, default=None)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--norm', choices=['max', 'sum'], default='max',
                    help="normalizacion por evento: 'max' (historico) o 'sum' (RchT, estable)")
    ap.add_argument('--scalers', action='store_true',
                    help='PNA con degree scalers (Corso 2020): escala por grado del nodo')
    ap.add_argument('--tag', default='')
    ap.add_argument('--eval', action='store_true', help='evaluar best.pth en la métrica real')
    ap.add_argument('--events', type=int, default=60_000, help='eventos/archivo en --eval')
    args = ap.parse_args()
    tag = f'_{args.tag}' if args.tag else ''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xs, ys = load_positions(PSIPM_PATH)
    nbr = get_neighbor_matrix(PSIPM_PATH)
    dxy = np.hypot(xs[:, None] - xs, ys[:, None] - ys)
    dist = dxy / np.median(dxy[dxy > 0])
    out_dir = Path(RUNS_BASE) / f'sandbox2_{args.arch}{tag}'

    if args.eval:
        eval_real(args.arch, out_dir, device, nbr, dist, args.events)
        return

    xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
    ys_t = torch.tensor(ys, dtype=torch.float32, device=device)
    model = build_model(args.arch, nbr, dist, dim=args.dim, layers=args.layers,
                        heads=args.heads, scalers=args.scalers).to(device)
    dim_used = args.dim or {'pna': 64, 'unet2': 128, 'xformer': 256}[args.arch]
    sc = ('  |  degree scalers: ON' if args.scalers else '') +          (f'  |  norm: {args.norm}' if args.norm != 'max' else '')
    print(f"RONDA 2 — {args.arch}  |  dim={dim_used}  |  params: {count_params(model):,}{sc}")
    print("Selección por val_dR_p90 (métrica alineada), warmup+coseno.")

    train_files, val_files, _ = get_file_split(GOOD_DIR)
    X_val = np.concatenate([load_dat_to_dense(f, max_events=MAX_EVENTS // len(val_files))
                            for f in val_files])
    val_loader = DataLoader(SiPMImputationDataset(X_val, seed=VAL_MASK_SEED,
                                                 norm_mode=args.norm),
                            batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    def lr_factor(e):                                   # warmup lineal + coseno
        if e < WARMUP:
            return (e + 1) / WARMUP
        t = (e - WARMUP) / max(args.epochs - WARMUP, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    out_dir.mkdir(parents=True, exist_ok=True)
    best = 1e9
    hist = []
    for epoch in range(1, args.epochs + 1):
        f = train_files[(epoch - 1) % len(train_files)]
        X = load_dat_to_dense(f, max_events=MAX_EVENTS)
        loader = DataLoader(SiPMImputationDataset(X, seed=epoch, norm_mode=args.norm),
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
        vloss, vmae, vdr90 = evaluate(model, val_loader, device, xs_t, ys_t)
        hist.append({'epoch': epoch, 'val_loss': vloss, 'val_mae_mod': vmae, 'val_dr_p90': vdr90})
        flag = ''
        if vdr90 < best:                                  # ← selección por dR p90
            best = vdr90
            torch.save({'arch': args.arch, 'model_state': model.state_dict(),
                        'epoch': epoch, 'val_mae_mod': vmae, 'val_dr_p90': vdr90,
                        'dim': dim_used, 'layers': args.layers, 'heads': args.heads,
                        'scalers': args.scalers, 'norm_mode': args.norm},
                       out_dir / 'best.pth')
            flag = '  ✓ best'
        print(f"Epoch {epoch:3d}/{args.epochs} | lr={opt.param_groups[0]['lr']:.2e} "
              f"val_mae_mod={vmae:.4f} val_dR_p90={vdr90:.4f}{flag}")

    (out_dir / 'history.json').write_text(json.dumps(hist, indent=2), encoding='utf-8')
    print(f"\nMEJOR val_dR_p90 = {best:.4f}  →  ahora: python sandbox_arch2.py {args.arch} --eval")


if __name__ == '__main__':
    main()
