"""
bad_validate.py  (punto 2 — validación del detector de canales malos)
=====================================================================
¿Cómo de bueno es el detector de bad_detect.py? Aquí lo medimos con GROUND TRUTH,
matando canales nosotros en módulos GOOD (sabemos la verdad) y trazando la ROC.

Diseño limpio (sin leakage):
  1. Split por MÓDULO: BASELINE-set (construye la referencia) vs TEST-set (se juzga).
     El baseline NUNCA ve los módulos que evalúa.
  2. En cada módulo de TEST, para cada canal:
       - NEGATIVO: su score estando SANO  (debería ser ~0).
       - POSITIVO: su score tras DEGRADARLO a severidad s.
     Degradar a severidad s = el canal "muere" en una fracción s de los eventos.
     En esperanza eso escala frac_active y neighbor_ratio por (1−s): s=1 muerto
     total, s=0.5 medio pocho. (Cálculo exacto en esperanza, sin ruido de muestreo.)
  3. Barremos el umbral Z → curva TPR (detecta lo que maté) vs FPR (falsea sanos),
     y el AUC (independiente del umbral). Todo desglosado por SEVERIDAD y por
     POSICIÓN (centro vs borde) → prueba directa de que el borde no se falsea.

Uso:
    conda activate tfm
    python bad_validate.py
    python bad_validate.py --test-frac 0.3 --max-events 120000 --seed 42

Autor: Miguel Escudero (TFM)
"""

import sys
import glob
import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dataset import load_dat_to_dense, load_positions, N_ACTIVE, IDX_TO_ICH
from hex_geometry import get_neighbor_matrix
from bad_detect import module_stats, build_baseline
from eval_multidead import grow_cluster

GOOD_DIR   = r'E:\Datos TFM\Good\Good'
PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'
OUT_DIR    = Path(r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\reports')

SEVERITIES = (1.0, 0.7, 0.5, 0.3)   # 1.0 = muerto total; 0.3 = pierde el 30%
CLUSTER_K   = (1, 2, 3, 4)          # tamaños de clúster contiguo degradado (caso físico real)
CLUSTER_SEV = (1.0, 0.7, 0.5)       # severidad del clúster: total vs parcial (el caso difícil)
Z_OP       = 2.0                    # umbral de operación (ROC + FPR estable con module-norm)
EPS        = 1e-9


def channel_scores(f, r, base, split=False):
    """z-score robusto por canal para actividad (f) y ratio (r) → score = min de ambos."""
    z_f = (f - base['frac_median'])  / base['frac_spread']
    z_r = (r - base['ratio_median']) / base['ratio_spread']
    return (z_f, z_r) if split else np.minimum(z_f, z_r)


def cluster_scores(fa, q, nbr, dead_idx, severity, base):
    """
    Score de los canales de un CLÚSTER muerto (k canales contiguos apagados a la vez).

    Aquí NO vale escalar el ratio por (1−s) como en el caso de un canal aislado: si los
    vecinos también mueren, el denominador de ρ baja a la vez que el numerador y el ratio
    deja de ser anómalo (el estadístico se vuelve CIEGO). Por eso recalculamos ρ con el
    vector de cargas ya degradado. Devuelve (score, z_f, z_ratio) de los canales muertos.
    """
    q2 = q.copy()
    q2[dead_idx] *= (1 - severity)                    # los k canales pierden carga a la vez
    f2 = fa.copy()
    f2[dead_idx] *= (1 - severity)

    r2 = np.zeros(N_ACTIVE)
    for i in dead_idx:
        vec = [int(j) for j in nbr[i] if j >= 0]
        r2[i] = q2[i] / (np.mean(q2[vec]) + EPS) if vec else 0.0

    z_f, z_r = channel_scores(f2, r2, base, split=True)
    sel = np.asarray(dead_idx)
    return np.minimum(z_f[sel], z_r[sel]), z_f[sel], z_r[sel]


def roc_auc(pos, neg):
    """
    ROC de un detector que marca cuando score < −Z. Trabajamos con la 'anomalía'
    a = −score (mayor = más muerto). Devuelve (fpr, tpr, thr_Z, auc).
    """
    a_pos, a_neg = -np.asarray(pos), -np.asarray(neg)
    thr = np.sort(np.unique(np.concatenate([a_pos, a_neg])))[::-1]
    tpr = np.array([(a_pos >= t).mean() for t in thr])
    fpr = np.array([(a_neg >= t).mean() for t in thr])
    # extremos (0,0) y (1,1) para cerrar la curva
    fpr = np.concatenate([[0], fpr, [1]]); tpr = np.concatenate([[0], tpr, [1]])
    thr = np.concatenate([[np.inf], thr, [-np.inf]])
    trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))   # numpy 2.0 renombró trapz→trapezoid
    auc = trapz(tpr, fpr)
    return fpr, tpr, thr, auc


def rate_at_Z(pos, neg, Z):
    """TPR y FPR en el umbral score < −Z (marca malo)."""
    tpr = (np.asarray(pos) < -Z).mean() if len(pos) else 0.0
    fpr = (np.asarray(neg) < -Z).mean() if len(neg) else 0.0
    return float(tpr), float(fpr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-frac', type=float, default=0.30)
    ap.add_argument('--max-events', type=int, default=120_000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_sipm, y_sipm = load_positions(PSIPM_PATH)
    nbr = get_neighbor_matrix(PSIPM_PATH)
    r_center = np.hypot(x_sipm, y_sipm)                      # distancia al centro por canal
    edge = r_center >= np.percentile(r_center, 66)          # tercio exterior = "borde"

    # ── Split por módulo (baseline vs test), sin solapamiento ──
    files = sorted(glob.glob(str(Path(GOOD_DIR) / '*.dat')))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(files))
    n_test = max(10, int(len(files) * args.test_frac))
    test_files = [files[i] for i in perm[:n_test]]
    base_files = [files[i] for i in perm[n_test:]]
    print(f"Split: baseline={len(base_files)} módulos | test={len(test_files)} módulos "
          f"(sin solapamiento → sin leakage)")

    base = build_baseline(base_files, nbr, args.max_events)

    # ── Recolectar scores en el TEST-set ──
    # negativos: canal SANO (severidad-independiente). positivos: canal DEGRADADO por severidad.
    neg, neg_edge = [], []
    pos = {s: [] for s in SEVERITIES}
    pos_edge = {s: [] for s in SEVERITIES}
    # Clústeres contiguos muertos (el caso físico real): score + las dos señales por separado
    clu = {(k, s): {'score': [], 'z_f': [], 'z_r': []}
           for k in CLUSTER_K for s in CLUSTER_SEV}
    rng_c = np.random.default_rng(args.seed)
    print(f"Evaluando {len(test_files)} módulos de test...")
    for k, f in enumerate(test_files):
        X = load_dat_to_dense(f, max_events=args.max_events)
        st = module_stats(X, nbr)
        fa, ra, q = st['frac_active'], st['neigh_ratio'], st['mean_charge']

        # Negativos: score del canal sano
        s_healthy = channel_scores(fa, ra, base)
        neg.extend(s_healthy.tolist())
        neg_edge.extend(edge.tolist())

        # Positivos (canal AISLADO): degradar cada canal a cada severidad (escala f y r por (1−s))
        for s in SEVERITIES:
            s_deg = channel_scores(fa * (1 - s), ra * (1 - s), base)
            pos[s].extend(s_deg.tolist())
            pos_edge[s].extend(edge.tolist())

        # Positivos (CLÚSTER contiguo, muerte total): k canales pegados mueren a la vez
        for kk in CLUSTER_K:
            for seed_ch in range(N_ACTIVE):
                dead = grow_cluster(nbr, seed_ch, kk, rng_c)
                for sev in CLUSTER_SEV:
                    sc, zf, zr = cluster_scores(fa, q, nbr, dead, sev, base)
                    clu[(kk, sev)]['score'].extend(sc.tolist())
                    clu[(kk, sev)]['z_f'].extend(zf.tolist())
                    clu[(kk, sev)]['z_r'].extend(zr.tolist())
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(test_files)} módulos")

    neg = np.array(neg); neg_edge = np.array(neg_edge, dtype=bool)

    # ── ROC/AUC por severidad + operating point en Z_OP ──
    print("\n=== VALIDACIÓN DEL DETECTOR (TEST-set, ground truth por inyección) ===")
    print(f"{'severidad':>10} {'AUC':>6} {f'TPR@Z{Z_OP}':>8} {f'FPR@Z{Z_OP}':>8} {'TPR borde':>10} {'TPR centro':>11}")
    results = {}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    for s in SEVERITIES:
        p = np.array(pos[s]); pe = np.array(pos_edge[s], dtype=bool)
        fpr, tpr, thr, auc = roc_auc(p, neg)
        tpr_op, fpr_op = rate_at_Z(p, neg, Z_OP)
        tpr_edge, _   = rate_at_Z(p[pe],  neg[neg_edge],  Z_OP)
        tpr_core, _   = rate_at_Z(p[~pe], neg[~neg_edge], Z_OP)
        results[f's{s}'] = {'severity': s, 'auc': round(float(auc), 4),
                            'tpr_at_Z4': round(tpr_op, 4), 'fpr_at_Z4': round(fpr_op, 4),
                            'tpr_edge': round(tpr_edge, 4), 'tpr_core': round(tpr_core, 4)}
        print(f"{s:>10.1f} {auc:>6.3f} {tpr_op:>8.3f} {fpr_op:>8.3f} {tpr_edge:>10.3f} {tpr_core:>11.3f}")
        axL.plot(fpr, tpr, lw=2, label=f'severity {int(s*100)}%  (AUC {auc:.3f})')

    # FPR global y en borde (severidad-independiente)
    fpr_all  = (neg < -Z_OP).mean()
    fpr_edge = (neg[neg_edge] < -Z_OP).mean()
    fpr_core = (neg[~neg_edge] < -Z_OP).mean()
    print(f"\n  Falsos positivos @Z{Z_OP}: global={fpr_all:.3f}  borde={fpr_edge:.3f}  centro={fpr_core:.3f}")
    print(f"  (clave: borde ≈ centro → el detector NO falsea los bordes sanos)")

    axL.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    axL.set_xlabel('False positive rate (healthy flagged)')
    axL.set_ylabel('True positive rate (dead detected)')
    axL.set_title('ROC by failure severity')
    axL.legend(fontsize=9); axL.grid(alpha=0.3)
    axL.set_xlim(-0.01, 1); axL.set_ylim(0, 1.01)

    # Panel derecho: TPR borde vs centro por severidad (la prueba del borde)
    sv = [int(s*100) for s in SEVERITIES]
    axR.plot(sv, [results[f's{s}']['tpr_edge'] for s in SEVERITIES], 'o-', color='#c0392b', label='edge channels')
    axR.plot(sv, [results[f's{s}']['tpr_core'] for s in SEVERITIES], 's-', color='#2471a3', label='core channels')
    axR.set_xlabel('Failure severity (% of signal lost)')
    axR.set_ylabel(f'Detection rate @ Z={Z_OP}')
    axR.set_title(f'Detection vs severity — edge vs core (Z={Z_OP})')
    axR.set_ylim(0, 1.02); axR.grid(alpha=0.3); axR.legend(fontsize=9)
    fig.tight_layout()
    roc_path = OUT_DIR / f'bad_validation_roc_z{Z_OP}.png'
    fig.savefig(roc_path, dpi=150, bbox_inches='tight'); plt.close(fig)

    # ── Elegir el umbral CON DATOS (no a ojo) ──
    # Guardamos los scores para poder re-analizar sin recomputar.
    np.savez(OUT_DIR / 'bad_validation_scores.npz', neg=neg, neg_edge=neg_edge,
             **{f'pos_{s}': np.array(pos[s]) for s in SEVERITIES},
             **{f'pos_edge_{s}': np.array(pos_edge[s], dtype=bool) for s in SEVERITIES})

    print("\n=== BARRIDO DE UMBRAL (elegir Z con datos) ===")
    print(f"{'Z':>5} {'FPR':>7} " + ' '.join(f'TPR_s{int(s*100)}'.rjust(9) for s in SEVERITIES))
    Z_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    sweep = {}
    for Z in Z_grid:
        fp = float((neg < -Z).mean())
        tps = [float((np.array(pos[s]) < -Z).mean()) for s in SEVERITIES]
        sweep[str(Z)] = {'fpr': round(fp, 4),
                         **{f'tpr_s{int(s*100)}': round(t, 4) for s, t in zip(SEVERITIES, tps)}}
        print(f"{Z:>5.1f} {fp:>7.4f} " + ' '.join(f'{t:>9.3f}' for t in tps))

    # Z óptimo: el MENOR umbral (más sensible) que mantiene FPR <= 1%
    cands = [Z for Z in np.arange(0.5, 6.01, 0.05) if (neg < -Z).mean() <= 0.01]
    z_opt = float(min(cands)) if cands else Z_OP
    tpr_opt = {f's{int(s*100)}': round(float((np.array(pos[s]) < -z_opt).mean()), 4)
               for s in SEVERITIES}
    fpr_opt = float((neg < -z_opt).mean())
    print(f"\n  Z ÓPTIMO (FPR<=1%): Z={z_opt:.2f} → FPR={fpr_opt:.4f}, "
          f"TPR muerte total={tpr_opt['s100']:.3f}, 50%={tpr_opt['s50']:.3f}")

    # ── Clústeres contiguos: ¿aguanta el detector cuando mueren vecinos juntos? ──
    print(f"\n=== CLÚSTERES CONTIGUOS MUERTOS (Z={z_opt:.2f}) ===")
    print("  Al morir los vecinos, el ratio deja de ser anómalo (numerador y denominador")
    print("  bajan juntos) → hay que ver si la ACTIVIDAD sostiene la detección.")
    print(f"{'sev':>5} {'k':>3} {'TPR':>7} {'z_frac':>8} {'z_ratio':>9} {'ratio ciego':>12}")
    clus_res = {}
    for sev in CLUSTER_SEV:
        for kk in CLUSTER_K:
            d = clu[(kk, sev)]
            sc = np.array(d['score']); zf = np.array(d['z_f']); zr = np.array(d['z_r'])
            tpr = float((sc < -z_opt).mean())
            blind = float((zr > -z_opt).mean())   # fracción en que el ratio NO detectaría
            clus_res[f'k{kk}_s{int(sev*100)}'] = {
                'k': kk, 'severity': sev, 'tpr': round(tpr, 4),
                'z_frac_mean': round(float(zf.mean()), 2),
                'z_ratio_mean': round(float(zr.mean()), 2),
                'ratio_blind_frac': round(blind, 4)}
            print(f"{sev:>5.1f} {kk:>3} {tpr:>7.3f} {zf.mean():>8.1f} {zr.mean():>9.1f} {blind:>11.1%}")
        print()

    out = {'seed': args.seed, 'test_frac': args.test_frac, 'max_events': args.max_events,
           'clusters': clus_res,
           'threshold_sweep': sweep, 'z_optimal': round(z_opt, 2),
           'fpr_at_optimal': round(fpr_opt, 4), 'tpr_at_optimal': tpr_opt,
           'n_baseline_modules': len(base_files), 'n_test_modules': len(test_files),
           'Z_operating': Z_OP, 'n_neg': int(len(neg)), 'n_pos_per_sev': int(len(pos[SEVERITIES[0]])),
           'fpr_global': round(float(fpr_all), 4), 'fpr_edge': round(float(fpr_edge), 4),
           'fpr_core': round(float(fpr_core), 4), 'by_severity': results}
    (OUT_DIR / 'bad_validation.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"\n  guardado: reports/bad_validation.json y {roc_path.name}")


if __name__ == '__main__':
    main()
