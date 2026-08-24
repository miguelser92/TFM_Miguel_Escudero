"""
eval_espectro.py
================
Conservación del ESPECTRO DE ENERGÍA tras la imputación (punto 4 del roadmap).

Idea: el espectro es el histograma de RchT (suma de las 61 cargas del evento,
proporcional a la energía del gamma). Apagar un sensor le roba carga a los
eventos donde tenía señal → su RchT baja → el espectro se corre a la izquierda.
La imputación debería devolverlo a su sitio. Medimos si lo hace.

Diseño DIFERENCIAL EMPAREJADO: los mismos eventos en tres versiones (original /
degradado / imputado). Todos los artefactos del espectro (singles, sin calibrar,
Compton, desalineación de zonas) son idénticos en las tres → se cancelan al
comparar. No hace falta calibración ni ver un fotopico bonito: todo va en ADC y %.

Zonas = METAHEXÁGONO: bandas hexagonales concéntricas según la distancia
hexagonal del evento (máxima proyección sobre las 3 normales a los lados del
detector). Es la estratificación físicamente correcta: el efecto de borde lo
causa el adhesivo de los LADOS → lo que importa es la distancia al lado más
cercano, y sus curvas de nivel son hexágonos, no círculos.

Métricas por (sensor de fallo × zona), solo eventos modified:
  - corrimiento de la media de RchT (degradado e imputado vs original; el del
    imputado ≈ el bias de la red expresado en el observable físico)
  - distancia de Wasserstein orig↔deg y orig↔imp → recuperación espectral %
  - posición del fotopico (ajuste gaussiano local, donde sea localizable) [bonus]

Salidas (runs/<run>/<out>/, por defecto ESPECTRO, sin sobrescritura):
  espectros.png (rejilla zonas×sensores), espectro_resumen.png,
  eval_espectro_report.pdf, eval_espectro_metrics.json y subida a W&B.

Uso:
    conda activate tfm
    python eval_espectro.py imputer_hexcnn_s_mse
    python eval_espectro.py imputer_hexcnn_s_mse --out ESPECTRO_v2 --events 200000
    python eval_espectro.py imputer_hexcnn_s_mse --quick

Autor: Miguel Escudero (TFM)
"""

import sys
import json
import time
import datetime
import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wasserstein_distance
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dataset import load_dat_to_dense, load_positions, get_file_split, ICH_TO_IDX
from imputation_eval import load_model, impute_channel, compute_xy

# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════

RUNS_BASE  = r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\runs'
GOOD_DIR   = r'E:\Datos TFM\Good\Good'
PSIPM_PATH = r'E:\Datos TFM\psipm.tsv'

TEST_MAX_EVENTS = None      # None = todos los eventos (limitable con --events N)

# Sensores de fallo representativos: centro, medio (el histórico), mitad de lado,
# esquina, y el canal anómalo detectado en el EVAL TOTAL.
FAIL_ICH = [13, 30, 32, 27, 31]

# Bandas del metahexágono (sobre la distancia hexagonal normalizada del evento)
ZONES = [('core', 0.00, 0.45), ('middle', 0.45, 0.75), ('edge', 0.75, 9.99)]

USE_WANDB     = True
WANDB_PROJECT = 'TFM-SiPM-imputation'


# ════════════════════════════════════════════════════════════
#  METAHEXÁGONO: distancia hexagonal de cada evento
# ════════════════════════════════════════════════════════════

def hex_distance(px, py):
    """
    Distancia "hexagonal" de cada posición al centro: máxima proyección sobre las
    3 normales a los lados del detector (0°, 60°, 120°). Sus curvas de nivel son
    hexágonos concéntricos con la MISMA orientación que el detector → un evento
    cerca del centro de un lado y otro cerca de una esquina se comparan por su
    distancia real al adhesivo, no por el radio circular.
    """
    d = np.zeros_like(px)
    for ang in (0.0, np.pi / 3, 2 * np.pi / 3):
        d = np.maximum(d, np.abs(px * np.cos(ang) + py * np.sin(ang)))
    return d


def zone_masks(rhex_norm):
    """Máscaras booleanas de las bandas del metahexágono."""
    return {name: (rhex_norm >= lo) & (rhex_norm < hi) for name, lo, hi in ZONES}


# ════════════════════════════════════════════════════════════
#  FOTOPICO (bonus): localización robusta + ajuste gaussiano local
# ════════════════════════════════════════════════════════════

def _gauss(x, a, mu, sig):
    return a * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def find_photopeak(rcht, hi_cut=99.0, bins=200):
    """
    Intenta localizar el fotopico: histograma suavizado → pico prominente de MAYOR
    RchT (el ruido/Compton domina a baja energía) → ajuste gaussiano local.
    Devuelve mu (posición del pico) o None si no es localizable. Lección del 22/06:
    el máximo simple cae en el ruido; por eso pico prominente + ajuste local.
    """
    try:
        hi = np.percentile(rcht, hi_cut)
        h, edges = np.histogram(rcht, bins=bins, range=(0, hi))
        centers = (edges[:-1] + edges[1:]) / 2
        hs = gaussian_filter1d(h.astype(float), sigma=2)
        # ignorar el 15% inferior del rango (suelo de ruido)
        lo_idx = int(0.15 * bins)
        peaks, props = find_peaks(hs[lo_idx:], prominence=0.02 * hs.max())
        if len(peaks) == 0:
            return None
        p = peaks[-1] + lo_idx                      # el pico prominente de mayor RchT
        # ajuste gaussiano en una ventana local de ±15% del rango
        w = int(0.15 * bins)
        sl = slice(max(p - w, 0), min(p + w, bins))
        popt, _ = curve_fit(_gauss, centers[sl], hs[sl],
                            p0=[hs[p], centers[p], 0.1 * hi], maxfev=5000)
        mu = float(popt[1])
        if not (0 < mu < hi):
            return None
        return mu
    except Exception:
        return None


# ════════════════════════════════════════════════════════════
#  NÚCLEO: espectros orig / deg / imp por escenario y zona
# ════════════════════════════════════════════════════════════

def eval_spectrum_model(run_name, X_list, zones_list, device, fail_ich):
    """
    Para cada sensor de fallo: RchT original / degradado / imputado de los eventos
    modified, estratificado por banda del metahexágono. Devuelve métricas + los
    arrays de RchT (para las figuras).
    """
    from imputation_eval import load_ckpt_meta
    ckpt_path = Path(RUNS_BASE) / run_name / 'best_model.pth'
    model = load_model(ckpt_path, device)
    ckpt = load_ckpt_meta(ckpt_path)

    scenarios, spectra = [], {}
    for k, ich in enumerate(fail_ich):
        ch = ICH_TO_IDX[ich]
        t0 = time.time()
        acc = {z: {'orig': [], 'deg': [], 'imp': []} for z, _, _ in ZONES}
        for X, zmasks in zip(X_list, zones_list):
            mod = X[:, ch] > 0
            if mod.sum() == 0:
                continue
            _, pred = impute_channel(model, X, ch, device)   # predicción en crudo
            rcht_o = X.sum(axis=1)
            rcht_d = rcht_o - X[:, ch]                        # quitar el canal
            rcht_i = rcht_d + pred                            # reponer la predicción
            for zname, zm in zmasks.items():
                m = mod & zm
                if m.sum() == 0:
                    continue
                acc[zname]['orig'].append(rcht_o[m])
                acc[zname]['deg'].append(rcht_d[m])
                acc[zname]['imp'].append(rcht_i[m])

        zres = {}
        for zname in acc:
            if not acc[zname]['orig']:
                continue
            o = np.concatenate(acc[zname]['orig'])
            d = np.concatenate(acc[zname]['deg'])
            i = np.concatenate(acc[zname]['imp'])
            spectra[(ich, zname)] = (o, d, i)

            w_deg = float(wasserstein_distance(o, d))
            w_imp = float(wasserstein_distance(o, i))
            rec = (w_deg - w_imp) / w_deg * 100 if w_deg > 0 else 0.0
            pk_o, pk_d, pk_i = find_photopeak(o), find_photopeak(d), find_photopeak(i)
            zres[zname] = {
                'n': int(o.size),
                'mean_orig': float(o.mean()),
                'shift_deg': float(d.mean() - o.mean()),   # cuánto roba el fallo
                'shift_imp': float(i.mean() - o.mean()),   # residuo tras imputar (≈ bias)
                'wasserstein_deg': w_deg,
                'wasserstein_imp': w_imp,
                'spectral_recovery_pct': rec,
                'photopeak': {'orig': pk_o, 'deg': pk_d, 'imp': pk_i},
            }
            print(f"  [Ich {ich:2d}] {zname:6s}: n={o.size:>9,}  "
                  f"shift deg={zres[zname]['shift_deg']:+.3f} imp={zres[zname]['shift_imp']:+.3f} ADC  "
                  f"W: {w_deg:.3f}->{w_imp:.3f}  recovery={rec:5.1f}%", flush=True)
        scenarios.append({'ich': ich, 'zones': zres})
        print(f"  [Ich {ich:2d}] hecho en {time.time()-t0:.0f}s  ({k+1}/{len(fail_ich)})", flush=True)

    return scenarios, spectra, ckpt


# ════════════════════════════════════════════════════════════
#  FIGURAS
# ════════════════════════════════════════════════════════════

def make_figures(run_name, scenarios, spectra, fail_ich, out_dir):
    """espectros.png (rejilla zonas × sensores) + espectro_resumen.png + PDF."""
    znames = [z for z, _, _ in ZONES]

    # ── Rejilla de espectros: filas = zonas, columnas = sensores de fallo ──
    nrow, ncol = len(znames), len(fail_ich)
    figS, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow))
    axes = np.atleast_2d(axes)
    for r, zname in enumerate(znames):
        for c, ich in enumerate(fail_ich):
            ax = axes[r, c]
            key = (ich, zname)
            if key not in spectra:
                ax.axis('off'); continue
            o, d, i = spectra[key]
            hi = np.percentile(o, 99)
            bins = np.linspace(0, hi, 150)
            ax.hist(o, bins=bins, color='0.55', alpha=0.55, label='original')
            ax.hist(d, bins=bins, histtype='step', lw=1.6, color='coral', label='degraded')
            ax.hist(i, bins=bins, histtype='step', lw=1.6, color='steelblue', label='imputed')
            zres = next(s for s in scenarios if s['ich'] == ich)['zones'][zname]
            ax.set_title(f"Ich {ich} — {zname}\n"
                         f"recovery {zres['spectral_recovery_pct']:.0f}%", fontsize=9.5)
            ax.set_xlim(0, hi)
            ax.tick_params(labelsize=7)
            if r == nrow - 1:
                ax.set_xlabel('RchT [ADC]', fontsize=8)
            if c == 0:
                ax.set_ylabel('events', fontsize=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=7)
    figS.suptitle(f'Energy spectrum (RchT) of affected events: original vs degraded vs imputed — '
                  f'{run_name}', fontsize=13, fontweight='bold')
    figS.tight_layout(rect=[0, 0, 1, 0.95])
    pS = out_dir / 'espectros.png'
    figS.savefig(pS, dpi=180, bbox_inches='tight')

    # ── Resumen: recuperación espectral + corrimientos de la media ──
    figR, axesR = plt.subplots(1, 2, figsize=(15.5, 5))
    width = 0.8 / len(znames)
    xs_ = np.arange(len(fail_ich))
    zcolors = {'core': '#2e7d52', 'middle': '#e0a63c', 'edge': '#c0532f'}
    for j, zname in enumerate(znames):
        recs, sh_d, sh_i = [], [], []
        for ich in fail_ich:
            zres = next(s for s in scenarios if s['ich'] == ich)['zones'].get(zname)
            recs.append(zres['spectral_recovery_pct'] if zres else np.nan)
            sh_d.append(zres['shift_deg'] if zres else np.nan)
            sh_i.append(zres['shift_imp'] if zres else np.nan)
        axesR[0].bar(xs_ + j * width, recs, width, color=zcolors.get(zname, '0.5'), label=zname)
        axesR[1].bar(xs_ + j * width, sh_d, width, color=zcolors.get(zname, '0.5'),
                     alpha=0.30)
        axesR[1].bar(xs_ + j * width, sh_i, width, color=zcolors.get(zname, '0.5'),
                     label=zname)
    for ax, ttl, yl in [(axesR[0], 'Spectral recovery (Wasserstein)', 'recovery [%]'),
                        (axesR[1], 'Mean RchT shift (light = degraded, solid = imputed)',
                         'shift [ADC]')]:
        ax.set_xticks(xs_ + width)
        ax.set_xticklabels([f'Ich {i}' for i in fail_ich])
        ax.set_title(ttl); ax.set_ylabel(yl)
        ax.grid(True, axis='y', alpha=0.3); ax.legend(title='zone', fontsize=8)
    axesR[1].axhline(0, color='k', lw=0.8)
    figR.suptitle(run_name, fontsize=12, fontweight='bold')
    pR = out_dir / 'espectro_resumen.png'
    figR.savefig(pR, dpi=200, bbox_inches='tight')

    with PdfPages(out_dir / 'eval_espectro_report.pdf') as pdf:
        pdf.savefig(figS, bbox_inches='tight')
        pdf.savefig(figR, bbox_inches='tight')
    plt.close(figS); plt.close(figR)
    print(f"  Guardado: {pS}\n  Guardado: {pR}\n  Guardado: {out_dir / 'eval_espectro_report.pdf'}")
    return [pS, pR]


# ════════════════════════════════════════════════════════════
#  W&B
# ════════════════════════════════════════════════════════════

def log_to_wandb(run_name, scenarios, pngs, out_name):
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb no instalado; no se sube el eval de espectro.")
        return
    suffix = '' if out_name == 'ESPECTRO' else f"_{out_name.replace('ESPECTRO_', '')}"
    run = wandb.init(project=WANDB_PROJECT, name=f'{run_name}_ESPECTRO{suffix}',
                     job_type='eval_espectro',
                     config={'run': run_name, 'fail_ich': [s['ich'] for s in scenarios],
                             'zones': [z for z, _, _ in ZONES]})
    recs = []
    for s in scenarios:
        for zname, z in s['zones'].items():
            run.summary[f"espectro/ich{s['ich']}/{zname}/recovery_pct"] = z['spectral_recovery_pct']
            run.summary[f"espectro/ich{s['ich']}/{zname}/shift_imp"] = z['shift_imp']
            recs.append(z['spectral_recovery_pct'])
    run.summary['espectro/recovery_pct_mean'] = float(np.mean(recs))
    run.log({p.stem: wandb.Image(str(p)) for p in pngs})
    print(f"  ✓ Espectro subido a W&B: {run.url}")
    run.finish()


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    argv = sys.argv[1:]
    quick = '--quick' in argv
    no_wandb = '--no-wandb' in argv or quick

    def flag_value(name, default):
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default

    out_name = flag_value('--out', 'ESPECTRO_quick' if quick else 'ESPECTRO')
    ev_arg = flag_value('--events', None)
    max_ev = 20_000 if quick else (int(ev_arg) if ev_arg else TEST_MAX_EVENTS)
    fail_ich = FAIL_ICH[:2] if quick else FAIL_ICH

    skip = set()
    for j, a in enumerate(argv):
        if a in ('--out', '--events'):
            skip.add(j); skip.add(j + 1)
        elif a.startswith('--'):
            skip.add(j)
    runs = [a for j, a in enumerate(argv) if j not in skip] or ['imputer_hexcnn_s_mse']

    print(f"EVAL ESPECTRO sobre {len(runs)} run(s): {runs}")
    print(f"  sensores de fallo: {fail_ich}  |  campaña: runs/<run>/{out_name}/")
    if quick:
        print("MODO --quick: 20k eventos/archivo, 2 sensores, sin W&B")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_sipm, y_sipm = load_positions(PSIPM_PATH)

    # La partición debe ser la MISMA con la que se entrenó cada run: evaluar con
    # otra significaría medir sobre módulos que el modelo sí vio. Se exige que
    # todos los runs de la tanda la compartan, porque los datos de test se cargan
    # una sola vez y se reutilizan. Mismo arreglo que eval_total (11/08) y
    # eval_resolution (23/08); ver E11 del registro de errores.
    def _split_seed_de(run):
        from imputation_eval import load_ckpt_meta
        try:
            return load_ckpt_meta(Path(RUNS_BASE) / run / 'best_model.pth').get('split_seed', 42)
        except Exception:
            return 42

    _seeds = {r: _split_seed_de(r) for r in runs}
    if len(set(_seeds.values())) > 1:
        raise SystemExit(f"Los runs no comparten partición: {_seeds}. Evalúalos por separado.")
    split_seed = next(iter(_seeds.values()))
    _, _, test_files = get_file_split(GOOD_DIR, seed=split_seed)
    if split_seed != 42:
        print(f"  PARTICION NO ESTANDAR (split_seed={split_seed}) -> test: "
              f"{[f.name for f in test_files]}")

    ev_str = f"{max_ev:,}" if max_ev else 'TODOS los'
    print(f"Cargando {len(test_files)} archivos de test ({ev_str} eventos c/u)...")
    X_list = [load_dat_to_dense(f, max_events=max_ev) for f in test_files]
    print(f"  total eventos: {sum(len(X) for X in X_list):,}")

    # ── Metahexágono: distancia hexagonal de cada evento (posición ORIGINAL) ──
    rhex_list = []
    for X in X_list:
        ox, oy = compute_xy(X, x_sipm, y_sipm)
        rhex_list.append(hex_distance(ox, oy))
    scale = np.percentile(np.concatenate(rhex_list), 99.5)   # escala robusta del borde
    zones_list = [zone_masks(rh / scale) for rh in rhex_list]
    pooled = np.concatenate([rh / scale for rh in rhex_list])
    zm_all = zone_masks(pooled)
    print("Bandas del metahexágono (eventos):",
          {z: f"{int(m.sum()):,}" for z, m in zm_all.items()})

    for run_name in runs:
        print(f"\n{'='*64}\nEVAL ESPECTRO: {run_name}\n{'='*64}")
        out_dir = Path(RUNS_BASE) / run_name / out_name
        if (out_dir / 'eval_espectro_metrics.json').exists():
            print(f"  YA EXISTE la campaña {out_name} → salto este run "
                  f"(usa --out <NOMBRE> para otra).")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        scenarios, spectra, ckpt = eval_spectrum_model(
            run_name, X_list, zones_list, device, fail_ich)

        recs = [z['spectral_recovery_pct'] for s in scenarios for z in s['zones'].values()]
        print(f"\n  RESUMEN: recuperación espectral media = {np.mean(recs):.1f}%  "
              f"(min {np.min(recs):.1f}%)")

        metrics = {
            'run': run_name, 'arch': ckpt.get('arch'),
            'generated': datetime.datetime.now().isoformat(timespec='seconds'),
            'test_files': [f.name for f in test_files],
            'max_events_per_file': max_ev if max_ev else 'all',
            'campaign': out_name,
            'zone_definition': {'bands': [{'name': n, 'lo': lo, 'hi': hi} for n, lo, hi in ZONES],
                                'hex_scale_p995': float(scale)},
            'scenarios': scenarios,
        }
        with open(out_dir / 'eval_espectro_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  Métricas: {out_dir / 'eval_espectro_metrics.json'}")

        pngs = make_figures(run_name, scenarios, spectra, fail_ich, out_dir)

        if USE_WANDB and not no_wandb:
            log_to_wandb(run_name, scenarios, pngs, out_name)

    print("\n✓ EVAL ESPECTRO terminado.")


if __name__ == '__main__':
    main()
