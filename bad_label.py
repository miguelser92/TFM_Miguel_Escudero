"""
bad_label.py — etiquetado MANUAL interactivo de sensores averiados en los módulos BAD
====================================================================================
El estadístico automático (bad_detect.py) resuelve bien los canales completamente
muertos, pero no separa las degradaciones parciales. Esta herramienta permite
construir la verdad de referencia a mano: se recorre módulo a módulo, se inspecciona
el flood map y el patrón de actividad, y se marcan con el ratón los sensores que
fallan. El resultado se guarda en un JSON reutilizable por el resto del pipeline.

QUÉ SE VE EN PANTALLA
---------------------
  · Panel izquierdo   : mapa hexagonal de los 61 SiPM, CLICABLE. Cada hexágono se
                        colorea por la métrica elegida (z-score, fracción activa o
                        carga media). Al pinchar un sensor se marca/desmarca como
                        averiado (borde rojo grueso).
  · Panel central     : flood map de la ventana de eventos seleccionada. Un canal
                        muerto deforma la retícula y deja un hueco: es la evidencia
                        visual que se busca.
  · Panel derecho     : flood map tras imputar con la red los canales marcados. Si
                        la retícula se recompone, el diagnóstico era correcto.
  · Banda inferior    : z-score por canal frente al baseline de los módulos Good.
                        Las barras por debajo de la línea roja son las sospechosas.

CONTROLES
---------
  Ratón   : clic en un hexágono → marcar / desmarcar el sensor como averiado.
  Sliders : primer evento de la ventana y tamaño de la ventana. Mover el inicio con
            una ventana pequeña permite detectar fallos INTERMITENTES (un canal que
            se cae a mitad de la adquisición no se ve en el promedio global).
  ← / →   : módulo anterior / siguiente (guarda automáticamente el actual).
  i       : imputar los canales marcados y refrescar el panel derecho.
  + / -   : subir o bajar la RESOLUCIÓN del flood map. Con pocos bins se ve la
            retícula entera; con muchos, el hueco de un sensor concreto.
  z       : ZOOM al entorno de los sensores marcados (y de vuelta). En vista
            completa el hueco de un sensor ocupa unos pocos píxeles.
  d       : panel derecho entre IMPUTADO y DIFERENCIA (imputado − crudo). La
            diferencia muestra en rojo dónde la imputación añade cuentas y en azul
            dónde las quita, que se lee mucho mejor que comparar dos mapas casi
            iguales a ojo.
  u       : precargar la sugerencia del estadístico automático (punto de partida).
  c       : limpiar todas las marcas del módulo.
  r       : marcar el módulo como revisado sin canales averiados.
  s       : guardar el JSON ahora.
  q       : guardar y salir.

Los dos flood maps llevan color propio (azul el medido, verde el reconstruido,
morado la diferencia) en el título y en el borde, porque son casi idénticos y sin
esa marca es fácil creer que se está mirando el otro.

USO
---
    conda activate tfm
    python bad_label.py --tag manual1
    python bad_label.py --tag manual1 --labels reports/bad_labels_manual1.json   # continuar
    python bad_label.py --tag rapido --max-events 80000 --no-model               # sin inferencia
    python bad_label.py --tag manual1 --run imputer_hexcnn_s_mse_dead1-8

SALIDA
------
    reports/bad_labels_<tag>.json   con, por módulo, los Ich marcados a mano, los
    que sugería el estadístico, y si el módulo ya ha sido revisado.

Autor: Miguel Escudero (TFM)
"""

import sys
import json
import glob
import argparse
import datetime
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')                       # backend interactivo: hace falta ventana
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import Normalize, PowerNorm

from dataset import load_dat_to_dense, load_positions, N_ACTIVE, IDX_TO_ICH, ICH_TO_IDX
from hex_geometry import get_neighbor_matrix

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BAD_DIR      = r'E:\Datos TFM\Bad\Bad'
PSIPM_PATH   = r'E:\Datos TFM\psipm.tsv'
OUT_DIR      = Path(r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\reports')
RUNS_BASE    = Path(r'C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código\runs')
BASELINE_NPZ = OUT_DIR / 'good_baseline.npz'
DEFAULT_RUN  = 'imputer_hexcnn_s_mse_dead1-8'   # multi-dead: admite k canales a la vez

Z_SUGGEST    = 2.0        # umbral del estadístico para la sugerencia automática
IMPUTE_CAP   = 60_000     # tope de eventos que se imputan de una vez (fluidez)
EPS          = 1e-9


# ════════════════════════════════════════════════════════════
#  ESTADÍSTICOS  (replicados de bad_detect.py a propósito)
#
#  bad_detect.py fija matplotlib en backend 'Agg' al importarse, lo que dejaría esta
#  ventana sin GUI. Como las dos funciones que hacen falta son cortas, se replican
#  aquí en lugar de importarlas. Si se cambia el estadístico allí, actualizar aquí.
# ════════════════════════════════════════════════════════════

def module_stats(X, nbr):
    """Fracción de eventos con señal, carga media y cociente con los vecinos, por canal."""
    frac_active = (X > 0).mean(axis=0)
    mean_charge = X.mean(axis=0)
    neigh_ratio = np.zeros(N_ACTIVE, dtype=np.float64)
    for i in range(N_ACTIVE):
        vecinos = [int(j) for j in nbr[i] if j >= 0]
        mn = mean_charge[vecinos].mean() if vecinos else 0.0
        neigh_ratio[i] = mean_charge[i] / (mn + EPS)
    return frac_active, neigh_ratio, mean_charge


def module_zscore(X, nbr, base):
    """
    z-score robusto por canal frente al baseline de los Good, normalizado por módulo.

    Negativo = el canal rinde por debajo de lo esperado EN SU POSICIÓN. Se resta la
    mediana del módulo para que un detector globalmente frío no marque todo a la vez.
    """
    frac, ratio, charge = module_stats(X, nbr)
    z_frac  = (frac  - base['frac_median'])  / base['frac_spread']
    z_ratio = (ratio - base['ratio_median']) / base['ratio_spread']
    z_frac  = z_frac  - np.median(z_frac)
    z_ratio = z_ratio - np.median(z_ratio)
    return np.minimum(z_frac, z_ratio), frac, charge


def compute_xy(X, x_sipm, y_sipm):
    """Posición del evento por centro de gravedad Rch² (lógica de Anger)."""
    w = X ** 2
    wsum = w.sum(axis=1, keepdims=True)
    wsum[wsum == 0] = 1.0
    return (w * x_sipm).sum(axis=1) / wsum[:, 0], (w * y_sipm).sum(axis=1) / wsum[:, 0]


# ════════════════════════════════════════════════════════════
#  CARGA DE MÓDULOS (con caché en disco)
# ════════════════════════════════════════════════════════════

def load_module(path, max_events, cache_dir):
    """
    Devuelve la matriz densa (N,61) del módulo, cacheada en .npy para que la segunda
    visita sea instantánea. Etiquetar 62 módulos implica volver atrás muchas veces.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cf = cache_dir / f'{Path(path).stem}_{max_events}.npy'
        if cf.exists():
            return np.load(cf)
        X = load_dat_to_dense(path, max_events=max_events)
        np.save(cf, X)
        return X
    return load_dat_to_dense(path, max_events=max_events)


# ════════════════════════════════════════════════════════════
#  INFERENCIA (import diferido: torch tarda en cargar)
# ════════════════════════════════════════════════════════════

_MODEL = {'net': None, 'device': None, 'impute_set': None}


def get_model(run_name):
    """
    Carga el modelo de imputación la primera vez que se pide.

    imputation_eval y eval_multidead fijan matplotlib en 'Agg' al importarse, así que
    se restaura el backend interactivo justo después. La figura ya creada conserva su
    canvas de Tk; el cambio solo afectaría a figuras nuevas, que aquí no se crean.
    """
    if _MODEL['net'] is not None:
        return _MODEL

    prev = matplotlib.get_backend()
    import torch
    from imputation_eval import load_model
    from eval_multidead import impute_set
    matplotlib.use(prev, force=True)           # deshacer el 'Agg' de esos módulos

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = RUNS_BASE / run_name / 'best_model.pth'
    if not ckpt.exists() and not (RUNS_BASE / run_name / 'ensemble.json').exists():
        raise FileNotFoundError(f'no existe el checkpoint {ckpt}')
    print(f'Cargando modelo {run_name} en {device}...')
    _MODEL.update(net=load_model(ckpt, device), device=device, impute_set=impute_set)
    return _MODEL


def impute_dead(model_pack, X, dead_idx):
    """
    Imputa a la vez todos los canales de dead_idx.

    impute_set no contempla la normalización por canal, así que si el modelo se
    entrenó con ella se aplica aquí: se escala la entrada al espacio equalizado y se
    deshace la escala sobre la salida.
    """
    net, device, impute_set = model_pack['net'], model_pack['device'], model_pack['impute_set']
    chs = getattr(net, '_chan_scale', None) if getattr(net, '_chan_norm', False) else None
    if chs is None:
        X_imp, _ = impute_set(net, X, dead_idx, device)
        return X_imp
    X_imp, _ = impute_set(net, X / chs, dead_idx, device)
    return X_imp * chs


# ════════════════════════════════════════════════════════════
#  SESIÓN DE ETIQUETADO
# ════════════════════════════════════════════════════════════

class Labeler:
    """Estado y ventana del etiquetador manual."""

    def __init__(self, files, args, base, nbr, x_sipm, y_sipm, labels, out_json):
        self.files, self.args, self.base, self.nbr = files, args, base, nbr
        self.x_sipm, self.y_sipm = x_sipm, y_sipm
        self.labels, self.out_json = labels, out_json
        self.cache_dir = Path(args.cache_dir) if args.cache_dir else None

        # Geometría del hexágono de dibujo, deducida del paso real de la retícula
        d = np.hypot(x_sipm[:, None] - x_sipm, y_sipm[:, None] - y_sipm)
        self.pitch = float(np.median(np.sort(d, axis=1)[:, 1]))
        self.hex_r = self.pitch / np.sqrt(3) * 0.97
        m = self.pitch
        self.extent = [x_sipm.min() - m, x_sipm.max() + m,
                       y_sipm.min() - m, y_sipm.max() + m]

        self.mi = 0                    # índice del módulo actual
        self.X = None                  # matriz del módulo actual
        self.pos = None                # centroides precalculados (todo el módulo)
        self.marked = set()            # índices densos marcados a mano
        self.X_imp_cache = None        # flood imputado ya calculado
        self.color_mode = 'z-score'
        # Resolucion del flood map, ajustable en caliente con + y -. Una resolucion
        # fija no sirve: para localizar el hueco de un sensor hace falta fino, y para
        # ver la reticula completa conviene grueso.
        self.bins = None               # se fija al arrancar desde args.bins
        # Zoom: None = detector completo. Con sensores marcados, 'z' encuadra su zona,
        # que es donde hay que mirar para decidir si el sensor falla.
        self.zoom = None
        # Panel derecho: 'imputed' (flood reconstruido) o 'diff' (imputado - crudo).
        # El modo diferencia resalta DONDE cambia el mapa, que es mas informativo que
        # comparar dos floods casi identicos a ojo.
        self.right_mode = 'imputed'

        self.bins = int(self.args.bins)
        self._build_figure()
        self.load_current(first=True)

    # ── Persistencia ────────────────────────────────────────

    @property
    def fname(self):
        """
        Nombre del módulo REALMENTE cargado, no el que indique el índice.

        Las marcas se guardan bajo este nombre, así que anclarlo a la carga (y no a
        self.mi) evita escribirlas en el módulo equivocado si el índice se cambia sin
        recargar los datos.
        """
        return getattr(self, '_loaded_name', None) or Path(self.files[self.mi]).name

    def store_current(self, reviewed=True):
        """Vuelca las marcas del módulo actual al diccionario de etiquetas."""
        idx = sorted(self.marked)
        self.labels[self.fname] = {
            'dead_ich':      [int(IDX_TO_ICH[i]) for i in idx],
            'dead_idx':      [int(i) for i in idx],
            'n_dead':        len(idx),
            'suggested_ich': [int(IDX_TO_ICH[i]) for i in np.where(self.z < -Z_SUGGEST)[0]],
            'reviewed':      reviewed,
            'timestamp':     datetime.datetime.now().isoformat(timespec='seconds'),
        }

    def save(self, verbose=True):
        self.store_current()
        done = sum(1 for v in self.labels.values() if v.get('reviewed'))
        payload = {
            'tag': self.args.tag,
            'created': datetime.datetime.now().isoformat(timespec='seconds'),
            'bad_dir': BAD_DIR,
            'max_events': self.args.max_events,
            'z_suggest': Z_SUGGEST,
            'n_modules_total': len(self.files),
            'n_modules_reviewed': done,
            'labels': self.labels,
        }
        self.out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        if verbose:
            print(f'  guardado {self.out_json.name}  ({done}/{len(self.files)} módulos revisados)')

    # ── Carga de módulo ─────────────────────────────────────

    def load_current(self, first=False):
        """Carga el módulo actual, sus estadísticos y las marcas previas si las hubiera."""
        path = self.files[self.mi]
        self.ax_hex.set_title(f'Cargando {Path(path).name}...', fontsize=11)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        self.X = load_module(path, self.args.max_events, self.cache_dir)
        self._loaded_name = Path(path).name
        px, py = compute_xy(self.X, self.x_sipm, self.y_sipm)
        self.pos = (px, py)
        self.X_imp_cache = None

        prev = self.labels.get(self.fname)
        self.marked = set(prev['dead_idx']) if prev else set()

        n = len(self.X)
        self.s_start.valmax = max(n - 1, 1)
        self.s_start.ax.set_xlim(0, self.s_start.valmax)
        self.s_start.set_val(0)
        self.s_win.valmax = n
        self.s_win.ax.set_xlim(1000, max(n, 2000))
        self.s_win.set_val(min(n, self.args.max_events))
        self.refresh(recompute=True)

    def window(self):
        """Rango [a,b) de eventos que muestran los sliders."""
        a = int(self.s_start.val)
        b = min(a + int(self.s_win.val), len(self.X))
        return a, max(b, a + 1)

    # ── Figura ──────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(17.5, 9.5))
        self.fig.canvas.manager.set_window_title('bad_label — etiquetado manual de SiPM averiados')
        gs = self.fig.add_gridspec(2, 3, height_ratios=[3.1, 1.0],
                                   left=0.035, right=0.985, top=0.93, bottom=0.20,
                                   hspace=0.28, wspace=0.16)
        self.ax_hex  = self.fig.add_subplot(gs[0, 0])
        self.ax_raw  = self.fig.add_subplot(gs[0, 1])
        self.ax_imp  = self.fig.add_subplot(gs[0, 2])
        self.ax_prof = self.fig.add_subplot(gs[1, :])

        # Sliders de ventana de eventos
        self.s_start = Slider(self.fig.add_axes([0.10, 0.115, 0.55, 0.022]),
                              'First event', 0, 1, valinit=0, valstep=1000, color='#34495e')
        self.s_win   = Slider(self.fig.add_axes([0.10, 0.075, 0.55, 0.022]),
                              'Window size', 1000, 2000, valinit=1000, valstep=1000, color='#16a085')
        self.s_start.on_changed(lambda v: self.refresh(recompute=True))
        self.s_win.on_changed(lambda v: self.refresh(recompute=True))

        # Selector de métrica del mapa hexagonal
        self.radio = RadioButtons(self.fig.add_axes([0.695, 0.055, 0.10, 0.10]),
                                  ('z-score', 'active frac', 'mean charge'), active=0)
        self.radio.on_clicked(self._on_radio)

        # Botones
        def mkbtn(x, label, cb, color='0.85'):
            b = Button(self.fig.add_axes([x, 0.10, 0.075, 0.045]), label, color=color)
            b.on_clicked(cb)
            return b

        def mkbtn2(x, label, cb, color='0.85'):
            b = Button(self.fig.add_axes([x, 0.045, 0.075, 0.045]), label, color=color)
            b.on_clicked(cb)
            return b

        self.b_prev = mkbtn(0.815, '< Prev', lambda e: self.step(-1))
        self.b_next = mkbtn(0.900, 'Next >', lambda e: self.step(+1))
        self.b_imp  = mkbtn2(0.815, 'Impute (i)', lambda e: self.do_impute(), '#aed6f1')
        self.b_save = mkbtn2(0.900, 'Save (s)', lambda e: self.save(), '#a9dfbf')

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    # ── Eventos ─────────────────────────────────────────────

    def _on_radio(self, label):
        self.color_mode = label
        self.refresh()

    def _on_click(self, event):
        """Clic sobre el mapa hexagonal: alterna el sensor más cercano."""
        if event.inaxes is not self.ax_hex or event.xdata is None:
            return
        d = np.hypot(self.x_sipm - event.xdata, self.y_sipm - event.ydata)
        i = int(np.argmin(d))
        if d[i] > self.pitch * 0.6:            # clic fuera de cualquier sensor
            return
        self.marked.symmetric_difference_update({i})
        self.X_imp_cache = None                # la imputación anterior ya no vale
        self.refresh()

    def _on_key(self, event):
        k = event.key
        if k == 'right':
            self.step(+1)
        elif k == 'left':
            self.step(-1)
        elif k == 'i':
            self.do_impute()
        elif k == 'c':
            self.marked.clear(); self.X_imp_cache = None; self.refresh()
        elif k == 'u':
            self.marked = set(int(i) for i in np.where(self.z < -Z_SUGGEST)[0])
            self.X_imp_cache = None; self.refresh()
        elif k == 'r':
            self.marked.clear(); self.store_current(reviewed=True)
            self.X_imp_cache = None; self.refresh()
        elif k in ('+', '='):                  # mas resolucion
            self.bins = min(int(self.bins * 1.4), 600); self.refresh()
        elif k in ('-', '_'):                  # menos resolucion
            self.bins = max(int(self.bins / 1.4), 20); self.refresh()
        elif k == 'z':                         # zoom a los sensores marcados
            self.toggle_zoom()
        elif k == 'd':                         # panel derecho: imputado / diferencia
            self.right_mode = 'diff' if self.right_mode == 'imputed' else 'imputed'
            self.refresh()
        elif k == 's':
            self.save()
        elif k == 'q':
            self.save(); plt.close(self.fig)

    def toggle_zoom(self):
        """
        Alterna entre el detector completo y un encuadre alrededor de los sensores
        marcados. Con la vista completa el hueco de un solo sensor ocupa unos pocos
        pixeles y no se distingue; ampliando su entorno si.
        """
        if self.zoom is not None:
            self.zoom = None
        elif self.marked:
            xs = [self.x_sipm[i] for i in self.marked]
            ys = [self.y_sipm[i] for i in self.marked]
            m = self.pitch * 2.2                      # margen: ~2 anillos de vecinos
            x0, x1 = min(xs) - m, max(xs) + m
            y0, y1 = min(ys) - m, max(ys) + m
            # Cuadrar la ventana: con aspect igual, un recuadro alargado deja bandas
            # muertas a los lados y se aprovecha peor el panel.
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            r = max(x1 - x0, y1 - y0) / 2
            x0, x1, y0, y1 = cx - r, cx + r, cy - r, cy + r
            # Y recortarla al detector: fuera de el solo hay fondo vacio.
            x0 = max(x0, self.extent[0]); x1 = min(x1, self.extent[1])
            y0 = max(y0, self.extent[2]); y1 = min(y1, self.extent[3])
            self.zoom = [[x0, x1], [y0, y1]]
        else:
            print('  (marca algun sensor antes de hacer zoom)')
            return
        self.refresh()

    def step(self, d):
        """Cambia de módulo guardando el actual."""
        self.store_current()
        self.save(verbose=False)
        self.mi = (self.mi + d) % len(self.files)
        self.load_current()

    def do_impute(self):
        """Imputa los canales marcados sobre la ventana visible."""
        if not self.marked:
            self.ax_imp.set_title('Mark at least one sensor first', fontsize=10, color='#c0392b')
            self.fig.canvas.draw_idle()
            return
        if self.args.no_model:
            self.ax_imp.set_title('Inference disabled (--no-model)', fontsize=10, color='#c0392b')
            self.fig.canvas.draw_idle()
            return
        a, b = self.window()
        b = min(b, a + IMPUTE_CAP)             # tope de eventos para que no bloquee la GUI
        self.ax_imp.set_title('Running inference...', fontsize=10)
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()
        try:
            pack = get_model(self.args.run)
            dead = np.array(sorted(self.marked), dtype=np.int64)
            X_imp = impute_dead(pack, self.X[a:b], dead)
            self.X_imp_cache = (compute_xy(X_imp, self.x_sipm, self.y_sipm),
                                (a, b), tuple(int(i) for i in dead))
        except Exception as exc:
            print(f'  [inferencia] {type(exc).__name__}: {exc}')
            self.ax_imp.set_title(f'Inference failed: {exc}', fontsize=8, color='#c0392b')
            self.fig.canvas.draw_idle()
            return
        self.refresh()

    # ── Dibujo ──────────────────────────────────────────────

    def valid_imputation(self, a, b):
        """
        El flood imputado solo sigue siendo válido si se calculó sobre el MISMO tramo
        de eventos y con los MISMOS canales que hay marcados ahora. En cuanto se mueve
        un slider o se marca otro sensor, deja de ser comparable y se descarta.
        """
        if self.X_imp_cache is None:
            return None
        _, (ca, cb), dead = self.X_imp_cache
        if dead != tuple(sorted(self.marked)) or ca != a or cb > b:
            return None
        return self.X_imp_cache

    def refresh(self, recompute=False):
        a, b = self.window()
        if recompute or not hasattr(self, 'z'):
            self.z, self.frac, self.charge = module_zscore(self.X[a:b], self.nbr, self.base)
        # Con imputación activa se recorta el crudo al mismo tramo: dos flood maps con
        # distinta estadística no se pueden comparar a ojo.
        cache = self.valid_imputation(a, b)
        b_raw = cache[1][1] if cache else b
        self._draw_hex()
        self._draw_flood(a, b_raw)
        self._draw_imputed(cache)
        self._draw_profile(b - a)
        n_rev = sum(1 for v in self.labels.values() if v.get('reviewed'))
        # Se anuncia el tramo REALMENTE dibujado: con imputación activa puede ser menor
        # que el de los sliders, y decir otra cosa induciría a error.
        cut = '' if b_raw == b else f' (capped from {b})'
        self.fig.suptitle(
            f'{self.fname}   [{self.mi+1}/{len(self.files)}]   '
            f'events {a}–{b_raw}{cut} of {len(self.X)}   ·   marked: '
            f'{", ".join(f"Ich{IDX_TO_ICH[i]}" for i in sorted(self.marked)) or "none"}   ·   '
            f'reviewed {n_rev}/{len(self.files)}   ·   '
            f'{self.bins} bins{"  · ZOOM" if self.zoom is not None else ""}',
            fontsize=12, fontweight='bold')
        self.fig.canvas.draw_idle()

    def _draw_hex(self):
        ax = self.ax_hex
        ax.clear()
        if self.color_mode == 'z-score':
            vals, cmap, norm = self.z, plt.cm.RdYlGn, Normalize(-6, 2)
            cbl = 'z-score vs Good baseline'
        elif self.color_mode == 'active frac':
            vals, cmap = self.frac * 100, plt.cm.viridis
            norm, cbl = Normalize(0, max(vals.max(), 1e-9)), 'Active fraction (%)'
        else:
            vals, cmap = self.charge, plt.cm.viridis
            norm, cbl = Normalize(0, max(vals.max(), 1e-9)), 'Mean charge (ADC)'

        for i in range(N_ACTIVE):
            on = i in self.marked
            ax.add_patch(RegularPolygon(
                (self.x_sipm[i], self.y_sipm[i]), 6, radius=self.hex_r,
                orientation=np.pi / 6, facecolor=cmap(norm(vals[i])),
                edgecolor='#c0392b' if on else 'k', lw=3.0 if on else 0.4, zorder=2))
            ax.text(self.x_sipm[i], self.y_sipm[i], str(IDX_TO_ICH[i]),
                    ha='center', va='center', fontsize=6.5, zorder=3,
                    fontweight='bold' if on else 'normal',
                    color='#7b241c' if on else '0.25')
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f'Click a sensor to flag it  —  {cbl}', fontsize=10)

        if getattr(self, '_cb', None) is None:
            self._cb = self.fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                                         ax=ax, fraction=0.046, pad=0.02)
        else:
            self._cb.update_normal(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
        self._cb.set_label(cbl, fontsize=9)

    def _rango(self):
        """Ventana espacial a dibujar: detector completo, o zoom a los marcados."""
        if self.zoom is None:
            return [[self.extent[0], self.extent[1]], [self.extent[2], self.extent[3]]]
        return self.zoom

    def _flood(self, ax, px, py, title, color='#2471a3'):
        """
        Un flood map. El parametro 'color' da identidad visual al panel: los dos
        mapas son casi identicos y sin una marca clara es facil confundirlos, que
        es justo lo que pasaba antes.
        """
        ax.clear()
        rng = self._rango()
        ax.hist2d(px, py, bins=self.bins, range=rng,
                  cmap='plasma', norm=PowerNorm(0.55))
        ax.scatter(self.x_sipm, self.y_sipm, facecolors='none', edgecolors='white',
                   s=45, linewidths=0.4, alpha=0.35, zorder=3)
        for i in self.marked:                  # sensores marcados, sin tapar el mapa
            ax.add_patch(Circle((self.x_sipm[i], self.y_sipm[i]), self.pitch * 0.45,
                                facecolor='none', edgecolor=color, lw=2.2, zorder=4))
        ax.set_xlim(rng[0]); ax.set_ylim(rng[1])
        ax.set_aspect('equal')
        ax.set_xlabel('X [mm]', fontsize=9); ax.set_ylabel('Y [mm]', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        for sp in ax.spines.values():          # borde del panel del mismo color
            sp.set_edgecolor(color); sp.set_linewidth(1.8)

    def _flood_diff(self, ax, px_i, py_i, px_r, py_r, title, color='#8e44ad'):
        """
        Diferencia imputado menos crudo. Rojo donde la imputacion ANADE cuentas y
        azul donde las quita, de modo que se ve de un vistazo que zona del mapa
        cambia. Comparar dos flood maps casi iguales a ojo no funciona.
        """
        ax.clear()
        rng = self._rango()
        Hi, xe, ye = np.histogram2d(px_i, py_i, bins=self.bins, range=rng)
        Hr, _, _   = np.histogram2d(px_r, py_r, bins=self.bins, range=rng)
        D = Hi - Hr
        v = np.percentile(np.abs(D), 99.5) or 1.0
        ax.imshow(D.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]],
                  cmap='coolwarm', vmin=-v, vmax=v, aspect='equal')
        ax.scatter(self.x_sipm, self.y_sipm, facecolors='none', edgecolors='0.35',
                   s=45, linewidths=0.4, alpha=0.5, zorder=3)
        for i in self.marked:
            ax.add_patch(Circle((self.x_sipm[i], self.y_sipm[i]), self.pitch * 0.45,
                                facecolor='none', edgecolor=color, lw=2.2, zorder=4))
        ax.set_xlim(rng[0]); ax.set_ylim(rng[1])
        ax.set_xlabel('X [mm]', fontsize=9); ax.set_ylabel('Y [mm]', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(1.8)

    RAW_C  = '#2471a3'      # azul: mapa medido
    IMP_C  = '#27ae60'      # verde: mapa reconstruido
    DIFF_C = '#8e44ad'      # morado: diferencia

    def _draw_flood(self, a, b):
        px, py = self.pos
        self._flood(self.ax_raw, px[a:b], py[a:b],
                    f'RAW — measured ({b-a} events)', self.RAW_C)

    def _draw_imputed(self, cache):
        ax = self.ax_imp
        if cache is None:
            ax.clear(); ax.set_aspect('equal'); ax.axis('off')
            msg = ('Press "i" to impute the flagged sensors'
                   if self.marked else 'Flag sensors, then press "i"')
            ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=11,
                    color='0.45', transform=ax.transAxes)
            c = self.DIFF_C if self.right_mode == 'diff' else self.IMP_C
            ax.set_title('DIFF imputed − raw' if self.right_mode == 'diff' else 'IMPUTED',
                         fontsize=10, color=c, fontweight='bold')
            return
        (px, py), (a, b), dead = cache
        # Con muchos canales la lista no cabe en el título: se resume.
        ich = (', '.join(f'Ich{IDX_TO_ICH[i]}' for i in dead) if len(dead) <= 4
               else f'{len(dead)} channels')
        if self.right_mode == 'diff':
            pr, pyr = self.pos
            self._flood_diff(ax, px, py, pr[a:b], pyr[a:b],
                             f'DIFF imputed − raw [{ich}]   red=added  blue=removed',
                             self.DIFF_C)
        else:
            self._flood(ax, px, py, f'IMPUTED [{ich}]  ({b-a} events)', self.IMP_C)

    def _draw_profile(self, n_events):
        ax = self.ax_prof
        ax.clear()
        colors = ['#c0392b' if i in self.marked else
                  ('#e67e22' if self.z[i] < -Z_SUGGEST else '#7f8c8d') for i in range(N_ACTIVE)]
        ax.bar(np.arange(N_ACTIVE), self.z, color=colors, width=0.75)
        ax.axhline(-Z_SUGGEST, color='#c0392b', ls='--', lw=1.0,
                   label=f'suggestion threshold (z = -{Z_SUGGEST})')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(np.arange(N_ACTIVE))
        ax.set_xticklabels([str(IDX_TO_ICH[i]) for i in range(N_ACTIVE)], fontsize=6, rotation=90)
        ax.set_xlim(-0.7, N_ACTIVE - 0.3)
        ax.set_ylim(min(-8, float(self.z.min()) - 1), max(3, float(self.z.max()) + 1))
        ax.set_xlabel('Channel (Ich)', fontsize=9)
        ax.set_ylabel('z-score', fontsize=9)
        ax.tick_params(axis='y', labelsize=8)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(axis='y', alpha=0.25)
        # Con pocos eventos el z-score se vuelve ruidoso: conviene avisar antes de que
        # una fluctuación estadística se etiquete como sensor averiado.
        warn = '   ⚠ ventana pequeña: z-score ruidoso' if n_events < 20_000 else ''
        ax.set_title('Per-channel deviation from the Good baseline  '
                     '(orange = suggested by the statistic, red = flagged by hand)'
                     + warn, fontsize=9,
                     color='#c0392b' if warn else 'black')


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='Etiquetado manual de SiPM averiados en los BAD')
    ap.add_argument('--tag', default='manual', help='identificador del JSON de salida')
    ap.add_argument('--labels', default=None, help='JSON previo que se quiere continuar')
    ap.add_argument('--max-events', type=int, default=200_000, help='eventos leídos por módulo')
    ap.add_argument('--bins', type=int, default=140, help='bins del flood map')
    ap.add_argument('--run', default=DEFAULT_RUN, help='run del modelo de imputación')
    ap.add_argument('--no-model', action='store_true', help='desactiva la inferencia')
    ap.add_argument('--cache-dir', default=None, help='carpeta para cachear los .npy')
    ap.add_argument('--start', default=None, help='módulo por el que empezar (p.ej. datas017.dat)')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(Path(BAD_DIR) / '*.dat')))
    if not files:
        sys.exit(f'No se han encontrado .dat en {BAD_DIR}')

    if not BASELINE_NPZ.exists():
        sys.exit(f'Falta el baseline {BASELINE_NPZ}.\n'
                 f'Genéralo con:  python bad_detect.py --build-only')
    base = dict(np.load(BASELINE_NPZ))

    x_sipm, y_sipm = load_positions(PSIPM_PATH)
    nbr = get_neighbor_matrix(PSIPM_PATH)

    # Etiquetas previas: permite retomar el trabajo donde se dejó
    out_json = OUT_DIR / f'bad_labels_{args.tag}.json'
    src = Path(args.labels) if args.labels else out_json
    labels = {}
    if src.exists():
        labels = json.loads(src.read_text(encoding='utf-8')).get('labels', {})
        print(f'Etiquetas previas cargadas de {src.name}: {len(labels)} módulos')

    print(f'{len(files)} módulos BAD  ·  salida: {out_json}')
    print('Controles: clic = marcar sensor · ←/→ módulo · i imputar · u sugerir · '
          'c limpiar · r sin fallos · s guardar · q salir\n')

    lab = Labeler(files, args, base, nbr, x_sipm, y_sipm, labels, out_json)
    if args.start:
        names = [Path(f).name for f in files]
        if args.start in names:
            lab.mi = names.index(args.start)
            lab.load_current()
        else:
            print(f'  aviso: {args.start} no está en la lista, se empieza por el primero')

    plt.show()
    lab.save()          # guardado final al cerrar la ventana
    print('Sesión terminada.')


if __name__ == '__main__':
    main()
