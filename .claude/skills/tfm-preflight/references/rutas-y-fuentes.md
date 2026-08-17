# Rutas, fuentes y convenciones

## Raíces

| Qué | Ruta |
|---|---|
| Repo del TFM | `C:\Users\Miguel\OneDrive\MASTER\11_TFM\Código` |
| Datos | `E:\Datos TFM\` → `Good\Good`, `Bad\Bad`, `psipm.tsv` |
| Python | `E:\envs\tfm\python.exe` (conda `tfm`, Python 3.11) |
| Vault de Obsidian | `C:\Users\Miguel\Documents\Obsidian Vault` |
| Excel de métricas | `Código\Metricas_runs_TFM.xlsx` |

En PowerShell, entrecomilla siempre: la ruta tiene espacios y un acento (`Código`).

**Los datos viven en el disco externo `E:`, fuera de OneDrive.** No los muevas al repo.

## Convención de nombres — usa la que ya existe

**No inventes IDs `RUN-AAAAMMDD-nn`.** El proyecto identifica los runs por el nombre de carpeta, y **ese nombre es un argumento de los evaluadores**, así que cambiarlo rompe el pipeline:

```
runs/imputer_<arch>_<size>_<loss>[_<sufijos>]_<tag>/
```

Ejemplos reales: `imputer_hexcnn_s_mse_full1`, `imputer_hexcnn_s_mse_banda11`, `imputer_hexcnn_s_mse_cvB`.

El `<tag>` lo pone `--tag` y es la parte que distingue una réplica o un experimento. **Es el hilo de trazabilidad**: aparece igual en la carpeta del run, en la fila del Excel, en la bitácora y en W&B. Elige tags que digan de qué familia son (`full1..3`, `banda11..17`, `cvB..cvE`), no fechas.

Los evaluadores se invocan con ese nombre:

```powershell
& $py eval_total.py imputer_hexcnn_s_mse_banda11 --events 750000
```

## Estructura real del proyecto

```
Código/
├── train.py, dataset.py, eval_*.py, bad_*.py     # el código
├── tandas/                # lanzadores .ps1 encadenados + README index
│   └── logs/              # logs de tanda (gitignored)
├── runs/                  # salidas por run (gitignored)
│   └── imputer_.../
│       ├── best_model.pth
│       ├── history.json
│       ├── PREPROC.json   # contrato del run: preprocesado, split, protocolo
│       └── TOTAL/eval_total_metrics.json
├── memoria/               # LaTeX
└── Metricas_runs_TFM.xlsx # gitignored, binario
```

**No crees `informes/`, `lanzadores/` ni `configs/`.** Los lanzadores van a `tandas/`, y no hay YAML: la configuración son constantes en la cabecera de `train.py` más flags de línea de comandos.

**Los informes de preflight van al vault de Obsidian**, no al repo (el repo no versiona markdown de trabajo; el vault es donde está la bitácora).

## Qué leer en la Fase 0

1. **`references/errores-conocidos.md`** — siempre, entero.
2. **`Bítacora.md`** (vault) — el registro cronológico. Es largo (~1.900 líneas); lee el final de la línea temporal y la sección de pendientes.
3. **`Checkpoint_Activo.md`** (vault) — el tablero vivo. **Empieza por aquí si tienes prisa**: dice qué está en curso, qué está cerrado y qué no reintentar.
4. **`Auditoria.md`** (vault) — la auditoría del 05/08, origen de E03–E06.
5. `git log --oneline -15` y `git status`.
6. El `PREPROC.json` del run anterior de la misma familia, para decir **qué cambia exactamente**.

## Cómo leer el Excel

Está en el `.gitignore` (binario reproducible), así que no aparece en `git status`. **Haz copia antes de escribir**: `Metricas_runs_TFM_backup_<dd>aug.xlsx`.

```python
import openpyxl
wb = openpyxl.load_workbook(r'...\Metricas_runs_TFM.xlsx')
for ws in wb.worksheets:
    print(ws.title, ws.max_row, ws.max_column)
```

Hojas: `resumen` (secciones numeradas, la última es el estado actual), `replicas` (bandas de error), `arquitecturas`, `runs` (una fila por run), `multidead`.

Si la última sección del `resumen` no corresponde al último run de `runs/`, hay desfase en el registro: señálalo. El Excel alimenta las cifras de la memoria, así que un Excel incompleto se propaga al TFM.

## Leer los logs de tanda

**Vienen con codificación mezclada**: las líneas de `Log()` en UTF-8 y la salida capturada por `Tee-Object` en UTF-16. Leerlos en un solo `decode` da basura. Patrón que funciona:

```python
txt = open(log, 'rb').read().replace(b'\x00', b'').decode('utf-8', errors='replace')
```

**La salida en rojo de la consola no significa error.** PowerShell 5.1 envuelve cada línea de stderr en un `ErrorRecord` cuando se usa `2>&1` sobre un ejecutable nativo, y wandb escribe **todo** por stderr. Para saber si hubo un fallo real, busca `Traceback (most recent call last)`, no el color. `$LASTEXITCODE` sigue siendo fiable.

## Reglas de ejecución

- **Nunca lances un entrenamiento desde el agente.** Miguel los lanza siempre en su consola. Lo que produces es el `.ps1` en `tandas/` y la línea para ejecutarlo.
- Smoke tests sí: segundos, sin escribir checkpoints.
- Comentarios y docstrings **en español**; etiquetas y títulos de figuras **en inglés**.
- Todo script de salida debe admitir un **tag/identificador en el nombre del fichero**.

## OneDrive

El repo está sincronizado. `runs/` y `*.pth` están en `.gitignore`, pero **siguen dentro de la carpeta de OneDrive**, así que los checkpoints se sincronizan. Hasta ahora no ha dado problemas (los checkpoints son de ~150 KB, no GB). Si algún experimento fuera a escribir muchos GB, propón sacar la salida a disco local.
