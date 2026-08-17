---
name: tfm-preflight
description: Auditoría de experimentos del TFM (imputación de SiPM averiados en detector PET hexagonal) antes y después de entrenar. Úsala SIEMPRE que haya que crear o modificar un script de entrenamiento, el dataloader, un calendario o plan de entrenamientos, o una tanda .ps1, y también al interpretar los resultados de un run, al comparar modelos o al escribir cifras en la memoria. Verifica el contrato de datos, revisa el registro de errores conocidos para no repetir bugs ya corregidos, comprueba qué versión del código se ejecuta de verdad, y entrega un .ps1 para lanzar en consola en lugar de entrenar dentro del agente. Actívala también ante entrenar, lanzar un run, nueva tanda, nuevo experimento, tocar dataset.py o train.py, revisar métricas, barra de error, réplicas, semillas, cobertura, preflight, checkpoint, peor canal, recuperación de posición, flood map o SiPM.
---

# Preflight TFM

## Por qué existe esta skill

Un entrenamiento mal planteado no falla: termina. Devuelve curvas, métricas y checkpoints, y solo días después se descubre que el dataloader usaba 40 de 149 módulos, o que el script lanzado era una copia anterior a la corrección. El coste no es el error; es el tiempo de GPU y las conclusiones construidas encima.

El fallo más caro documentado en este proyecto fue exactamente ese: se detectó que solo se cargaban 40 de 149 módulos de entrenamiento, se corrigió, y **el lanzamiento siguiente volvió a arrastrar el error**. Una regresión. Por eso esta skill no se limita a "revisar si el código tiene sentido": mantiene memoria explícita de los errores ya vistos y los vuelve a comprobar uno por uno antes de cada lanzamiento.

Hay un segundo modo de fallo, igual de caro y menos evidente, que esta skill también cubre: **concluir de más a partir de lo que el código parece hacer.** Una barra de error medida con una variable congelada (E08) o una afirmación sobre el generador aleatorio deducida leyendo en vez de ejecutando (E09) no rompen ningún run — envenenan la memoria, que es peor, porque se descubre al final. De ahí la regla de la Fase 2: **evidencia o no ha ocurrido.**

Miguel decide siempre. Esta skill informa, no bloquea. Pero informa sin suavizar: si un run está condenado, dilo en la primera línea.

## Reglas duras

**No entrenar dentro del agente.** Los entrenamientos lanzados desde la sesión han provocado errores. Lo que produces es un `.ps1` en `tandas/` y la línea de consola para ejecutarlo. Miguel lo lanza en su terminal, siempre.

Lo que sí puedes ejecutar:
- lectura e inspección de ficheros, Excel, logs, `git log`, `git status`
- análisis estático (grep de patrones sospechosos, recuento de ficheros del dataset)
- **smoke tests**: 1–2 batches, un forward/backward, verificación de formas y recuentos, medición de tiempo por paso. Segundos, no minutos.

Si dudas de si algo es smoke test o entrenamiento: si tarda más de ~2 minutos o escribe checkpoints, es entrenamiento. Va al `.ps1`.

**Entorno Windows + OneDrive.** El repo vive en una ruta con espacios y acentos (`...\11_TFM\Código`). Entrecomilla siempre las rutas en PowerShell. OneDrive puede bloquear ficheros grandes durante la sincronización: si los checkpoints o los datos generados se escriben dentro de la carpeta sincronizada, señálalo como riesgo (corrupción o `PermissionError` a mitad de run) y sugiere una ruta local fuera de OneDrive.

**Idioma.** Informes, comentarios de código, mensajes de commit y conversación en español. Nombres de variables y funciones en inglés si el repo ya lo hace así: respeta la convención existente, no la cambies a mitad del TFM.

## Convención de identificadores

**Usa la que ya existe: el `--tag`.** No inventes IDs nuevos — el nombre de carpeta del run es un argumento de los evaluadores, así que cambiarlo rompe el pipeline.

```
runs/imputer_<arch>_<size>_<loss>[_<sufijos>]_<tag>/
```

El tag es el hilo de trazabilidad y aparece igual en la carpeta del run, la fila del Excel, la bitácora y W&B. Elige tags que digan de qué familia son (`full1..3`, `banda11..17`, `cvB..cvE`), no fechas — lo que importa después es agrupar réplicas del mismo experimento.

Detalle de rutas y estructura en `references/rutas-y-fuentes.md`. Resumen: los lanzadores van a `tandas/`, los informes de preflight al vault de Obsidian, y **no** existen `informes/`, `lanzadores/` ni `configs/` en el repo.

## Flujo

Las fases 0–5 son el preflight de un lanzamiento nuevo. La 6 es post-run. La 7 es al escribir la memoria. No hace falta recorrerlas todas si la petición es acotada, pero **la Fase 0 nunca se salta**: proponer un config sin haber leído el historial es exactamente el fallo que esta skill previene.

### Fase 0 — Orientación (obligatoria)

Antes de escribir una línea de código:

1. Lee `references/errores-conocidos.md`. Es el registro de regresiones.
2. Lee el histórico de evaluaciones (Excel) y el diario en Obsidian. Rutas y método en `references/rutas-y-fuentes.md`.
3. `git log --oneline -15` y `git status`. Qué se cambió desde el último run y si hay trabajo sin commitear.
4. Localiza el config y el script del run anterior para poder decir **qué cambia exactamente** respecto a él.

Al terminar, enuncia en dos frases: cuál fue el último run, qué resultado dio, y qué hipótesis concreta pone a prueba el nuevo. Si no puedes formular la hipótesis, no hay experimento: hay un entrenamiento. Dilo.

### Fase 1 — Contrato de datos

Aquí es donde se pierden las semanas. El principio: **no te fíes de lo que dice el config; comprueba lo que el código carga de verdad.**

Establece explícitamente, en números:

- Módulos esperados vs. módulos que el bucle recorre de hecho (cuéntalos ejecutando `get_file_split`, no leyendo la constante). **Ojo con el round-robin**: si `N_EPOCHS < nº de ficheros` no da la vuelta y solo se ven los primeros — es E01.
- Eventos por época y presupuesto total. Los datos se generan al vuelo: comprobar además **con qué semilla** y qué fuentes de azar quedan fijas (importa para E08).
- Reparto train/val/test: **por fichero**, `seed=42`, y el test intacto. Aquí no hay "posición de fuente" por la que particionar: son datos reales y cada fichero es un detector físico.
- Normalización: por evento, y **qué estadísticas globales se calculan y con qué ficheros** (E04).
- **Coherencia del preprocesado entre entrenamiento y evaluación** (E03): es el fallo más caro del proyecto y no da ningún síntoma. El bloque `preproc` del checkpoint debe existir y coincidir.
- Canales: 61 activos de 64. Una dimensión distinta es señal de alarma.

Patrones a buscar con grep antes de dar por bueno un pipeline: `[:40]`, `[:n]`, `max_files`, `limit`, `n_files`, `head(`, `break` dentro del bucle de carga, `sample(`, `subset`, `DEBUG`, `TODO`. Cualquier recorte que fuese temporal y se quedó.

Y la contramedida estructural, que es lo que evita la repetición: **el script de entrenamiento debe afirmar sus propias cifras al arrancar** y abortar si no cuadran.

```python
# Contrato de datos: si esto falla, el run no arranca.
N_MODULOS_TRAIN = 149
n_vistos = min(N_EPOCHS * MIX_MODULES, len(train_files))
assert len(train_files) == N_MODULOS_TRAIN, \
    f"Se esperaban {N_MODULOS_TRAIN} módulos de train, hay {len(train_files)}"
assert n_vistos == len(train_files), \
    f"COBERTURA PARCIAL: se verían {n_vistos} de {len(train_files)} módulos (E01)"
```

**Estado actual: `train.py` imprime la línea de `COBERTURA` pero no aborta.** Convertir ese print en assert es la propuesta pendiente número uno; habría hecho imposible E01 y su regresión. Mientras no exista, la comprobación se hace **a posteriori sobre el log del run**, que es donde se verifica de verdad qué pasó:

```
COBERTURA: 149/149 módulos distintos (100%)  |  107,000 eventos/época  → 15,943,000 muestras totales
```

### Fase 2 — Chequeo de regresiones

Recorre `references/errores-conocidos.md` entrada por entrada y para cada una da un estado con evidencia: `OK`, `RIESGO` o `NO APLICA`, citando `fichero:línea` o la salida del comando que lo comprueba. No basta con "revisado": sin evidencia, la comprobación no ha ocurrido.

Añadido a esto, la comprobación que habría evitado la regresión del proyecto: **¿qué versión del código se va a ejecutar?**

- `git rev-parse --short HEAD` y árbol limpio o no.
- El `.ps1` apunta al script del repo, no a una copia suelta en otra carpeta ni a un fichero con sufijo `_v2`, `_old`, `_backup`.
- El fix correspondiente al último error corregido está presente en el fichero que se va a ejecutar. Compruébalo leyendo la línea, no confiando en el commit.

### Fase 3 — Aritmética y coste

Cuentas que se hacen en treinta segundos y ahorran días:

- **Presupuesto total** = épocas × eventos por época. Compáralo con las ~15,9 M de referencia: si un experimento cambia la cobertura *y* el coste a la vez, no se podrá interpretar. El histórico está diseñado a coste constante a propósito.
- **El cosine se agota exactamente al final.** Hoy es `CosineAnnealingLR(T_max=N_EPOCHS)`, ligado a la variable, así que `--epochs` lo ajusta solo. Si alguien fija un `T_max` literal, es un error silencioso clásico: comprobarlo.
- **Tiempo estimado** con las referencias medidas: **~155 min por entrenamiento** completo y **~77 min por `eval_total` a 750k**. Una tanda de 3 trains + 3 evals son ~11,5 h. Si sale muy distinto, algo cambió.
- **Que las carpetas de destino no existan ya** — una tanda que repite un tag pisa resultados anteriores.
- Espacio en disco: los checkpoints son pequeños (~150 KB), no es un problema hoy. Si un experimento fuera a escribir GB, sacar la salida de OneDrive.

### Fase 4 — Smoke test

Ejecuta lo mínimo que demuestra que el pipeline vive:

1. **Que los flags parsean de verdad** (E07). El parseo de `train.py` vive dentro de `if __name__ == '__main__'`, así que importar el módulo **no lo ejecuta** y comprobarlo así da un falso negativo. El patrón que sí funciona: ejecutar ese bloque con la línea real de la tanda y verificar las variables resultantes.

   ```python
   import train, textwrap
   src = open('train.py', encoding='utf-8').read().split('\n')
   bloque = '\n'.join(src[591:744])          # el bloque __main__ SIN la llamada a main()
   sys.argv = ['train.py', 'hexcnn', 's', 'mse', '--seed', '811', '--rotseed', '11', '--tag', 'x']
   g = dict(vars(train)); g['__name__'] = '__main__'
   exec(textwrap.dedent(bloque), g)
   assert g['TORCH_SEED'] == 811 and g['ROT_SEED'] == 11 and g['SPLIT_SEED'] == 42
   ```

2. Instanciar el dataset y **contar los módulos reales** que enumera el split.
3. Sacar un batch: formas, dtype, rango, NaN/Inf, y que el target corresponda a la entrada. Recuerda la tupla: `(x_in (2,61), target (61,), dead (61,), is_modified)` — `[1]` es el target, **no** la máscara.
4. Un forward + backward: pérdida finita, gradientes no nulos ni NaN.
5. **Reproducibilidad**: misma semilla dos veces → idéntico. Semillas distintas → distinto. Este es el test que faltó en E09; hazlo siempre que una conclusión dependa de qué mueve una semilla.
6. Que las combinaciones de una tanda sean **distintas entre sí** (dos réplicas con el mismo orden no son dos réplicas).

Overfitting a un solo batch es la comprobación más informativa que existe si sospechas del pipeline: si el modelo no consigue memorizar 8 muestras, el problema no es el hiperparámetro, es el dato o el objetivo. Cabe en un smoke test corto y merece la pena proponerlo cuando hay dudas.

### Fase 5 — Veredicto, informe y lanzador

Tres salidas, en este orden:

**a) Veredicto en consola**, 5–8 líneas como máximo:

```
tanda noche_banda (tags banda11/13/17) | VEREDICTO: LANZAR CON RESERVAS
Hipótesis: la banda del peor canal medida solo con --seed subestima la variabilidad real.
Contrato: 149/149 módulos, 15.9 M muestras, partición 149/5/5 seed 42 (OK).
Regresiones: E01 OK (COBERTURA verificada en el log del run previo) · E07 OK (flags probados
  end-to-end, carpetas libres) · E02 RIESGO: árbol sucio, 3 ficheros sin commitear.
Coste: ~11,5 h (3 × 155 min de train + 3 × 77 min de eval).
Reserva: con n=3 la sd seguirá siendo un estimador pobre; no cerrará el caso, solo lo acotará.
Lanzar con:  .\tandas\noche_banda.ps1
```

Veredictos posibles: `LANZAR`, `LANZAR CON RESERVAS`, `NO LANZAR`. Usa `NO LANZAR` cuando corresponda; un veredicto que siempre es verde no aporta información.

**b) Informe** en el vault de Obsidian, siguiendo `assets/plantilla-informe.md`. Para una tanda rutinaria basta con el veredicto en consola más la entrada en la bitácora; el informe completo se justifica cuando la tanda es cara o el diseño es discutible.

**c) Lanzador** en `tandas/`, a partir de `assets/plantilla-lanzador.ps1`, y **añadir su fila al `tandas/README.md`**, que es el índice de qué resolvió cada tanda. El lanzador registra el commit y el estado del árbol al arrancar.

Si lo que pide Miguel es un **calendario o plan de varios entrenamientos**, aplica lo mismo por cada run del plan, y añade el orden: primero los que pueden invalidar a los siguientes. No tiene sentido barrer hiperparámetros si un experimento previo puede revelar que el objetivo está mal definido.

### Fase 6 — Post-run

Al analizar resultados, el sesgo por defecto es la desconfianza; un resultado limpio también puede ser un bug.

- **Lo primero: la línea de COBERTURA del log.** Es la que habría cazado E01 el primer día.
- **El rojo de la consola no es señal de nada.** PowerShell envuelve stderr en `ErrorRecord` y wandb escribe todo por stderr. Busca `Traceback (most recent call last)`.
- Compara con el histórico del Excel, y **comprueba que las campañas tienen el mismo `max_events_per_file`** (E06). Dos modelos evaluados con distinto tamaño de muestra no son comparables: la diferencia por ese solo motivo es del orden de 0,26 puntos.
- Métrica sospechosamente buena → fuga entre particiones. Compruébalo antes de celebrarlo.
- Métrica idéntica a un run anterior → checkpoint o tag equivocado.
- Pérdida plana desde el principio → learning rate, normalización o etiquetas.
- **Un punto que cae a muchas sd de una banda:** la hipótesis por defecto es **banda mal estimada**, no dato anómalo (E08). Antes de buscarle una explicación física, enumera qué fuentes de azar varían en esa banda y cuáles están congeladas.
- **Antes de anunciar un hallazgo, mira la n.** Dos conclusiones de este proyecto ("nuevo mejor modelo", "anti-acumulación") se anunciaron con n=1 y las desmintieron las réplicas. Criterio adoptado: **por debajo de 2 sd no es concluyente**.

Cierra actualizando la fila del Excel y la bitácora con el mismo tag, y si apareció un error nuevo, ve a la sección siguiente.

### Fase 7 — Coherencia con la memoria

Al escribir la memoria: toda cifra del texto debe ser trazable a un **tag** concreto y a la fila del Excel. Cuando encuentres un número en el LaTeX que no puedas atar a un run, márcalo. Las cifras heredadas de borradores anteriores son la fuente habitual de incoherencias entre tabla, texto y figura — ya pasó con las cifras de los módulos BAD (49/281/16 en el `.tex` frente a 48/278/12 en el Excel, que era el correcto).

Dos comprobaciones específicas de este TFM:

- **Toda cifra necesita su n y su banda.** Si viene de un solo run, o se dice explícitamente, o no se escribe. La partición domina la incertidumbre (sd 2,35 entre particiones frente a 0,1–0,2 entre réplicas), así que la cifra honesta de recuperación es la de la validación cruzada, no la de una partición favorable.
- **El peor canal es la métrica más frágil del proyecto** (E08): ±5 puntos de banda real y el canal del mínimo cambia entre entrenamientos. No sostiene ninguna afirmación de robustez sin su barra de error.

## Cuando aparece un error nuevo

Este es el mecanismo que hace que la skill mejore con el uso. Cada vez que se descubre un bug real, añade una entrada a `references/errores-conocidos.md` con el formato de ese fichero: qué pasó, el síntoma observable, y **una receta concreta de detección** (comando, grep o assert). Una entrada sin receta de detección no sirve: la próxima vez tampoco se detectará.

Propón también el assert que lo haga imposible en el futuro. La entrada del registro es la red de seguridad; el assert en el código es el arreglo.

## Ficheros de referencia

- `references/errores-conocidos.md` — registro de regresiones (E01–E09). Leer en Fase 0, comprobar en Fase 2, ampliar cuando aparezca un error nuevo.
- `references/rutas-y-fuentes.md` — rutas, convención de tags, estructura real del repo, cómo leer el Excel y los logs de tanda.
- `references/contexto-detector.md` — qué problema es este TFM, el detector, el conjunto de datos, el protocolo y los órdenes de magnitud normales. Consultar cuando una cifra o unidad tenga que cuadrar; es contexto, no criterio de éxito.
- `assets/plantilla-informe.md` — estructura del informe de preflight.
- `assets/plantilla-lanzador.ps1` — plantilla de tanda `.ps1`.
