# Registro de errores conocidos

Registro de bugs ya detectados en el proyecto. Se lee en la Fase 0 y se comprueba entrada por entrada en la Fase 2, **con evidencia** (`fichero:línea` o salida de comando). Un "revisado" sin evidencia no cuenta.

Este fichero es acumulativo: no se borran entradas aunque estén corregidas. Una entrada corregida sigue siendo la lista de comprobación que evita la regresión.

**Las entradas E03–E09 salen de la auditoría completa del 05/08/2026** (`Obsidian Vault/Auditoria.md`) y de los hallazgos posteriores. Veredicto de aquella auditoría: ningún resultado publicado estaba mal calculado, pero los defectos de abajo eran reales y algunos habrían envenenado resultados futuros en silencio.

## Formato de entrada

```
### E<nn> — <título corto>
- **Qué pasó:** descripción en una o dos frases.
- **Síntoma observable:** qué se veía en el log/métrica (o por qué no se veía nada).
- **Detección:** comando, grep o assert concreto que lo caza.
- **Arreglo:** qué se cambió y dónde.
- **Estado:** corregido (fecha) / abierto / recurrente.
```

---

### E01 — Solo se cargaban 40 de 149 módulos de entrenamiento

- **Qué pasó:** la rotación de ficheros es un round-robin y con `N_EPOCHS < nº de ficheros` **no da la vuelta**: solo se veían los `N_EPOCHS` primeros de la lista, y como iba ordenada por nombre, siempre los mismos 40 módulos de numeración baja (y anormalmente homogéneos entre sí).
- **Síntoma observable:** ninguno directo. El run terminaba sin errores. El coste era el mismo, así que ni siquiera los tiempos delataban nada.
- **Detección:**
  - En el log del run, la línea `COBERTURA: n/149 módulos distintos`. Debe decir **149/149**.
  - `grep -n "N_EPOCHS\s*=" train.py` y comparar con el número de ficheros de train.
  - `grep -rn "\[:40\]\|\[:[0-9]\+\]\|max_files\|n_files\|limit\|subset\|sample(\|DEBUG" *.py`
- **Arreglo:** `N_EPOCHS = 149` por defecto (commit `bea2002`) y línea de COBERTURA impresa al arrancar. **Falta aún el assert que aborte**: hoy solo se imprime.
- **Estado:** corregido 15/08, pero **recurrente** — reapareció en un lanzamiento posterior (ver E02). Efecto medido: la media no cambia, el peor canal sube 2,2 puntos.

### E02 — Regresión: el fix existía pero el run relanzado no lo incluía

- **Qué pasó:** después de corregir E01 se preparó un nuevo entrenamiento que volvió a arrastrar el error. La corrección estaba hecha, pero el código efectivamente ejecutado no era el corregido.
- **Síntoma observable:** el mismo que E01, y la sensación de que "esto ya lo habíamos arreglado".
- **Detección:**
  - `git rev-parse --short HEAD` y `git status` antes de lanzar. Árbol sucio = riesgo declarado en el veredicto.
  - Comprobar que el `.ps1` invoca el script del repo y no una copia (`_v2`, `_old`, `_backup`).
  - Leer la línea del fix en el fichero que se va a ejecutar. No basta con que el commit exista.
  - **Después de lanzar**, confirmar en el log que el run declara lo que se esperaba (cobertura, semillas, presupuesto).
- **Arreglo:** el lanzador registra el commit y el estado del árbol junto a los logs.
- **Estado:** corregido a nivel de proceso. Comprobar en **todos** los preflights.

### E03 — Preprocesado de evaluación distinto al de entrenamiento (`norm_mode`, `channel_norm`)

- **Qué pasó:** un modelo entrenado con una normalización y evaluado con otra. Ha ocurrido **dos veces**: primero con `norm_mode` (el caso `pna_nsum`), después con `impute_set`, que no sabía deshacer la equalización por canal de `--chnorm` mientras `impute_channel` sí.
- **Síntoma observable:** **ninguno.** No hay excepción ni aviso: el modelo recibe entradas que no reconoce y rinde mal, y parece un problema de arquitectura. Es el modo de fallo más caro del proyecto.
- **Detección:**
  - El checkpoint lleva un bloque `preproc` con `norm_mode`, `channel_norm`, `clip_negativos`, `norm_por_evento` y `orden`. `load_model` debe verificarlo. Comprobar que el checkpoint que se evalúa lo tiene: los anteriores a agosto **no**.
  - Prueba cruzada: imputar un canal por las dos rutas (`impute_channel` e `impute_set`) sobre los mismos eventos y exigir diferencia 0,0.
- **Arreglo:** `preproc` viaja dentro del checkpoint; `impute_set` corregido. Verificado: ambas rutas dan resultados idénticos.
- **Estado:** corregido 05/08. **Recurrente por diseño** — cada vez que se añada una variante de preprocesado, vuelve a ser posible.

### E04 — Fuga de validación en la escala por canal (`get_channel_scale`)

- **Qué pasó:** la función que estima cuánta carga recoge típicamente cada sensor tomaba los 8 primeros ficheros del directorio, y uno (`datas013`) es de **validación**.
- **Síntoma observable:** ninguno. Lo que se filtra es una media agregada, no eventos.
- **Detección:** comprobar que toda estadística global (escalas, baselines, normalizaciones) se calcula **solo con ficheros de train**, obtenidos de `dataset.get_file_split`. `grep -n "get_channel_scale\|glob\|sorted(" dataset.py` y mirar de dónde sale la lista.
- **Arreglo:** solo ficheros de train. **Ojo: el fichero cacheado en disco conserva el valor antiguo**; hay que borrarlo para regenerarlo limpio.
- **Estado:** corregido 05/08. Impacto real despreciable (solo la usan los modelos `--chnorm`, que rindieron peor) y **el test nunca estuvo implicado**. Recordatorio conceptual: una fuga **infla** resultados, nunca los deprime — no puede ser la causa de que algo no mejore.

### E05 — Numerador y denominador con distinto número de eventos

- **Qué pasó:** el error relativo de carga se calculaba con el MAE de campañas de 600–700k eventos y la carga media de 150k.
- **Síntoma observable:** cifra plausible pero sesgada; el error relativo salía ~0,8 puntos bajo.
- **Detección:** en cualquier métrica que sea un cociente, comprobar que numerador y denominador vienen del **mismo** número de eventos. La carga media **no es estable**: baja de forma monótona con el tamaño de muestra (2.128 → 2.070 ADC de 150k a 600k), lo que además revela una deriva real del detector a lo largo de cada adquisición.
- **Arreglo:** mismo tamaño de muestra en ambos. La cifra de memoria pasa a "~30% de error relativo". El **ranking entre modelos no cambia**.
- **Estado:** corregido 05/08.

### E06 — Comparar modelos evaluados con campañas de distinto tamaño

- **Qué pasó:** la tabla comparativa tomaba para cada modelo la primera evaluación disponible de una lista de preferencia, mezclando campañas de 600k y 700k **sin avisar**.
- **Síntoma observable:** diferencias entre modelos del orden de 0,26 puntos que son puro tamaño de muestra, indistinguibles de una mejora real.
- **Detección:** el JSON de cada eval lleva `max_events_per_file`. Antes de comparar dos modelos, **comprobar que coincide**. Si no, no son comparables.
- **Arreglo:** parámetro para forzar campaña única, registro del tamaño de muestra por modelo y aviso explícito cuando la comparación no es homogénea.
- **Estado:** corregido 05/08.

### E07 — Flag añadido sin probar: fallo silencioso en el parseo

- **Qué pasó:** se añadió el valor `near` al flag `--deadmode` con un `replace` sobre el fichero que **no encontró** el texto a sustituir. El script siguió siendo sintácticamente válido, `ast.parse` pasó, y el run murió a los 4 segundos por un assert que no se había actualizado. Ha ocurrido **tres veces** con distintas variantes.
- **Síntoma observable:** el run muere enseguida, o peor, **ignora el flag y entrena otra cosa** sin avisar.
- **Detección:**
  - Todo `replace` sobre un fichero necesita `assert old in s` **antes** de sustituir.
  - Todo flag nuevo se prueba **end-to-end** antes de meterlo en una tanda: ejecutar el bloque `__main__` de `train.py` con la línea real y comprobar que las variables quedan como se espera. Ver el patrón en la Fase 4.
  - Comprobar que las carpetas de destino del run **no existen** ya (no pisar resultados).
- **Arreglo:** ninguno estructural — es disciplina de proceso. Por eso está aquí.
- **Estado:** **recurrente.** Comprobar en todo preflight que genere o modifique un lanzador.

### E08 — Barra de error medida con una variable congelada

- **Qué pasó:** se midieron tres réplicas variando solo `--seed` y se reportó la sd del peor canal como **±0,21**. Pero las tres compartían `rot_seed = 7`, o sea el **mismo orden de módulos**. Al variar semilla y orden a la vez, la sd real resultó ser **±5,3**: veinticinco veces mayor. El canal del mínimo además **salta** (Ich 60 → Ich 31).
- **Síntoma observable:** una banda sospechosamente estrecha, y un run antiguo con el mismo protocolo que cae a 14 sd de ella (cov149, 47,76 frente a 44,78 ± 0,21). **Ese outlier inexplicable era la pista.**
- **Detección:**
  - Antes de reportar una sd, enumerar **qué fuentes de azar varían y cuáles no**. Si alguna queda fija, la banda mide reproducibilidad, no variabilidad.
  - En este proyecto: `--seed` mueve inicialización de pesos **y** máscaras (el `rng` del `Dataset` se consume en `__getitem__`, así que el orden del `shuffle` decide qué máscara toca a cada evento). `--rotseed` mueve el orden de módulos. `--splitseed` mueve la partición.
  - Si un punto queda fuera de la banda por muchas sd, la hipótesis por defecto es **banda mal estimada**, no dato anómalo.
- **Arreglo:** las réplicas de banda varían semilla y orden juntos (`tandas/noche_banda.ps1`).
- **Estado:** detectado 17/08. **La causa mecánica está identificada:** con `MIX_MODULES = 1` cada época es un solo módulo, los módulos difieren mucho entre sí (sd 7,9) y la época seleccionada cae en un módulo distinto según el orden, así que el modelo final arrastra un sesgo del último módulo visto.

### E09 — Afirmar el comportamiento del código sin ejecutarlo

- **Qué pasó:** al leer `SiPMImputationDataset(X_train, seed=epoch, ...)` se **dedujo** que las máscaras estaban atadas al índice del evento y se escribió como hecho verificado en tres documentos y un commit. Era falso: el `rng` es único y se consume por orden de acceso, así que `--seed` sí las mueve. La comprobación costaba dos minutos.
- **Síntoma observable:** ninguno en el código. El daño es documental: conclusiones erróneas propagadas al Excel, la bitácora y el checkpoint, y tiempo perdido rehaciéndolas.
- **Detección:** cualquier afirmación sobre *qué hace* el código —especialmente si es aleatoriedad, orden o estado compartido— se acompaña de la salida del comando que lo demuestra. Patrón mínimo: ejecutar dos veces con la misma semilla (debe dar 100% de coincidencia) y dos veces con semillas distintas (debe bajar hacia el azar).
- **Arreglo:** disciplina. Si el informe dice "verificado", tiene que haber una salida de comando al lado.
- **Estado:** **recurrente.** Es el fallo que esta skill existe para evitar en su versión de análisis, no solo de lanzamiento.

---

### E10 — Evaluar con la métrica o el comparador que no responden la pregunta

- **Qué pasó:** dos veces seguidas al montar tandas sobre el modelo `near`.
  1. Se encadenó `eval_total` (que mide **un solo canal apagado**) para validar una afirmación sobre **los tres regímenes de fallo multi-canal**, que se mide con `eval_multidead`. El modelo perdía en k=1 por construcción, y eso no decía nada de la pregunta.
  2. Corregido lo anterior, se comparó `near` contra la **referencia entrenada con k=1**, que en multi-dead se desploma por construcción. Un hombre de paja: ganarle no demuestra que `near` sea mejor que las **otras variantes multi-dead**, que es lo que estaba en discusión.
- **Síntoma observable:** un resultado espectacular y en la dirección esperada (+24 puntos). **Una ventaja enorme frente a un comparador es señal de alarma, no de éxito**: suele significar que el comparador no era el rival. La pista aquí fueron las sd del comparador (±11,7), imposibles en un modelo sano.
- **Detección**, antes de escribir la tanda:
  - Escribir la afirmación exacta que se quiere poder defender, y preguntarse **qué dos números la sostienen**. Si la frase es "A es mejor que B en el escenario X", el eval tiene que medir X y B tiene que ser el rival real.
  - **El comparador debe diferir en una sola cosa.** `near` vs referencia difiere en régimen *y* en si es multi-dead: dos ejes, resultado ininterpretable. `near` vs `cluster` multi-dead difiere solo en el régimen.
  - Comprobar que el comparador tiene el **mismo protocolo** (cobertura, presupuesto) y la **misma campaña** de evaluación (`n_seeds`, `max_events_per_file`).
- **Arreglo:** ninguno estructural, es disciplina de diseño. La plantilla de informe lo fuerza con el campo *«qué decisión depende del resultado»*: si no se puede rellenar, la tanda no está lista.
- **Estado:** **recurrente.** Detectado el 19/08. Ambas veces la parte cara (el entrenamiento) estaba bien hecha y solo hubo que repetir evaluaciones, pero costó dos noches.

---

### E11 — Un arreglo aplicado a un solo fichero cuando el patrón está en varios

- **Qué pasó:** el 11/08 se corrigió `eval_total.py` para que leyera `split_seed` del checkpoint en vez de dar por hecha la partición por defecto (si no, evalúa sobre módulos que el modelo sí vio). **El mismo patrón seguía sin corregir en `eval_resolution.py`** hasta el 23/08. Con los modelos de validación cruzada, `eval_resolution` habría medido sobre `datas057/116/126/202/214` cuando el test real de `cvB` es `datas037/045/101/122/172`: cinco módulos de entrenamiento.
- **Síntoma observable:** ninguno. Devuelve un número plausible y algo mejor de lo real, que es lo peor que puede pasar.
- **Detección:**
  - Cuando se arregle un bug en un evaluador, **buscar el mismo patrón en todos los hermanos**: `grep -n "get_file_split" eval_*.py` y comprobar cuáles pasan `seed=` y cuáles no.
  - Antes de lanzar cualquier evaluación, confirmar que el test que usará coincide con el `split_seed` del checkpoint.
- **Arreglo:** `eval_resolution.py` lee ahora `split_seed` del checkpoint y avisa cuando la partición no es la estándar. Verificado que para `split_seed=42` el resultado es **idéntico bit a bit** al anterior, así que no invalida ninguna cifra ya publicada.
- **Estado:** corregido 23/08. Quedan por revisar el resto de evaluadores con la misma receta.

---

<!-- Entradas nuevas debajo de esta línea, numeración correlativa. Cada una necesita su receta de detección. -->
