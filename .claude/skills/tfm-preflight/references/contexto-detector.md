# Contexto del detector y del problema

Este fichero existe para que las cifras y las unidades cuadren, no para juzgar si un resultado es "suficientemente bueno". El objetivo es que los experimentos sean correctos; la comparación con trabajos previos es orientación, no criterio de aceptación.

## Qué problema es este TFM

**No es reconstrucción de posición.** Es **imputación de canales de SiPM averiados**: dado un evento al que se le ha apagado uno o varios sensores, reconstruir la carga que habrían recogido. Es autosupervisado: la verdad es el vector original de 61 canales antes de apagar nada.

Consecuencia práctica para el preflight: **la métrica no es FWHM en mm.** Es la **recuperación de posición p90 (%)**, o sea cuánto del error de posición introducido al apagar el canal se recupera al imputarlo. Secundarias: `MAE_mod` (canal imputado, en ADC normalizado) y el bias como guardarraíl. Si un informe habla de FWHM, se ha copiado de otro contexto.

## Detector (medido de los propios datos, verificado en la auditoría del 05/08)

- Cristal **monolítico de LYSO**, geometría **hexagonal**.
- **64 canales de SiPM, de los cuales 61 activos.** Los canales `1`, `16` y `18` están inactivos y se excluyen en todo el pipeline. Una dimensión que no sea 61 es señal de alarma.
- **Pitch** entre sensores: **3,75 mm**.
- Extensión: **X ∈ [−13, 13] mm, Y ∈ [−15, 15] mm**. Es decir **apotema 13, circunradio 15**; el cociente 13/15 = 0,867 = cos 30°, que confirma hexágono regular con vértices arriba y abajo y normales a 0°/60°/120°.
- Posición por **centroide tipo Anger** con pesos cuadráticos: `X = Σ(Rch²·x) / Σ(Rch²)`.

**⚠️ DATOS REALES DE LABORATORIO, no simulación.** Ficheros `.dat` de adquisición con fuente de **²²Na**, un fichero por módulo detector físico. Esto invalida varias comprobaciones típicas de simulación: no hay "posición de fuente" por la que particionar, no hay ground truth de DOI, y la variabilidad entre módulos es real y grande (**sd de 7,9 puntos de recuperación entre detectores**, frente a 0,1–0,2 entre entrenamientos del mismo protocolo).

### Datos aún sin confirmar

Pendientes de Lidia / Víctor, y **bloquean la versión final de Materiales y Métodos**. No inventarlos ni tomarlos de otro detector:

- modelo concreto de SiPM y dimensiones activas
- **espesor del cristal LYSO**
- actividad de la fuente de ²²Na, geometría de irradiación y duración de las adquisiciones
- modelo de ASIC y valor del umbral de discriminación

> Una versión anterior de este fichero afirmaba «lado 17,6 mm, espesor 13 mm, simulación GATE». **El lado no cuadra con los datos** (el circunradio medido es 15 mm, no 17,6) y los datos no son simulados. Esas cifras parecen del detector simulado del grupo, no de este. El espesor de 13 mm podría ser correcto pero **hay que confirmarlo antes de escribirlo en la memoria**.

## Conjunto de datos

| | |
|---|---|
| Módulos sanos (`Good`) | 159 → **149 train / 5 val / 5 test**, partición por fichero, `seed=42` |
| Módulos averiados (`Bad`) | 62, para validación del detector de canales malos |
| Eventos por módulo | ~1,1 millones |
| Test reservado | `datas057`, `datas116`, `datas126`, `datas202`, `datas214` |

**La partición es por fichero, no por evento**, y eso es deliberado: cada fichero es un detector físico distinto, así que separar por fichero mide generalización a **hardware no visto**. Separar por evento mediría algo mucho más fácil y sin interés.

## Protocolo de entrenamiento: cuidado con la palabra "época"

**Una época = un módulo**, no una pasada por el dataset. Cada época carga **un** detector y toma 107.000 de sus ~1,1 M eventos.

- `N_EPOCHS = 149` = número de módulos de train, cada uno visto **una vez**. No es un hiperparámetro ajustado a ojo.
- Presupuesto: 149 × 107k = **15.943.000** muestras, frente a los 40 × 400k = 16.000.000 del protocolo histórico. **−0,36%: la cobertura se amplió a coste constante.**
- No sobra: el óptimo cae en la época ~134 de 149 y parar en la 40 cuesta un 2,6–4,5% de `val_mae`.
- Sin corte temprano (`PATIENCE = N_EPOCHS`), justificado porque cada época trae datos nuevos.

Al redactar: **"149 pasadas, una por módulo, de 107k eventos"**, no "149 épocas" a secas.

## Modelo de referencia

**HexGNN** (`hexcnn` en el código), tamaño `s`, loss MSE: **38.305 parámetros**, `hidden=48`, `n_blocks=4`. Paso de mensajes sobre el grafo de vecindad real (61 nodos, 312 aristas dirigidas, grados 37×6 / 18×4 / 6×3, 100% simétrico). Agregación por media, isótropa.

## Órdenes de magnitud (para detectar disparates, no para aprobar o suspender)

| Métrica | Rango normal |
|---|---|
| recuperación p90 macro | 57–59 % |
| recuperación media macro | 51–53 % |
| MAE_mod macro | 0,61–0,63 |
| peor canal (p90 mínimo) | **37–48 %**, con ±5 de barra de error real (ver E08) |
| baseline regresión lineal | ~11,5 puntos por debajo del modelo |
| ablación de grafo barajado | cae de ~58 a ~29 |

Si un run devuelve recuperación del 90% o del 10%, algo está roto (fuga entre particiones, o preprocesado incoherente entre entrenamiento y evaluación — ver E03 y E04).

## Trabajos previos del grupo

Contexto, no criterio. **Ojo: son de simulación**, así que sus cifras no son comparables directamente con las de aquí.

- **Lidia Jiménez Algarra (TFM 2024-25):** CNN1D+MLP y redes residuales con atención. FWHM ≈ 1,52 mm en simulación frente a ≈ 4 mm en datos reales. Esa brecha simulación→real es en sí misma un dato relevante.
- **Hidalgo-Torres (TFM 2022):** MLP sobre el mismo detector.
