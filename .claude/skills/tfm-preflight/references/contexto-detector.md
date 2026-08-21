# Contexto del detector y del problema

Este fichero existe para que las cifras y las unidades cuadren, no para juzgar si un resultado es "suficientemente bueno". El objetivo es que los experimentos sean correctos; la comparación con trabajos previos es orientación, no criterio de aceptación.

## Qué problema es este TFM

**No es reconstrucción de posición.** Es **imputación de canales de SiPM averiados**: dado un evento al que se le ha apagado uno o varios sensores, reconstruir la carga que habrían recogido. Es autosupervisado: la verdad es el vector original de 61 canales antes de apagar nada.

Consecuencia práctica para el preflight: **la métrica no es FWHM en mm.** Es la **recuperación de posición p90 (%)**, o sea cuánto del error de posición introducido al apagar el canal se recupera al imputarlo. Secundarias: `MAE_mod` (canal imputado, en ADC normalizado) y el bias como guardarraíl. Si un informe habla de FWHM, se ha copiado de otro contexto.

## Detector (medido de los propios datos, verificado en la auditoría del 05/08)

- Cristal **monolítico de LYSO**, geometría **hexagonal**, lado 17,6 mm y espesor 13 mm, grabado con láser en 364 estructuras.
- **64 canales de SiPM, de los cuales 61 activos.** Los canales `1`, `16` y `18` están inactivos y se excluyen en todo el pipeline. Una dimensión que no sea 61 es señal de alarma.
- **Pitch** entre sensores: **3,75 mm**.
- Extensión: **X ∈ [−13, 13] mm, Y ∈ [−15, 15] mm**. Es decir **apotema 13, circunradio 15**; el cociente 13/15 = 0,867 = cos 30°, que confirma hexágono regular con vértices arriba y abajo y normales a 0°/60°/120°.
- Posición por **centroide tipo Anger** con pesos cuadráticos: `X = Σ(Rch²·x) / Σ(Rch²)`.

**⚠️ DATOS REALES DE LABORATORIO, no simulación.** Ficheros `.dat` de adquisición con fuente de **²²Na**, un fichero por módulo detector físico. Esto invalida varias comprobaciones típicas de simulación: no hay "posición de fuente" por la que particionar, no hay ground truth de DOI, y la variabilidad entre módulos es real y grande (**sd de 7,9 puntos de recuperación entre detectores**, frente a 0,1–0,2 entre entrenamientos del mismo protocolo).

### El cristal (confirmado el 21/08)

- **Lado del hexágono: 17,6 mm. Espesor: 13 mm.**
- Grabado con láser en **364 estructuras** tipo panal, que limitan cómo se reparte la luz antes de llegar al fotosensor.

> **Corrección de una versión anterior de este fichero.** Escribí que el lado de 17,6 mm «no cuadraba con los datos» porque el circunradio que yo había medido era de 15 mm. **Estaba comparando magnitudes distintas**: los 15 mm son el circunradio de los *centros de los SiPM*, y los 17,6 mm el lado del *cristal*. Los sensores no llegan al borde del cristal, así que las dos cifras son compatibles y ambas son correctas. Miguel lo confirmó.
>
> Lección para el registro: antes de declarar que dos cifras se contradicen, comprobar que miden lo mismo.

### El fotosensor (del TFM de Jiménez Algarra, 21/08)

- Fabricante: **Hamamatsu**
- **3.980 microceldas** por celda SiPM
- PDE ≈ **35 %** a la longitud de onda de emisión del LYSO
- La salida del ASIC son cuentas ADC; la relación con fotones detectados es
  `N_fired = N_pixels · (1 − exp(−N_photons · PDE / N_pixels))`

### Datos aún sin confirmar

- **modelo o part number del SiPM** — Jiménez Algarra dice sólo «los fotomultiplicadores
  Hamamatsu», sin referencia
- **modelo de ASIC** y valor del umbral de discriminación por canal
- **actividad de la fuente, geometría de irradiación y duración** de las adquisiciones
  de *este* conjunto de datos

### ⚠️ NO tomar del TFM de Jiménez Algarra: es otro experimento

Aquel trabajo da actividad de **286 kBq**, colimador de **1,2 mm** y **10 min por
posición**. Son de un **banco de haz colimado** sobre **un** módulo, apuntando a los
centros de los 61 SiPM, con ~10.192 eventos por posición.

Los datos de este TFM son **159 módulos distintos, ~1,1 M de eventos cada uno, en
inundación**: adquisiciones de calibración de producción. **Mismo detector y misma
electrónica, protocolo de adquisición distinto.** Copiar esas cifras sería describir el
montaje de otra persona.

Del mismo modo, su **umbral analógico de 330 keV** es un corte por energía *del evento*
que aplican para descartar dispersados, no el umbral por canal que hace que un SiPM
registre cero en nuestros ficheros. No confundirlos.

### Contexto de resultados de aquel trabajo (para la Discusión, no para Métodos)

FWHM ≈ **1,52 mm en simulación GATE** frente a ≈ **4 mm en datos reales**, con haces de
1,6 y 1,2 mm de diámetro respectivamente. La brecha simulación→real es en sí misma un
dato citable. Su métrica es FWHM de posicionamiento; la nuestra es recuperación de
posición tras imputar. **No son comparables directamente.**

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
