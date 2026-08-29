# Candidatas a apéndice

> **Estado (28 ago).** El apéndice ya está escrito en `main_WIP - appx.tex`, con
> cinco secciones (A arquitecturas, B particiones, C módulos Bad, D espectros,
> E estadística por sensor). De esta carpeta entraron **`A1`, `A2` y `C1`**;
> el resto sigue aquí por si se recupera. Lo demás del apéndice son tablas
> nuevas, no figuras.

Copias de figuras que **no están en el cuerpo del paper** (salvo `A1`, que sí
está y se plantea mover aquí). Los originales siguen en su sitio: esta carpeta
se puede borrar entera sin perder nada.

El criterio con el que están elegidas: **el apéndice es para evidencia primaria
y reproducibilidad, no para lo que no cupo.** Y la regla que decide cada una:
*si el texto no la cita, no va*. Una figura huérfana en un apéndice parece
relleno y resta.

Cada bloque va marcado con lo que yo haría:

- **[SÍ]** — la metería
- **[SI HAY SITIO]** — aporta, pero el paper aguanta sin ella
- **[NO]** — está aquí porque la pediste, no porque la recomiende

---

## A — Espectro de energía

Es el bloque más fuerte. El resultado espectral es de los dos que hacen físico
el trabajo, y en Resultados sólo aparece resumido en una cifra. Aquí está el
dato crudo que la sostiene.

| Fichero | [ ] | Qué es | Origen |
|---|---|---|---|
| `A1_espectros_zonas_k1.png` | **SÍ** | Los 15 histogramas por sensor y zona con un canal suprimido. **Es la figura 10 actual del paper.** | `TFM FINAL/imagenes/espectros_zonas.png` |
| `A2_espectros_zonas_k4_near.png` | **SÍ** | Lo mismo con **cuatro** canales suprimidos, régimen `near`. Nunca ha salido de `runs/`. | `runs/imputer_hexcnn_s_mse_dead1-4_near_banda11/ESPECTRO_k4_near/espectros.png` |
| `A3_resumen_espectro_k4_cluster.png` | SI HAY SITIO | Resumen agregado, régimen contiguo, k=4 | `.../ESPECTRO_k4_cluster/espectro_resumen.png` |
| `A4_resumen_espectro_k4_near.png` | SI HAY SITIO | Íd., régimen cercano | `.../ESPECTRO_k4_near/espectro_resumen.png` |
| `A5_resumen_espectro_k4_scatter.png` | SI HAY SITIO | Íd., régimen disperso | `.../ESPECTRO_k4_scatter/espectro_resumen.png` |

**A1** es la que dijiste que ocupa mucho y no se ve. Aquí abajo se lee sin
competir con el argumento del capítulo.

**A2** es la que yo pondría sí o sí, porque el texto afirma que el espectro *no
se degrada al crecer el daño* (83,1 % con un canal → 88,7 % con cuatro) y ahora
mismo esa afirmación no tiene ninguna figura detrás. Poner A1 y A2 juntas hace
el argumento solo: se ven las dos y las curvas siguen superpuestas.

**A3–A5** son las que sostienen la inversión de orden entre regímenes en k=4
(disperso 93,3 % > cercano 88,7 % > contiguo 85,6 %, al revés que en posición).
Si el texto menciona esa inversión —y es el argumento más fuerte que tienes para
`near`— van las tres o ninguna, porque la gracia está en compararlas.

## B — Multi-canal, sensor a sensor

| Fichero | [ ] | Qué es | Origen |
|---|---|---|---|
| `B1_multidead_mapas_cluster.png` | SI HAY SITIO | Mapa hexagonal de recuperación P90 por semilla de fallo, k=1..4, régimen contiguo | `.../MULTIDEAD_3MODOS/multidead_maps_cluster.png` |
| `B2_multidead_mapas_near.png` | SI HAY SITIO | Íd., cercano | `.../multidead_maps_near.png` |
| `B3_multidead_mapas_scatter.png` | SI HAY SITIO | Íd., disperso | `.../multidead_maps_scatter.png` |
| `B4_multidead_curva_4paneles.png` | **NO** | La figura 8 **antigua**: 4 paneles, un solo run, sin banda de réplicas | `TFM FINAL/imagenes/multidead_curve.png` |

**B1–B3** enseñan lo que la figura 8 promedia: dónde del cristal duele cada
régimen. Es información real que la curva esconde. Van juntas o no van.

**B4 no la metería.** Es de **un solo run** y sin banda de incertidumbre, y la
figura 8 nueva es su versión honesta con las tres réplicas. Tener las dos en el
mismo documento invita a preguntar por qué no coinciden exactamente. Está aquí
sólo para que la veas antes de decidir.

## C — Comportamiento por sensor

| Fichero | [ ] | Qué es | Origen |
|---|---|---|---|
| `C1_estadistica_y_mae_por_sensor.png` | **SÍ** | Dos hexágonos: eventos modificados por sensor (de 459k a 1.892k) y MAE de imputación por sensor | `runs/imputer_hexcnn_s_mse_banda11/TOTAL/stats_maps.png` |
| `C2_perfiles_1d.png` | SI HAY SITIO | Perfiles 1D del flood map | `.../TOTAL/line_profiles.png` |

**C1** es la que más me gusta de todo lo que no está en el paper. El panel
izquierdo responde por adelantado a la objeción obvia —*«¿el borde va peor
porque hay menos estadística?»*— y el derecho enseña el gradiente
centro-borde en unidades físicas (ADC), no en porcentaje de recuperación. Es
exactamente el tipo de figura que un tribunal agradece.

**C2** solapa con la figura 5 del paper (`line_floodmap_diff`), que es la
versión con el flood map al lado. Sólo si quieres el perfil limpio.

## D — Arquitecturas

| Fichero | [ ] | Qué es | Origen |
|---|---|---|---|
| `D1_ablacion_arquitecturas.png` | SI HAY SITIO | Comparativa de las arquitecturas evaluadas | `TFM FINAL/imagenes/ablacion.png` |

Ya la habías retirado con buen criterio («la ablación son tres números que ya
están en el texto»). En un apéndice el argumento cambia un poco: no cuesta
columna y documenta trabajo hecho que el texto sólo resume.

---

## Lo que NO he copiado, y por qué

| | Por qué no |
|---|---|
| `training_curves.png` | No sostiene ninguna afirmación del texto. Nadie las mira. |
| `memoria/figuras_informe/*` | Protocolo viejo de **40 módulos**. Las cifras del paper no salen de ahí. Meterlas sería arrastrar un fallo. |
| `line_floodmap_diff`, `mapa_recuperacion` | Ya están en Resultados (figuras 5 y 6). Duplicarlas resta. |
| `floodmap.pdf` | Es el que produce las líneas blancas. Pendiente de borrar, no de reubicar. |

## Y lo que metería aunque no sea una imagen

Esto vale más que la mitad de las figuras de arriba, y no existe todavía:

1. **Configuración completa de las arquitecturas comparadas** — capas,
   dimensiones, learning rate, épocas. La tabla de Resultados da parámetros y
   métricas, pero no la configuración: sin eso la comparación no es
   reproducible. Sale de los `config.json` de cada run.
2. **Los módulos de cada partición de validación cruzada** — cinco líneas, y
   convierte la banda de ±2,3 en algo que alguien puede verificar.
3. **El listado de los 62 módulos Bad y su canal etiquetado** — de
   `reports/bad_labels_manual.json`. Documenta la parte manual del trabajo.

## Nota de formato

En IEEEtran a dos columnas, `A1` y `A2` necesitan `figure*` y probablemente
página completa apaisada (`\begin{figure*}[p]` + `\rotatebox{90}`). Si van a
dos columnas normales tendrás el mismo problema de legibilidad que ahora, sólo
que al final del documento.

El apéndice va después de `\section{Conclusions}` y antes de la bibliografía,
abierto con `\appendix` (o `\appendices` en IEEEtran, que numera A, B, C...).
Ahora mismo el `.tex` **no tiene ninguno de los dos**.
