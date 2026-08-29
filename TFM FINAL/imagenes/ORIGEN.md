# Origen de las figuras

De dónde sale cada imagen y cómo regenerarla. Sirve para dos cosas: que una
figura no se quede desfasada respecto a las cifras del texto, y que cualquiera
pueda rehacerlas sin adivinar qué run las produjo.

**Todas las figuras de resultados son del protocolo corregido** (149 módulos),
es decir de los runs `banda11/13/17`. Las que había antes en
`memoria/figuras_informe/` son del 10 de agosto, del protocolo de 40 módulos, y
**no deben usarse**: el texto cita cifras que no corresponden a esos modelos.

## Portada y plantilla

| Fichero | Origen |
|---|---|
| `logo_UC3M.png` | Plantilla oficial UC3M |
| `creativecommons.png` | Plantilla oficial UC3M |

## Introducción

| Fichero | Fig. | Origen |
|---|---|---|
| `escaner.png` | 1 | Reproducida de Pérez-Benito 2019 |
| `cristales.png` | 2 | Reproducida de Enlow 2023, CC BY 4.0 |
| `floodmap_intro.png/.pdf` | 3 | `python fig_floodmap_intro.py` — `datas057`, Ich 13, 600k eventos |

## Materiales y Métodos

| Fichero | Fig. | Origen |
|---|---|---|
| `regimenes_diagrama.png/.pdf` | 4 | `python fig_regimenes_diagrama.py` — semilla 23, k=3. Usa las funciones reales de `dataset.py` |

## Resultados

Todas de `runs/imputer_hexcnn_s_mse_banda11/` salvo donde se indique. **Se
copian, nunca se mueven**: esos directorios son salida de entrenamientos ya
ejecutados.

| Fichero | Fig. | Origen | Nota |
|---|---|---|---|
| `line_floodmap_diff.png` | 5 | `.../TOTAL/line_floodmap_diff.png` | recortada la cabecera con el nombre del run |
| `mapa_recuperacion.png` | 6 | `.../TOTAL/recovery_map.png` | íd. |
| `floodmap.png/.pdf` | 7 | generada con `imputation_eval` sobre `banda11` | |
| `multidead_curve.png` | 8 | `runs/imputer_hexcnn_s_mse_dead1-4_near_banda11/MULTIDEAD_3MODOS/` | modelo entrenado en `near` |
| `resolucion.png` | 9 | `reports/resolution_pointspread_banda11.png` | el título interno decía «narrower = better resolution», corregido a «less error introduced» |
| `espectros_zonas.png` | 10 | `.../ESPECTRO/espectros.png` | 15 paneles, recortada la cabecera |

## Retiradas

| Fichero | Por qué |
|---|---|
| `ablacion.png` | La ablación son tres números que ya están en el texto |
| `espectro.png` | Sustituida por `espectros_zonas.png`, que enseña los histogramas en vez del resumen agregado |

Se dejan en la carpeta por si se recuperan, pero **el `.tex` no las cita**.

## Apéndice

Usadas por `main_WIP - appx.tex`, que es la versión con apéndice. El `main_WIP.tex`
normal no las cita.

| Fichero | Fig. | Origen | Nota |
|---|---|---|---|
| `espectros_zonas.png` | 10 | ya descrita arriba | **movida** de Resultados al Apéndice D |
| `espectros_zonas_k4.png` | 11 | `runs/imputer_hexcnn_s_mse_dead1-4_near_banda11/ESPECTRO_k4_near/espectros.png` | recortados 81 px de cabecera, que es justo lo que se le quitó a la de k=1: así las dos quedan a la misma escala y se comparan panel a panel |
| `stats_por_sensor.png` | 12 | `runs/imputer_hexcnn_s_mse_banda11/TOTAL/stats_maps.png` | recortados 90 px de cabecera |

`apendice_candidatas/` tiene copias de todas las que se valoraron, con un README
que explica qué sostiene cada una y por qué entró o no. Son copias: los
originales no se han movido y la carpeta se puede borrar sin consecuencias.

## Al regenerar cualquiera

1. Copiar desde `runs/`, no mover.
2. Recortar la cabecera si lleva el nombre interno del run: no pinta nada en un
   paper y contradice el pie.
3. Comprobar que el run del que sale es el mismo del que salen las cifras del
   párrafo que la cita.
