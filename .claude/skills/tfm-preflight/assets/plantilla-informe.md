# Preflight — <tanda o familia de tags>

**Fecha:** <fecha> · **Veredicto:** LANZAR / LANZAR CON RESERVAS / NO LANZAR
**Commit:** `<hash corto>` · **Árbol:** limpio / sucio (`<n>` ficheros)
**Tags que produce:** `<tag1>`, `<tag2>`, … → `runs/imputer_hexcnn_s_mse_<tag>/`

## 1. Hipótesis y cambio respecto al run anterior

- **Run anterior de esta familia:** `<tag>` — <resultado en una línea, con su cifra>
- **Hipótesis:** <qué se pone a prueba, en una frase falsable>
- **Qué cambia exactamente:** <un solo eje si es posible: modelo / datos / protocolo / semillas>
- **Qué se mantiene fijo:** <para que la comparación sea interpretable>
- **Qué decisión depende del resultado:** <si no hay ninguna, replantear la tanda>

## 2. Fuentes consultadas

| Fuente | Qué aportó |
|---|---|
| `references/errores-conocidos.md` | <n> entradas comprobadas |
| `Checkpoint_Activo.md` | <estado del tablero relevante> |
| `Bítacora.md` | <entrada previa relevante> |
| Excel, hoja `<hoja>` | <histórico contra el que se comparará> |
| `PREPROC.json` del run anterior | <configuración de la que se parte> |
| Git | <últimos commits relevantes> |

## 3. Contrato de datos

| Magnitud | Esperado | Observado | Estado |
|---|---|---|---|
| Módulos de train | 149 | <n> | OK / RIESGO |
| Partición train/val/test | 149/5/5, `seed=42`, por fichero | <observado> | |
| Canales activos | 61 (de 64; inactivos 1, 16, 18) | <observado> | |
| Eventos por época | 107.000 | <observado> | |
| Presupuesto total | ~15,9 M muestras | <observado> | |
| Normalización | por evento, `max` post-máscara | <observado> | |
| Régimen de fallo | `n_dead`, `dead_mode` | <observado> | |

**Línea de COBERTURA que debe aparecer en el log:** `COBERTURA: 149/149 módulos distintos (100%)`

## 4. Chequeo de regresiones

Una fila por entrada del registro. **Evidencia obligatoria**: `fichero:línea` o salida del comando.

| ID | Error | Estado | Evidencia |
|---|---|---|---|
| E01 | 40/149 módulos | OK / RIESGO / NO APLICA | |
| E02 | Fix no incluido en el run relanzado | | |
| E03 | Preprocesado eval ≠ train | | |
| E04 | Fuga en `get_channel_scale` | | |
| E05 | Numerador/denominador con distinto n | | |
| E06 | Campañas de distinto tamaño | | |
| E07 | Flag añadido sin probar | | |
| E08 | Barra de error con variable congelada | | |
| E09 | Afirmar sin ejecutar | | |

## 5. Aritmética y coste

- Épocas × eventos = **<n> muestras** (comparar con las ~15,9 M de referencia)
- Pasos por época: 107.000 / 512 = **~209**; pasos totales: **<n>**
- Tiempo estimado: <n> min/train × <n> trains + <n> min/eval × <n> evals = **<h> h**
  (referencia medida: ~155 min por entrenamiento, ~77 min por `eval_total` a 750k)
- Carpetas de destino: **¿existen ya?** (si existen, la tanda las pisaría)

## 6. Smoke test

| Comprobación | Resultado |
|---|---|
| Los flags parsean: variables tras el bloque `__main__` | |
| Recuento real de módulos que enumera el split | |
| Formas, dtype, rango, NaN/Inf de un batch | |
| Forward/backward: pérdida finita, gradientes no nulos | |
| Reproducibilidad: misma semilla → mismo batch | |
| Las combinaciones de la tanda son distintas entre sí | |

## 7. Riesgos abiertos

1. <riesgo, con la consecuencia concreta si se materializa>

## 8. Veredicto

**<VEREDICTO>** — <justificación en dos líneas, sin suavizar>

```powershell
.\tandas\<nombre>.ps1
```

## 9. Qué mirar cuando termine

- La línea de COBERTURA de cada entrenamiento.
- `Traceback` en el log (el rojo de la consola **no** es indicador).
- <la comparación con el histórico que decidirá si el resultado es creíble>
- <qué cifra del Excel y de la bitácora hay que actualizar>
