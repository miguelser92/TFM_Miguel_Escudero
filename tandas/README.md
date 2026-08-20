# Tandas nocturnas

Scripts de PowerShell que encadenan entrenamientos y evaluaciones para dejarlos
corriendo sin supervision. Cada uno registra tiempos y codigos de salida, y no se
detiene si un paso falla.

| script | fecha | que resolvio |
|---|---|---|
| eval_noche.ps1 | 9 ago | Tabla de arquitecturas homogenea a 600k y cov149 en las demas metricas |
| train_noche.ps1 | 10 ago | Mezcla de modulos en el lote (no ayuda) y combinaciones con multi-dead |
| noche_curva.ps1 | 11 ago | Curva de aprendizaje 40/80/148 modulos y cuarta replica |
| noche_presupuesto_cv.ps1 | 12 ago | Presupuesto x2 (techo cerrado) y validacion cruzada B y C |
| noche_cv_near.ps1 | 13 ago | Particiones D y E, y el regimen de fallo cercano |
| noche_produccion.ps1 | 14 ago | Modelo de produccion y acumulabilidad de la robustez |
| noche_replicas.ps1 | 14 ago | Replicas de los dos hallazgos del dia anterior |
| noche_protocolo_bueno.ps1 | 15 ago | Modelo de referencia con cobertura completa (149 modulos) y su banda: la media no cambia, el peor canal sube +2.2 pts |
| noche_banda.ps1 | 17 ago | Barra de error REAL del peor canal variando semilla y orden juntos: 42,3 +/- 4,5 (n=4), 21x mas ancha que la de --seed sola. Cierra el caso cov149 y retira el "techo de robustez" |
| noche_near_completo.ps1 | 18 ago | 'near' con cobertura completa, 3 replicas (15,9 h, no 11,5: los train de near tardan 240 min). En k=1 queda por debajo de la referencia, pero k=1 no es la metrica que decide |
| noche_multidead_near.ps1 | 19 ago | near vs cluster en los tres regimenes con cobertura completa. Cae la ventaja EN LOS TRES: near solo gana en el suyo (+1,99, z=2,9) y empata en los extremos. Lo determinante es entrenar multi-dead: +24 pts sobre k=1 |
| noche_mix_estabilidad.ps1 | 20 ago | NO estabiliza: sd 3,31 vs 4,45 pero F=1,81 p=0,750, y la media empeora -0,65. El minimo sobre 61 canales es ruidoso de por si |

**Fase experimental cerrada el 20/08.** Cinco noches, ~68 h de computo. Lo que queda es redaccion.

Los logs van a `tandas/logs/` y no se versionan: son reproducibles relanzando el
script, y las conclusiones estan en la bitacora y en el Excel.

Uso: desde la raiz del proyecto, lanzar el script indicando la subcarpeta, o
copiarlo a la raiz si se prefiere ejecutarlo desde alli.
