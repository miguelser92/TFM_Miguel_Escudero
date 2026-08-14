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

Los logs van a  y no se versionan: son reproducibles relanzando el script,
y las conclusiones estan en la bitacora y en el Excel.

Uso:  y despues , o copiar el script a la raiz si
se prefiere lanzarlo desde alli.
