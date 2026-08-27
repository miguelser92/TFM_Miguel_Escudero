# =====================================================================
#  noche_replicas.ps1 - confirmar los dos resultados de ayer
# =====================================================================
#  Ayer aparecieron dos resultados que cambian conclusiones del trabajo,
#  y los dos descansan sobre UN solo entrenamiento cada uno. Antes de
#  llevarlos a la memoria conviene repetirlos con otra semilla.
#
#  1) REPLICA DE 'm + 32M'. Es el mejor modelo del proyecto: iguala al
#     ensemble heterogeneo en recuperacion media (53.17 frente a 53.21) y
#     lo supera en error de carga, con la tercera parte del coste de
#     inferencia. Si la replica lo confirma, sustituye al ensemble como
#     recomendacion y deja de haber ninguna mejora que dependa de
#     triplicar el coste.
#     Este mismo run responde ademas a la segunda pregunta: su peor canal
#     fue 43.6, por debajo de las vias por separado (47.6 y 48.4), lo que
#     sugiere que no se acumulan. Esa diferencia esta a unas dos
#     desviaciones, al limite, y la replica la confirma o la tumba.
#
#  2) REPLICA DE 'presupuesto 32M'. Tiene el mejor peor-canal medido
#     (48.4) y es el termino de comparacion del punto anterior, asi que
#     conviene saber cuanto ruido tiene esa cifra.
#
#  OJO al reparto de tiempo: el modelo mediano tarda casi cuatro horas en
#  evaluarse (231 min medidos), frente a 80 del pequeno. El bloque 1 se
#  lanza primero por ser el mas importante.
#
#  Uso:  .\noche_replicas.ps1
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = 'E:\envs\tfm\python.exe'
$log = Join-Path $PSScriptRoot ("noche_repl_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

$tareas = @(
    @{ n = '1a) TRAIN replica de m + 32M (semilla 601)'
       a = @('train.py','hexcnn','m','mse','--maxev','800000','--seed','601','--tag','combR2') },
    @{ n = '1b) EVAL  replica de m + 32M'
       a = @('eval_total.py','imputer_hexcnn_m_mse_combR2','--events','750000') },

    @{ n = '2a) TRAIN replica de presupuesto 32M (semilla 602)'
       a = @('train.py','hexcnn','s','mse','--maxev','800000','--seed','602','--tag','presup32MR2') },
    @{ n = '2b) EVAL  replica de presupuesto 32M'
       a = @('eval_total.py','imputer_hexcnn_s_mse_presup32MR2','--events','750000') }
)

function Log($msg) {
    $linea = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $linea
    Add-Content -Path $log -Value $linea -Encoding utf8
}

$t0 = Get-Date
Log "=== REPLICAS DE LOS DOS RESULTADOS DE AYER ==="
Log "log: $log"
Log ""

$resumen = @()
foreach ($t in $tareas) {
    Log ("---- {0} ----" -f $t.n)
    Log ("  {0} {1}" -f $py, ($t.a -join ' '))
    $ini = Get-Date
    & $py $t.a 2>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    $min  = [math]::Round(((Get-Date) - $ini).TotalMinutes, 1)
    $estado = if ($code -eq 0) { 'OK' } else { "FALLO (codigo $code)" }
    Log ("  -> {0} en {1} min" -f $estado, $min)
    Log ""
    $resumen += [pscustomobject]@{ Tarea = $t.n; Estado = $estado; Minutos = $min }
}

$tot = [math]::Round(((Get-Date) - $t0).TotalHours, 2)
Log "=== RESUMEN ($tot h) ==="
foreach ($r in $resumen) { Log ("  {0,-48} {1,-20} {2,6} min" -f $r.Tarea, $r.Estado, $r.Minutos) }
Log ""
Log "Lectura 1: si la replica de m+32M repite una recuperacion media en torno a"
Log "           53, el modelo sustituye al ensemble en la memoria."
Log "Lectura 2: si su peor canal vuelve a quedar claramente por debajo de 47,"
Log "           la anti-acumulacion de las vias a la robustez queda confirmada."
