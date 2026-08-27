# =====================================================================
#  noche_produccion.ps1 - el modelo final y la ultima hipotesis abierta
# =====================================================================
#  1) MODELO DE PRODUCCION CANDIDATO: regimen 'near' + presupuesto doble.
#     Junta las dos cosas que sabemos que funcionan y que actuan sobre
#     aspectos distintos: el regimen 'near' domina en los tres modos de
#     fallo, y duplicar el presupuesto sube el peor canal nueve puntos.
#     Si las dos se acumulan, este es el modelo que va a la memoria como
#     recomendacion practica. Se evalua en multi-dead (tres regimenes).
#
#  2) CAPACIDAD + PRESUPUESTO: la ultima hipotesis del tablero. Sabemos
#     que capacidad, cobertura y presupuesto llevan por separado al mismo
#     techo de robustez (~48% en el peor canal). Lo que no sabemos es si
#     son ACUMULABLES: si el modelo mediano con presupuesto doble supera
#     ese 48, el techo de robustez cede; si se queda ahi, queda confirmado
#     como limite real y no como coincidencia.
#
#  Uso:  .\noche_produccion.ps1
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = 'E:\envs\tfm\python.exe'
$log = Join-Path $PSScriptRoot ("noche_prod_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

$tareas = @(
    @{ n = '1a) TRAIN produccion: near + 32M muestras'
       a = @('train.py','hexcnn','s','mse','--dead','1-4','--deadmode','near',
             '--maxev','800000','--seed','501','--tag','prod') },
    @{ n = '1b) EVAL  multi-dead (3 regimenes)'
       a = @('eval_multidead.py','imputer_hexcnn_s_mse_dead1-4_near_prod','--seeds','20') },
    @{ n = '1c) EVAL  total (fallo unico)'
       a = @('eval_total.py','imputer_hexcnn_s_mse_dead1-4_near_prod','--events','750000') },

    @{ n = '2a) TRAIN capacidad + presupuesto (modelo m, 32M)'
       a = @('train.py','hexcnn','m','mse','--maxev','800000','--seed','502','--tag','comb') },
    @{ n = '2b) EVAL  capacidad + presupuesto'
       a = @('eval_total.py','imputer_hexcnn_m_mse_comb','--events','750000') }
)

function Log($msg) {
    $linea = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $linea
    Add-Content -Path $log -Value $linea -Encoding utf8
}

$t0 = Get-Date
Log "=== MODELO DE PRODUCCION + ACUMULABILIDAD DE LA ROBUSTEZ ==="
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
Log "Lectura 1: si el modelo de produccion mantiene la ventaja de 'near' en los"
Log "           tres regimenes y ademas sube el peor canal, es el que va a la memoria."
Log "Lectura 2: si el peor canal del modelo mediano con presupuesto doble supera"
Log "           el 48 por ciento, el techo de robustez cede; si no, queda confirmado."
