# =====================================================================
#  noche_presupuesto_cv.ps1 - el ultimo flanco del techo + validacion cruzada
# =====================================================================
#  1) PRESUPUESTO x2 (lo mas importante). El techo esta demostrado frente a
#     arquitectura (nueve palancas) y frente a diversidad de modulos (la curva
#     del 11 ago), pero NO frente a cantidad total de datos: de 4 a 16 millones
#     de muestras se ganaron 3.3 puntos y nadie ha medido que pasa despues.
#     40 epocas x 800k = 32 millones, el doble del protocolo historico.
#     Si no mejora, el techo queda cerrado por los tres flancos.
#
#  2) VALIDACION CRUZADA por modulos (2 particiones nuevas). Da la barra de
#     error de particion completa, que es lo que eval_modulos no puede dar.
#     OJO: --splitseed cambia que 5 modulos van a test, y el evaluador lee esa
#     particion del checkpoint, de modo que cada modelo se mide sobre SUS
#     modulos reservados y no sobre los que vio.
#
#  Uso:  .\noche_presupuesto_cv.ps1
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = 'E:\envs\tfm\python.exe'
$log = Join-Path $PSScriptRoot ("noche_presup_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

$tareas = @(
    @{ n = '1a) TRAIN presupuesto x2 (40 ep x 800k = 32M muestras)'
       a = @('train.py','hexcnn','s','mse','--maxev','800000','--seed','303','--tag','presup32M') },
    @{ n = '1b) EVAL  presupuesto x2'
       a = @('eval_total.py','imputer_hexcnn_s_mse_presup32M','--events','750000') },

    @{ n = '2a) TRAIN CV particion B'
       a = @('train.py','hexcnn','s','mse','--splitseed','43','--seed','401','--tag','cvB') },
    @{ n = '2b) EVAL  CV particion B'
       a = @('eval_total.py','imputer_hexcnn_s_mse_cvB','--events','750000') },

    @{ n = '3a) TRAIN CV particion C'
       a = @('train.py','hexcnn','s','mse','--splitseed','44','--seed','402','--tag','cvC') },
    @{ n = '3b) EVAL  CV particion C'
       a = @('eval_total.py','imputer_hexcnn_s_mse_cvC','--events','750000') }
)

function Log($msg) {
    $linea = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $linea
    Add-Content -Path $log -Value $linea -Encoding utf8
}

$t0 = Get-Date
Log "=== PRESUPUESTO x2 + VALIDACION CRUZADA ==="
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
foreach ($r in $resumen) { Log ("  {0,-52} {1,-20} {2,6} min" -f $r.Tarea, $r.Estado, $r.Minutos) }
Log ""
Log "Lectura 1: si 32M no supera a 16M (52.60 recMean), el techo queda cerrado"
Log "           por los tres flancos: arquitectura, diversidad y presupuesto."
Log "Lectura 2: la dispersion entre las tres particiones es la barra de error de"
Log "           particion, que es la que faltaba y la que domina."
