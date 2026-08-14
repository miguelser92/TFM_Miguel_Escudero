# =====================================================================
#  noche_curva.ps1 — curva de aprendizaje + replicas reproducibles
# =====================================================================
#  1) CURVA DE APRENDIZAJE (lo que convierte el techo en una curva)
#     Hasta ahora el techo se apoya en UN punto: 40 modulos frente a 149.
#     Lo estandar en la literatura es una curva: entrenar con fracciones
#     crecientes de datos y ver si satura. Se hace a coste constante
#     (mismo numero total de muestras y de pasos), variando solo cuantos
#     modulos distintos entran:
#         10 epocas x 4 modulos  =  40 modulos   (el protocolo historico)
#         20 epocas x 4 modulos  =  80 modulos
#         37 epocas x 4 modulos  = 148 modulos   (practicamente todos)
#     Los tres con --mix 4 para que la unica variable sea la cobertura.
#
#  2) REPLICAS CON SEMILLA FIJA. Ahora --seed existe: las mismas tres
#     replicas de siempre, pero reproducibles bit a bit.
#
#  Uso:  .\noche_curva.ps1
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = 'E:\envs\tfm\python.exe'
$log = Join-Path $PSScriptRoot ("noche_curva_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

$tareas = @(
    @{ n = '1a) curva 40 modulos'
       a = @('train.py','hexcnn','s','mse','--epochs','10','--mix','4','--maxev','400000',
             '--rotseed','7','--seed','101','--tag','curva040') },
    @{ n = '1b) eval 40'
       a = @('eval_total.py','imputer_hexcnn_s_mse_mix4_curva040','--events','750000') },

    @{ n = '2a) curva 80 modulos'
       a = @('train.py','hexcnn','s','mse','--epochs','20','--mix','4','--maxev','200000',
             '--rotseed','7','--seed','101','--tag','curva080') },
    @{ n = '2b) eval 80'
       a = @('eval_total.py','imputer_hexcnn_s_mse_mix4_curva080','--events','750000') },

    @{ n = '3a) curva 148 modulos'
       a = @('train.py','hexcnn','s','mse','--epochs','37','--mix','4','--maxev','108000',
             '--rotseed','7','--seed','101','--tag','curva148') },
    @{ n = '3b) eval 148'
       a = @('eval_total.py','imputer_hexcnn_s_mse_mix4_curva148','--events','750000') },

    @{ n = '4a) replica reproducible seed=202'
       a = @('train.py','hexcnn','s','mse','--seed','202','--tag','seed202') },
    @{ n = '4b) eval replica'
       a = @('eval_total.py','imputer_hexcnn_s_mse_seed202','--events','750000') }
)

function Log($msg) {
    $linea = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $linea
    Add-Content -Path $log -Value $linea -Encoding utf8
}

$t0 = Get-Date
Log "=== CURVA DE APRENDIZAJE + REPLICA REPRODUCIBLE ==="
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
foreach ($r in $resumen) { Log ("  {0,-36} {1,-20} {2,6} min" -f $r.Tarea, $r.Estado, $r.Minutos) }
Log ""
Log "Lectura: si los tres puntos de la curva dan lo mismo, el techo deja de"
Log "apoyarse en una comparacion y pasa a ser una curva saturada."
