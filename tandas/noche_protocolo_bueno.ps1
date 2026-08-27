# =====================================================================
#  noche_protocolo_bueno.ps1 - el modelo de referencia, bien entrenado
# =====================================================================
#  Tres entrenamientos con el protocolo corregido: cobertura completa
#  (los 149 modulos), mismo presupuesto de siempre (~16M muestras).
#  Ya no hace falta pasar ningun flag: es el default.
#
#  Dan dos cosas de golpe:
#   - el modelo de referencia entrenado como se debe entrenar
#   - la BANDA DEL PEOR CANAL, que nunca hemos tenido: todas las cifras
#     de robustez del proyecto salen de un unico entrenamiento cada una,
#     y el 15/08 se midio que esa metrica varia 5.4 puntos entre semillas
#
#  Uso:  .\tandas\noche_protocolo_bueno.ps1
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = Join-Path $PSScriptRoot '..\..\..\..\..\envs\tfm\python.exe'
if (-not (Test-Path $py)) { $py = 'E:\envs\tfm\python.exe' }
$raiz = Split-Path $PSScriptRoot -Parent
$log  = Join-Path $raiz ("noche_protocolo_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

$tareas = @()
foreach ($s in 801, 802, 803) {
    $tag = "full$($s - 800)"
    $tareas += @{ n = "TRAIN cobertura completa (semilla $s)"
                  a = @('train.py','hexcnn','s','mse','--seed',"$s",'--tag',$tag) }
    $tareas += @{ n = "EVAL  $tag"
                  a = @('eval_total.py',"imputer_hexcnn_s_mse_$tag",'--events','750000') }
}

function Log($msg) {
    $linea = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $linea
    Add-Content -Path $log -Value $linea -Encoding utf8
}

Push-Location $raiz
$t0 = Get-Date
Log "=== PROTOCOLO CORREGIDO: 149 modulos, 3 semillas ==="
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
foreach ($r in $resumen) { Log ("  {0,-44} {1,-20} {2,6} min" -f $r.Tarea, $r.Estado, $r.Minutos) }
Log ""
Log "Con los tres: modelo de referencia con el protocolo correcto y, por primera vez,"
Log "una banda de error para el peor canal."
Pop-Location
