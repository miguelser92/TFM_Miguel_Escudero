# =====================================================================
#  train_noche.ps1 - tanda de ENTRENAMIENTOS + sus evaluaciones
# =====================================================================
#  Orden por valor esperado. Cada entrenamiento va seguido de su
#  eval_total, para que por la manana haya numeros y no solo pesos.
#
#  1) mix8      40 epocas x 8 modulos mezclados. MISMO coste que el
#               protocolo original (16M muestras) y MISMA cobertura que
#               cov149 (149 modulos): aisla el efecto de mezclar
#               detectores dentro del lote. Es la hipotesis nueva.
#  2) cov149md  cobertura completa + regimen multi-dead. Combina las dos
#               mejoras conocidas en vez de intercambiarlas.
#  3) mix8md    las tres cosas a la vez, si diera tiempo.
#
#  Uso:  .\train_noche.ps1
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = 'E:\envs\tfm\python.exe'
$log = Join-Path $PSScriptRoot ("train_noche_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

$tareas = @(
    @{ n = '1a) TRAIN mix8 (40 ep x 8 modulos mezclados)'
       a = @('train.py','hexcnn','s','mse','--mix','8','--rotseed','7','--tag','mix8') },
    @{ n = '1b) EVAL  mix8'
       a = @('eval_total.py','imputer_hexcnn_s_mse_mix8_mix8') },

    @{ n = '2a) TRAIN cov149md (cobertura completa + multi-dead 1-4)'
       a = @('train.py','hexcnn','s','mse','--epochs','149','--maxev','107000',
             '--rotseed','7','--dead','1-4','--tag','cov149md') },
    @{ n = '2b) EVAL  cov149md'
       a = @('eval_total.py','imputer_hexcnn_s_mse_dead1-4_cov149md') },

    @{ n = '3a) TRAIN mix8md (mezcla + multi-dead)'
       a = @('train.py','hexcnn','s','mse','--mix','8','--rotseed','7',
             '--dead','1-4','--tag','mix8md') },
    @{ n = '3b) EVAL  mix8md'
       a = @('eval_total.py','imputer_hexcnn_s_mse_mix8_dead1-4_mix8md') }
)

function Log($msg) {
    $linea = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $linea
    Add-Content -Path $log -Value $linea -Encoding utf8
}

$t0 = Get-Date
Log "=== TANDA DE ENTRENAMIENTOS ==="
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
Log "La comparacion clave es mix8 vs cov149: misma cobertura y mismo coste,"
Log "solo cambia si los modulos se ven mezclados o de uno en uno."
