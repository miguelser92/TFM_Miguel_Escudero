# =====================================================================
#  noche_rotseed.ps1 — la banda de verdad del peor canal
# =====================================================================
#  La tanda del 15/08 (full1/2/3, semillas 801-803) dio sd = 0.21 en el
#  peor canal. Esa banda esta SUBESTIMADA: entre aquellas tres replicas
#  solo cambiaba la inicializacion de pesos.
#
#   - el masking depende de la EPOCA (seed=epoch en train.py), no de --seed
#   - el orden de modulos es fijo (rot_seed = 7 por defecto)
#
#  Las tres veian los mismos modulos, en el mismo orden, con las mismas
#  mascaras. Aqui se fija --seed y se varia --rotseed.
#
#  Matiz: con cobertura completa las 149 epocas recorren los 149 modulos,
#  asi que --rotseed no cambia QUE modulos se ven (se ven todos). Cambia
#  el ORDEN, y como la mascara depende de la epoca, cambia el
#  EMPAREJAMIENTO modulo <-> mascara y en que modulo cae la epoca que
#  acaba seleccionada. Eso es justo la variabilidad que --seed no captura.
#
#  Contraste que lo motiva: cov149, con el mismo protocolo nominal, da
#  peor canal 47.76 frente a 44.78 +/- 0.21. Catorce sd fuera de banda.
#
#  Uso:  .\tandas\noche_rotseed.ps1        (~11.5 h)
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = Join-Path $PSScriptRoot '..\..\..\..\..\envs\tfm\python.exe'
if (-not (Test-Path $py)) { $py = 'E:\envs\tfm\python.exe' }
$raiz = Split-Path $PSScriptRoot -Parent
$log  = Join-Path $PSScriptRoot ("logs\noche_rotseed_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

$tareas = @()
foreach ($rs in 11, 13, 17) {
    $tag = "rot$rs"
    # --seed 801 fijo: la inicializacion deja de ser la variable, y lo unico
    # que cambia entre los tres es el orden de modulos y el masking.
    $tareas += @{ n = "TRAIN rotseed $rs (semilla fija 801)"
                  a = @('train.py','hexcnn','s','mse','--seed','801','--rotseed',"$rs",'--tag',$tag) }
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
Log "=== BANDA REAL: rotseed 11/13/17, semilla fija 801 ==="
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
Log "Comparar la sd de estos tres con la de full1/2/3 (0.21). Si es mucho mayor,"
Log "queda demostrado que la banda buena es esta y que el peor canal no se puede"
Log "reportar con la barra de --seed."
Pop-Location
