# =====================================================================
#  noche_rotseed.ps1 — la banda de verdad del peor canal
# =====================================================================
#  La tanda del 15/08 (full1/2/3, semillas 801-803) dio peor canal
#  44.78 +/- 0.21. Esa banda es legitima: --seed mueve la inicializacion
#  de pesos Y las mascaras (el rng del Dataset se consume dentro de
#  __getitem__, asi que el orden del shuffle decide que mascara toca a
#  cada evento; verificado el 16/08).
#
#  El problema es otro: cov149, con el MISMO protocolo y sin mas
#  diferencia que no haber fijado semilla, da 47.76. Catorce sd fuera.
#  Deberia caer dentro. Lo mas probable es que con n=3 la sd este mal
#  estimada y el minimo sobre 61 canales tenga colas largas.
#
#  Esta tanda hace dos cosas a la vez:
#   1. duplica la muestra (n=3 -> n=6) para estimar la sd en condiciones
#   2. varia --rotseed, la unica fuente de azar que --seed NO toca
#
#  Sobre el punto 2: con cobertura completa las 149 epocas recorren los
#  149 modulos, asi que --rotseed no cambia QUE modulos se ven (se ven
#  todos), sino su ORDEN, y con el en que modulo cae la epoca que acaba
#  seleccionada como mejor.
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
Log "Juntar estos tres con full1/2/3 y recalcular la sd con n=6. Si la banda se"
Log "ensancha hasta abarcar el 47.8 de cov149, el problema era la estimacion con"
Log "n=3 y el caso queda cerrado. Si no, hay algo no identificado y toca buscarlo."
Pop-Location
