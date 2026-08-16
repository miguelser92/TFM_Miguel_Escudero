# =====================================================================
#  noche_banda.ps1 — la barra de error del peor canal, bien muestreada
# =====================================================================
#  QUE PROBLEMA RESUELVE
#
#  La tanda del 15/08 (full1/2/3) dio peor canal 44.78 +/- 0.21. Esa banda
#  es legitima pero PARCIAL: las tres compartian rot_seed = 7, o sea el
#  mismo orden de modulos. Y cov149, con el mismo protocolo y sin mas
#  diferencia que no haber fijado semilla, da 47.76: catorce sd fuera.
#
#  Con una variable congelada no se puede estimar la variabilidad total.
#  Aqui cada replica varia TODO lo estocastico a la vez -semilla global y
#  orden de modulos-, que es lo que hace un experimentador cuando reporta
#  "media +/- sd sobre N repeticiones".
#
#  QUE SE OBTIENE
#
#  Tres entrenamientos plenamente independientes. Junto con cov149, que
#  tambien varia todo, dan n=4 de la variabilidad REAL, que es la cifra
#  que puede ir a la memoria como barra de error del peor canal.
#
#  Lo que NO se toca: split_seed (la particion debe ser fija; su efecto ya
#  se midio aparte con la validacion cruzada: 56.60 +/- 2.35).
#
#  Uso:  .\tandas\noche_banda.ps1        (~11.5 h)
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = Join-Path $PSScriptRoot '..\..\..\..\..\envs\tfm\python.exe'
if (-not (Test-Path $py)) { $py = 'E:\envs\tfm\python.exe' }
$raiz = Split-Path $PSScriptRoot -Parent
$log  = Join-Path $PSScriptRoot ("logs\noche_banda_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

# semilla global y orden de modulos cambian JUNTOS en cada replica
$combis = @(
    @{ seed = 811; rot = 11 },
    @{ seed = 812; rot = 13 },
    @{ seed = 813; rot = 17 }
)

$tareas = @()
foreach ($c in $combis) {
    $tag = "banda$($c.rot)"
    $tareas += @{ n = "TRAIN replica independiente (seed $($c.seed), rotseed $($c.rot))"
                  a = @('train.py','hexcnn','s','mse','--seed',"$($c.seed)",
                        '--rotseed',"$($c.rot)",'--tag',$tag) }
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
Log "=== BANDA REAL DEL PEOR CANAL: 3 replicas independientes ==="
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
Log "Siguiente paso: sd del peor canal sobre estas tres + cov149 (n=4, todas con"
Log "variacion completa). Si la banda abarca el 47.8 de cov149, el caso se cierra"
Log "y esa es la barra de error que va a la memoria."
Pop-Location
