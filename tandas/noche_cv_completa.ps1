# =====================================================================
#  noche_cv_completa.ps1 - la validacion cruzada con el protocolo bueno
# =====================================================================
#  QUE ARREGLA. La cifra que el paper usa para describir el rendimiento en
#  hardware nuevo -56,56 +/- 2,32 %- sale de cinco particiones entrenadas
#  con 40 de los 149 modulos. Es la ULTIMA cifra del protocolo viejo que
#  sigue en el texto, mientras el resto del capitulo usa 149. Esta tanda
#  la rehace entera.
#
#  QUE NO ARREGLA, y conviene saberlo antes de gastar 19 horas: la BANDA
#  seguira siendo de +/-2 puntos. Esa anchura no viene de la cobertura del
#  entrenamiento sino de que hay solo CINCO modulos en cada test, y la
#  variabilidad entre detectores es de 7,9 puntos: 7,9/sqrt(5) = 3,5. Para
#  estrecharla harian falta test mas grandes, no entrenamientos mas largos.
#  Lo que se gana aqui es coherencia de protocolo, no precision.
#
#  DISENO. Cinco particiones, y semilla y orden FIJOS en las cinco. Es lo
#  contrario que noche_banda: alli se variaba todo menos la particion para
#  medir el ruido de entrenamiento; aqui se varia solo la particion para
#  aislar su efecto. Si se variaran las dos cosas a la vez, la banda
#  resultante mezclaria ambas fuentes y no seria comparable con la de
#  noche_banda.
#
#  eval_total lee el split_seed del checkpoint desde el 11/08, asi que cada
#  run se evalua con SU particion. Sin eso mediria sobre modulos que el
#  modelo si vio.
#
#  Uso:  .\tandas\noche_cv_completa.ps1        (~19 h, mas de una noche)
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = Join-Path $PSScriptRoot '..\..\..\..\..\envs\tfm\python.exe'
if (-not (Test-Path $py)) { $py = 'E:\envs\tfm\python.exe' }
$raiz = Split-Path $PSScriptRoot -Parent
$log  = Join-Path $PSScriptRoot ("logs\noche_cvfull_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

Push-Location $raiz
$commit = (git rev-parse --short HEAD).Trim()
$sucio  = (git status --porcelain)

# Se salta lo que ya este hecho: la tanda es relanzable y se puede partir
# en varias noches sin repetir trabajo.
$tareas = @()
foreach ($sp in 42, 43, 44, 45, 46) {
    $tag = "cvf$sp"
    $run = "imputer_hexcnn_s_mse_$tag"
    if (-not (Test-Path (Join-Path $raiz "runs\$run\best_model.pth"))) {
        $tareas += @{ n = "TRAIN particion $sp"
                      a = @('train.py','hexcnn','s','mse','--splitseed',"$sp",
                            '--seed','901','--rotseed','7','--tag',$tag) }
    } else { Write-Host "  ya entrenado, se salta: $tag" }
    if (-not (Test-Path (Join-Path $raiz "runs\$run\TOTAL\eval_total_metrics.json"))) {
        $tareas += @{ n = "EVAL  particion $sp"
                      a = @('eval_total.py',$run,'--events','750000') }
    } else { Write-Host "  ya evaluado, se salta: $tag" }
}
if ($tareas.Count -eq 0) { Write-Host "Nada pendiente: las cinco particiones ya estan."; Pop-Location; exit 0 }
Write-Host "Pendientes: $($tareas.Count) de 10"

function Log($msg) {
    $linea = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $linea
    Add-Content -Path $log -Value $linea -Encoding utf8
}

$t0 = Get-Date
Log "=== VALIDACION CRUZADA CON COBERTURA COMPLETA: 5 particiones ==="
Log "commit: $commit"
if ($sucio) { Log "AVISO: arbol SUCIO, el run no sera reproducible desde $commit" }
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
foreach ($r in $resumen) { Log ("  {0,-34} {1,-20} {2,6} min" -f $r.Tarea, $r.Estado, $r.Minutos) }
Log ""
Log "Al terminar, la media y sd de las cinco sustituyen al 56,56 +/- 2,32 del"
Log "protocolo viejo. Comprobar en el log que cada train declara COBERTURA 149/149"
Log "y que cada eval imprime la particion que le toca."
Pop-Location
