# =====================================================================
#  noche_multidead_near.ps1 — la comparacion que de verdad decide near
# =====================================================================
#  HIPOTESIS: 'near' sigue ganando en los tres regimenes de fallo cuando
#  ambos modelos se entrenan con cobertura completa.
#
#  POR QUE ESTA TANDA. La del 18/08 entreno los tres 'near' con el
#  protocolo correcto, pero los evaluo con eval_total, que mide UN SOLO
#  canal apagado (k=1). En k=1 'near' pierde por construccion: se entrena
#  para 1-4 canales, asi que reparte capacidad en casos que k=1 no premia.
#  La afirmacion del TFM -"entrenar en el regimen intermedio generaliza a
#  los extremos"- se mide con eval_multidead, y no estaba hecho.
#
#  Los modelos YA ESTAN ENTRENADOS: esto es solo evaluacion. Lo caro (12 h
#  de GPU) no se repite.
#
#  QUE SE COMPARA. Pareado y con banda, por primera vez:
#     3 x near      (dead 1-4, deadmode near)  cobertura completa
#     3 x referencia (dead 1, cluster)         cobertura completa
#  mismos rotseed en ambos grupos (11/13/17), asi que cada near tiene su
#  referencia con el mismo orden de modulos.
#
#  OJO CON LA LECTURA. La evidencia vieja (13/08, 40 modulos, n=1) daba
#  ventaja de +0,72 en contiguo, +2,35 en el propio y +1,86 en disperso.
#  Con la sd que hoy conocemos (~0,6 en p90) el +0,72 es ruido. Lo que hay
#  que ver aqui es si el +2,35 aguanta con n=3 y protocolo bueno.
#
#  Uso:  .\tandas\noche_multidead_near.ps1        (~8 h)
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = Join-Path $PSScriptRoot '..\..\..\..\..\envs\tfm\python.exe'
if (-not (Test-Path $py)) { $py = 'E:\envs\tfm\python.exe' }
$raiz = Split-Path $PSScriptRoot -Parent
$log  = Join-Path $PSScriptRoot ("logs\noche_mdnear_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

Push-Location $raiz
$commit = (git rev-parse --short HEAD).Trim()
$sucio  = (git status --porcelain)

# Se SALTA lo que ya este calculado, asi que la tanda se puede relanzar tantas
# veces como haga falta sin repetir trabajo ni tener que editar nada a mano.
$tareas = @()
foreach ($m in @(
        @{ etq = 'near';       run = 'imputer_hexcnn_s_mse_dead1-4_near_banda' },
        @{ etq = 'referencia'; run = 'imputer_hexcnn_s_mse_banda' })) {
    foreach ($rot in 11, 13, 17) {
        $run  = "$($m.run)$rot"
        $hecho = Join-Path $raiz "runs\$run\MULTIDEAD_3MODOS\eval_multidead_metrics.json"
        if (Test-Path $hecho) {
            Write-Host "  ya hecho, se salta: $($m.etq) rot$rot"
            continue
        }
        $tareas += @{ n = "MULTIDEAD $($m.etq) rot$rot"
                      a = @('eval_multidead.py',$run,'--out','MULTIDEAD_3MODOS','--seeds','20') }
    }
}
if ($tareas.Count -eq 0) { Write-Host "Nada pendiente: los seis evals ya estan."; Pop-Location; exit 0 }
Write-Host "Pendientes: $($tareas.Count) de 6"

function Log($msg) {
    $linea = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $linea
    Add-Content -Path $log -Value $linea -Encoding utf8
}

$t0 = Get-Date
Log "=== MULTIDEAD: near vs referencia, ambos con cobertura completa ==="
Log "commit: $commit"
if ($sucio) { Log "AVISO: arbol SUCIO" }
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
foreach ($r in $resumen) { Log ("  {0,-40} {1,-20} {2,6} min" -f $r.Tarea, $r.Estado, $r.Minutos) }
Log ""
Log "Comparar recov_p90 por regimen (cluster / near / scatter), near contra"
Log "referencia, pareando por rotseed. Con n=3 en cada grupo ya hay banda:"
Log "una ventaja menor que 2 sd NO es concluyente."
Pop-Location
