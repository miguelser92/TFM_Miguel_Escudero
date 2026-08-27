# =====================================================================
#  noche_cv_near.ps1 - cerrar la barra de particion + fallo realista
# =====================================================================
#  1 y 2) DOS PARTICIONES MAS DE VALIDACION CRUZADA (D y E).
#     Con tres particiones la desviacion entre ellas (1.55 puntos) sigue
#     siendo un estimador pobre, y ese numero es justo el que va a la
#     memoria como barra de error del valor absoluto. Con cinco queda
#     mucho mas asentado.
#
#  3) REGIMEN DE FALLO CERCANO. Al etiquetar los modulos averiados reales
#     aparecen tres casos: un sensor suelto, un grupo pegado, y dos o tres
#     separados por una o dos posiciones. Ese tercero no estaba cubierto:
#     'cluster' los pone pegados y 'scatter' los reparte por todo el
#     detector. El modo 'near' queda en medio (1.8 saltos de media frente
#     a 1.2 y 17.6) y es el que corresponde a lo observado.
#     Se entrena con el y se evalua en los TRES regimenes.
#
#  Uso:  .\noche_cv_near.ps1
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = 'E:\envs\tfm\python.exe'
$log = Join-Path $PSScriptRoot ("noche_cvnear_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

$tareas = @(
    @{ n = '1a) TRAIN CV particion D'
       a = @('train.py','hexcnn','s','mse','--splitseed','45','--seed','403','--tag','cvD') },
    @{ n = '1b) EVAL  CV particion D'
       a = @('eval_total.py','imputer_hexcnn_s_mse_cvD','--events','750000') },

    @{ n = '2a) TRAIN CV particion E'
       a = @('train.py','hexcnn','s','mse','--splitseed','46','--seed','404','--tag','cvE') },
    @{ n = '2b) EVAL  CV particion E'
       a = @('eval_total.py','imputer_hexcnn_s_mse_cvE','--events','750000') },

    @{ n = '3a) TRAIN fallo cercano (dead 1-4, modo near)'
       a = @('train.py','hexcnn','s','mse','--dead','1-4','--deadmode','near',
             '--seed','405','--tag','near14') },
    @{ n = '3b) EVAL  multidead en los TRES regimenes'
       a = @('eval_multidead.py','imputer_hexcnn_s_mse_dead1-4_near_near14','--seeds','20') }
)

function Log($msg) {
    $linea = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $linea
    Add-Content -Path $log -Value $linea -Encoding utf8
}

$t0 = Get-Date
Log "=== CV (particiones D y E) + REGIMEN DE FALLO CERCANO ==="
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
Log "Lectura 1: con cinco particiones la barra de error del valor absoluto"
Log "           queda asentada y es la que debe ir en la memoria."
Log "Lectura 2: si el modelo entrenado en 'near' gana a los otros dos en su"
Log "           propio regimen, conviene entrenar con el modo de fallo que"
Log "           realmente se observa en los modulos averiados."
