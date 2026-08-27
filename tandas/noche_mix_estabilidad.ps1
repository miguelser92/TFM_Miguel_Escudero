# =====================================================================
#  noche_mix_estabilidad.ps1 - mezclar modulos estabiliza el peor canal?
# =====================================================================
#  HIPOTESIS: --mix reduce la DISPERSION del peor canal, aunque el 10/08
#  ya se midiera que no mejora la media.
#
#  POR QUE. El 17/08 se identifico la causa de que el peor canal varie
#  +/-4,5: con MIX_MODULES=1 cada epoca es UN modulo, los modulos difieren
#  mucho entre si (sd 7,9 puntos) y el orden decide en que modulo cae la
#  epoca que acaba seleccionada (110/119/129/133 en las cuatro replicas).
#  El modelo final arrastra un sesgo del ultimo modulo visto, y el cosine
#  lo agrava porque apaga el LR justo al final.
#
#  --mix 4 ataca exactamente eso: cada epoca mezcla 4 modulos, asi que
#  ningun modulo domina el gradiente final.
#
#  OJO CON LA COMPARACION. Se mide sobre el modelo de REFERENCIA (dead 1,
#  cluster), no sobre near, porque la banda con la que hay que comparar es
#  la de referencia: 42,29 +/- 4,45 (n=4). Con near no habria contra que
#  comparar.
#
#  LO QUE SE MIRA ES LA sd, NO LA MEDIA. Que la media no mejore ya se sabe
#  y no invalida nada. Si la sd baja de 4,45 a ~1, se recupera la capacidad
#  de decir algo sobre robustez. Si no baja, se reporta que la metrica es
#  intrinsecamente ruidosa y se cierra el tema.
#
#  Coste constante: --mix reparte MAX_EVENTS entre los modulos, asi que el
#  presupuesto sigue siendo 15.943.000 muestras (verificado).
#
#  Uso:  .\tandas\noche_mix_estabilidad.ps1        (~12 h)
#        Lanzar DESPUES de noche_near_completo.ps1.
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = Join-Path $PSScriptRoot '..\..\..\..\..\envs\tfm\python.exe'
if (-not (Test-Path $py)) { $py = 'E:\envs\tfm\python.exe' }
$raiz = Split-Path $PSScriptRoot -Parent
$log  = Join-Path $PSScriptRoot ("logs\noche_mix_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

Push-Location $raiz
$commit = (git rev-parse --short HEAD).Trim()
$sucio  = (git status --porcelain)

# mismos rotseed que la banda de referencia, para que la comparacion sea pareada
$combis = @(
    @{ seed = 831; rot = 11 },
    @{ seed = 832; rot = 13 },
    @{ seed = 833; rot = 17 }
)

$tareas = @()
foreach ($c in $combis) {
    $tag = "banda$($c.rot)"
    # carpeta resultante: imputer_hexcnn_s_mse_mix4_<tag>  (verificado)
    $tareas += @{ n = "TRAIN mix4 (seed $($c.seed), rot $($c.rot))"
                  a = @('train.py','hexcnn','s','mse','--mix','4',
                        '--seed',"$($c.seed)",'--rotseed',"$($c.rot)",'--tag',$tag) }
    $tareas += @{ n = "EVAL  mix4 $tag"
                  a = @('eval_total.py',"imputer_hexcnn_s_mse_mix4_$tag",'--events','750000') }
}

function Log($msg) {
    $linea = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $linea
    Add-Content -Path $log -Value $linea -Encoding utf8
}

$t0 = Get-Date
Log "=== MIX 4: baja la dispersion del peor canal? ==="
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
foreach ($r in $resumen) { Log ("  {0,-56} {1,-20} {2,6} min" -f $r.Tarea, $r.Estado, $r.Minutos) }
Log ""
Log "Comparar la SD del peor canal de estos tres contra 4,45 (referencia sin mix)."
Log "La media NO es el criterio: ya se sabe que --mix no la mejora (-0,34 el 10/08)."
Log "Con n=3 la sd es un estimador pobre: solo vale si la diferencia es grande."
Pop-Location
