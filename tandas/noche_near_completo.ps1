# =====================================================================
#  noche_near_completo.ps1 — el modelo recomendado, con el protocolo bueno
# =====================================================================
#  HIPOTESIS: 'near' sigue siendo el mejor modelo general con cobertura
#  completa, y su banda de error permite reportarlo en la memoria.
#
#  POR QUE. El TFM propone 'near' (--dead 1-4 --deadmode near) como modelo
#  de produccion: entrenar en el regimen intermedio generaliza a los
#  extremos. Pero los dos runs que existen (near_prod y near_near14) estan
#  entrenados con 40 de los 149 modulos, o sea con el protocolo que el
#  propio texto declara defectuoso. Eso es indefendible en una lectura
#  atenta: "dice que hay que usar los 149, y su modelo recomendado usa 40".
#
#  Ademas el 17/08 se vio que una cifra de un solo run no vale: la barra
#  real del peor canal es +/-4,5. Asi que tres replicas, y con variacion
#  COMPLETA (semilla y orden juntos), como en noche_banda.
#
#  LANZAR ESTA ANTES QUE noche_mix_estabilidad: esta tiene valor seguro
#  (arregla el modelo que va en la memoria), la otra es exploratoria.
#
#  Uso:  .\tandas\noche_near_completo.ps1        (~11,5 h)
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = Join-Path $PSScriptRoot '..\..\..\..\..\envs\tfm\python.exe'
if (-not (Test-Path $py)) { $py = 'E:\envs\tfm\python.exe' }
$raiz = Split-Path $PSScriptRoot -Parent
$log  = Join-Path $PSScriptRoot ("logs\noche_near_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

Push-Location $raiz
$commit = (git rev-parse --short HEAD).Trim()
$sucio  = (git status --porcelain)

# semilla y orden cambian juntos: es lo que mide la variabilidad real
$combis = @(
    @{ seed = 821; rot = 11 },
    @{ seed = 822; rot = 13 },
    @{ seed = 823; rot = 17 }
)

$tareas = @()
foreach ($c in $combis) {
    $tag = "banda$($c.rot)"
    # carpeta resultante: imputer_hexcnn_s_mse_dead1-4_near_<tag>  (verificado)
    $tareas += @{ n = "TRAIN near cobertura completa (seed $($c.seed), rot $($c.rot))"
                  a = @('train.py','hexcnn','s','mse','--dead','1-4','--deadmode','near',
                        '--seed',"$($c.seed)",'--rotseed',"$($c.rot)",'--tag',$tag) }
    $tareas += @{ n = "EVAL  near $tag"
                  a = @('eval_total.py',"imputer_hexcnn_s_mse_dead1-4_near_$tag",'--events','750000') }
}

function Log($msg) {
    $linea = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $linea
    Add-Content -Path $log -Value $linea -Encoding utf8
}

$t0 = Get-Date
Log "=== NEAR CON COBERTURA COMPLETA: 3 replicas independientes ==="
Log "commit: $commit"
if ($sucio) { Log "AVISO: arbol SUCIO, el run no sera reproducible desde $commit" }
Log "log: $log"
Log ""

$resumen = @()
foreach ($t in $tareas) {
    Log ("---- {0} ----" -f $t.n)
    Log ("  {0} {1}" -f $py, ($t.a -join ' '))
    $ini = Get-Date
    # El rojo de la consola es cosmetico (PowerShell envuelve stderr y wandb
    # escribe todo por ahi). Para saber si fallo algo: buscar 'Traceback'.
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
Log "Al terminar: verificar 'COBERTURA: 149/149' en los tres, y comparar la media"
Log "de los tres con la referencia (58,04 +/- 0,37). Si near mantiene su ventaja"
Log "con el protocolo bueno, es la recomendacion del TFM y ya se puede escribir."
Pop-Location
