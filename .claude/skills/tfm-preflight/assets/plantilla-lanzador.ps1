# =====================================================================
#  <nombre>.ps1 — <una linea: que pregunta responde esta tanda>
# =====================================================================
#  HIPOTESIS: <que se pone a prueba, en una frase falsable>
#  Preflight: <nota del vault de Obsidian>
#
#  <Parrafo explicando POR QUE se lanza esto: que resultado previo lo
#   motiva y que decision depende del resultado. Si no se puede escribir
#   este parrafo, no hay experimento: hay un entrenamiento.>
#
#  Uso:  .\tandas\<nombre>.ps1        (~<n> h)
# =====================================================================

$ErrorActionPreference = 'Continue'   # una tarea que falla no aborta la tanda
$py  = Join-Path $PSScriptRoot '..\..\..\..\..\envs\tfm\python.exe'
if (-not (Test-Path $py)) { $py = 'E:\envs\tfm\python.exe' }
$raiz = Split-Path $PSScriptRoot -Parent
$log  = Join-Path $PSScriptRoot ("logs\<nombre>_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

# --- Trazabilidad: que codigo se ejecuta de verdad (E02) ---
Push-Location $raiz
$commit = (git rev-parse --short HEAD).Trim()
$sucio  = (git status --porcelain)

# --- Las tareas, en orden. Primero las que pueden invalidar a las siguientes ---
$tareas = @()
foreach ($x in <valores>) {
    $tag = "<familia>$x"
    $tareas += @{ n = "TRAIN <descripcion> ($x)"
                  a = @('train.py','hexcnn','s','mse','--seed',"$x",'--tag',$tag) }
    $tareas += @{ n = "EVAL  $tag"
                  a = @('eval_total.py',"imputer_hexcnn_s_mse_$tag",'--events','750000') }
}

function Log($msg) {
    $linea = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $linea
    Add-Content -Path $log -Value $linea -Encoding utf8
}

$t0 = Get-Date
Log "=== <TITULO> ==="
Log "commit: $commit"
if ($sucio) { Log "AVISO: arbol de trabajo SUCIO, el run no sera reproducible desde $commit" }
Log "log: $log"
Log ""

$resumen = @()
foreach ($t in $tareas) {
    Log ("---- {0} ----" -f $t.n)
    Log ("  {0} {1}" -f $py, ($t.a -join ' '))
    $ini = Get-Date
    # Nota: 2>&1 hace que PowerShell pinte stderr en ROJO (wandb escribe todo por
    # stderr). Es cosmetico; $LASTEXITCODE sigue siendo fiable. Para saber si hubo
    # fallo real, buscar 'Traceback' en el log, no mirar el color.
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
Log "<Que hacer con el resultado: la comparacion concreta que decide si la"
Log " hipotesis se sostiene, y contra que cifra del historico se compara.>"
Pop-Location

# Al terminar:
#  1) Verificar en el log la linea "COBERTURA: 149/149 modulos" de CADA train (E01).
#  2) Comprobar que no hay 'Traceback' en el log.
#  3) Fila en el Excel + entrada en la bitacora, con el mismo tag.
