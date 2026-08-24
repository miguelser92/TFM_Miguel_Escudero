# =====================================================================
#  noche_espectro_resolucion.ps1 — las dos metricas fisicas, con banda
# =====================================================================
#  HIPOTESIS: la conservacion del espectro y la recuperacion de resolucion
#  se mantienen con el protocolo corregido, y por primera vez con barra de
#  error en lugar de un unico run.
#
#  POR QUE. Las dos cifras que van al paper estan medidas SOLO sobre el
#  modelo viejo de 40 modulos y con n=1:
#     - conservacion del espectro 82.9%  (runs/imputer_hexcnn_s_mse/ESPECTRO)
#     - recuperacion de blur 48.8%       (reports/resolution_def.json)
#  El espectro por zonas es ademas la metrica que los tutores marcaron como
#  la clave del trabajo. Los tres modelos de referencia con cobertura
#  completa (banda11/13/17) no la tienen calculada.
#
#  Esto es SOLO EVALUACION: no reentrena nada, los modelos ya existen.
#
#  HOMOGENEIDAD (E06 del registro de errores). Los parametros estan
#  elegidos para que los resultados sean comparables con lo ya publicado,
#  no por comodidad:
#     - espectro   : TODOS los eventos, igual que las campanas ESPECTRO
#                    existentes ("max_events_per_file": "all")
#     - resolucion : 500.000 eventos, los mismos que resolution_def.json y
#                    resolution_cov149.json. Con 700k el MISMO modelo da
#                    48.35 en vez de 48.82: el tamano de muestra mueve la
#                    cifra medio punto, asi que hay que fijarlo.
#
#  NO PISA NADA. Cada paso comprueba si su salida ya existe y se la salta,
#  asi que la tanda se puede relanzar sin repetir trabajo. eval_resolution
#  escribe en reports/resolution_<tag>.json y por eso cada modelo lleva su
#  propio tag: sin tags distintos el tercero machacaria a los dos primeros.
#
#  Uso:  .\tandas\noche_espectro_resolucion.ps1        (~4-6 h estimadas)
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = Join-Path $PSScriptRoot '..\..\..\..\..\envs\tfm\python.exe'
if (-not (Test-Path $py)) { $py = 'E:\envs\tfm\python.exe' }
$raiz = Split-Path $PSScriptRoot -Parent
$log  = Join-Path $PSScriptRoot ("logs\noche_espres_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

Push-Location $raiz
$commit = (git rev-parse --short HEAD).Trim()
$sucio  = (git status --porcelain)

$modelos = @('imputer_hexcnn_s_mse_banda11',
             'imputer_hexcnn_s_mse_banda13',
             'imputer_hexcnn_s_mse_banda17')

$tareas = @()

# --- 1) ESPECTRO primero: es la metrica clave del trabajo ---
foreach ($m in $modelos) {
    $hecho = Join-Path $raiz "runs\$m\ESPECTRO\eval_espectro_metrics.json"
    if (Test-Path $hecho) { Write-Host "  ya hecho, se salta: ESPECTRO $m"; continue }
    $tareas += @{ n = "ESPECTRO $m"
                  a = @('eval_espectro.py', $m, '--out', 'ESPECTRO') }
}

# --- 2) RESOLUCION: tag propio por modelo, si no se pisan entre ellos ---
foreach ($m in $modelos) {
    $tag = $m -replace '^imputer_hexcnn_s_mse_', ''      # banda11 / banda13 / banda17
    $hecho = Join-Path $raiz "reports\resolution_$tag.json"
    if (Test-Path $hecho) { Write-Host "  ya hecho, se salta: RESOLUCION $tag"; continue }
    $tareas += @{ n = "RESOLUCION $tag"
                  a = @('eval_resolution.py', '--run', $m,
                        '--max-events', '500000', '--tag', $tag) }
}

if ($tareas.Count -eq 0) {
    Write-Host "Nada pendiente: las seis evaluaciones ya estan."
    Pop-Location; exit 0
}
Write-Host "Pendientes: $($tareas.Count) de 6"

function Log($msg) {
    $linea = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $linea
    Add-Content -Path $log -Value $linea -Encoding utf8
}

$t0 = Get-Date
Log "=== ESPECTRO Y RESOLUCION sobre los 3 modelos de referencia (149 modulos) ==="
Log "commit: $commit"
if ($sucio) { Log "AVISO: arbol SUCIO, el run no sera reproducible desde $commit" }
Log "log: $log"
Log ""

$resumen = @()
foreach ($t in $tareas) {
    Log ("---- {0} ----" -f $t.n)
    Log ("  {0} {1}" -f $py, ($t.a -join ' '))
    $ini = Get-Date
    # El rojo de consola es cosmetico: PowerShell envuelve stderr y wandb
    # escribe todo por ahi. Para saber si fallo algo, buscar 'Traceback'.
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
foreach ($r in $resumen) { Log ("  {0,-46} {1,-20} {2,6} min" -f $r.Tarea, $r.Estado, $r.Minutos) }
Log ""
Log "Al terminar habra, por primera vez con banda de error:"
Log "  - conservacion del espectro por zonas (nucleo / medio / borde)"
Log "  - recuperacion de blur y de sesgo en mm"
Log "Comparar contra 82.9% y 48.8%, que son los valores de un solo run con 40 modulos."
Pop-Location
