# =====================================================================
#  noche_espectro_multidead.ps1 - se conserva el espectro con varios
#  canales muertos?
# =====================================================================
#  QUE HUECO CIERRA. Toda la seccion de fidelidad fisica del paper esta
#  medida con UN canal apagado, mientras el resultado mas fuerte del
#  trabajo (+24,6 puntos por entrenar multi-dead) y los modulos averiados
#  reales -hasta siete canales en datas016- son multi-canal. Falta saber
#  si la senal reconstruida sigue siendo fisicamente admisible cuando
#  falla un grupo entero.
#
#  Hay motivo para dudarlo: con un canal, los vecinos acotan bien la carga
#  que falta; con un grupo contiguo, la red tiene que inventar la de una
#  region entera y el error de carga total podria acumularse en vez de
#  cancelarse.
#
#  MODELO. Los tres 'near' con cobertura completa, que son el modelo de
#  produccion y el que mejor aguanta el multi-dead. n=3, asi que las
#  cifras salen con banda como el resto del capitulo.
#
#  COMPARABILIDAD. Los conjuntos de canales los construye build_dead_sets
#  de eval_multidead, con su misma semilla: el fallo que se apaga aqui para
#  un (sensor, k, regimen) dado es EXACTAMENTE el mismo que en las curvas
#  de recuperacion. Y los eventos son todos, igual que las campanas
#  ESPECTRO ya publicadas.
#
#  DISENO. k = 1..4 en el regimen 'near', que da la curva directamente
#  comparable con la de recuperacion; mas k=4 en 'cluster' y en 'scatter',
#  que son los dos extremos, para poder decir si el regimen influye.
#
#  Uso:  .\tandas\noche_espectro_multidead.ps1        (~3 h)
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = Join-Path $PSScriptRoot '..\..\..\..\..\envs\tfm\python.exe'
if (-not (Test-Path $py)) { $py = 'E:\envs\tfm\python.exe' }
$raiz = Split-Path $PSScriptRoot -Parent
$log  = Join-Path $PSScriptRoot ("logs\noche_espmd_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))

Push-Location $raiz
$commit = (git rev-parse --short HEAD).Trim()
$sucio  = (git status --porcelain)

$modelos = @('imputer_hexcnn_s_mse_dead1-4_near_banda11',
             'imputer_hexcnn_s_mse_dead1-4_near_banda13',
             'imputer_hexcnn_s_mse_dead1-4_near_banda17')

# (k, regimen): la curva completa en 'near' y los dos extremos en k=4
$configs = @(
    @{ k = 1; modo = 'near'    },
    @{ k = 2; modo = 'near'    },
    @{ k = 3; modo = 'near'    },
    @{ k = 4; modo = 'near'    },
    @{ k = 4; modo = 'cluster' },
    @{ k = 4; modo = 'scatter' }
)

$tareas = @()
foreach ($c in $configs) {
    foreach ($m in $modelos) {
        $camp = if ($c.k -eq 1) { "ESPECTRO_k1_$($c.modo)" } else { "ESPECTRO_k$($c.k)_$($c.modo)" }
        $hecho = Join-Path $raiz "runs\$m\$camp\eval_espectro_metrics.json"
        if (Test-Path $hecho) { Write-Host "  ya hecho, se salta: $camp $($m.Substring($m.Length-8))"; continue }
        $tareas += @{ n = "ESPECTRO k=$($c.k) $($c.modo) - $($m.Substring($m.Length-8))"
                      a = @('eval_espectro.py', $m, '--dead', "$($c.k)",
                            '--deadmode', $c.modo, '--out', $camp) }
    }
}
if ($tareas.Count -eq 0) { Write-Host "Nada pendiente: las 18 campanas ya estan."; Pop-Location; exit 0 }
Write-Host "Pendientes: $($tareas.Count) de 18"

function Log($msg) {
    $linea = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $linea
    Add-Content -Path $log -Value $linea -Encoding utf8
}

$t0 = Get-Date
Log "=== CONSERVACION DEL ESPECTRO CON VARIOS CANALES MUERTOS ==="
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
foreach ($r in $resumen) { Log ("  {0,-48} {1,-20} {2,6} min" -f $r.Tarea, $r.Estado, $r.Minutos) }
Log ""
Log "Al terminar: recuperacion espectral frente a k, y frente al regimen en k=4."
Log "Comparar el k=1 de aqui con el 84,7 +/- 0,9 % del modelo de referencia; ojo,"
Log "son modelos distintos (near vs referencia), asi que no tienen por que coincidir."
Pop-Location
