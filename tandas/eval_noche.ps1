# =====================================================================
#  eval_noche.ps1 - tanda de evaluaciones desatendida
# =====================================================================
#  Lanza en orden las evaluaciones pendientes y deja un log con tiempos
#  y codigos de salida. NO se detiene si una falla: las demas siguen,
#  que es lo que interesa para aprovechar la noche.
#
#  Los parametros estan elegidos para que los resultados sean COMPARABLES
#  con las campanas ya existentes (misma base de eventos y de semillas):
#    - multidead : 200k eventos y 20 semillas, como los 7 ya evaluados
#    - espectro  : todos los eventos, como las 12 campanas ESPECTRO
#    - baselines : 700k, como reports/baselines_300k.json
#    - resolucion: 500k, como reports/resolution_def.json
#    - eval_total: 600k, para homogeneizar los MLP y HexGNN m/l, que
#                  estaban a 200k (TODO(4) del capitulo)
#
#  Uso:  .\eval_noche.ps1
# =====================================================================

$ErrorActionPreference = 'Continue'
$py  = 'E:\envs\tfm\python.exe'
$log = Join-Path $PSScriptRoot ("eval_noche_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmm'))
$cov = 'imputer_hexcnn_s_mse_cov149'

# Orden deliberado: primero lo que mas aporta, por si no diera tiempo a todo.
$tareas = @(
    @{ n = '1/5 multidead cov149 (200k, 20 semillas)'
       a = @('eval_multidead.py', $cov, '--seeds', '20') },

    @{ n = '2/5 eval_total homogeneo de MLP y HexGNN m/l (600k)'
       a = @('eval_total.py', 'imputer_deepmlp_mse', 'imputer_resmlp_mse',
             'imputer_hexcnn_m_mse', 'imputer_hexcnn_l_mse',
             '--out', 'TOTAL600', '--events', '600000') },

    @{ n = '3/5 espectro cov149 (todos los eventos)'
       a = @('eval_espectro.py', $cov) },

    @{ n = '4/5 resolucion cov149 (500k)'
       a = @('eval_resolution.py', '--run', $cov, '--max-events', '500000',
             '--tag', 'cov149') },

    @{ n = '5/5 baselines cov149 (700k)'
       a = @('eval_baselines.py', '--run', $cov, '--max-events', '700000',
             '--tag', 'cov149') }
)

function Log($msg) {
    $linea = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $linea
    Add-Content -Path $log -Value $linea -Encoding utf8
}

$t0 = Get-Date
Log "=== TANDA DE EVALUACIONES ==="
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
Log "=== RESUMEN ($tot h en total) ==="
foreach ($r in $resumen) { Log ("  {0,-52} {1,-22} {2,6} min" -f $r.Tarea, $r.Estado, $r.Minutos) }
Log ""
Log "Manana: pasale el log a Claude para que rellene Excel, bitacora y memoria."
