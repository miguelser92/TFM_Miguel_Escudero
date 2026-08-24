# =====================================================================
#  compilar.ps1 - compila el TFM dejando la carpeta limpia
# =====================================================================
#  LaTeX necesita generar .aux, .bbl, .out y compania: son las notas que
#  se deja entre pasadas para resolver las referencias cruzadas y la
#  bibliografia. Overleaf genera exactamente los mismos ficheros, solo
#  que en su servidor y sin ensenartelos.
#
#  Este script los manda todos a .build\ y deja en la carpeta unicamente
#  el PDF. Si algo falla, el log esta en .build\<nombre>.log
#
#  Uso:  .\compilar.ps1              -> compila main_WIP
#        .\compilar.ps1 main         -> compila main
#        .\compilar.ps1 -Limpiar     -> borra .build y los PDF
# =====================================================================

param(
    [string]$Documento = 'main_WIP',
    [switch]$Limpiar
)

$raiz  = $PSScriptRoot
$build = Join-Path $raiz '.build'

if ($Limpiar) {
    if (Test-Path $build) { Remove-Item $build -Recurse -Force }
    Get-ChildItem $raiz -Filter '*.pdf' | Where-Object { $_.Name -like 'main*' } |
        Remove-Item -Force
    Write-Host "Limpiado: .build y los PDF de main*"
    return
}

if (-not (Test-Path (Join-Path $raiz "$Documento.tex"))) { throw "No existe $Documento.tex" }
New-Item -ItemType Directory -Force -Path $build | Out-Null

Push-Location $raiz
Write-Host "Compilando $Documento ..."

# La configuracion (auxiliares a .build, PDF en la carpeta) vive en .latexmkrc,
# que latexmk lee solo. Asi el editor y este script hacen lo mismo.
# OJO: se le pasa el NOMBRE del .tex, no la ruta absoluta, o el aux_dir se
# descoloca. Y nada de 2>&1: PowerShell 5.1 envuelve stderr en ErrorRecord y
# aborta aunque latexmk devuelva 0.
& latexmk -pdf "$Documento.tex" | Out-Null
$code = $LASTEXITCODE

$log = Join-Path $build "$Documento.log"
$errores = 0
if (Test-Path $log) {
    $errores = (Select-String -Path $log -Pattern '^!' -AllMatches).Count
    $citas   = (Select-String -Path $log -Pattern 'Citation .* undefined').Count
    $refs    = (Select-String -Path $log -Pattern 'Reference .* undefined').Count
    $paginas = (Select-String -Path $log -Pattern 'Output written on .* \((\d+) pages' |
                Select-Object -First 1).Matches.Groups[1].Value
}

if ($code -eq 0 -and $errores -eq 0) {
    # el PDF ya queda en la carpeta: .latexmkrc pone out_dir en '.'
    Write-Host "  OK  $Documento.pdf  ($paginas paginas)"
    if ($citas) { Write-Host "  AVISO: $citas citas sin resolver" -ForegroundColor Yellow }
    if ($refs)  { Write-Host "  AVISO: $refs referencias sin resolver" -ForegroundColor Yellow }
} else {
    Write-Host "  FALLO. Primeros errores:" -ForegroundColor Red
    Select-String -Path $log -Pattern '^!' | Select-Object -First 5 |
        ForEach-Object { Write-Host "    $($_.Line)" -ForegroundColor Red }
    Write-Host "  log completo: .build\$Documento.log"
}
Pop-Location
