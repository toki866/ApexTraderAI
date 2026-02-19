$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Join-Path $scriptRoot 'package_diagnostics_to_onedrive.ps1'

Write-Host "[diagnostics-wrapper] script_root=$scriptRoot"
Write-Host "[diagnostics-wrapper] target=$target"

if (-not (Test-Path $target)) {
  Write-Error "Diagnostics target script not found: $target"
  exit 1
}

& powershell -NoProfile -ExecutionPolicy Bypass -File $target
$rc = $LASTEXITCODE

if ($rc -ne 0) {
  Write-Error "Diagnostics target script failed with exit code: $rc"
  exit $rc
}

Write-Host '[diagnostics-wrapper] completed successfully.'
