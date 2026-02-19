param(
  [Parameter(Mandatory = $true)]
  [string]$DiagDir
)

$ErrorActionPreference = 'Stop'

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runId = if ($env:GITHUB_RUN_ID) { $env:GITHUB_RUN_ID } else { 'local' }
$attempt = if ($env:GITHUB_RUN_ATTEMPT) { $env:GITHUB_RUN_ATTEMPT } else { '0' }

$ws = $env:GITHUB_WORKSPACE
if ([string]::IsNullOrWhiteSpace($ws)) {
  throw 'GITHUB_WORKSPACE is not set.'
}

$runnerRoot = (Resolve-Path (Join-Path $ws '..\..\..')).Path
$runnerDiag = Join-Path $runnerRoot '_diag'
if (!(Test-Path $runnerDiag)) {
  throw "runnerDiag missing: $runnerDiag"
}

$DiagDir = [System.IO.Path]::GetFullPath($DiagDir)
$null = New-Item -Path $DiagDir -ItemType Directory -Force
Write-Host "runnerDiag=$runnerDiag"
Write-Host "diagDir=$DiagDir"

$zipName = "diag_${runId}_${attempt}_${timestamp}.zip"
$zipPath = Join-Path $DiagDir $zipName
Compress-Archive -Path (Join-Path $runnerDiag '*') -DestinationPath $zipPath -Force
if (!(Test-Path $zipPath)) {
  throw "zip not created: $zipPath"
}
Write-Host "zipPath=$zipPath"

Get-ChildItem -Path $DiagDir | Sort-Object LastWriteTime -Descending
