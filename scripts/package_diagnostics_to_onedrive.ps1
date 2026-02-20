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
  Write-Warning "runnerDiag missing: $runnerDiag"
}

$DiagDir = [System.IO.Path]::GetFullPath($DiagDir)
$null = New-Item -Path $DiagDir -ItemType Directory -Force
Write-Host "runnerDiag=$runnerDiag"
Write-Host "diagDir=$DiagDir"

$zipName = "diag_${runId}_${attempt}_${timestamp}.zip"
$zipPath = Join-Path $DiagDir $zipName
$zipSources = @()
if (Test-Path $runnerDiag) {
  $zipSources += (Join-Path $runnerDiag '*')
}

$workspaceTemp = Join-Path $ws 'temp'
$oneTapReport = Join-Path $workspaceTemp 'ONE_TAP_ERROR_REPORT.txt'
if (Test-Path $oneTapReport) {
  $zipSources += $oneTapReport
}

$consoleLogCandidates = @(
  (Join-Path $workspaceTemp 'run_all_local_then_copy_console.log'),
  (Join-Path $env:RUNNER_TEMP 'run_all_local_then_copy_console.log')
) | Where-Object { $_ -and (Test-Path $_) }
$zipSources += $consoleLogCandidates

if ($zipSources.Count -eq 0) {
  $placeholder = Join-Path $workspaceTemp 'diag_placeholder.txt'
  $null = New-Item -Path $workspaceTemp -ItemType Directory -Force
  @(
    '[WARN] runner _diag and known logs were unavailable; packaging placeholder only.',
    ('timestamp={0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'))
  ) | Set-Content -Path $placeholder -Encoding UTF8
  $zipSources += $placeholder
}

Compress-Archive -Path $zipSources -DestinationPath $zipPath -Force
if (!(Test-Path $zipPath)) {
  throw "zip not created: $zipPath"
}
Write-Host "zipPath=$zipPath"

Get-ChildItem -Path $DiagDir | Sort-Object LastWriteTime -Descending
