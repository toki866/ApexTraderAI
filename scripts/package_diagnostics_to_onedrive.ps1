param(
  [Parameter(Mandatory = $true)]
  [string]$DiagDir
)

$ErrorActionPreference = 'Stop'

function Add-OneTapWarning {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Message
  )

  try {
    $workspace = $env:GITHUB_WORKSPACE
    if ([string]::IsNullOrWhiteSpace($workspace)) { return }
    $workspaceTemp = Join-Path $workspace 'temp'
    $null = New-Item -Path $workspaceTemp -ItemType Directory -Force
    $reportPath = Join-Path $workspaceTemp 'ONE_TAP_ERROR_REPORT.txt'
    Add-Content -Path $reportPath -Encoding UTF8 -Value ("[WARN] publish skipped: {0}" -f $Message)
  } catch {
    Write-Warning ("Unable to append ONE_TAP warning: {0}" -f $_.Exception.Message)
  }
}

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
$zipSources = New-Object System.Collections.Generic.List[string]
$sourceItems = @{}
if (Test-Path $runnerDiag) {
  $runnerDiagWildcard = Join-Path $runnerDiag '*'
  $sourceItems[$runnerDiagWildcard] = $true
}

$workspaceTemp = Join-Path $ws 'temp'
$oneTapReport = Join-Path $workspaceTemp 'ONE_TAP_ERROR_REPORT.txt'
if (Test-Path $oneTapReport) {
  $sourceItems[$oneTapReport] = $true
}

$consoleLogCandidates = @((Join-Path $workspaceTemp 'run_all_local_then_copy_console.log'))
if (-not [string]::IsNullOrWhiteSpace($env:RUNNER_TEMP)) {
  $consoleLogCandidates += (Join-Path $env:RUNNER_TEMP 'run_all_local_then_copy_console.log')
}
$consoleLogCandidates = $consoleLogCandidates | Where-Object { $_ -and (Test-Path $_) }
foreach ($candidate in $consoleLogCandidates) {
  if (-not [string]::IsNullOrWhiteSpace($candidate)) {
    $sourceItems[$candidate] = $true
  }
}

foreach ($key in $sourceItems.Keys) {
  if (-not [string]::IsNullOrWhiteSpace($key) -and (Test-Path $key)) {
    $zipSources.Add($key)
  }
}

if ($zipSources.Count -eq 0) {
  Add-OneTapWarning -Message 'runner _diag and known logs were unavailable; packaging placeholder only.'
  $placeholder = Join-Path $workspaceTemp 'diag_placeholder.txt'
  $null = New-Item -Path $workspaceTemp -ItemType Directory -Force
  @(
    '[WARN] runner _diag and known logs were unavailable; packaging placeholder only.',
    ('timestamp={0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'))
  ) | Set-Content -Path $placeholder -Encoding UTF8
  $zipSources += $placeholder
}

try {
  Compress-Archive -Path $zipSources -DestinationPath $zipPath -Force -ErrorAction Stop
} catch {
  Write-Warning "Primary diagnostics archive failed: $($_.Exception.Message)"
  Add-OneTapWarning -Message ("Primary diagnostics archive failed: {0}" -f $_.Exception.Message)
  $placeholder = Join-Path $workspaceTemp 'diag_zip_fallback_notice.txt'
  @(
    '[WARN] Primary diagnostics archive failed; created fallback package only.',
    ('error={0}' -f $_.Exception.Message),
    ('timestamp={0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'))
  ) | Set-Content -Path $placeholder -Encoding UTF8

  $fallbackSources = @($placeholder)
  if (Test-Path $oneTapReport) {
    $fallbackSources += $oneTapReport
  }
  $fallbackSources += $consoleLogCandidates
  $fallbackSources = $fallbackSources | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

  if ($fallbackSources.Count -eq 0) {
    throw 'Fallback diagnostics archive has no valid sources.'
  }

  try {
    Compress-Archive -Path $fallbackSources -DestinationPath $zipPath -Force -ErrorAction Stop
  } catch {
    throw "Fallback diagnostics archive failed: $($_.Exception.Message)"
  }
}

if (!(Test-Path $zipPath)) {
  Add-OneTapWarning -Message ("zip not created: {0}" -f $zipPath)
  throw "zip not created: $zipPath"
}
Write-Host "zipPath=$zipPath"

Get-ChildItem -Path $DiagDir | Sort-Object LastWriteTime -Descending
