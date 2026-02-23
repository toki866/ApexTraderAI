param(
  [Parameter(Mandatory = $true)]
  [string]$DiagDir,
  [string]$RunDir,
  [string]$OutputRoot
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
    Add-Content -Path $reportPath -Encoding UTF8 -Value ("[WARN] publish issue: {0}" -f $Message)
  } catch {
    Write-Warning ("Unable to append ONE_TAP warning: {0}" -f $_.Exception.Message)
  }
}

function Add-PublishSummaryWarning {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Message,
    [Parameter(Mandatory = $true)]
    [string]$SummaryPath
  )

  try {
    Add-Content -Path $SummaryPath -Encoding UTF8 -Value ("[WARN] publish issue: {0}" -f $Message)
  } catch {
    Write-Warning ("Unable to append summary warning: {0}" -f $_.Exception.Message)
  }

  Add-OneTapWarning -Message $Message
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
$summaryPath = Join-Path $DiagDir 'diag_publish_summary.txt'
@(
  ('timestamp={0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')),
  ('run_id={0}' -f $runId),
  ('attempt={0}' -f $attempt)
) | Set-Content -Path $summaryPath -Encoding UTF8

$outputRootSource = 'input_parameter'

if ([string]::IsNullOrWhiteSpace($RunDir) -or -not (Test-Path $RunDir)) {
  $runsRoot = 'C:\work\apex_work\runs'
  if (Test-Path $runsRoot) {
    $latestRun = Get-ChildItem -Path $runsRoot -Directory -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 1
    if ($latestRun) {
      $RunDir = $latestRun.FullName
    }
  }
}

if ([string]::IsNullOrWhiteSpace($OutputRoot) -or -not (Test-Path $OutputRoot)) {
  if (-not [string]::IsNullOrWhiteSpace($RunDir)) {
    $OutputRoot = Join-Path $RunDir 'output'
    $outputRootSource = 'run_dir_fallback'
  }
}

Add-Content -Path $summaryPath -Encoding UTF8 -Value ("[PUBLISH] run_dir={0}" -f $RunDir)
Add-Content -Path $summaryPath -Encoding UTF8 -Value ("[PUBLISH] output_root={0}" -f $OutputRoot)
Add-Content -Path $summaryPath -Encoding UTF8 -Value ("[PUBLISH] output_root_source={0}" -f $outputRootSource)

if ([string]::IsNullOrWhiteSpace($RunDir)) {
  Add-PublishSummaryWarning -SummaryPath $summaryPath -Message 'run_dir unresolved; falling back to diagnostics-only packaging.'
}
if ([string]::IsNullOrWhiteSpace($OutputRoot) -or -not (Test-Path $OutputRoot)) {
  Add-PublishSummaryWarning -SummaryPath $summaryPath -Message ("output_root unavailable: {0}" -f $OutputRoot)
}

$dPrimeModeNames = @('sim','live','display')
$dPrimeStateFound = $false
$dPrimeEmbeddingsFound = $false
if (-not [string]::IsNullOrWhiteSpace($OutputRoot) -and (Test-Path $OutputRoot)) {
  foreach ($modeName in $dPrimeModeNames) {
    $dPrimeBase = Join-Path $OutputRoot (Join-Path 'stepD_prime' $modeName)
    if (-not (Test-Path $dPrimeBase)) { continue }
    $stateHits = @(Get-ChildItem -Path $dPrimeBase -File -Filter 'stepDprime_state_*_*.csv' -ErrorAction SilentlyContinue)
    if ($stateHits.Count -gt 0) { $dPrimeStateFound = $true }
    $embRoot = Join-Path $dPrimeBase 'embeddings'
    if (Test-Path $embRoot) {
      $embHits = @(Get-ChildItem -Path $embRoot -File -Recurse -Filter 'stepDprime_*_embeddings*.csv' -ErrorAction SilentlyContinue)
      if ($embHits.Count -gt 0) { $dPrimeEmbeddingsFound = $true }
    }
  }
}
if (-not $dPrimeStateFound) {
  Add-PublishSummaryWarning -SummaryPath $summaryPath -Message "D' state CSV missing under output/stepD_prime/<mode>."
}
if (-not $dPrimeEmbeddingsFound) {
  Add-PublishSummaryWarning -SummaryPath $summaryPath -Message "D' embeddings CSV missing under output/stepD_prime/<mode>/embeddings."
}

Write-Host "runnerDiag=$runnerDiag"
Write-Host "diagDir=$DiagDir"

$zipName = "diag_${runId}_${attempt}_${timestamp}.zip"
$zipPath = Join-Path $DiagDir $zipName
$zipSources = New-Object System.Collections.Generic.List[string]
$sourceItems = @{}
if ($null -eq $sourceItems) { $sourceItems = @{} }
if (Test-Path $runnerDiag) {
  $runnerDiagWildcard = Join-Path $runnerDiag '*'
  $sourceItems[$runnerDiagWildcard] = $true
}

$workspaceTemp = Join-Path $ws 'temp'
$oneTapReport = Join-Path $workspaceTemp 'ONE_TAP_ERROR_REPORT.txt'
if (Test-Path $oneTapReport) {
  $sourceItems[$oneTapReport] = $true
}
if (Test-Path $summaryPath) {
  $sourceItems[$summaryPath] = $true
}

$evalDir = Join-Path $workspaceTemp 'eval'
if (Test-Path $evalDir) {
  $evalArtifacts = @(
    Get-ChildItem -Path $evalDir -File -Filter '*.md' -ErrorAction SilentlyContinue
    Get-ChildItem -Path $evalDir -File -Filter '*.json' -ErrorAction SilentlyContinue
    Get-ChildItem -Path $evalDir -File -Filter '*.txt' -ErrorAction SilentlyContinue
    Get-ChildItem -Path $evalDir -File -Filter '*.csv' -ErrorAction SilentlyContinue
    Get-ChildItem -Path $evalDir -File -Filter '*.png' -ErrorAction SilentlyContinue
  ) | Where-Object { $_ -ne $null }

  foreach ($item in $evalArtifacts) {
    if ($null -eq $sourceItems) { $sourceItems = @{} }
    $sourceItems[$item.FullName] = $true
  }
}

$consoleLogCandidates = @((Join-Path $workspaceTemp 'run_all_local_then_copy_console.log'))
if (-not [string]::IsNullOrWhiteSpace($env:RUNNER_TEMP)) {
  $consoleLogCandidates += (Join-Path $env:RUNNER_TEMP 'run_all_local_then_copy_console.log')
}
$consoleLogCandidates = $consoleLogCandidates | Where-Object { $_ -and (Test-Path $_) }
foreach ($candidate in $consoleLogCandidates) {
  if (-not [string]::IsNullOrWhiteSpace($candidate)) {
    if ($null -eq $sourceItems) { $sourceItems = @{} }
    $sourceItems[$candidate] = $true
  }
}

if ($null -eq $sourceItems) { $sourceItems = @{} }
foreach ($key in @($sourceItems.Keys)) {
  if (-not [string]::IsNullOrWhiteSpace($key) -and (Test-Path $key)) {
    $zipSources.Add($key)
  }
}

if ($zipSources.Count -eq 0) {
  Add-PublishSummaryWarning -SummaryPath $summaryPath -Message 'runner _diag and known logs were unavailable; packaging placeholder only.'
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
  Add-PublishSummaryWarning -SummaryPath $summaryPath -Message ("Primary diagnostics archive failed: {0}" -f $_.Exception.Message)
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
  Add-PublishSummaryWarning -SummaryPath $summaryPath -Message ("zip not created: {0}" -f $zipPath)
  throw "zip not created: $zipPath"
}
Write-Host "zipPath=$zipPath"

Get-ChildItem -Path $DiagDir | Sort-Object LastWriteTime -Descending
