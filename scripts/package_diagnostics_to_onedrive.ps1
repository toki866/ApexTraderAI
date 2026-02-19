param(
  [string]$DiagDir
)

$ErrorActionPreference = 'Stop'

function Resolve-OneDriveRoot {
  $candidates = @(
    $env:OneDrive,
    $env:OneDriveCommercial,
    $env:OneDriveConsumer,
    (Join-Path $env:USERPROFILE 'OneDrive')
  )

  foreach ($candidate in $candidates) {
    if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path $candidate)) {
      return (Get-Item $candidate).FullName
    }
  }

  throw 'Unable to resolve OneDrive root from OneDrive/OneDriveCommercial/OneDriveConsumer/USERPROFILE fallback.'
}

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runId = if ($env:GITHUB_RUN_ID) { $env:GITHUB_RUN_ID } else { 'local' }
$attempt = if ($env:GITHUB_RUN_ATTEMPT) { $env:GITHUB_RUN_ATTEMPT } else { '0' }

$workspace = if ($env:GITHUB_WORKSPACE) { $env:GITHUB_WORKSPACE } else { (Get-Location).Path }
$runnerTemp = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [System.IO.Path]::GetTempPath() }

$consoleLog = Join-Path $runnerTemp 'run_all_local_then_copy_console.log'
$workRoot = 'C:\work\apex_work\runs'
$resolvedRunLog = $null

if (Test-Path $consoleLog) {
  $match = Select-String -Path $consoleLog -Pattern 'log=(.+\.log)' -AllMatches | Select-Object -Last 1
  if ($match -and $match.Matches.Count -gt 0) {
    $candidate = $match.Matches[0].Groups[1].Value.Trim().Trim('"')
    if ($candidate -and (Test-Path $candidate)) {
      $resolvedRunLog = (Get-Item $candidate).FullName
      Write-Host "Resolved run log from console regex: $resolvedRunLog"
    }
  }
}

if (-not $resolvedRunLog -and (Test-Path $workRoot)) {
  $fallback = Get-ChildItem -Path (Join-Path $workRoot '*\logs\run_*.log') -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

  if ($fallback) {
    $resolvedRunLog = $fallback.FullName
    Write-Host "Resolved run log from fallback scan: $resolvedRunLog"
  }
}

if (-not $resolvedRunLog) {
  Write-Warning 'Run log could not be resolved from console log or fallback scan.'
}

if ([string]::IsNullOrWhiteSpace($DiagDir)) {
  $oneDriveRoot = Resolve-OneDriveRoot
  $DiagDir = Join-Path $oneDriveRoot 'ApexTraderAI\diagnostics'
}

$DiagDir = [System.IO.Path]::GetFullPath($DiagDir)
$null = New-Item -Path $DiagDir -ItemType Directory -Force

$zipName = "diag_${runId}_${attempt}_${timestamp}.zip"
$summaryName = "diag_${runId}_${attempt}_${timestamp}_summary.txt"
$zipPath = Join-Path $DiagDir $zipName
$summaryPath = Join-Path $DiagDir $summaryName

$summaryLines = @(
  "timestamp=$timestamp",
  "run_id=$($env:GITHUB_RUN_ID)",
  "run_attempt=$($env:GITHUB_RUN_ATTEMPT)",
  "sha=$($env:GITHUB_SHA)",
  "workspace=$workspace",
  "runner_temp=$runnerTemp",
  "console_log=$consoleLog",
  "resolved_run_log=$resolvedRunLog",
  ''
)

$summaryLines += '=== tail run log (300 lines) ==='
if ($resolvedRunLog -and (Test-Path $resolvedRunLog)) {
  $summaryLines += Get-Content -Path $resolvedRunLog -Tail 300
} else {
  $summaryLines += '<run log not found>'
}
$summaryLines += ''

$summaryLines += '=== tail console log (200 lines) ==='
if (Test-Path $consoleLog) {
  $summaryLines += Get-Content -Path $consoleLog -Tail 200
} else {
  $summaryLines += '<console log not found>'
}

$summaryLines | Set-Content -Path $summaryPath -Encoding UTF8

$zipInputs = @()
if ($resolvedRunLog -and (Test-Path $resolvedRunLog)) {
  $zipInputs += $resolvedRunLog
  $runLogsDir = Split-Path -Path $resolvedRunLog -Parent
  if ($runLogsDir -and (Test-Path $runLogsDir)) {
    $zipInputs += $runLogsDir
  }
}
if (Test-Path $consoleLog) {
  $zipInputs += $consoleLog
}
if (Test-Path $summaryPath) {
  $zipInputs += $summaryPath
}

if ($zipInputs.Count -le 0) {
  throw 'No diagnostic files were found to zip.'
}

Write-Host ("ZIP input count: {0}" -f $zipInputs.Count)
$zipInputs | ForEach-Object { Write-Host ("ZIP input: {0}" -f $_) }

Compress-Archive -Path $zipInputs -DestinationPath $zipPath -Force
if (!(Test-Path $zipPath)) {
  throw "zip not created: $zipPath"
}

$workspaceDiagRoot = Join-Path $workspace '_diagnostics'
$null = New-Item -Path $workspaceDiagRoot -ItemType Directory -Force
Copy-Item -Path $zipPath -Destination (Join-Path $workspaceDiagRoot (Split-Path -Path $zipPath -Leaf)) -Force
if (Test-Path $summaryPath) {
  Copy-Item -Path $summaryPath -Destination (Join-Path $workspaceDiagRoot (Split-Path -Path $summaryPath -Leaf)) -Force
}

Write-Host "diagDir=$DiagDir"
Write-Host "zipPath=$zipPath"
Get-ChildItem -Path $DiagDir |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 10 |
  Format-Table -AutoSize
