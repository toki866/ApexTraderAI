param(
  [string]$DiagDir
)

$ErrorActionPreference = 'Stop'

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runId = if ($Env:GITHUB_RUN_ID) { $Env:GITHUB_RUN_ID } else { 'local' }
$attempt = if ($Env:GITHUB_RUN_ATTEMPT) { $Env:GITHUB_RUN_ATTEMPT } else { '0' }

$workspace = $Env:GITHUB_WORKSPACE
if (-not $workspace) {
  $workspace = (Get-Location).Path
}

$runnerTemp = $Env:RUNNER_TEMP
if (-not $runnerTemp) {
  $runnerTemp = [System.IO.Path]::GetTempPath()
}

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

$oneDriveRoot = if ($Env:OneDrive) { $Env:OneDrive } else { Join-Path $Env:USERPROFILE 'OneDrive' }
$diagnosticsRoot = if ([string]::IsNullOrWhiteSpace($DiagDir)) {
  Join-Path (Join-Path $oneDriveRoot 'ApexTraderAI') 'diagnostics'
} else {
  $DiagDir
}

Write-Host "OneDrive root candidate: $oneDriveRoot"
Write-Host "Diagnostics destination root: $diagnosticsRoot"

if (-not (Test-Path $oneDriveRoot)) {
  Write-Warning "OneDrive root does not exist yet: $oneDriveRoot"
}

$zipName = "diag_${runId}_${attempt}_${timestamp}.zip"
$summaryName = "diag_${runId}_${attempt}_${timestamp}_summary.txt"
$zipPath = Join-Path $diagnosticsRoot $zipName
$summaryPath = Join-Path $diagnosticsRoot $summaryName

try {
  $null = New-Item -Path $diagnosticsRoot -ItemType Directory -Force
  Write-Host "Ensured diagnostics directory exists: $diagnosticsRoot"

  $summaryLines = @()
  $summaryLines += "timestamp=$timestamp"
  $summaryLines += "run_id=$($Env:GITHUB_RUN_ID)"
  $summaryLines += "run_attempt=$($Env:GITHUB_RUN_ATTEMPT)"
  $summaryLines += "sha=$($Env:GITHUB_SHA)"
  $summaryLines += "workspace=$workspace"
  $summaryLines += "runner_temp=$runnerTemp"
  $summaryLines += "console_log=$consoleLog"
  $summaryLines += "resolved_run_log=$resolvedRunLog"
  $summaryLines += ''

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
  foreach ($input in $zipInputs) {
    Write-Host ("ZIP input: {0}" -f $input)
  }

  Compress-Archive -Path $zipInputs -DestinationPath $zipPath -Force
  Write-Host "Created diagnostics ZIP: $zipPath"

  if (-not (Test-Path $zipPath)) {
    throw ("Diagnostics ZIP was not created at expected path: {0}" -f $zipPath)
  }

  $workspaceDiagRoot = Join-Path $workspace '_diagnostics'
  $null = New-Item -Path $workspaceDiagRoot -ItemType Directory -Force

  $workspaceZip = Join-Path $workspaceDiagRoot (Split-Path -Path $zipPath -Leaf)
  Copy-Item -Path $zipPath -Destination $workspaceZip -Force
  Write-Host "Copied diagnostics ZIP to workspace: $workspaceZip"

  if (Test-Path $summaryPath) {
    $workspaceSummary = Join-Path $workspaceDiagRoot (Split-Path -Path $summaryPath -Leaf)
    Copy-Item -Path $summaryPath -Destination $workspaceSummary -Force
    Write-Host "Copied diagnostics summary to workspace: $workspaceSummary"
  }

  Write-Host "Diagnostics summary path: $summaryPath"
  Write-Host "Diagnostics ZIP path: $zipPath"
  Write-Host "Workspace diagnostics root: $workspaceDiagRoot"
}
finally {
  Write-Host "diagDir=$diagnosticsRoot"
  if (Test-Path $diagnosticsRoot) {
    Get-ChildItem -Path $diagnosticsRoot |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 10
  } else {
    Write-Warning "Diagnostics directory does not exist: $diagnosticsRoot"
  }
}
