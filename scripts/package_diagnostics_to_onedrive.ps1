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
$workspace = (Resolve-Path $workspace).Path
$runnerTemp = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [System.IO.Path]::GetTempPath() }

$runnerRoot = (Resolve-Path (Join-Path $workspace '..\..\..')).Path
$runnerDiag = Join-Path $runnerRoot '_diag'
if (!(Test-Path $runnerDiag)) {
  throw "Runner diagnostics directory not found: $runnerDiag"
}

if ([string]::IsNullOrWhiteSpace($DiagDir)) {
  $oneDriveRoot = Resolve-OneDriveRoot
  $DiagDir = Join-Path $oneDriveRoot 'ApexTraderAI\diagnostics'
}

$DiagDir = [System.IO.Path]::GetFullPath($DiagDir)
$null = New-Item -Path $DiagDir -ItemType Directory -Force
Write-Host "diagDir=$DiagDir"

$zipName = "diag_${runId}_${attempt}_${timestamp}.zip"
$zipPath = Join-Path $DiagDir $zipName

$stagingRoot = Join-Path $runnerTemp ("runner_diag_staging_{0}" -f [System.Guid]::NewGuid().ToString('N'))
$null = New-Item -Path $stagingRoot -ItemType Directory -Force

try {
  Write-Host "runnerRoot=$runnerRoot"
  Write-Host "runnerDiag=$runnerDiag"
  Copy-Item -Path (Join-Path $runnerDiag '*') -Destination $stagingRoot -Recurse -Force

  $stagedItems = Get-ChildItem -Path $stagingRoot -Force -ErrorAction SilentlyContinue
  if (-not $stagedItems -or $stagedItems.Count -eq 0) {
    throw "No runner diagnostics content staged from: $runnerDiag"
  }

  Compress-Archive -Path (Join-Path $stagingRoot '*') -DestinationPath $zipPath -Force
  if (!(Test-Path $zipPath)) {
    throw "zip not created: $zipPath"
  }
}
finally {
  if (Test-Path $stagingRoot) {
    Remove-Item -Path $stagingRoot -Recurse -Force -ErrorAction SilentlyContinue
  }
}

$workspaceDiagRoot = Join-Path $workspace '_diagnostics'
$null = New-Item -Path $workspaceDiagRoot -ItemType Directory -Force
Copy-Item -Path $zipPath -Destination (Join-Path $workspaceDiagRoot (Split-Path -Path $zipPath -Leaf)) -Force

Write-Host "zipPath=$zipPath"
Get-ChildItem -Path $DiagDir |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 10 |
  Format-Table -AutoSize
