param(
  [string]$OutputRoot,
  [string]$RunDir,
  [string]$StageDir,
  [string]$BranchName = 'output-latest',
  [string]$ZipName = 'output_latest.zip',
  [string]$TestStartDate
)

$ErrorActionPreference = 'Stop'

function Resolve-OutputRoot {
  param([string]$InputOutputRoot, [string]$InputRunDir)

  if (-not [string]::IsNullOrWhiteSpace($InputOutputRoot) -and (Test-Path $InputOutputRoot)) {
    return (Resolve-Path $InputOutputRoot).Path
  }

  if (-not [string]::IsNullOrWhiteSpace($env:EFFECTIVE_WIN_OUTPUT) -and (Test-Path $env:EFFECTIVE_WIN_OUTPUT)) {
    return (Resolve-Path $env:EFFECTIVE_WIN_OUTPUT).Path
  }

  if (-not [string]::IsNullOrWhiteSpace($InputRunDir)) {
    $runOutput = Join-Path $InputRunDir 'output'
    if (Test-Path $runOutput) {
      return (Resolve-Path $runOutput).Path
    }
  }

  return $null
}

function Copy-IfExists {
  param([string[]]$Candidates, [string]$DestinationPath, [string]$FallbackText)

  foreach ($candidate in $Candidates) {
    if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path $candidate)) {
      Copy-Item -Path $candidate -Destination $DestinationPath -Force
      return $true
    }
  }

  if (-not [string]::IsNullOrWhiteSpace($FallbackText)) {
    Set-Content -Path $DestinationPath -Encoding UTF8 -Value $FallbackText
  }
  return $false
}

$outputResolved = Resolve-OutputRoot -InputOutputRoot $OutputRoot -InputRunDir $RunDir
if ([string]::IsNullOrWhiteSpace($outputResolved)) {
  throw "Output root not found. OutputRoot=$OutputRoot RunDir=$RunDir"
}

if ([string]::IsNullOrWhiteSpace($StageDir)) {
  $runnerTemp = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [System.IO.Path]::GetTempPath() }
  $StageDir = Join-Path $runnerTemp 'output_latest_publish'
}
if (Test-Path $StageDir) { Remove-Item -Path $StageDir -Recurse -Force }
New-Item -Path $StageDir -ItemType Directory -Force | Out-Null

$zipSourceDir = Join-Path $StageDir 'output_root'
New-Item -Path $zipSourceDir -ItemType Directory -Force | Out-Null

$excludeDirs = @('__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','.cache','tmp','temp')
$excludeFiles = @('*.tmp','*.temp','*.bak','*.env','*.pem','*.pfx','*.key','id_rsa*','id_ed25519*')

$xd = @()
foreach ($d in $excludeDirs) { $xd += @('/XD', $d) }
$xf = @()
foreach ($f in $excludeFiles) { $xf += @('/XF', $f) }

& robocopy "$outputResolved" "$zipSourceDir" /E /R:1 /W:1 /NFL /NDL /NJH /NJS @xd @xf | Out-Null
$rc = $LASTEXITCODE
if ($rc -gt 7) {
  throw "robocopy failed while staging output_root (exit=$rc)"
}

$zipPath = Join-Path $StageDir $ZipName
Compress-Archive -Path (Join-Path $zipSourceDir '*') -DestinationPath $zipPath -Force

$hash = (Get-FileHash -Path $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -Path (Join-Path $StageDir 'output_latest.sha256') -Encoding UTF8 -Value ("{0}  {1}" -f $hash, $ZipName)

$workspace = $env:GITHUB_WORKSPACE
$evalDir = if ($workspace) { Join-Path $workspace 'temp\eval' } else { $null }
$runOutput = $outputResolved

Copy-IfExists -Candidates @(
  (Join-Path $runOutput 'EVAL_REPORT.md'),
  (Join-Path $runOutput 'csv\EVAL_REPORT.md'),
  (if ($evalDir) { Join-Path $evalDir 'EVAL_REPORT.md' } else { $null })
) -DestinationPath (Join-Path $StageDir 'EVAL_REPORT.md') -FallbackText 'EVAL_REPORT.md not found.' | Out-Null

Copy-IfExists -Candidates @(
  (Join-Path $runOutput 'CSV_INDEX.md'),
  (Join-Path $runOutput 'csv\CSV_INDEX.md'),
  (if ($evalDir) { Join-Path $evalDir 'CSV_INDEX.md' } else { $null })
) -DestinationPath (Join-Path $StageDir 'CSV_INDEX.md') -FallbackText 'CSV_INDEX.md not found.' | Out-Null

Copy-IfExists -Candidates @(
  (Join-Path $workspace 'temp\ONE_TAP_ERROR_REPORT.txt'),
  (if ($RunDir) { Join-Path $RunDir 'logs\ONE_TAP_ERROR_REPORT.txt' } else { $null })
) -DestinationPath (Join-Path $StageDir 'ONE_TAP_ERROR_REPORT.txt') -FallbackText 'ONE_TAP_ERROR_REPORT.txt not found.' | Out-Null

Copy-IfExists -Candidates @(
  (Join-Path $runOutput 'run_manifest.json')
) -DestinationPath (Join-Path $StageDir 'run_manifest.json') -FallbackText '{}' | Out-Null

Copy-IfExists -Candidates @(
  (Join-Path $runOutput 'split_summary.json')
) -DestinationPath (Join-Path $StageDir 'split_summary.json') -FallbackText '{}' | Out-Null

Copy-IfExists -Candidates @(
  (Join-Path $runOutput 'timings.csv')
) -DestinationPath (Join-Path $StageDir 'timings.csv') -FallbackText 'timings.csv not found.' | Out-Null

Copy-IfExists -Candidates @(
  (Join-Path $workspace 'reports\metrics_summary.csv'),
  (Join-Path $runOutput 'metrics_summary.csv'),
  (if ($evalDir) { Join-Path $evalDir 'metrics_summary.csv' } else { $null })
) -DestinationPath (Join-Path $StageDir 'metrics_summary.csv') -FallbackText 'metrics_summary.csv not found.' | Out-Null

$metricsMdCandidates = @(
  (Join-Path $workspace 'reports\metrics_summary.md'),
  (Join-Path $runOutput 'metrics_summary.md')
)
if ($evalDir) { $metricsMdCandidates += (Join-Path $evalDir 'metrics_summary.md') }
$metricsMd = $metricsMdCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if ($metricsMd) {
  Copy-Item -Path $metricsMd -Destination (Join-Path $StageDir 'metrics_summary.md') -Force
}

$zipBytes = (Get-Item $zipPath).Length
$zipMb = [Math]::Round(($zipBytes / 1MB), 3)
$runInfo = [ordered]@{
  run_id = if ($env:GITHUB_RUN_ID) { $env:GITHUB_RUN_ID } else { 'local' }
  sha = if ($env:GITHUB_SHA) { $env:GITHUB_SHA } else { 'unknown' }
  test_start_date = if (-not [string]::IsNullOrWhiteSpace($TestStartDate)) { $TestStartDate } elseif ($env:INPUT_TEST_START_DATE) { $env:INPUT_TEST_START_DATE } else { '' }
  output_root = $outputResolved
  zip_name = $ZipName
  zip_size_bytes = $zipBytes
  zip_size_mb = $zipMb
  updated_at = (Get-Date).ToString('o')
  branch_name = $BranchName
}
$runInfo | ConvertTo-Json | Set-Content -Path (Join-Path $StageDir 'latest_run_info.json') -Encoding UTF8

Remove-Item -Path $zipSourceDir -Recurse -Force
Write-Host "Prepared output latest bundle at: $StageDir"
