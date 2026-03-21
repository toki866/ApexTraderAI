param(
  [Parameter(Mandatory = $true)]
  [string]$OutputRoot,
  [string]$CopyToOneDrive = '0',
  [string]$OutputRootSource = '',
  [string]$ReuseMatchFound = '',
  [string]$Mode,
  [string]$Symbols,
  [string]$TestStartDate,
  [string]$TrainYears,
  [string]$TestMonths,
  [switch]$EmitJson
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Resolve-OneDriveDestination {
  if (-not [string]::IsNullOrWhiteSpace($env:ONE_DRIVE_RUNS_ROOT)) {
    return Join-Path $env:ONE_DRIVE_RUNS_ROOT $env:APEX_RUN_ID
  }
  if (-not [string]::IsNullOrWhiteSpace($env:OneDrive)) {
    return Join-Path $env:OneDrive "ApexTraderAI\runs\$env:APEX_RUN_ID"
  }
  return $null
}

$resolvedOutputRoot = (Resolve-Path $OutputRoot).Path
if (-not (Test-Path $resolvedOutputRoot)) {
  throw "output_root_missing path='$OutputRoot'"
}

$outputRootName = Split-Path -Leaf $resolvedOutputRoot
$runnerTemp = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [System.IO.Path]::GetTempPath() }
$zipLocalName = if ([string]::IsNullOrWhiteSpace($env:APEX_RUN_ID)) { 'output_local.zip' } else { "output_$($env:APEX_RUN_ID).zip" }
$zipLocalPath = Join-Path $runnerTemp $zipLocalName
if (Test-Path $zipLocalPath) { Remove-Item -Path $zipLocalPath -Force }

$stageRoot = Join-Path $runnerTemp ("output_zip_stage_" + [guid]::NewGuid().ToString('N'))
$stageOutputDir = Join-Path $stageRoot $outputRootName
$null = New-Item -Path $stageOutputDir -ItemType Directory -Force

$robocopyArgs = @(
  $resolvedOutputRoot,
  $stageOutputDir,
  '/E','/R:1','/W:1','/NFL','/NDL','/NJH','/NJS'
)
& robocopy @robocopyArgs | Out-Null
$robocopyRc = $LASTEXITCODE
if ($robocopyRc -ge 8) {
  throw "zip_stage_robocopy_failed rc=$robocopyRc"
}

[System.IO.Compression.ZipFile]::CreateFromDirectory($stageRoot, $zipLocalPath)
Remove-Item -Path $stageRoot -Recurse -Force -ErrorAction SilentlyContinue

if (-not (Test-Path $zipLocalPath)) {
  throw "output_zip_missing path='$zipLocalPath'"
}

$zipInfo = Get-Item -LiteralPath $zipLocalPath
if ($zipInfo.Length -le 0) {
  throw "output_zip_empty path='$zipLocalPath'"
}

$primarySymbol = 'SOXL'
if (-not [string]::IsNullOrWhiteSpace($Symbols)) {
  $primarySymbol = (($Symbols -split ',')[0]).Trim().ToUpperInvariant()
  if ([string]::IsNullOrWhiteSpace($primarySymbol)) { $primarySymbol = 'SOXL' }
}

$oneDriveDestinationDir = Resolve-OneDriveDestination
$exportRequested = ($CopyToOneDrive -eq '1')
$exportedToOneDrive = $false
$exportResult = if ($exportRequested) { 'warn' } else { 'skipped' }
$exportWarning = ''
$outputZipName = 'output.zip'

$runManifestPath = Join-Path $resolvedOutputRoot 'run_manifest.json'
$donePath = Join-Path $resolvedOutputRoot 'DONE.txt'
$latestRunInfoPath = Join-Path $runnerTemp 'latest_run_info.json'

$runInfo = [ordered]@{
  run_id = $env:APEX_RUN_ID
  mode = $Mode
  symbols = @($Symbols -split ',' | ForEach-Object { $_.Trim().ToUpperInvariant() } | Where-Object { $_ })
  primary_symbol = $primarySymbol
  test_start_date = $TestStartDate
  train_years = $TrainYears
  test_months = $TestMonths
  output_root = $resolvedOutputRoot
  output_root_name = $outputRootName
  output_zip_name = $outputZipName
  output_zip_local_path = $zipLocalPath
  output_zip_size_bytes = [int64]$zipInfo.Length
  exported_to_onedrive = $false
  onedrive_destination_dir = $oneDriveDestinationDir
  exported_at = (Get-Date).ToString('o')
  github_run_id = $env:GITHUB_RUN_ID
  github_run_attempt = $env:GITHUB_RUN_ATTEMPT
  github_sha = $env:GITHUB_SHA
  reuse_match_found = $ReuseMatchFound
  output_root_source = $OutputRootSource
}
$runInfo | ConvertTo-Json -Depth 6 | Set-Content -Path $latestRunInfoPath -Encoding UTF8

if ($exportRequested) {
  if ([string]::IsNullOrWhiteSpace($oneDriveDestinationDir)) {
    $exportResult = 'warn'
    $exportWarning = 'onedrive_not_configured'
  } else {
    try {
      $null = New-Item -Path $oneDriveDestinationDir -ItemType Directory -Force -ErrorAction Stop
      Copy-Item -Path $zipLocalPath -Destination (Join-Path $oneDriveDestinationDir $outputZipName) -Force -ErrorAction Stop
      Copy-Item -Path $latestRunInfoPath -Destination (Join-Path $oneDriveDestinationDir 'latest_run_info.json') -Force -ErrorAction Stop
      if (Test-Path $donePath) {
        Copy-Item -Path $donePath -Destination (Join-Path $oneDriveDestinationDir 'DONE.txt') -Force -ErrorAction Stop
      }
      if (Test-Path $runManifestPath) {
        Copy-Item -Path $runManifestPath -Destination (Join-Path $oneDriveDestinationDir 'run_manifest.json') -Force -ErrorAction Stop
      }
      $exportedToOneDrive = $true
      $exportResult = 'success'
      $exportWarning = ''
    } catch {
      $exportResult = 'warn'
      $exportWarning = "onedrive_copy_failed: $($_.Exception.Message)"
    }
  }
}

$runInfo.exported_to_onedrive = $exportedToOneDrive
$runInfo.onedrive_destination_dir = $oneDriveDestinationDir
$runInfo.exported_at = (Get-Date).ToString('o')
$runInfo | ConvertTo-Json -Depth 6 | Set-Content -Path $latestRunInfoPath -Encoding UTF8

$result = [ordered]@{
  zip_local_path = $zipLocalPath
  zip_local_name = $zipLocalName
  output_zip_name = $outputZipName
  zip_size_bytes = [int64]$zipInfo.Length
  onedrive_destination_dir = $oneDriveDestinationDir
  export_requested = $exportRequested
  export_result = $exportResult
  export_warning = $exportWarning
  exported_to_onedrive = $exportedToOneDrive
  latest_run_info_path = $latestRunInfoPath
  output_root = $resolvedOutputRoot
  output_root_name = $outputRootName
}

if ($EmitJson) {
  $result | ConvertTo-Json -Depth 5 -Compress
} else {
  Write-Output $zipLocalPath
}
