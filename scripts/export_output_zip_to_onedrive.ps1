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

function Resolve-OneDriveExportDestination {
  if (-not [string]::IsNullOrWhiteSpace($env:ONE_DRIVE_RUNS_ROOT)) {
    return Join-Path $env:ONE_DRIVE_RUNS_ROOT 'export'
  }
  if (-not [string]::IsNullOrWhiteSpace($env:OneDrive)) {
    return Join-Path $env:OneDrive 'ApexTraderAI\runs\export'
  }
  return $null
}

function Get-LocalDateStamp {
  return (Get-Date).ToString('yyyyMMdd')
}

function Get-NextZipSequence {
  param(
    [Parameter(Mandatory = $true)]
    [string]$DestinationDir,
    [Parameter(Mandatory = $true)]
    [string]$DateStamp
  )

  $maxSeq = 0
  if (Test-Path $DestinationDir) {
    $pattern = '^output_{0}_(?<seq>\d{{3}})\.zip$' -f [regex]::Escape($DateStamp)
    Get-ChildItem -Path $DestinationDir -Filter ("output_${DateStamp}_*.zip") -File -ErrorAction SilentlyContinue |
      ForEach-Object {
        if ($_.Name -match $pattern) {
          $seq = [int]$Matches['seq']
          if ($seq -gt $maxSeq) {
            $maxSeq = $seq
          }
        }
      }
  }
  return ($maxSeq + 1)
}

function Write-JsonFile {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    $Value,
    [int]$Depth = 8
  )

  $dir = Split-Path -Parent $Path
  if ($dir) {
    $null = New-Item -Path $dir -ItemType Directory -Force -ErrorAction Stop
  }
  $Value | ConvertTo-Json -Depth $Depth | Set-Content -Path $Path -Encoding UTF8
}

function Get-SymbolList {
  param([string]$SymbolsRaw)

  if ([string]::IsNullOrWhiteSpace($SymbolsRaw)) {
    return @()
  }

  $list = New-Object System.Collections.Generic.List[string]
  foreach ($part in ($SymbolsRaw -split ',')) {
    if ($null -eq $part) {
      continue
    }
    $normalized = ('{0}' -f $part).Trim().ToUpperInvariant()
    if (-not [string]::IsNullOrWhiteSpace($normalized)) {
      $list.Add($normalized)
    }
  }
  return $list.ToArray()
}

$resolvedOutputRoot = (Resolve-Path $OutputRoot).Path
if (-not (Test-Path $resolvedOutputRoot)) {
  throw "output_root_missing path='$OutputRoot'"
}

$outputRootName = Split-Path -Leaf $resolvedOutputRoot
$normalizedSymbols = @(Get-SymbolList -SymbolsRaw $Symbols)
$primarySymbol = if ($normalizedSymbols.Count -gt 0) { $normalizedSymbols[0] } else { 'SOXL' }

$exportRequested = ($CopyToOneDrive -eq '1')
$exportedToOneDrive = $false
$exportResult = if ($exportRequested) { 'warn' } else { 'skipped' }
$exportWarning = ''
$oneDriveDestinationDir = Resolve-OneDriveExportDestination
$localDateStamp = Get-LocalDateStamp
$assignedSequenceNumber = $null
$outputZipName = $null
$outputZipPath = $null
$createdAt = (Get-Date).ToString('o')
$exportDate = $null
$exportSequence = $null
$zipSizeBytes = [int64]0

if ($exportRequested) {
  if ([string]::IsNullOrWhiteSpace($oneDriveDestinationDir)) {
    $exportResult = 'warn'
    $exportWarning = 'onedrive_not_configured'
  } else {
    try {
      $null = New-Item -Path $oneDriveDestinationDir -ItemType Directory -Force -ErrorAction Stop
      $nextSequence = Get-NextZipSequence -DestinationDir $oneDriveDestinationDir -DateStamp $localDateStamp
      $assignedSequenceNumber = $nextSequence.ToString('000')
      $outputZipName = 'output_{0}_{1}.zip' -f $localDateStamp, $assignedSequenceNumber
      $outputZipPath = Join-Path $oneDriveDestinationDir $outputZipName
      if (Test-Path $outputZipPath) {
        Remove-Item -Path $outputZipPath -Force -ErrorAction Stop
      }
      [System.IO.Compression.ZipFile]::CreateFromDirectory($resolvedOutputRoot, $outputZipPath)
      if (-not (Test-Path $outputZipPath)) {
        throw "output_zip_missing path='$outputZipPath'"
      }
      $zipInfo = Get-Item -LiteralPath $outputZipPath
      if ($zipInfo.Length -le 0) {
        throw "output_zip_empty path='$outputZipPath'"
      }
      $zipSizeBytes = [int64]$zipInfo.Length
      $exportedToOneDrive = $true
      $exportResult = 'success'
      $exportWarning = ''
      $createdAt = (Get-Date).ToString('o')
      $exportDate = $localDateStamp
      $exportSequence = $assignedSequenceNumber
      Write-Host ("[ZIP_EXPORT_AUDIT] source_canonical_output_path={0}" -f $resolvedOutputRoot)
      Write-Host ("[ZIP_EXPORT_AUDIT] destination_onedrive_output_path={0}" -f $outputZipPath)
      Write-Host ("[ZIP_EXPORT_AUDIT] export_date={0}" -f $exportDate)
      Write-Host ("[ZIP_EXPORT_AUDIT] export_sequence={0}" -f $exportSequence)
      Write-Host ("[ZIP_EXPORT_AUDIT] created_at={0}" -f $createdAt)
    } catch {
      $exportResult = 'warn'
      $exportWarning = "onedrive_zip_create_failed: $($_.Exception.Message)"
    }
  }
}

$latestRunInfoPath = Join-Path $resolvedOutputRoot 'latest_run_info.json'
$exportAuditPath = Join-Path $resolvedOutputRoot 'onedrive_zip_export_audit.json'

$runInfo = [ordered]@{
  run_id = $env:APEX_RUN_ID
  mode = $Mode
  symbols = $normalizedSymbols
  primary_symbol = $primarySymbol
  test_start_date = $TestStartDate
  train_years = $TrainYears
  test_months = $TestMonths
  local_canonical_output_path = $resolvedOutputRoot
  output_root = $resolvedOutputRoot
  output_root_name = $outputRootName
  output_root_source = $OutputRootSource
  reuse_match_found = $ReuseMatchFound
  onedrive_export_requested = $exportRequested
  onedrive_export_destination_dir = $oneDriveDestinationDir
  onedrive_export_name = $outputZipName
  onedrive_zip_name = $outputZipName
  onedrive_zip_path = $outputZipPath
  onedrive_zip_local_date = $localDateStamp
  onedrive_zip_sequence = $assignedSequenceNumber
  onedrive_zip_size_bytes = $zipSizeBytes
  source_canonical_output_path = $resolvedOutputRoot
  destination_onedrive_output_path = $outputZipPath
  export_date = $exportDate
  export_sequence = $exportSequence
  exported_to_onedrive = $exportedToOneDrive
  export_result = $exportResult
  export_warning = $exportWarning
  created_at = $createdAt
  github_run_id = $env:GITHUB_RUN_ID
  github_run_attempt = $env:GITHUB_RUN_ATTEMPT
  github_sha = $env:GITHUB_SHA
}

$exportAudit = [ordered]@{
  local_canonical_output_path = $resolvedOutputRoot
  source_canonical_output_path = $resolvedOutputRoot
  onedrive_export_name = $outputZipName
  destination_onedrive_output_path = $outputZipPath
  destination_onedrive_zip_path = $outputZipPath
  export_date = $exportDate
  export_sequence = $exportSequence
  assigned_sequence_number = $assignedSequenceNumber
  created_at = $createdAt
  export_result = $exportResult
  export_warning = $exportWarning
}

Write-JsonFile -Path $latestRunInfoPath -Value $runInfo
Write-JsonFile -Path $exportAuditPath -Value $exportAudit

$result = [ordered]@{
  local_canonical_output_path = $resolvedOutputRoot
  output_zip_path = $outputZipPath
  output_zip_name = $outputZipName
  onedrive_export_name = $outputZipName
  zip_size_bytes = $zipSizeBytes
  onedrive_destination_dir = $oneDriveDestinationDir
  export_requested = $exportRequested
  export_result = $exportResult
  export_warning = $exportWarning
  exported_to_onedrive = $exportedToOneDrive
  latest_run_info_path = $latestRunInfoPath
  export_audit_path = $exportAuditPath
  output_root = $resolvedOutputRoot
  output_root_name = $outputRootName
  output_zip_sequence = $assignedSequenceNumber
  output_zip_local_date = $localDateStamp
}

if ($EmitJson) {
  $result | ConvertTo-Json -Depth 5 -Compress
} elseif ($outputZipPath) {
  Write-Output $outputZipPath
}
