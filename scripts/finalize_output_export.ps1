param(
  [Parameter(Mandatory = $true)]
  [string]$OutputRoot,
  [Parameter(Mandatory = $true)]
  [string]$ReportPath,
  [string]$CopyToOneDrive = '0',
  [string]$OutputRootName = '',
  [string]$OutputRootSource = '',
  [string]$ReuseMatchFound = '',
  [string]$NamingDate = '',
  [string]$NamingSeq = '',
  [string]$Mode = '',
  [string]$Symbols = '',
  [string]$TestStartDate = '',
  [string]$TrainYears = '',
  [string]$TestMonths = '',
  [switch]$EmitJson
)

$ErrorActionPreference = 'Stop'

function New-StringList {
  return [System.Collections.Generic.List[string]]::new()
}

function Add-LogLines {
  param(
    [Parameter(Mandatory = $true)]
    [System.Collections.Generic.List[string]]$Lines
  )

  foreach ($line in $Lines) {
    Write-Host $line
    Add-Content -Path $ReportPath -Value $line
  }
}

function Add-SummaryLines {
  param(
    [Parameter(Mandatory = $true)]
    [System.Collections.Generic.List[string]]$Lines
  )

  if ([string]::IsNullOrWhiteSpace($env:GITHUB_STEP_SUMMARY)) {
    return
  }

  Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value $Lines.ToArray()
}

try {
  $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
  if (-not (Test-Path -LiteralPath $resolvedOutputRoot)) {
    $null = New-Item -Path $resolvedOutputRoot -ItemType Directory -Force -ErrorAction Stop
  }

  $resolvedReportDir = Split-Path -Parent $ReportPath
  if (-not [string]::IsNullOrWhiteSpace($resolvedReportDir)) {
    $null = New-Item -Path $resolvedReportDir -ItemType Directory -Force -ErrorAction Stop
  }
  if (-not (Test-Path -LiteralPath $ReportPath)) {
    $null = New-Item -Path $ReportPath -ItemType File -Force -ErrorAction Stop
  }

  $effectiveMode = if ([string]::IsNullOrWhiteSpace($Mode)) { $env:MODE } else { $Mode }
  $effectiveSymbols = if ([string]::IsNullOrWhiteSpace($Symbols)) { 'SOXL' } else { $Symbols }
  $primarySymbol = (($effectiveSymbols -split ',' | ForEach-Object { $_.Trim().ToUpperInvariant() } | Where-Object { $_ }) | Select-Object -First 1)
  if ([string]::IsNullOrWhiteSpace($primarySymbol)) {
    $primarySymbol = 'SOXL'
  }

  $effectiveOutputRootName = if ([string]::IsNullOrWhiteSpace($OutputRootName)) {
    Split-Path -Leaf $resolvedOutputRoot
  } else {
    $OutputRootName
  }
  $effectiveOutputRootSource = if ([string]::IsNullOrWhiteSpace($OutputRootSource)) {
    'env_effective_win_output'
  } else {
    $OutputRootSource
  }
  $copyEffective = if ($CopyToOneDrive -eq '1') { '1' } else { '0' }

  $doneLines = New-StringList
  $doneLines.Add(('run_id={0}' -f $env:APEX_RUN_ID))
  $doneLines.Add(('finished_at={0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')))
  $doneLines.Add(('commit={0}' -f $env:GITHUB_SHA))
  $doneLines.Add(('mode={0}' -f $effectiveMode))
  Set-Content -Path (Join-Path $resolvedOutputRoot 'DONE.txt') -Value $doneLines.ToArray() -Encoding UTF8 -ErrorAction Stop

  $timingLogLines = New-StringList
  $timingLogLines.Add(('[TIMING] events_path={0}' -f (Join-Path $resolvedOutputRoot (Join-Path 'timing' (Join-Path $effectiveMode 'timing_events.jsonl')))))
  $timingLogLines.Add(('[TIMING] csv_path={0}' -f (Join-Path $resolvedOutputRoot 'timings.csv')))
  Add-LogLines -Lines $timingLogLines

  $strictRequired = @(
    (Join-Path $resolvedOutputRoot 'run_manifest.json'),
    (Join-Path $resolvedOutputRoot 'reuse_signature.json'),
    (Join-Path $resolvedOutputRoot 'timings.csv')
  )
  $missingStrict = @($strictRequired | Where-Object { -not (Test-Path -LiteralPath $_) })
  if ($missingStrict.Count -gt 0) {
    $missingText = $missingStrict -join ','
    Add-Content -Path $ReportPath -Value ('[NG] strict_required_missing={0}' -f $missingText)
    throw ('Missing strict required artifacts: {0}' -f $missingText)
  }

  $outputLogLines = New-StringList
  $outputLogLines.Add(('[OUTPUT_ROOT] local_canonical_output_path={0}' -f $resolvedOutputRoot))
  $outputLogLines.Add(('[OUTPUT_ROOT] output_root_name={0}' -f $effectiveOutputRootName))
  $outputLogLines.Add(('[OUTPUT_ROOT] output_root_source={0}' -f $effectiveOutputRootSource))
  $outputLogLines.Add(('[OUTPUT_ROOT] mode={0}' -f $effectiveMode))
  $outputLogLines.Add(('[OUTPUT_ROOT] symbol={0}' -f $primarySymbol))
  $outputLogLines.Add(('[OUTPUT_ROOT] test_start_date={0}' -f $TestStartDate))
  $outputLogLines.Add(('[OUTPUT_ROOT] reuse_match_found={0}' -f $ReuseMatchFound))
  $outputLogLines.Add(('[OUTPUT_ROOT] copy_to_onedrive_effective={0}' -f $copyEffective))
  if (-not [string]::IsNullOrWhiteSpace($NamingDate)) {
    $outputLogLines.Add(('[OUTPUT_ROOT] naming_date={0}' -f $NamingDate))
  }
  if (-not [string]::IsNullOrWhiteSpace($NamingSeq)) {
    $outputLogLines.Add(('[OUTPUT_ROOT] naming_seq={0}' -f $NamingSeq))
  }
  Add-LogLines -Lines $outputLogLines

  $zipExportScript = Join-Path $env:REPO_ROOT 'scripts\export_output_zip_to_onedrive.ps1'
  if (-not (Test-Path -LiteralPath $zipExportScript)) {
    throw ('export_output_zip_to_onedrive script missing: {0}' -f $zipExportScript)
  }

  $zipExportArgs = @(
    '-OutputRoot', $resolvedOutputRoot,
    '-CopyToOneDrive', $copyEffective,
    '-OutputRootSource', $effectiveOutputRootSource,
    '-ReuseMatchFound', $ReuseMatchFound,
    '-Mode', $effectiveMode,
    '-Symbols', $effectiveSymbols,
    '-TestStartDate', $TestStartDate,
    '-TrainYears', $TrainYears,
    '-TestMonths', $TestMonths,
    '-EmitJson'
  )
  $zipExportJson = & powershell -NoProfile -ExecutionPolicy Bypass -File $zipExportScript @zipExportArgs | Select-Object -Last 1
  $zipExportResult = $zipExportJson | ConvertFrom-Json
  $zipExportStatus = '{0}' -f $zipExportResult.export_result
  $zipExportWarning = '{0}' -f $zipExportResult.export_warning
  $oneDriveExportName = '{0}' -f $zipExportResult.output_zip_name
  $oneDriveExportPath = '{0}' -f $zipExportResult.output_zip_path

  $zipLogLines = New-StringList
  $zipLogLines.Add(('[ZIP_EXPORT] source_canonical_output_path={0}' -f $resolvedOutputRoot))
  $zipLogLines.Add(('[ZIP_EXPORT] oneDrive_export_name={0}' -f $oneDriveExportName))
  $zipLogLines.Add(('[ZIP_EXPORT] destination_onedrive_output_path={0}' -f $oneDriveExportPath))
  $zipLogLines.Add(('[ZIP_EXPORT] zip_size_bytes={0}' -f $zipExportResult.zip_size_bytes))
  $zipLogLines.Add(('[ZIP_EXPORT] onedrive_dest={0}' -f $zipExportResult.onedrive_destination_dir))
  $zipLogLines.Add(('[ZIP_EXPORT] result={0}' -f $zipExportStatus))
  Add-LogLines -Lines $zipLogLines
  if (-not [string]::IsNullOrWhiteSpace($zipExportWarning)) {
    Write-Warning ('[ZIP_EXPORT] warning={0}' -f $zipExportWarning)
    Add-Content -Path $ReportPath -Value ('[ZIP_EXPORT] warning={0}' -f $zipExportWarning)
  }

  $summaryLines = New-StringList
  $summaryLines.Add(('- local canonical output path ({0}): `{1}`' -f $effectiveMode, $resolvedOutputRoot))
  $summaryLines.Add(('- output_root_name ({0}): `{1}`' -f $effectiveMode, $effectiveOutputRootName))
  $summaryLines.Add(('- output_root_source ({0}): `{1}`' -f $effectiveMode, $effectiveOutputRootSource))
  $summaryLines.Add(('- mode ({0}): `{1}`' -f $effectiveMode, $effectiveMode))
  $summaryLines.Add(('- symbol ({0}): `{1}`' -f $effectiveMode, $primarySymbol))
  $summaryLines.Add(('- test_start_date ({0}): `{1}`' -f $effectiveMode, $TestStartDate))
  $summaryLines.Add(('- onedrive export name ({0}): `{1}`' -f $effectiveMode, $oneDriveExportName))
  $summaryLines.Add(('- onedrive output zip path ({0}): `{1}`' -f $effectiveMode, $oneDriveExportPath))
  $summaryLines.Add(('- output_zip_size_bytes ({0}): `{1}`' -f $effectiveMode, $zipExportResult.zip_size_bytes))
  $summaryLines.Add(('- onedrive_export_result ({0}): `{1}`' -f $effectiveMode, $zipExportStatus))
  Add-SummaryLines -Lines $summaryLines

  Add-Content -Path $ReportPath -Value ('[OK] finalize output_root={0}' -f $resolvedOutputRoot)
  $finalizeStepResult = 'success'
  Write-Host ('[FINALIZE] zip_export_result={0}' -f $zipExportStatus)
  if (-not [string]::IsNullOrWhiteSpace($zipExportWarning)) {
    Write-Host ('[FINALIZE] zip_export_warning={0}' -f $zipExportWarning)
  }
  Write-Host ('[FINALIZE] finalize_step_result={0}' -f $finalizeStepResult)
  Add-Content -Path $ReportPath -Value ('[FINALIZE] zip_export_result={0}' -f $zipExportStatus)
  if (-not [string]::IsNullOrWhiteSpace($zipExportWarning)) {
    Add-Content -Path $ReportPath -Value ('[FINALIZE] zip_export_warning={0}' -f $zipExportWarning)
  }
  Add-Content -Path $ReportPath -Value ('[FINALIZE] finalize_step_result={0}' -f $finalizeStepResult)

  $result = [ordered]@{
    output_root = $resolvedOutputRoot
    output_root_name = $effectiveOutputRootName
    output_root_source = $effectiveOutputRootSource
    mode = $effectiveMode
    symbol = $primarySymbol
    test_start_date = $TestStartDate
    output_zip_path = $zipExportResult.output_zip_path
    output_zip_name = $zipExportResult.output_zip_name
    zip_size_bytes = $zipExportResult.zip_size_bytes
    onedrive_destination_dir = $zipExportResult.onedrive_destination_dir
    export_requested = $zipExportResult.export_requested
    export_result = $zipExportStatus
    export_warning = $zipExportWarning
    exported_to_onedrive = $zipExportResult.exported_to_onedrive
    latest_run_info_path = $zipExportResult.latest_run_info_path
    export_audit_path = $zipExportResult.export_audit_path
    output_zip_sequence = $zipExportResult.output_zip_sequence
    output_zip_local_date = $zipExportResult.output_zip_local_date
  }

  if ($EmitJson) {
    $result | ConvertTo-Json -Depth 6 -Compress
  } else {
    Write-Output $resolvedOutputRoot
  }
} catch {
  $errorMessage = $_.Exception.Message
  Write-Error ('finalize failed: {0}' -f $errorMessage)
  if (Test-Path -LiteralPath $ReportPath) {
    Add-Content -Path $ReportPath -Value ('[NG] finalize exception={0}' -f $errorMessage)
    Add-Content -Path $ReportPath -Value '[FINALIZE] zip_export_result=fail'
    Add-Content -Path $ReportPath -Value '[FINALIZE] finalize_step_result=fail'
  }
  exit 1
}
