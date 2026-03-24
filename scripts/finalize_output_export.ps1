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

function Get-ExceptionMessage {
  param($ErrorRecord)

  if ($null -eq $ErrorRecord) {
    return 'unknown_error_record'
  }
  if ($null -ne $ErrorRecord.Exception -and -not [string]::IsNullOrWhiteSpace($ErrorRecord.Exception.Message)) {
    return $ErrorRecord.Exception.Message
  }
  return ('{0}' -f $ErrorRecord)
}

function New-ZipExportFallbackResult {
  param(
    [Parameter(Mandatory = $true)]
    [string]$ResolvedOutputRoot,
    [Parameter(Mandatory = $true)]
    [string]$ExportWarning
  )

  return [pscustomobject]@{
    local_canonical_output_path = $ResolvedOutputRoot
    output_zip_path = ''
    output_zip_name = ''
    zip_size_bytes = 0
    onedrive_destination_dir = ''
    export_requested = $false
    export_result = 'warn'
    export_warning = $ExportWarning
    exported_to_onedrive = $false
    latest_run_info_path = ''
    export_audit_path = ''
    output_zip_sequence = ''
    output_zip_local_date = ''
  }
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
  $effectiveNamingDate = if ([string]::IsNullOrWhiteSpace($NamingDate)) {
    Get-Date -Format 'yyyyMMdd'
  } else {
    $NamingDate
  }
  $effectiveRunId = if ([string]::IsNullOrWhiteSpace($env:APEX_RUN_ID)) { 'unknown' } else { $env:APEX_RUN_ID }

  $doneLines = New-StringList
  $doneLines.Add(('run_id={0}' -f $effectiveRunId))
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
  $outputLogLines.Add(('[OUTPUT_ROOT] naming_date={0}' -f $effectiveNamingDate))
  $outputLogLines.Add(('[OUTPUT_ROOT] naming_date_source={0}' -f $(if ([string]::IsNullOrWhiteSpace($NamingDate)) { 'default_current_date' } else { 'argument' })))
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
  $zipExportOutput = @(& powershell -NoProfile -ExecutionPolicy Bypass -File $zipExportScript @zipExportArgs)
  $zipExportJson = $zipExportOutput | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Last 1
  $zipExportResult = $null
  if (-not [string]::IsNullOrWhiteSpace($zipExportJson)) {
    try {
      $zipExportResult = $zipExportJson | ConvertFrom-Json -ErrorAction Stop
    } catch {
      $zipExportResult = New-ZipExportFallbackResult -ResolvedOutputRoot $resolvedOutputRoot -ExportWarning ('export_json_parse_failed: {0}' -f (Get-ExceptionMessage $_))
    }
  }
  if ($null -eq $zipExportResult) {
    $zipExportResult = New-ZipExportFallbackResult -ResolvedOutputRoot $resolvedOutputRoot -ExportWarning 'export_json_missing'
  }

  $zipExportStatus = if ([string]::IsNullOrWhiteSpace(('{0}' -f $zipExportResult.export_result))) { 'warn' } else { '{0}' -f $zipExportResult.export_result }
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
    naming_date = $effectiveNamingDate
    naming_date_source = if ([string]::IsNullOrWhiteSpace($NamingDate)) { 'default_current_date' } else { 'argument' }
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
  $errorMessage = Get-ExceptionMessage $_
  $diagLines = @(
    ('[FINALIZE][DIAG] run_id={0}' -f $(if ([string]::IsNullOrWhiteSpace($env:APEX_RUN_ID)) { '<null_or_empty>' } else { $env:APEX_RUN_ID })),
    ('[FINALIZE][DIAG] output_root={0}' -f $(if ([string]::IsNullOrWhiteSpace($OutputRoot)) { '<null_or_empty>' } else { $OutputRoot })),
    ('[FINALIZE][DIAG] output_root_name={0}' -f $(if ([string]::IsNullOrWhiteSpace($OutputRootName)) { '<null_or_empty>' } else { $OutputRootName })),
    ('[FINALIZE][DIAG] export_path={0}' -f $(if ([string]::IsNullOrWhiteSpace($oneDriveExportPath)) { '<null_or_empty>' } else { $oneDriveExportPath })),
    ('[FINALIZE][DIAG] naming_date={0}' -f $(if ([string]::IsNullOrWhiteSpace($NamingDate)) { '<null_or_empty>' } else { $NamingDate })),
    ('[FINALIZE][DIAG] local_canonical_output_exists={0}' -f $(if (($resolvedOutputRoot) -and (Test-Path -LiteralPath $resolvedOutputRoot)) { 'true' } else { 'false' })),
    ('[FINALIZE][DIAG] finalize_exception={0}' -f $errorMessage)
  )
  foreach ($diag in $diagLines) {
    Write-Warning $diag
  }
  Write-Error ('finalize failed: {0}' -f $errorMessage)
  if (Test-Path -LiteralPath $ReportPath) {
    Add-Content -Path $ReportPath -Value '[FINALIZE] stage=postprocess_finalize_export'
    foreach ($diag in $diagLines) {
      Add-Content -Path $ReportPath -Value $diag
    }
    Add-Content -Path $ReportPath -Value '[FINALIZE] note=core_pipeline_outputs_may_already_exist_rerun_finalize_only_possible'
    Add-Content -Path $ReportPath -Value ('[NG] finalize exception={0}' -f $errorMessage)
    Add-Content -Path $ReportPath -Value '[FINALIZE] zip_export_result=fail'
    Add-Content -Path $ReportPath -Value '[FINALIZE] finalize_step_result=fail'
  }
  exit 1
}
