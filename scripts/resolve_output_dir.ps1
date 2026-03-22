param(
  [string]$Mode,
  [string]$Symbol,
  [string]$TestStartDate,
  [string]$CanonicalOutputRoot = 'C:\work\apex_work\output',
  [string]$PreferredOutputRoot,
  [string]$RunDir,
  [string]$DateStamp,
  [switch]$AllocateNew,
  [switch]$CreateDirectory,
  [switch]$EmitJson
)

$ErrorActionPreference = 'Stop'

function Get-NormalizedPath {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
  try {
    return [System.IO.Path]::GetFullPath($Path)
  } catch {
    return $Path
  }
}

function Resolve-CanonicalBase {
  param(
    [string]$CanonicalRoot,
    [string]$ModeName,
    [string]$SymbolName
  )

  $resolvedMode = if ([string]::IsNullOrWhiteSpace($ModeName)) { 'sim' } else { $ModeName.Trim().ToLowerInvariant() }
  $resolvedSymbol = if ([string]::IsNullOrWhiteSpace($SymbolName)) { 'SOXL' } else { $SymbolName.Trim().ToUpperInvariant() }
  return Join-Path (Join-Path $CanonicalRoot $resolvedMode) $resolvedSymbol
}

function Resolve-LatestExistingOutput {
  param(
    [string]$CanonicalBase,
    [string]$TestStart
  )

  if ([string]::IsNullOrWhiteSpace($CanonicalBase) -or -not (Test-Path $CanonicalBase)) {
    return $null
  }

  $safeTestStart = if ([string]::IsNullOrWhiteSpace($TestStart)) { 'unknown_test_start' } else { $TestStart.Trim() }
  $exactLegacyPath = Join-Path $CanonicalBase $safeTestStart
  if (Test-Path $exactLegacyPath) {
    return (Get-Item -LiteralPath $exactLegacyPath)
  }

  # Backward-compatible fallback for legacy canonical outputs that were allocated as
  # <test_start>_<YYYYMMDD>_<NNN>. New runs must no longer allocate these names.
  $prefix = '{0}_' -f $safeTestStart
  $legacyCandidates = Get-ChildItem -Path $CanonicalBase -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "$prefix*" } |
    Sort-Object -Property @{ Expression = 'LastWriteTime'; Descending = $true }, @{ Expression = 'Name'; Descending = $true }
  if ($legacyCandidates) {
    return ($legacyCandidates | Select-Object -First 1)
  }

  return $null
}

$canonicalBase = Resolve-CanonicalBase -CanonicalRoot $CanonicalOutputRoot -ModeName $Mode -SymbolName $Symbol
$safeTestStartDate = if ([string]::IsNullOrWhiteSpace($TestStartDate)) { 'unknown_test_start' } else { $TestStartDate.Trim() }
$result = [ordered]@{
  path = $null
  source = ''
  output_root_name = ''
  naming_date = ''
  naming_seq = ''
  canonical_base = $canonicalBase
}

$preferredCandidates = @(
  $PreferredOutputRoot,
  $env:EFFECTIVE_WIN_OUTPUT
) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

foreach ($candidate in $preferredCandidates) {
  if (Test-Path $candidate) {
    $resolved = Resolve-Path $candidate
    $result.path = $resolved.Path
    $result.source = if ($candidate -eq $PreferredOutputRoot) { 'preferred_output_root' } else { 'env_effective_win_output' }
    $result.output_root_name = (Split-Path -Leaf $result.path)
    break
  }
}

if (-not $result.path -and -not $AllocateNew) {
  $existing = Resolve-LatestExistingOutput -CanonicalBase $canonicalBase -TestStart $safeTestStartDate
  if ($existing) {
    $result.path = $existing.FullName
    $result.source = 'canonical_pattern_search'
    $result.output_root_name = $existing.Name
  }
}

if (-not $result.path -and -not [string]::IsNullOrWhiteSpace($RunDir)) {
  $runOutput = Join-Path $RunDir 'output'
  if (Test-Path $runOutput) {
    $result.path = (Resolve-Path $runOutput).Path
    $result.source = 'run_dir_fallback'
    $result.output_root_name = Split-Path -Leaf $result.path
  }
}

if ($AllocateNew) {
  $result.naming_date = ''
  $result.naming_seq = ''
  $result.output_root_name = $safeTestStartDate
  $result.path = Join-Path $canonicalBase $safeTestStartDate
  $result.source = 'allocated_new'
  if ($CreateDirectory) {
    $null = New-Item -Path $result.path -ItemType Directory -Force
  }
}

$result.path = Get-NormalizedPath -Path $result.path
if (-not $result.output_root_name -and $result.path) {
  $result.output_root_name = Split-Path -Leaf $result.path
}

if ($EmitJson) {
  $result | ConvertTo-Json -Depth 4 -Compress
} elseif ($result.path) {
  Write-Output $result.path
}
