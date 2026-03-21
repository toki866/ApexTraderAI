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
  $candidates = @()
  $exactLegacyPath = Join-Path $CanonicalBase $safeTestStart
  if (Test-Path $exactLegacyPath) {
    $candidates += (Get-Item -LiteralPath $exactLegacyPath)
  }

  $prefix = '{0}_' -f $safeTestStart
  $namedCandidates = Get-ChildItem -Path $CanonicalBase -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "$prefix*" } |
    Sort-Object -Property @{ Expression = 'LastWriteTime'; Descending = $true }, @{ Expression = 'Name'; Descending = $true }
  if ($namedCandidates) {
    $candidates += $namedCandidates
  }

  return $candidates | Select-Object -First 1
}

function Resolve-NextOutputName {
  param(
    [string]$CanonicalBase,
    [string]$TestStart,
    [string]$NameDate
  )

  $safeTestStart = if ([string]::IsNullOrWhiteSpace($TestStart)) { 'unknown_test_start' } else { $TestStart.Trim() }
  $resolvedDate = if ([string]::IsNullOrWhiteSpace($NameDate)) { (Get-Date).ToString('yyyyMMdd') } else { $NameDate.Trim() }
  $prefix = '{0}_{1}_' -f $safeTestStart, $resolvedDate
  $maxSeq = 0

  if (Test-Path $CanonicalBase) {
    foreach ($dir in Get-ChildItem -Path $CanonicalBase -Directory -ErrorAction SilentlyContinue) {
      if ($dir.Name -match ('^{0}(?<seq>\d{{3}})$' -f [regex]::Escape($prefix))) {
        $seq = [int]$Matches['seq']
        if ($seq -gt $maxSeq) { $maxSeq = $seq }
      }
    }
  }

  $nextSeq = $maxSeq + 1
  return [ordered]@{
    output_root_name = '{0}{1}' -f $prefix, $nextSeq.ToString('000')
    naming_date = $resolvedDate
    naming_seq = $nextSeq.ToString('000')
  }
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
  $naming = Resolve-NextOutputName -CanonicalBase $canonicalBase -TestStart $safeTestStartDate -NameDate $DateStamp
  $result.naming_date = $naming.naming_date
  $result.naming_seq = $naming.naming_seq
  $result.output_root_name = $naming.output_root_name
  $result.path = Join-Path $canonicalBase $naming.output_root_name
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
