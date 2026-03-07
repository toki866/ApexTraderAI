param(
  [Parameter(Mandatory = $true)][string]$RepoRoot,
  [Parameter(Mandatory = $true)][string]$WorkRoot,
  [Parameter(Mandatory = $true)][string]$ApexRunId,
  [Parameter(Mandatory = $true)][string]$WslDistro,
  [Parameter(Mandatory = $true)][string]$WslPython,
  [Parameter(Mandatory = $true)][string]$Symbols,
  [Parameter(Mandatory = $true)][string]$DataStart,
  [Parameter(Mandatory = $true)][string]$DataEnd,
  [Parameter(Mandatory = $true)][string]$GithubWorkspace,
  [Parameter(Mandatory = $true)][string]$RunnerTemp,
  [Parameter(Mandatory = $true)][string]$GithubEnvPath
)

$ErrorActionPreference = 'Stop'

$tempDir = Join-Path $GithubWorkspace 'temp'
$null = New-Item -Path $tempDir -ItemType Directory -Force
$bootstrap = Join-Path $tempDir ("bootstrap_prepare_{0}.log" -f $ApexRunId)
$null = New-Item -ItemType File -Path $bootstrap -Force

function Add-GithubEnv([string]$Name, [string]$Value) {
  ("{0}={1}" -f $Name, $Value) | Out-File -FilePath $GithubEnvPath -Append -Encoding utf8
}

function Write-Bootstrap([string]$Message) {
  Add-Content -Path $bootstrap -Encoding UTF8 -Value $Message
  Write-Host $Message
}

$prepareLog = $null
function Write-PrepareLog([string]$Message) {
  if ($prepareLog) {
    Add-Content -Path $prepareLog -Encoding UTF8 -Value $Message
  }
  Write-Bootstrap $Message
}

function Invoke-WslCapture([string[]]$WslArgs, [string]$Label) {
  $stdoutPath = Join-Path $RunnerTemp ("prepare_{0}_{1}_stdout.log" -f $ApexRunId, $Label)
  $stderrPath = Join-Path $RunnerTemp ("prepare_{0}_{1}_stderr.log" -f $ApexRunId, $Label)
  if (Test-Path $stdoutPath) { Remove-Item -Path $stdoutPath -Force }
  if (Test-Path $stderrPath) { Remove-Item -Path $stderrPath -Force }

  $proc = Start-Process -FilePath 'wsl.exe' -ArgumentList $WslArgs -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
  $stdoutRaw = if (Test-Path $stdoutPath) { Get-Content -Path $stdoutPath -Raw } else { '' }
  $stderrRaw = if (Test-Path $stderrPath) { Get-Content -Path $stderrPath -Raw } else { '' }

  [pscustomobject]@{
    Command  = ('wsl.exe ' + ($WslArgs -join ' '))
    ExitCode = $proc.ExitCode
    StdOutRaw = $stdoutRaw
    StdErrRaw = $stderrRaw
  }
}

function Resolve-WslPathStrict([string]$Label, [string]$WindowsPath) {
  if ([string]::IsNullOrWhiteSpace($WindowsPath)) {
    throw "[$Label] Windows path is null or empty"
  }

  $capture = Invoke-WslCapture -WslArgs @('-d', $WslDistro, '--', 'wslpath', '-u', '--', $WindowsPath) -Label ("wslpath_{0}" -f $Label)
  if ($capture.ExitCode -ne 0) {
    throw "[$Label] wslpath failed rc=$($capture.ExitCode) path=$WindowsPath stderr=$($capture.StdErrRaw.Trim())"
  }

  $resolved = $capture.StdOutRaw.Trim()
  if ([string]::IsNullOrWhiteSpace($resolved)) {
    throw "[$Label] resolved path is empty for $WindowsPath"
  }
  return $resolved
}

function Invoke-PrepareData([string]$WslDataDir, [string]$WslRepoRoot) {
  $prepScriptWin = Join-Path $RunnerTemp ("prepare_data_{0}.sh" -f $ApexRunId)
  @(
    '#!/usr/bin/env bash',
    'set -euo pipefail',
    'cd "$1"',
    'shift',
    'exec "$@"'
  ) | Set-Content -Path $prepScriptWin -Encoding UTF8

  $wslPrepScript = Resolve-WslPathStrict -Label 'prepare_script' -WindowsPath $prepScriptWin
  $args = @(
    '-d', $WslDistro,
    '--',
    'bash',
    $wslPrepScript,
    $WslRepoRoot,
    $WslPython,
    'tools/run_with_python.py',
    'tools/prepare_data.py',
    '--symbols', $Symbols,
    '--start', $DataStart,
    '--end', $DataEnd,
    '--force',
    '--data-dir', $WslDataDir
  )

  $capture = Invoke-WslCapture -WslArgs $args -Label 'prepare_data'
  Write-PrepareLog ("[PREPARE] prepare_data_command={0}" -f $capture.Command)
  Write-PrepareLog ("[PREPARE] prepare_data_rc={0}" -f $capture.ExitCode)

  if (-not [string]::IsNullOrWhiteSpace($capture.StdOutRaw)) {
    $capture.StdOutRaw.TrimEnd() | Tee-Object -FilePath $bootstrap -Append | Write-Host
  }
  if (-not [string]::IsNullOrWhiteSpace($capture.StdErrRaw)) {
    $capture.StdErrRaw.TrimEnd() | Tee-Object -FilePath $bootstrap -Append | Write-Host
  }

  if ($capture.ExitCode -ne 0) {
    throw "prepare_data failed rc=$($capture.ExitCode)"
  }
}

try {
  Add-GithubEnv -Name 'BOOTSTRAP_LOG' -Value $bootstrap
  Write-Bootstrap '[PREPARE] status=run'
  Write-Bootstrap ("[PREPARE] repo_root={0}" -f $RepoRoot)
  Write-Bootstrap ("[PREPARE] work_root={0}" -f $WorkRoot)
  Write-Bootstrap ("[PREPARE] run_id={0}" -f $ApexRunId)
  Write-Bootstrap ("[PREPARE] symbols={0} start={1} end={2}" -f $Symbols, $DataStart, $DataEnd)

  & where.exe python 2>&1 | Tee-Object -FilePath $bootstrap -Append | Write-Host
  python --version 2>&1 | Tee-Object -FilePath $bootstrap -Append | Write-Host

  $winRunDir = Join-Path $WorkRoot $ApexRunId
  $winDataDir = Join-Path $winRunDir 'data'
  $winOutputDir = Join-Path $winRunDir 'output'
  $winLogDir = Join-Path $winRunDir 'logs'
  New-Item -Path $winDataDir, $winOutputDir, $winLogDir -ItemType Directory -Force | Out-Null

  Add-GithubEnv -Name 'WIN_RUN_DIR' -Value $winRunDir
  Add-GithubEnv -Name 'WIN_DATA_DIR' -Value $winDataDir
  Add-GithubEnv -Name 'WIN_OUTPUT_DIR' -Value $winOutputDir
  Add-GithubEnv -Name 'WIN_LOG_DIR' -Value $winLogDir

  $prepareLog = Join-Path $winLogDir 'prepare_environment.log'
  $null = New-Item -ItemType File -Path $prepareLog -Force
  Add-GithubEnv -Name 'PREPARE_LOG' -Value $prepareLog

  $wslRepoRoot = Resolve-WslPathStrict -Label 'repo_root' -WindowsPath $RepoRoot
  $wslDataDir = Resolve-WslPathStrict -Label 'data_dir' -WindowsPath $winDataDir
  $wslOutputDir = Resolve-WslPathStrict -Label 'output_dir' -WindowsPath $winOutputDir

  Add-GithubEnv -Name 'WSL_REPO_ROOT' -Value $wslRepoRoot
  Add-GithubEnv -Name 'WSL_DATA_DIR' -Value $wslDataDir
  Add-GithubEnv -Name 'WSL_OUTPUT_DIR' -Value $wslOutputDir

  Write-PrepareLog ("[PREPARE] win_run_dir={0}" -f $winRunDir)
  Write-PrepareLog ("[PREPARE] wsl_repo_root={0}" -f $wslRepoRoot)
  Write-PrepareLog ("[PREPARE] wsl_data_dir={0}" -f $wslDataDir)
  Write-PrepareLog ("[PREPARE] wsl_output_dir={0}" -f $wslOutputDir)

  python -m pip install -U pip setuptools wheel 2>&1 | Tee-Object -FilePath $bootstrap -Append | Write-Host
  python -m pip install -r (Join-Path $RepoRoot 'requirements.txt') 2>&1 | Tee-Object -FilePath $bootstrap -Append | Write-Host

  Invoke-PrepareData -WslDataDir $wslDataDir -WslRepoRoot $wslRepoRoot

  Write-PrepareLog '[PREPARE] status=success'
} catch {
  $message = $_.Exception.Message
  Write-Bootstrap '[PREPARE] status=failed'
  Write-Bootstrap ("[PREPARE] error={0}" -f $message)
  if ($prepareLog) {
    Add-Content -Path $prepareLog -Encoding UTF8 -Value '[PREPARE] status=failed'
    Add-Content -Path $prepareLog -Encoding UTF8 -Value ("[PREPARE] error={0}" -f $message)
  }
  throw
}
