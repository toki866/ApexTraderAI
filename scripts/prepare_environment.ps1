param(
  [Parameter(Mandatory = $true)][string]$RepoRoot,
  [Parameter(Mandatory = $true)][string]$WorkRoot,
  [Parameter(Mandatory = $true)][string]$ApexRunId,
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

function Invoke-PrepareData([string]$DataDir) {
  $runner = Join-Path $RepoRoot 'tools\run_with_python.py'
  $script = Join-Path $RepoRoot 'tools\prepare_data.py'
  $cmd = @(
    'python', $runner, $script,
    '--symbols', $Symbols,
    '--start', $DataStart,
    '--end', $DataEnd,
    '--force',
    '--data-dir', $DataDir
  )
  Write-PrepareLog ("[PREPARE] prepare_data_command={0}" -f ($cmd -join ' '))

  Push-Location $RepoRoot
  try {
    & python $runner $script --symbols $Symbols --start $DataStart --end $DataEnd --force --data-dir $DataDir 2>&1 | Tee-Object -FilePath $bootstrap -Append | Write-Host
    $rc = $LASTEXITCODE
  } finally {
    Pop-Location
  }

  Write-PrepareLog ("[PREPARE] prepare_data_rc={0}" -f $rc)
  if ($rc -ne 0) {
    throw "prepare_data failed rc=$rc"
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

  $consoleLog = Join-Path $RunnerTemp 'run_all_local_then_copy_console.log'
  $null = New-Item -ItemType File -Path $consoleLog -Force
  Add-Content -Path $consoleLog -Encoding UTF8 -Value ("[BOOTSTRAP] console_log_initialized run_id={0}" -f $ApexRunId)
  Add-Content -Path $consoleLog -Encoding UTF8 -Value ("[BOOTSTRAP] prepare_environment_start={0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'))
  Add-GithubEnv -Name 'RUN_CONSOLE_LOG' -Value $consoleLog

  $runLog = Join-Path $winLogDir ("run_{0}.log" -f $ApexRunId)
  $null = New-Item -ItemType File -Path $runLog -Force
  Add-Content -Path $runLog -Encoding UTF8 -Value ("[BOOTSTRAP] run_log_initialized run_id={0}" -f $ApexRunId)
  Add-GithubEnv -Name 'RUN_LOG_PATH' -Value $runLog

  Add-GithubEnv -Name 'WIN_RUN_DIR' -Value $winRunDir
  Add-GithubEnv -Name 'WIN_DATA_DIR' -Value $winDataDir
  Add-GithubEnv -Name 'WIN_OUTPUT_DIR' -Value $winOutputDir
  Add-GithubEnv -Name 'WIN_LOG_DIR' -Value $winLogDir

  $prepareLog = Join-Path $winLogDir 'prepare_environment.log'
  $null = New-Item -ItemType File -Path $prepareLog -Force
  Add-GithubEnv -Name 'PREPARE_LOG' -Value $prepareLog

  # Convert Windows paths to WSL /mnt/ paths for use inside WSL bash commands
  $wslDistro    = if ($env:WSL_DISTRO) { $env:WSL_DISTRO } else { 'Ubuntu' }
  $wslRepoRoot  = (wsl.exe -d $wslDistro wslpath -u $RepoRoot 2>$null).Trim()
  $wslDataDir   = (wsl.exe -d $wslDistro wslpath -u $winDataDir 2>$null).Trim()
  $wslOutputDir = (wsl.exe -d $wslDistro wslpath -u $winOutputDir 2>$null).Trim()
  foreach ($entry in @(
    @('WSL_REPO_ROOT',  $wslRepoRoot,  $RepoRoot),
    @('WSL_DATA_DIR',   $wslDataDir,   $winDataDir),
    @('WSL_OUTPUT_DIR', $wslOutputDir, $winOutputDir)
  )) {
    if (-not ($entry[1] -match '^/mnt/')) {
      throw ("wslpath failed for {0}: win='{1}' -> wsl='{2}' (expected /mnt/ path). Check WSL distro='{3}'." -f $entry[0], $entry[2], $entry[1], $wslDistro)
    }
  }
  Add-GithubEnv -Name 'WSL_REPO_ROOT'  -Value $wslRepoRoot
  Add-GithubEnv -Name 'WSL_DATA_DIR'   -Value $wslDataDir
  Add-GithubEnv -Name 'WSL_OUTPUT_DIR' -Value $wslOutputDir

  Write-PrepareLog ("[PREPARE] win_run_dir={0}" -f $winRunDir)
  Write-PrepareLog ("[PREPARE] repo_root={0}" -f $RepoRoot)
  Write-PrepareLog ("[PREPARE] data_dir={0}" -f $winDataDir)
  Write-PrepareLog ("[PREPARE] output_dir={0}" -f $winOutputDir)
  Write-PrepareLog ("[PREPARE] wsl_repo_root={0}" -f $wslRepoRoot)
  Write-PrepareLog ("[PREPARE] wsl_data_dir={0}"  -f $wslDataDir)
  Write-PrepareLog ("[PREPARE] wsl_output_dir={0}" -f $wslOutputDir)
  Write-PrepareLog ("[PREPARE] console_log={0}" -f $consoleLog)
  Write-PrepareLog ("[PREPARE] run_log={0}" -f $runLog)

  python -m pip install -U pip setuptools wheel 2>&1 | Tee-Object -FilePath $bootstrap -Append | Write-Host
  python -m pip install -r (Join-Path $RepoRoot 'requirements.txt') 2>&1 | Tee-Object -FilePath $bootstrap -Append | Write-Host

  Invoke-PrepareData -DataDir $winDataDir

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
