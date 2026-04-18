param(
    [string]$Tag = "",
    [string]$PythonExe = "",
    [string]$ProjectRoot = "",
    [int]$Epochs = 120,
    [int]$EpisodesPerEpoch = 5,
    [string]$Device = "cuda",
    [int]$GruHiddenDim = 96
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Tag)) {
    $Tag = Get-Date -Format "yyyyMMdd_HHmmss"
}

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = $PSScriptRoot
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    throw "PythonExe must be provided explicitly."
}

Set-Location -LiteralPath $ProjectRoot

$queueDir = Join-Path $ProjectRoot "__agent_debug__\serial_runs\$Tag"
New-Item -ItemType Directory -Force -Path $queueDir | Out-Null
$queueLog = Join-Path $queueDir "queue.log"

function Write-QueueLog {
    param([string]$Message)
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $queueLog -Value $line
    Write-Output $line
}

$runs = @(
    @{
        Mode = "DD"
        RunName = "dd_gru_serial_$Tag"
        ExtraArgs = @()
    },
    @{
        Mode = "CD"
        RunName = "cd_gru_serial_$Tag"
        ExtraArgs = @("--uf_control_mode", "absolute")
    },
    @{
        Mode = "CC"
        RunName = "cc_gru_serial_$Tag"
        ExtraArgs = @("--uf_control_mode", "absolute")
    }
)

Write-QueueLog "Serial run start | tag=$Tag | epochs=$Epochs | episodes_per_epoch=$EpisodesPerEpoch | device=$Device"

foreach ($run in $runs) {
    $mode = $run.Mode
    $runName = $run.RunName
    $stdoutLog = Join-Path $queueDir "$runName.stdout.log"

    $args = @(
        "$ProjectRoot\train.py",
        "--algo", "td3",
        "--mode", $mode,
        "--use_gru_encoder",
        "--gru_hidden_dim", "$GruHiddenDim",
        "--curriculum", "safety_mass",
        "--epochs", "$Epochs",
        "--episodes_per_epoch", "$EpisodesPerEpoch",
        "--device", $Device,
        "--run_name", $runName
    ) + $run.ExtraArgs

    Write-QueueLog "Starting $mode | run_name=$runName"

    & $PythonExe @args *>&1 | Tee-Object -FilePath $stdoutLog
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-QueueLog "FAILED $mode | run_name=$runName | exit_code=$exitCode"
        exit $exitCode
    }

    Write-QueueLog "Completed $mode | run_name=$runName"
}

Write-QueueLog "All serial runs finished | tag=$Tag"
