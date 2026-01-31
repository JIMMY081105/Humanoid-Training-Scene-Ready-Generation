$ErrorActionPreference = 'Stop'

# Keep generation service alive until every retry has succeeded.  A fixed wall
# clock can pre-empt a queued five-hour allocation, while a mere empty squeue
# could hide a failed worker.  Poll Slurm only after a brief startup window and
# require every selected recovery job to be COMPLETED before handoff.
$jobs = @(
    @{ Room = 'classroom_05'; Worker = 170575; Proxy = 24500 },
    @{ Room = 'storage_room'; Worker = 170574; Proxy = 24400 },
    @{ Room = 'classroom_06'; Worker = 170459; Proxy = 24320 },
    @{ Room = 'library'; Worker = 170576; Proxy = 24110 }
)
$workerIds = ($jobs.Worker -join ',')
$notBefore = (Get-Date).AddMinutes(5)
while ((Get-Date) -lt $notBefore) { Start-Sleep -Seconds 30 }
while ($true) {
    $ErrorActionPreference = 'Continue'
    $active = & ssh.exe -o ClearAllForwardings=yes -o BatchMode=yes -o KbdInteractiveAuthentication=no -o PasswordAuthentication=no -o ConnectTimeout=20 paracloud "timeout 20 squeue -h -j $workerIds" 2>$null
    $queryExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($queryExit -ne 0) {
        Start-Sleep -Seconds 60
        continue
    }
    if ($active -match '\S') {
        Start-Sleep -Seconds 60
        continue
    }
    $ErrorActionPreference = 'Continue'
    $states = & ssh.exe -o ClearAllForwardings=yes -o BatchMode=yes -o KbdInteractiveAuthentication=no -o PasswordAuthentication=no -o ConnectTimeout=20 paracloud "timeout 20 sacct -X -n -j $workerIds --format=State -P" 2>$null
    $stateExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    $terminal = @($states | Where-Object { $_ -match '\S' } | ForEach-Object { ($_ -split '\|')[0].Trim() })
    if ($stateExit -eq 0 -and $terminal.Count -ge $jobs.Count -and @($terminal | Where-Object { $_ -ne 'COMPLETED' }).Count -eq 0) {
        break
    }
    Start-Sleep -Seconds 300
}

$factory = '/data/run01/scvj260/codex_factory'
$root = Split-Path -Parent $PSScriptRoot
$sshKey = Join-Path $HOME '.ssh\paracloud_ed25519'
$sshUser = 'scvj260@NC-N50R5'
$sshHost = 'ssh.cn-zhongwei-1.paracloud.com'
$releaseBatch = Join-Path $root 'api_tunnel/paracloud_room_release_gate.sbatch'
$postBatch = Join-Path $root 'api_tunnel/paracloud_school_post_room_pipeline.sbatch'
$bytes = [System.IO.File]::ReadAllBytes($releaseBatch)
$b64 = [Convert]::ToBase64String($bytes)
$sha = ([System.Security.Cryptography.SHA256]::Create().ComputeHash($bytes) | ForEach-Object { $_.ToString('x2') }) -join ''
$postBytes = [System.IO.File]::ReadAllBytes($postBatch)
$postB64 = [Convert]::ToBase64String($postBytes)
$postSha = ([System.Security.Cryptography.SHA256]::Create().ComputeHash($postBytes) | ForEach-Object { $_.ToString('x2') }) -join ''
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

# End only the old one-way bridge.  It was deliberately kept separate from
# the wake locks and from all other user SSH work.
$oldSsh = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'ssh.exe' -and $_.CommandLine -like '*127.0.0.1:18113:127.0.0.1:10809*'
}
foreach ($process in $oldSsh) {
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.ParentProcessId)" -ErrorAction SilentlyContinue
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    if ($parent -and $parent.Name -eq 'powershell.exe') { Stop-Process -Id $parent.ProcessId -Force -ErrorAction SilentlyContinue }
}
$releaseControllers = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'powershell.exe' -and $_.CommandLine -like '*submit_dependent_release_gates.ps1*'
}
foreach ($process in $releaseControllers) { Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue }

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('set -euo pipefail')
$lines.Add("FACTORY='$factory'")
$lines.Add('mkdir -p "$FACTORY/logs"')
$lines.Add('test -x "$FACTORY/ensure_paracloud_api_bridge.sh"')
$lines.Add('bash "$FACTORY/ensure_paracloud_api_bridge.sh"')
$lines.Add("printf '%s' '$b64' | base64 -d > `"`$FACTORY/paracloud_room_release_gate.sbatch.tmp_$stamp`"")
$lines.Add("test `$(sha256sum `"`$FACTORY/paracloud_room_release_gate.sbatch.tmp_$stamp`" | awk '{print `$1}') = '$sha'")
$lines.Add("test ! -e `"`$FACTORY/paracloud_room_release_gate.sbatch`" || cp `"`$FACTORY/paracloud_room_release_gate.sbatch`" `"`$FACTORY/paracloud_room_release_gate.sbatch.backup_$stamp`"")
$lines.Add("mv `"`$FACTORY/paracloud_room_release_gate.sbatch.tmp_$stamp`" `"`$FACTORY/paracloud_room_release_gate.sbatch`"")
$lines.Add("sed -i 's/\\r`$//' `"`$FACTORY/paracloud_room_release_gate.sbatch`"")
$lines.Add('bash -n "$FACTORY/paracloud_room_release_gate.sbatch"')
$lines.Add("printf '%s' '$postB64' | base64 -d > `"`$FACTORY/paracloud_school_post_room_pipeline.sbatch.tmp_$stamp`"")
$lines.Add("test `$(sha256sum `"`$FACTORY/paracloud_school_post_room_pipeline.sbatch.tmp_$stamp`" | awk '{print `$1}') = '$postSha'")
$lines.Add("test ! -e `"`$FACTORY/paracloud_school_post_room_pipeline.sbatch`" || cp `"`$FACTORY/paracloud_school_post_room_pipeline.sbatch`" `"`$FACTORY/paracloud_school_post_room_pipeline.sbatch.backup_$stamp`"")
$lines.Add("mv `"`$FACTORY/paracloud_school_post_room_pipeline.sbatch.tmp_$stamp`" `"`$FACTORY/paracloud_school_post_room_pipeline.sbatch`"")
$lines.Add("sed -i 's/\\r`$//' `"`$FACTORY/paracloud_school_post_room_pipeline.sbatch`"")
$lines.Add('bash -n "$FACTORY/paracloud_school_post_room_pipeline.sbatch"')
foreach ($entry in $jobs) {
    $name = "release_$($entry.Room)"
    $receipt = "`$FACTORY/$name.submission"
    $exports = "ALL,ROOM_ID=$($entry.Room),PROXY_PORT=$($entry.Proxy)"
    $lines.Add("if test -s $receipt; then echo EXISTING_$name; else sbatch --parsable --dependency=afterok:$($entry.Worker) --chdir=/tmp --output=/tmp/$name-%j.out --job-name=$name --export='$exports' `"`$FACTORY/paracloud_room_release_gate.sbatch`" > $receipt.tmp; mv $receipt.tmp $receipt; cat $receipt; fi")
}
$lines.Add('if test -x "$FACTORY/submit_post_room_pipeline_after_all_gates.sh" && ! pgrep -af "[s]ubmit_post_room_pipeline_after_all_gates.sh" >/dev/null; then nohup bash "$FACTORY/submit_post_room_pipeline_after_all_gates.sh" >"$FACTORY/logs/post_room_pipeline_current_watcher.out" 2>&1 </dev/null & echo POST_WATCHER_STARTED; fi')
$lines.Add('echo POST_WORKER_HANDOFF_READY')
$lines.Add("exec bash -c 'while :; do sleep 3600; done' # retain reverse API bridge")
$remote = $lines -join "`n"

$logDir = Join-Path $PSScriptRoot 'overnight_factory_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'post_worker_bridge_handoff.log'
while ($true) {
    $ErrorActionPreference = 'Continue'
    $remote | & ssh.exe `
        -F NUL `
        -i $sshKey `
        -l $sshUser `
        -o BatchMode=yes `
        -o KbdInteractiveAuthentication=no `
        -o PasswordAuthentication=no `
        -o ExitOnForwardFailure=yes `
        -o ConnectTimeout=90 `
        -o ServerAliveInterval=30 `
        -o ServerAliveCountMax=3 `
        -o KexAlgorithms=diffie-hellman-group14-sha256 `
        -o HostKeyAlias=ssh.cn-zhongwei-1.paracloud.com `
        -R 127.0.0.1:18113:127.0.0.1:10809 `
        $sshHost "tr -d '\r' | bash" 2>&1 | Tee-Object -FilePath $log -Append
    $ErrorActionPreference = 'Stop'
    Start-Sleep -Seconds 30
}
