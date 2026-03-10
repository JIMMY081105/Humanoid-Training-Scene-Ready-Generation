$ErrorActionPreference = 'Stop'

$Repo = 'E:\Researches\Humanoid-Training-Scene-Ready-Generation'
$Factory = '/data/run01/scvj260/codex_factory'
$Exec = "$Factory/room1_execution"
$Gateways = @('36.103.203.6','36.103.203.5')
$Gateway = $Gateways[0]
$Log = Join-Path $Repo '_remote_patch\night_visual_repair\overnight_dispatch.log'

function Write-DispatchLog([string]$Message) {
    "$(Get-Date -Format o) $Message" | Add-Content -LiteralPath $Log
}

trap {
    Write-DispatchLog ("supervisor_error " + $_.Exception.Message)
    Start-Sleep -Seconds 30
    Start-Process powershell.exe -ArgumentList @(
        '-NoLogo','-NoProfile','-ExecutionPolicy','Bypass','-File',$PSCommandPath
    ) -WindowStyle Hidden
    exit 1
}

function Invoke-CapturedProcess(
    [string]$Program,
    [string[]]$Arguments,
    [int]$TimeoutMilliseconds
) {
    $token = [guid]::NewGuid().ToString('N')
    $stdout = Join-Path $env:TEMP "codex-$token.out"
    $stderr = Join-Path $env:TEMP "codex-$token.err"
    try {
        $process = Start-Process $Program -ArgumentList $Arguments -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        if (-not $process.WaitForExit($TimeoutMilliseconds)) {
            $process.Kill()
            $process.WaitForExit()
            return [pscustomobject]@{ ExitCode = 124; Stdout = ''; Stderr = 'timeout' }
        }
        # Flush redirected streams before reading ExitCode; Windows PowerShell
        # can otherwise expose a transient null value after the timed overload.
        $process.WaitForExit()
        $process.Refresh()
        $exitCode = [int]$process.ExitCode
        return [pscustomobject]@{
            ExitCode = $exitCode
            Stdout = if (Test-Path $stdout) { Get-Content -Raw $stdout } else { '' }
            Stderr = if (Test-Path $stderr) { Get-Content -Raw $stderr } else { '' }
        }
    }
    finally {
        Remove-Item -LiteralPath $stdout,$stderr -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-Remote([string]$Command, [int]$TimeoutMilliseconds = 30000) {
    $common = @(
        '-o', "HostName=$script:Gateway",
        '-o', 'HostKeyAlias=ssh.cn-zhongwei-1.paracloud.com',
        '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=8',
        '-o', 'ClearAllForwardings=yes'
    )
    Invoke-CapturedProcess 'ssh.exe' ($common + @('paracloud', $Command)) $TimeoutMilliseconds
}

function Copy-Remote([string]$LocalPath, [string]$RemotePath) {
    $common = @(
        '-o', "HostName=$script:Gateway",
        '-o', 'HostKeyAlias=ssh.cn-zhongwei-1.paracloud.com',
        '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=8',
        '-o', 'ClearAllForwardings=yes'
    )
    Invoke-CapturedProcess 'scp.exe' ($common + @($LocalPath, "paracloud:$RemotePath")) 60000
}

$files = [ordered]@{
    'repair_visual_completion_v2.py' = Join-Path $Repo '_remote_patch\night_visual_repair\repair_visual_completion_v2.py'
    'paracloud_visual_completion_v2.sbatch' = Join-Path $Repo '_remote_patch\night_visual_repair\paracloud_visual_completion_v2.sbatch'
    'render_room_review_views.py' = Join-Path $Repo '_remote_patch\render_cutaway_fix\render_room_review_views.py'
    'cutaway_evidence_contract.py' = Join-Path $Repo '_remote_patch\night_visual_repair\cutaway_evidence_contract.py'
    'room1_baseline_ba67.json' = Join-Path $Repo '_remote_patch\night_visual_repair\latest\states\classroom_01.json'
    'rebind_one_manipuland_plan.py' = Join-Path $Repo '_remote_patch\night_visual_repair\rebind_one_manipuland_plan.py'
    'paracloud_manipuland_resume.sbatch' = Join-Path $Repo '_remote_patch\manipuland_resume\paracloud_manipuland_resume.sbatch'
    'submit_post_room_pipeline_after_all_gates.sh' = Join-Path $Repo '_remote_patch\room1_visual_gate_fix\submit_post_room_pipeline_after_all_gates.sh'
}

foreach ($path in $files.Values) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing deployment file: $path" }
}

Write-DispatchLog 'supervisor_started'
while ($true) {
    $healthy = $false
    foreach ($candidate in $Gateways) {
        $script:Gateway = $candidate
        $health = Invoke-Remote "test -r '$Factory/paracloud_room_release_gate.sbatch' && test -w '$Factory' && head -c 64 '$Factory/paracloud_room_release_gate.sbatch' >/dev/null && echo MOUNT_OK" 18000
        if ($health.ExitCode -eq 0 -and $health.Stdout -match 'MOUNT_OK') {
            Write-DispatchLog "mount_healthy gateway=$candidate"
            $healthy = $true
            break
        }
        Write-DispatchLog "mount_unavailable gateway=$candidate exit=$($health.ExitCode)"
    }
    if ($healthy) { break }
    Start-Sleep -Seconds 30
}

$deploymentAlreadyVerified = Select-String -LiteralPath $Log -SimpleMatch 'deployment_verified_v4' -Quiet -ErrorAction SilentlyContinue
if (-not $deploymentAlreadyVerified) {
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
  $stage = "$Factory/.overnight_v2_stage_$stamp"
  $mkdir = Invoke-Remote "set -e; mkdir -p '$stage'" 30000
  if ($mkdir.ExitCode -ne 0) { throw "Remote staging failed: $($mkdir.Stderr)" }

foreach ($entry in $files.GetEnumerator()) {
    $copy = Copy-Remote $entry.Value "$stage/$($entry.Key)"
    if ($copy.ExitCode -ne 0) { throw "Copy failed for $($entry.Key): $($copy.Stderr)" }
}

$checks = foreach ($entry in $files.GetEnumerator()) {
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $entry.Value).Hash.ToLowerInvariant()
    "$hash  $stage/$($entry.Key)"
}
$checkText = ($checks -join "`n") + "`n"
$checkFile = Join-Path $env:TEMP "codex-checks-$stamp.txt"
Set-Content -LiteralPath $checkFile -Value $checkText -NoNewline -Encoding ascii
try {
    $copyChecks = Copy-Remote $checkFile "$stage/SHA256SUMS"
    if ($copyChecks.ExitCode -ne 0) { throw "Checksum copy failed: $($copyChecks.Stderr)" }
}
finally { Remove-Item -LiteralPath $checkFile -Force -ErrorAction SilentlyContinue }

$apply = @"
set -euo pipefail
cd '$stage'
sha256sum -c SHA256SUMS
backup_install() {
  src=`$1
  dst=`$2
  if test -e "`$dst"; then cp -a "`$dst" "`$dst.backup_$stamp"; fi
  cp "`$src" "`$dst.tmp.$stamp"
  mv -f "`$dst.tmp.$stamp" "`$dst"
}
backup_install '$stage/repair_visual_completion_v2.py' '$Factory/repair_visual_completion_v2.py'
backup_install '$stage/paracloud_visual_completion_v2.sbatch' '$Factory/paracloud_visual_completion_v2.sbatch'
backup_install '$stage/render_room_review_views.py' '$Exec/scripts/render_room_review_views.py'
backup_install '$stage/cutaway_evidence_contract.py' '$Exec/scripts/cutaway_evidence_contract.py'
backup_install '$stage/room1_baseline_ba67.json' '$Factory/room1_baseline_ba67.json'
backup_install '$stage/rebind_one_manipuland_plan.py' '$Factory/rebind_one_manipuland_plan.py'
backup_install '$stage/paracloud_manipuland_resume.sbatch' '$Factory/paracloud_manipuland_resume.sbatch'
backup_install '$stage/submit_post_room_pipeline_after_all_gates.sh' '$Factory/submit_post_room_pipeline_after_all_gates.sh'
chmod 755 '$Factory/repair_visual_completion_v2.py' '$Factory/rebind_one_manipuland_plan.py' '$Factory/submit_post_room_pipeline_after_all_gates.sh'
/data/run01/scvj260/scenesmith/.venv/bin/python -m py_compile \
  '$Factory/repair_visual_completion_v2.py' \
  '$Factory/rebind_one_manipuland_plan.py' \
  '$Exec/scripts/render_room_review_views.py' \
  '$Exec/scripts/cutaway_evidence_contract.py'
bash -n '$Factory/paracloud_visual_completion_v2.sbatch' '$Factory/paracloud_manipuland_resume.sbatch' '$Factory/submit_post_room_pipeline_after_all_gates.sh'
echo DEPLOY_OK
"@
$applied = Invoke-Remote $apply 120000
if ($applied.ExitCode -ne 0 -or $applied.Stdout -notmatch 'DEPLOY_OK') {
    throw "Atomic deployment failed: $($applied.Stdout) $($applied.Stderr)"
}
  Write-DispatchLog 'deployment_verified_v4'
}

Write-DispatchLog 'bridge_owned_by_parallel_watchdog'

$submit = @"
set -euo pipefail
readonly F='$Factory'
mkdir -p "`$F/overnight_v2_submissions" "`$F/logs"
exec 9>"`$F/.overnight_v2_dispatch.lock"
flock -w 30 9
submit_room() {
  room=`$1; proxy=`$2; offset=`$3
  receipt="`$F/overnight_v2_submissions/`$room.txt"
  test -s "`$receipt" && return 0
  output=`$(sbatch --job-name="resume_`$room" \
    --export="ALL,ROOM_ID=`$room,GPU_PROXY_PORT=`$proxy,PORT_OFFSET=`$offset,RUN_NAME=resume_`$room,REBIND_PLAN=1" \
    "`$F/paracloud_manipuland_resume.sbatch")
  printf '%s\n' "`$output" > "`$receipt.tmp.`$`$"
  mv -f "`$receipt.tmp.`$`$" "`$receipt"
}
# Long-room resumes were submitted directly from LF-normalized batch snapshots
# as jobs 170129-170132 while shared-storage writes were degraded.  Do not
# duplicate them here; this supervisor now owns only v3 deployment, the seven
# short visual tasks, and the exact all-room watcher.
if ! test -s "`$F/overnight_v2_submissions/visual_completion_v2.txt"; then
  output=`$(sbatch "`$F/paracloud_visual_completion_v2.sbatch")
  printf '%s\n' "`$output" > "`$F/overnight_v2_submissions/visual_completion_v2.txt.tmp.`$`$"
  mv -f "`$F/overnight_v2_submissions/visual_completion_v2.txt.tmp.`$`$" "`$F/overnight_v2_submissions/visual_completion_v2.txt"
fi
if ! test -s "`$F/post_room_pipeline_submission_current.txt" && ! pgrep -af "[s]ubmit_post_room_pipeline_after_all_gates.sh" >/dev/null; then
  nohup bash "`$F/submit_post_room_pipeline_after_all_gates.sh" >"`$F/logs/post_room_pipeline_current_watcher.out" 2>&1 </dev/null &
  echo `$! > "`$F/post_room_pipeline_current_watcher.pid"
fi
squeue -u scvj260 -o '%i|%T|%M|%b|%j|%R'
"@
$submitted = Invoke-Remote $submit 120000
if ($submitted.ExitCode -ne 0) { throw "Dispatch failed: $($submitted.Stdout) $($submitted.Stderr)" }
Write-DispatchLog ("dispatch_complete " + ($submitted.Stdout -replace "`r?`n", ';'))
