$ErrorActionPreference = 'Stop'

# This is deliberately a single long-lived SSH session: it owns the reverse
# API bridge and submits the ready snapshots before the first allocation can
# start.  Keeping both responsibilities on one healthy session eliminates the
# gateway race that previously left Slurm workers without their proxy.
$root = Split-Path -Parent $PSScriptRoot
$factory = '/data/run01/scvj260/codex_factory'
$sshKey = Join-Path $HOME '.ssh\paracloud_ed25519'
$sshUser = 'scvj260@NC-N50R5'
$sshHost = 'ssh.cn-zhongwei-1.paracloud.com'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$files = @(
    @{ Local = (Join-Path $root 'night_visual_repair/repair_visual_completion_v2.py'); Remote = "$factory/repair_visual_completion_v2.py" },
    @{ Local = (Join-Path $root 'night_visual_repair/rebind_one_manipuland_plan.py'); Remote = "$factory/rebind_one_manipuland_plan.py" },
    @{ Local = (Join-Path $root 'api_tunnel/ensure_paracloud_api_bridge.sh'); Remote = "$factory/ensure_paracloud_api_bridge.sh" },
    @{ Local = (Join-Path $root 'night_visual_repair/paracloud_visual_completion_v2.sbatch'); Remote = "$factory/paracloud_visual_completion_v2.sbatch" },
    @{ Local = (Join-Path $root 'manipuland_resume/paracloud_manipuland_resume.sbatch'); Remote = "$factory/paracloud_manipuland_resume.sbatch" }
)

$remote = New-Object System.Collections.Generic.List[string]
$remote.Add('set -euo pipefail')
$remote.Add("FACTORY='$factory'")
$remote.Add('EXEC=/data/run01/scvj260/codex_factory/room1_execution')
$remote.Add('VENV=/data/run01/scvj260/scenesmith/.venv')
$remote.Add('RUN="$EXEC/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"')
$remote.Add('SCENE="$RUN/scene_000"')
$remote.Add('mkdir -p "$FACTORY/logs"')
foreach ($file in $files) {
    $bytes = [System.IO.File]::ReadAllBytes($file.Local)
    $b64 = [Convert]::ToBase64String($bytes)
    $sha = ([System.Security.Cryptography.SHA256]::Create().ComputeHash($bytes) | ForEach-Object { $_.ToString('x2') }) -join ''
    $tmp = "$($file.Remote).tmp_$stamp"
    $backup = "$($file.Remote).backup_$stamp"
    $remote.Add("printf '%s' '$b64' | base64 -d > '$tmp'")
    $remote.Add("test `$(sha256sum '$tmp' | awk '{print `$1}') = '$sha'")
    $remote.Add("test ! -e '$($file.Remote)' || cp '$($file.Remote)' '$backup'")
    $remote.Add("mv '$tmp' '$($file.Remote)'")
}
$remote.Add("sed -i 's/\\r`$//' `"`$FACTORY/paracloud_visual_completion_v2.sbatch`" `"`$FACTORY/paracloud_manipuland_resume.sbatch`"")
$remote.Add('python3 -m py_compile "$FACTORY/repair_visual_completion_v2.py"')
$remote.Add('bash -n "$FACTORY/paracloud_visual_completion_v2.sbatch" "$FACTORY/paracloud_manipuland_resume.sbatch"')
$remote.Add('echo SNAPSHOTS_STAGED_AND_VALIDATED')
$remote.Add('sleep 8 # allow the direct reverse forward to settle before relay setup')
$remote.Add('chmod 700 "$FACTORY/ensure_paracloud_api_bridge.sh"')
$remote.Add('bridge_ready=0; for attempt in $(seq 1 8); do if bash "$FACTORY/ensure_paracloud_api_bridge.sh"; then bridge_ready=1; break; fi; sleep 5; done; test "$bridge_ready" = 1')
$remote.Add('echo API_BRIDGE_READY')

# The allocator enforces the account GPU cap.  These pending requests make all
# available capacity productive without claiming more than the scheduler grants.
$jobs = @(
    @{ Name='resume8_classroom_05'; Room='classroom_05'; Proxy=24105; Offset=4105; Run='resume_classroom_05' },
    @{ Name='resume10_storage_room'; Room='storage_room'; Proxy=24400; Offset=4400; Run='resume_storage_room' },
    @{ Name='resume9_classroom_06'; Room='classroom_06'; Proxy=24320; Offset=4320; Run='resume_classroom_06' },
    @{ Name='resume8_library'; Room='library'; Proxy=24110; Offset=4110; Run='resume_library' }
)
foreach ($job in $jobs) {
    $exports = "ALL,ROOM_ID=$($job.Room),GPU_PROXY_PORT=$($job.Proxy),PORT_OFFSET=$($job.Offset),RUN_NAME=$($job.Run),REBIND_PLAN=1"
    $remote.Add("if squeue -h -u scvj260 -n '$($job.Name)' | grep -q .; then echo ACTIVE_$($job.Name); elif `"`$VENV/bin/python`" `"`$EXEC/scripts/select_room_resume_stage.py`" --scene-dir `"`$SCENE`" --room-id '$($job.Room)' --prompt-binding `"`$SCENE/quality_gates/room_prompt_binding.json`" | grep -qx manipuland; then sbatch --parsable --begin=now+2minutes --chdir=/tmp --output=/tmp/$($job.Name)-%j.out --job-name=$($job.Name) --export='$exports' `"`$FACTORY/paracloud_manipuland_resume.sbatch`"; else echo COMPLETE_OR_BLOCKED_$($job.Name); fi")
}
$remote.Add('if squeue -h -u scvj260 -n visual_completion_v2c | grep -q .; then echo ACTIVE_visual_completion_v2c; elif for room in classroom_01 classroom_02 classroom_03 classroom_04 boys_toilet girls_toilet main_corridor; do state="$SCENE/room_$room/scene_states/final_scene/scene_state.json"; receipt="$SCENE/room_$room/quality_gates/visual_completion_v2.json"; test -s "$state" && test -s "$receipt" && grep -q pass "$receipt" && current=$(sha256sum "$state" | cut -d " " -f1) && grep -q "$current" "$receipt" || exit 1; done; then echo COMPLETE_visual_completion_v2c; else sbatch --parsable --begin=now+3minutes --chdir=/tmp --output=/tmp/visual_completion_v2c-%A_%a.out --job-name=visual_completion_v2c "$FACTORY/paracloud_visual_completion_v2.sbatch"; fi')
$remote.Add('touch "$FACTORY/logs/overnight_factory_submitted_v2"')
$remote.Add('echo SUBMISSIONS_COMPLETE')
$remote.Add("exec bash -c 'while :; do sleep 3600; done' # retain reverse API bridge")
$remoteScript = $remote -join "`n"

$logDir = Join-Path $PSScriptRoot 'overnight_factory_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'remote_session.log'
while ($true) {
    $ErrorActionPreference = 'Continue'
    $remoteScript | & ssh.exe `
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
    Start-Sleep -Seconds 20
}
