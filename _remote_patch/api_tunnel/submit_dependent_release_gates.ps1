$ErrorActionPreference = 'Stop'

# Submit each release gate only after its exact recovery worker succeeds.  This
# preserves the strict render -> deterministic -> visual sequence while leaving
# allocations free for the independent room work in the meantime.
$root = Split-Path -Parent $PSScriptRoot
$factory = '/data/run01/scvj260/codex_factory'
$source = Join-Path $root 'api_tunnel/paracloud_room_release_gate.sbatch'
$bytes = [System.IO.File]::ReadAllBytes($source)
$b64 = [Convert]::ToBase64String($bytes)
$sha = ([System.Security.Cryptography.SHA256]::Create().ComputeHash($bytes) | ForEach-Object { $_.ToString('x2') }) -join ''
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

$rooms = @(
    @{ Room = 'classroom_05'; Worker = 170358; Proxy = 21705 },
    @{ Room = 'storage_room'; Worker = 170359; Proxy = 21711 },
    @{ Room = 'classroom_06'; Worker = 170360; Proxy = 21706 },
    @{ Room = 'library'; Worker = 170361; Proxy = 21710 }
)

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('set -euo pipefail')
$lines.Add("FACTORY='$factory'")
$lines.Add('mkdir -p "$FACTORY/logs"')
$lines.Add("printf '%s' '$b64' | base64 -d > `"`$FACTORY/paracloud_room_release_gate.sbatch.tmp_$stamp`"")
$lines.Add("test `$(sha256sum `"`$FACTORY/paracloud_room_release_gate.sbatch.tmp_$stamp`" | awk '{print `$1}') = '$sha'")
$lines.Add("test ! -e `"`$FACTORY/paracloud_room_release_gate.sbatch`" || cp `"`$FACTORY/paracloud_room_release_gate.sbatch`" `"`$FACTORY/paracloud_room_release_gate.sbatch.backup_$stamp`"")
$lines.Add("mv `"`$FACTORY/paracloud_room_release_gate.sbatch.tmp_$stamp`" `"`$FACTORY/paracloud_room_release_gate.sbatch`"")
$lines.Add("sed -i 's/\\r`$//' `"`$FACTORY/paracloud_room_release_gate.sbatch`"")
$lines.Add('bash -n "$FACTORY/paracloud_room_release_gate.sbatch"')
$lines.Add('if test -x "$FACTORY/submit_post_room_pipeline_after_all_gates.sh" && ! pgrep -af "[s]ubmit_post_room_pipeline_after_all_gates.sh" >/dev/null; then nohup bash "$FACTORY/submit_post_room_pipeline_after_all_gates.sh" >"$FACTORY/logs/post_room_pipeline_current_watcher.out" 2>&1 </dev/null & echo POST_WATCHER_STARTED; fi')
foreach ($entry in $rooms) {
    $name = "release_$($entry.Room)"
    $receipt = "`$FACTORY/$name.submission"
    $exports = "ALL,ROOM_ID=$($entry.Room),PROXY_PORT=$($entry.Proxy)"
    $lines.Add("if test -s $receipt; then echo EXISTING_$name; else sbatch --parsable --dependency=afterok:$($entry.Worker) --chdir=/tmp --output=/tmp/$name-%j.out --job-name=$name --export='$exports' `"`$FACTORY/paracloud_room_release_gate.sbatch`" > $receipt.tmp; mv $receipt.tmp $receipt; cat $receipt; fi")
}
$lines.Add('echo DEPENDENT_RELEASE_GATES_SUBMITTED')
$remote = $lines -join "`n"

$logDir = Join-Path $PSScriptRoot 'overnight_factory_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'dependent_release_gates.log'
while ($true) {
    $ErrorActionPreference = 'Continue'
    $remote | & ssh.exe `
        -o BatchMode=yes `
        -o KbdInteractiveAuthentication=no `
        -o PasswordAuthentication=no `
        -o ConnectTimeout=90 `
        -o KexAlgorithms=diffie-hellman-group14-sha256 `
        -o HostName=36.103.203.6 `
        -o HostKeyAlias=ssh.cn-zhongwei-1.paracloud.com `
        paracloud 'bash -s' 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -eq 0) { exit 0 }
    Start-Sleep -Seconds 30
}
