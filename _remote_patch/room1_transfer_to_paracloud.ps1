$ErrorActionPreference = "Stop"

$stamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
$log = Join-Path $PSScriptRoot "room1_transfer_to_paracloud.log"
$destinationRoot = "/data/run01/scvj260/codex_factory/room1_execution"
$sourcePrefix = "codex_room1_state_20260714.tgz.block"
$expectedArchiveSha256 = "26f226e035225dd2ab842bef256211fa0db251d4e77cb8af2cd57322106f1cb5"

function Write-TransferLog([string] $message) {
    "$((Get-Date).ToString('o')) $message" | Tee-Object -FilePath $log -Append
}

function Invoke-ChunkCopy([string] $part) {
    $source = "sqz:/tmp/$part"
    $destination = "paracloud:$destinationRoot/"
    $arguments = @(
        "-3",
        "-o", "ClearAllForwardings=yes",
        "-o", "BatchMode=yes",
        "-o", "KbdInteractiveAuthentication=no",
        "-o", "PasswordAuthentication=no",
        "-o", "ConnectTimeout=20",
        $source,
        $destination
    )
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        $process = Start-Process -FilePath "scp.exe" -ArgumentList $arguments -PassThru -WindowStyle Hidden
        if (-not $process.WaitForExit(60000)) {
            $process.Kill()
            $process.WaitForExit()
            Write-TransferLog "TIMEOUT $part attempt=$attempt"
        } elseif ($process.ExitCode -eq 0) {
            Write-TransferLog "COPIED $part attempt=$attempt"
            return
        } else {
            Write-TransferLog "RETRY $part attempt=$attempt exit=$($process.ExitCode)"
        }
        Start-Sleep -Seconds 3
    }
    throw "Could not transfer $part after four attempts"
}

try {
    Write-TransferLog "START $stamp"
    foreach ($index in 0..34) {
        $part = "{0}{1:D2}.block" -f $sourcePrefix, $index
        Invoke-ChunkCopy $part
    }

    $remoteScript = @'
set -e
EXEC=/data/run01/scvj260/codex_factory/room1_execution
cd "$EXEC"
cat codex_room1_state_20260714.tgz.block{00..34}.block > codex_room1_state_20260714.tgz
EXPECTED=26f226e035225dd2ab842bef256211fa0db251d4e77cb8af2cd57322106f1cb5
ACTUAL=$(sha256sum codex_room1_state_20260714.tgz | awk '{print $1}')
[ "$ACTUAL" = "$EXPECTED" ]
tar -xzf codex_room1_state_20260714.tgz
rm -f codex_room1_state_20260714.tgz codex_room1_state_20260714.tgz.block*.block
sha256sum outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710/scene_000/room_classroom_01/scene_states/final_scene/scene_state.json
touch .room1_transfer_complete
echo ROOM1_TRANSFER_COMPLETE
'@
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remoteScript))
    $remoteCommand = "echo $encoded | base64 -d | bash"
    & ssh.exe -o ClearAllForwardings=yes -o BatchMode=yes -o KbdInteractiveAuthentication=no -o PasswordAuthentication=no -o ConnectTimeout=20 paracloud $remoteCommand 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Remote extraction and verification failed with exit $LASTEXITCODE"
    }
    Write-TransferLog "COMPLETE"
} catch {
    Write-TransferLog "FAILED $($_.Exception.Message)"
    exit 1
}
