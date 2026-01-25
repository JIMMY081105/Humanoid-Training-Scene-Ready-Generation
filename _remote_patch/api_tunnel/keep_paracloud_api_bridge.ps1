$ErrorActionPreference = 'Stop'

# Keep only the API reverse forward alive.  Job submission is deliberately kept
# out of this process so a bridge reconnect can never create duplicate workers.
$sshKey = Join-Path $HOME '.ssh\paracloud_ed25519'
$remote = "exec bash -c 'while :; do sleep 3600; done'"
$logDir = Join-Path $PSScriptRoot 'api_bridge_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'paracloud_api_bridge.log'

while ($true) {
    $ErrorActionPreference = 'Continue'
    $remote | & ssh.exe `
        -F NUL `
        -i $sshKey `
        -l 'scvj260@NC-N50R5' `
        -o BatchMode=yes `
        -o KbdInteractiveAuthentication=no `
        -o PasswordAuthentication=no `
        -o ExitOnForwardFailure=yes `
        -o ConnectTimeout=45 `
        -o ServerAliveInterval=30 `
        -o ServerAliveCountMax=3 `
        -o KexAlgorithms=diffie-hellman-group14-sha256 `
        -o HostKeyAlias=ssh.cn-zhongwei-1.paracloud.com `
        -R 127.0.0.1:18113:127.0.0.1:10809 `
        ssh.cn-zhongwei-1.paracloud.com "tr -d '\r' | bash" 2>&1 | Tee-Object -FilePath $log -Append
    $ErrorActionPreference = 'Stop'
    Start-Sleep -Seconds 10
}
