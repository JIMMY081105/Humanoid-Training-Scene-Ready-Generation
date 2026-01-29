$ErrorActionPreference = 'SilentlyContinue'

$TunnelPattern = '18103:127\.0\.0\.1:10809'
$BridgeHelper = '/data/run01/scvj260/codex_factory/ensure_paracloud_api_bridge.sh'
$HealthyGateways = @('36.103.203.6', '36.103.203.5')
$GatewayIndex = 0
$HealthyGateway = $HealthyGateways[$GatewayIndex]
$BridgeFailureCount = 0

function Get-TunnelProcess {
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -eq 'ssh.exe' -and $_.CommandLine -match $TunnelPattern } |
        Select-Object -First 1
}

function Invoke-BridgeHelper([string] $Argument = '') {
    $remote = if ($Argument) { "$BridgeHelper $Argument" } else { $BridgeHelper }
    $process = Start-Process ssh.exe -ArgumentList @(
        '-o', 'BatchMode=yes',
        '-o', 'KbdInteractiveAuthentication=no',
        '-o', 'PasswordAuthentication=no',
        '-o', 'ConnectTimeout=8',
        '-o', "HostName=$HealthyGateway",
        '-o', 'HostKeyAlias=ssh.cn-zhongwei-1.paracloud.com',
        '-o', 'ClearAllForwardings=yes',
        'paracloud',
        $remote
    ) -WindowStyle Hidden -PassThru
    if (-not $process.WaitForExit(45000)) {
        $process.Kill()
        $process.WaitForExit()
        return $false
    }
    return $process.ExitCode -eq 0
}

while ($true) {
    $tunnel = Get-TunnelProcess
    if (-not $tunnel) {
        $tunnelProcess = Start-Process ssh.exe -ArgumentList @(
            '-N',
            '-o', 'BatchMode=yes',
            '-o', 'KbdInteractiveAuthentication=no',
            '-o', 'PasswordAuthentication=no',
            '-o', 'ExitOnForwardFailure=yes',
            '-o', 'ConnectTimeout=8',
            '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=3',
            '-o', "HostName=$HealthyGateway",
            '-o', 'HostKeyAlias=ssh.cn-zhongwei-1.paracloud.com',
            '-R', '127.0.0.1:18103:127.0.0.1:10809',
            'paracloud'
        ) -WindowStyle Hidden -PassThru
        Start-Sleep -Seconds 8
    }
    if (Get-TunnelProcess) {
        if (Invoke-BridgeHelper) {
            $BridgeFailureCount = 0
        } else {
            $BridgeFailureCount += 1
            if ($BridgeFailureCount -ge 2) {
                $failedTunnel = Get-TunnelProcess
                if ($failedTunnel) {
                    Stop-Process -Id $failedTunnel.ProcessId -Force -ErrorAction SilentlyContinue
                }
                $GatewayIndex = ($GatewayIndex + 1) % $HealthyGateways.Count
                $HealthyGateway = $HealthyGateways[$GatewayIndex]
                $BridgeFailureCount = 0
            }
        }
    }
    Start-Sleep -Seconds 20
}
