$ErrorActionPreference = 'SilentlyContinue'
$Pattern = '18103:127\.0\.0\.1:10809'

function Get-Tunnel {
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -eq 'ssh.exe' -and $_.CommandLine -match $Pattern } |
        Select-Object -First 1
}

while ($true) {
    if (-not (Get-Tunnel)) {
        Start-Process ssh.exe -ArgumentList @(
            '-N','-o','BatchMode=yes',
            '-o','KbdInteractiveAuthentication=no',
            '-o','PasswordAuthentication=no',
            '-o','ExitOnForwardFailure=yes',
            '-o','ConnectTimeout=8',
            '-o','ServerAliveInterval=30',
            '-o','ServerAliveCountMax=3',
            '-o','HostName=36.103.203.6',
            '-o','HostKeyAlias=ssh.cn-zhongwei-1.paracloud.com',
            '-R','127.0.0.1:18103:127.0.0.1:10809',
            'paracloud'
        ) -WindowStyle Hidden
    }
    Start-Sleep -Seconds 20
}
