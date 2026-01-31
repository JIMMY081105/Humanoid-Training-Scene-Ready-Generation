$ErrorActionPreference = 'Continue'
$bridge = Join-Path $PSScriptRoot 'http_connect_stdio.ps1'
& ssh.exe -vvv `
    -o BatchMode=yes `
    -o KbdInteractiveAuthentication=no `
    -o PasswordAuthentication=no `
    -o ConnectTimeout=10 `
    -o KexAlgorithms=diffie-hellman-group14-sha256 `
    -o HostName=36.103.203.6 `
    -o HostKeyAlias=ssh.cn-zhongwei-1.paracloud.com `
    paracloud 'true'
exit $LASTEXITCODE
