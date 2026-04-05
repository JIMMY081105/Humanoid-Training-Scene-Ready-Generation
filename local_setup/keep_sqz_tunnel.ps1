$ErrorActionPreference = "Continue"

$ssh = Join-Path $env:WINDIR "System32\OpenSSH\ssh.exe"
# When this supervisor does not own a tunnel, this lightweight round trip detects
# a healthy user-owned forward. It is never used to recycle an owned connection:
# two large downloads can legitimately delay a third SSH channel.
$probeCommand = 'curl --max-time 4 -sS http://127.0.0.1:17890/ -o /dev/null'
$lastDialAttempt = [datetime]::MinValue
$statusPath = Join-Path $env:TEMP "codex_sqz_tunnel_status.txt"
$ownedTunnel = $null
$tunnelArguments = @(
    "-N",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ControlMaster=no",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=10",
    "-o", "ServerAliveCountMax=2",
    "sqz"
)

function Write-SqzStatus {
    param([string]$Message)
    "$(Get-Date -Format o) $Message" | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

Add-Type @"
using System.Runtime.InteropServices;
public static class SqzExecutionState {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

# Keep the system awake while this supervisor is active; the display may still
# turn off normally. ES_CONTINUOUS | ES_SYSTEM_REQUIRED.
[void][SqzExecutionState]::SetThreadExecutionState([uint32]0x80000001)

try {
    while ($true) {
        $vpn = Get-VpnConnection -Name SQZ -ErrorAction SilentlyContinue
        if ($null -eq $vpn -or $vpn.ConnectionStatus -ne "Connected") {
            # Do not leave a half-live SSH process across a VPN route change.
            if ($null -ne $ownedTunnel) {
                try {
                    if (-not $ownedTunnel.HasExited) {
                        Stop-Process -Id $ownedTunnel.Id -Force -ErrorAction SilentlyContinue
                        $ownedTunnel.WaitForExit(5000)
                    }
                } catch {}
                $ownedTunnel = $null
            }
            Write-SqzStatus "vpn_disconnected"

            # Cached-credential redial only, with a five-minute backoff. Never
            # pass, infer, or repeatedly guess credentials from this supervisor.
            if (((Get-Date) - $lastDialAttempt).TotalSeconds -ge 300) {
                $lastDialAttempt = Get-Date
                & rasdial.exe SQZ *> $null
            }
            Start-Sleep -Seconds 15
            continue
        }

        # A process owned by this supervisor is governed by SSH keepalives and
        # ExitOnForwardFailure. Do not recycle it because an application-level
        # probe is slow while multi-gigabyte channels are active.
        $ownedAlive = $false
        if ($null -ne $ownedTunnel) {
            try {
                $ownedTunnel.Refresh()
                $ownedAlive = -not $ownedTunnel.HasExited
            } catch {
                $ownedAlive = $false
            }
        }
        if ($ownedAlive) {
            Write-SqzStatus "healthy_ssh owned_pid=$($ownedTunnel.Id)"
            Start-Sleep -Seconds 15
            continue
        }
        $ownedTunnel = $null

        # No owned process exists. Probe for a healthy user-owned forward before
        # attempting to bind the same reverse ports ourselves.
        & $ssh `
            -o BatchMode=yes `
            -o ConnectTimeout=10 `
            -o ClearAllForwardings=yes `
            sqz $probeCommand `
            2>$null

        if ($LASTEXITCODE -eq 0) {
            Write-SqzStatus "healthy external_owner"
            Start-Sleep -Seconds 15
            continue
        }

        Write-SqzStatus "starting_tunnel"
        $ownedTunnel = Start-Process `
            -FilePath $ssh `
            -ArgumentList $tunnelArguments `
            -PassThru `
            -WindowStyle Hidden

        Start-Sleep -Seconds 5
    }
} finally {
    if ($null -ne $ownedTunnel) {
        try {
            if (-not $ownedTunnel.HasExited) {
                Stop-Process -Id $ownedTunnel.Id -Force -ErrorAction SilentlyContinue
            }
        } catch {}
    }
    # ES_CONTINUOUS clears this thread's system-awake request.
    [void][SqzExecutionState]::SetThreadExecutionState([uint32]0x80000000)
    Write-SqzStatus "stopped"
}
