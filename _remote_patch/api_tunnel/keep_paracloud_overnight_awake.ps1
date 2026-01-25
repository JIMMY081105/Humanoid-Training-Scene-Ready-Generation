$ErrorActionPreference = 'Stop'

Add-Type @'
using System;
using System.Runtime.InteropServices;

public static class CodexExecutionState {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint executionState);
}
'@

$EsContinuous = [uint32]2147483648
$EsSystemRequired = [uint32]0x00000001
$state = $EsContinuous -bor $EsSystemRequired

try {
    while ($true) {
        if ([CodexExecutionState]::SetThreadExecutionState($state) -eq 0) {
            throw 'SetThreadExecutionState failed.'
        }
        Start-Sleep -Seconds 30
    }
}
finally {
    [void][CodexExecutionState]::SetThreadExecutionState($EsContinuous)
}
