param(
    [Parameter(Mandatory = $true)][string]$HostName,
    [Parameter(Mandatory = $true)][int]$Port
)

$ErrorActionPreference = 'Stop'
$client = [System.Net.Sockets.TcpClient]::new()
$diagnostic = Join-Path $PSScriptRoot 'http_connect_runtime.log'
try {
    $client.Connect('127.0.0.1', 10809)
    $network = $client.GetStream()
    $request = [System.Text.Encoding]::ASCII.GetBytes("CONNECT ${HostName}:${Port} HTTP/1.1`r`nHost: ${HostName}:${Port}`r`nProxy-Connection: Keep-Alive`r`n`r`n")
    $network.Write($request, 0, $request.Length)
    $header = [System.Collections.Generic.List[byte]]::new()
    $terminator = [byte[]](13, 10, 13, 10)
    while ($true) {
        $value = $network.ReadByte()
        if ($value -lt 0) { throw 'Proxy closed before CONNECT completed.' }
        $header.Add([byte]$value)
        if ($header.Count -ge 4 -and ($header.GetRange($header.Count - 4, 4).ToArray() -join ',') -eq ($terminator -join ',')) { break }
        if ($header.Count -gt 16384) { throw 'Proxy CONNECT response was too large.' }
    }
    $response = [System.Text.Encoding]::ASCII.GetString($header.ToArray())
    if ($response -notmatch '^HTTP/1\.[01] 200 ') { throw "Proxy CONNECT failed: $($response.Trim())" }
    [System.IO.File]::AppendAllText($diagnostic, "$(Get-Date -Format o) CONNECT_OK ${HostName}:${Port}`n")
    $stdin = [Console]::OpenStandardInput()
    $stdout = [Console]::OpenStandardOutput()
    $toNetwork = $stdin.CopyToAsync($network)
    $fromNetwork = $network.CopyToAsync($stdout)
    [System.Threading.Tasks.Task]::WaitAny([System.Threading.Tasks.Task[]]@($toNetwork, $fromNetwork)) | Out-Null
}
catch {
    [System.IO.File]::AppendAllText($diagnostic, "$(Get-Date -Format o) ERROR $($_.Exception.Message)`n")
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
finally {
    if ($client) { $client.Close() }
}
