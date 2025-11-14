#!/usr/bin/env pwsh
param(
    [string]$Kind = "dylib"  # or "cdylib"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$File = Join-Path $ScriptDir "..\Cargo.toml"

# Read all lines
$content = Get-Content -Raw -Path $File -Encoding UTF8
$lines = $content -split "`n"

$inLib = $false
$saw = $false
$result = @()

foreach ($line in $lines) {
    if ($line -match '^\[lib\]\s*$') {
        $inLib = $true
        $result += $line
        continue
    }
    elseif ($line -match '^\[' -and $line -notmatch '^\[lib\]\s*$') {
        $inLib = $false
    }

    if ($inLib -and $line -match '^\s*crate-type\s*=') {
        $result += "crate-type = [""$Kind""]"
        $saw = $true
        continue
    }

    $result += $line
}

if (-not $saw) {
    # Optionally add if not found
    # $result += ""
    # $result += "[lib]"
    # $result += "crate-type = [""$Kind""]"
}

# Write back to file (overwrite)
$result -join "`n" | Set-Content -Path $File -Encoding UTF8

Write-Host "Set [lib].crate-type = [`"$Kind`"] in Cargo.toml"
