#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build and stage the Windows release package for Pattern Match.

.DESCRIPTION
    Builds `pm-ui` in release mode (with the default `camera` feature) and
    copies the executable and README into a distribution directory.  With
    -Verify the packaged executable is launched briefly and stopped, which
    checks that it starts outside a cargo runner.

.EXAMPLE
    ./scripts/package.ps1
    ./scripts/package.ps1 -Verify
    ./scripts/package.ps1 -Jobs 1        # low-memory machines
#>
[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [int]$Jobs = 0,
    [switch]$NoBuild,
    [switch]$Verify
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $root "dist\PatternMatch"
}
$exe = Join-Path $root "target\release\pm-ui.exe"

Push-Location $root
try {
    if (-not $NoBuild) {
        $arguments = @("build", "--release", "-p", "pm-ui", "--locked")
        if ($Jobs -gt 0) {
            $arguments += @("--jobs", $Jobs)
        }
        Write-Host "cargo $($arguments -join ' ')"
        & cargo @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "cargo build failed with exit code $LASTEXITCODE"
        }
    }

    if (-not (Test-Path $exe)) {
        throw "missing release executable: $exe"
    }

    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    $packaged = Join-Path $OutputDirectory "PatternMatch.exe"
    Copy-Item -Path $exe -Destination $packaged -Force
    Copy-Item -Path (Join-Path $root "README.md") -Destination $OutputDirectory -Force

    $size = [math]::Round((Get-Item $packaged).Length / 1MB, 1)
    Write-Host "Packaged $packaged ($size MB)"
    Write-Host "  includes: PatternMatch.exe, README.md"

    if ($Verify) {
        Write-Host "Launching the packaged executable..."
        $process = Start-Process -FilePath $packaged -WorkingDirectory $OutputDirectory -PassThru
        $deadline = (Get-Date).AddSeconds(10)
        while (-not $process.HasExited -and
               $process.MainWindowHandle -eq 0 -and
               (Get-Date) -lt $deadline) {
            Start-Sleep -Milliseconds 200
            $process.Refresh()
        }
        if ($process.HasExited) {
            throw "PatternMatch.exe exited immediately with code $($process.ExitCode)"
        }
        if ($process.MainWindowHandle -eq 0) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            throw "PatternMatch.exe did not show a window within 10 seconds."
        }
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        Write-Host "Launch verified: the GUI window appeared and stayed up."
    }
}
finally {
    Pop-Location
}
