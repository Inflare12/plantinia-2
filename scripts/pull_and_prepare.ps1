$ErrorActionPreference='Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
git pull --ff-only
& "$PSScriptRoot\bootstrap_models.ps1"
