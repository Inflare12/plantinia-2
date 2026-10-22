$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

python -m PyInstaller --noconfirm --clean --onedir --windowed --name PlantiniaOffline --collect-all PySide6 --collect-all torch --collect-all torchvision --collect-all cv2 main.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

$Bundle = Join-Path $Root 'dist\PlantiniaOffline'
New-Item -ItemType Directory -Force -Path (Join-Path $Bundle 'models') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Bundle 'data') | Out-Null
if (Test-Path (Join-Path $Root 'models')) {
    Get-ChildItem (Join-Path $Root 'models') -File |
        Where-Object { $_.Extension -in '.pt', '.gguf' } |
        Copy-Item -Destination (Join-Path $Bundle 'models') -Force
}
Copy-Item (Join-Path $Root 'data\plant_kb.json') (Join-Path $Bundle 'data\plant_kb.json') -Force
Write-Host "Built: $Bundle\PlantiniaOffline.exe"
Write-Host "Copied available local models and plant knowledge base beside the executable."
