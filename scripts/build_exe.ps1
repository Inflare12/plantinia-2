$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Set-Location $Root
python -m PyInstaller --noconfirm --clean --onedir --windowed --name PlantiniaOffline --collect-all PySide6 --collect-all llama_cpp main.py
Write-Host "Built: $Root\dist\PlantiniaOffline\PlantiniaOffline.exe"
Write-Host "Copy the models folder beside the exe for offline AI inference. The dataset is only needed for retraining."
