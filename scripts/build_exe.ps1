param(
    [switch]$AllowMissingModels
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Vision = Join-Path $Root 'models\plantinia_vision.pt'
$Brain = Join-Path $Root 'models\qwen2.5-0.5b-instruct-q4_k_m.gguf'
$Missing = @()
if (-not (Test-Path $Vision)) { $Missing += 'models\plantinia_vision.pt' }
if (-not (Test-Path $Brain)) { $Missing += 'models\qwen2.5-0.5b-instruct-q4_k_m.gguf' }
if ($Missing.Count -gt 0 -and -not $AllowMissingModels) {
    throw "Production build blocked: missing model files: $($Missing -join ', '). Use -AllowMissingModels only for a UI/CI smoke build."
}

python -m PyInstaller --noconfirm --clean --onedir --windowed --name PlantiniaOffline --collect-all PySide6 --collect-all torch --collect-all torchvision --collect-all cv2 --collect-all llama_cpp main.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

$Bundle = Join-Path $Root 'dist\PlantiniaOffline'
$BundleModels = Join-Path $Bundle 'models'
$BundleData = Join-Path $Bundle 'data'
New-Item -ItemType Directory -Force -Path $BundleModels | Out-Null
New-Item -ItemType Directory -Force -Path $BundleData | Out-Null

foreach ($Model in @($Vision, $Brain)) {
    if (Test-Path $Model) {
        Copy-Item $Model -Destination $BundleModels -Force
    }
}
Copy-Item (Join-Path $Root 'data\plant_kb.json') (Join-Path $BundleData 'plant_kb.json') -Force

$Manifest = [ordered]@{
    product = 'Plantinia Offline'
    build_mode = if ($Missing.Count -eq 0) { 'production' } else { 'smoke' }
    vision_model_present = (Test-Path $Vision)
    brain_model_present = (Test-Path $Brain)
    knowledge_base_present = (Test-Path (Join-Path $Root 'data\plant_kb.json'))
    internet_required = $false
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
}
$Manifest | ConvertTo-Json | Set-Content (Join-Path $Bundle 'runtime_manifest.json') -Encoding UTF8

if ($Missing.Count -eq 0) {
    Write-Host "Production bundle ready: $Bundle\PlantiniaOffline.exe"
} else {
    Write-Warning "Smoke bundle built without: $($Missing -join ', ')"
}
