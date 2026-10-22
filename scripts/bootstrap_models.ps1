$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path "$Root\models","$Root\data\datasets" | Out-Null

Write-Host "Installing Python dependencies..."
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }

$llm = "$Root\models\qwen2.5-0.5b-instruct-q4_k_m.gguf"
if (!(Test-Path $llm)) {
    Write-Host "Downloading Qwen2.5 0.5B Q4_K_M (local LLM)..."
    curl.exe -L --fail --retry 3 -o $llm "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    if ($LASTEXITCODE -ne 0) { throw "Qwen model download failed." }
}

$dataset = "$Root\data\datasets\PlantVillage-Dataset"
if (!(Test-Path "$dataset\raw\color")) {
    Write-Host "Downloading PlantVillage dataset repository..."
    git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset.git $dataset
    if ($LASTEXITCODE -ne 0) { throw "PlantVillage dataset download failed." }
}

if (!(Test-Path "$Root\models\plantinia_vision.pt")) {
    Write-Host "Vision weights are not trained yet. Training all available classes..."
    python -u "$Root\scripts\train_vision.py" --epochs 3 --batch 8
    if ($LASTEXITCODE -ne 0) { throw "Vision training failed." }
}
Write-Host "Bootstrap complete. Verify the local model files before building the EXE."
