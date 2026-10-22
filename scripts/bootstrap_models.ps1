$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path "$Root\models","$Root\data\datasets" | Out-Null

Write-Host "Installing Python dependencies..."
python -m pip install -r requirements.txt

$llm = "$Root\models\qwen2.5-0.5b-instruct-q4_k_m.gguf"
if (!(Test-Path $llm)) {
    Write-Host "Downloading Qwen2.5 0.5B Q4_K_M (local LLM)..."
    curl.exe -L --fail --retry 3 -o $llm "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
}

$dataset = "$Root\data\datasets\PlantVillage-Dataset"
if (!(Test-Path "$dataset\raw\color")) {
    Write-Host "Downloading PlantVillage dataset repository..."
    git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset.git $dataset
}

if (!(Test-Path "$Root\models\plantinia_vision.pt")) {
    Write-Host "Vision weights are not trained yet. Training a first local classifier..."
    python "$Root\scripts\train_vision.py" --epochs 3 --batch 16 --max-classes 16
}

Write-Host "Bootstrap complete. The runtime is now local/offline; it does not contact a cloud AI service."
