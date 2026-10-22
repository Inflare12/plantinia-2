from pathlib import Path
import sys


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


ROOT = app_root()
MODELS = ROOT / "models"
DATA = ROOT / "data"
DATASETS = DATA / "datasets"
KB = DATA / "plant_kb.json"
VISION_WEIGHTS = MODELS / "plantinia_vision.pt"
LLM_MODEL = MODELS / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
