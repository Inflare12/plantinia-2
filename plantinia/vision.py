from __future__ import annotations

from pathlib import Path
from PIL import Image
import torch
from torchvision import models, transforms
from .config import VISION_WEIGHTS


class PlantVision:
    """Offline MobileNetV3-Small classifier with explicit confidence state."""

    IMAGE_SIZE = 160
    REVIEW_THRESHOLD = 0.50
    HIGH_CONFIDENCE_THRESHOLD = 0.80

    def __init__(self, weights: Path = VISION_WEIGHTS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes: list[str] = []
        self.ready = False
        self.error: str | None = None
        self.transform = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        if not weights.is_file():
            self.error = f"Trained vision weights not found: {weights}"
            return
        try:
            payload = torch.load(weights, map_location=self.device, weights_only=True)
            if "state_dict" not in payload or "classes" not in payload:
                raise ValueError("Checkpoint must contain state_dict and classes.")
            self.classes = list(payload["classes"])
            if not self.classes:
                raise ValueError("Checkpoint contains no class labels.")
            self.model = models.mobilenet_v3_small(weights=None)
            self.model.classifier[3] = torch.nn.Linear(
                self.model.classifier[3].in_features, len(self.classes)
            )
            self.model.load_state_dict(payload["state_dict"])
            self.model.to(self.device).eval()
            self.ready = True
        except Exception as exc:
            self.model = None
            self.classes = []
            self.error = f"Could not load vision model: {exc}"

    @property
    def status(self) -> str:
        if self.ready:
            return f"Ready • {len(self.classes)} classes • {self.device.type.upper()}"
        return f"Unavailable • {self.error or 'unknown error'}"

    @classmethod
    def confidence_state(cls, confidence: float) -> str:
        if confidence >= cls.HIGH_CONFIDENCE_THRESHOLD:
            return "high"
        if confidence >= cls.REVIEW_THRESHOLD:
            return "review"
        return "low"

    def predict(self, image: Image.Image, top_k: int = 5):
        if not self.ready or self.model is None:
            return [{
                "label": self.error or "vision model not loaded",
                "confidence": 0.0,
                "state": "unavailable",
            }]
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1)[0]
        values, indices = probabilities.topk(min(max(1, top_k), len(self.classes)))
        return [
            {
                "label": self.classes[int(index)],
                "confidence": float(value),
                "state": self.confidence_state(float(value)),
            }
            for value, index in zip(values, indices)
        ]
