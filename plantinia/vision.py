from __future__ import annotations
import json
from pathlib import Path
from PIL import Image
import torch
from torchvision import models, transforms
from .config import VISION_WEIGHTS


class PlantVision:
    """Local classifier. No network access is performed here."""
    def __init__(self, weights: Path = VISION_WEIGHTS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.mobilenet_v3_small(weights=None)
        self.classes = []
        if weights.exists():
            payload = torch.load(weights, map_location=self.device)
            self.classes = payload["classes"]
            self.model.classifier[3] = torch.nn.Linear(self.model.classifier[3].in_features, len(self.classes))
            self.model.load_state_dict(payload["state_dict"])
            self.model.to(self.device).eval()
            self.ready = True
        else:
            self.ready = False
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def predict(self, image: Image.Image, top_k: int = 5):
        if not self.ready:
            return [{"label":"vision model not trained", "confidence":0.0}]
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = self.model(x).softmax(dim=1)[0]
        vals, idx = probs.topk(min(top_k, len(self.classes)))
        return [{"label": self.classes[int(i)], "confidence": float(v)} for v,i in zip(vals,idx)]
