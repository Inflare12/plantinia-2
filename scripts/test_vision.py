from __future__ import annotations
import argparse
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from plantinia.vision import PlantVision

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "datasets" / "PlantVillage-Dataset" / "raw" / "color"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class ValidationDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(default_loader(path)), label


def main():
    parser = argparse.ArgumentParser(description="Evaluate Plantinia on held-out PlantVillage images.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-images-per-class", type=int, default=0,
                        help="Must match training; 0 means all images.")
    args = parser.parse_args()
    vision = PlantVision()
    if not vision.ready or vision.model is None:
        raise SystemExit(vision.error or "Vision model did not load.")
    if not DATA.is_dir():
        raise SystemExit(f"Dataset not found: {DATA}")
    rng = random.Random(42)
    samples = []
    for label, name in enumerate(vision.classes):
        folder = DATA / name
        paths = sorted(str(p) for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in EXTENSIONS) if folder.is_dir() else []
        rng.shuffle(paths)
        if args.max_images_per_class > 0:
            paths = paths[:args.max_images_per_class]
        if len(paths) > 1:
            split = max(1, min(int(len(paths) * 0.9), len(paths) - 1))
            samples.extend((p, label) for p in paths[split:])
    if not samples:
        raise SystemExit("No held-out validation images found.")
    tf = transforms.Compose([
        transforms.Resize((160, 160)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(ValidationDataset(samples, tf), batch_size=args.batch, shuffle=False, num_workers=0)
    correct = total = 0
    per_class = {name: [0, 0] for name in vision.classes}
    vision.model.eval()
    with torch.inference_mode():
        for index, (images, labels) in enumerate(loader, 1):
            predictions = vision.model(images.to(vision.device)).argmax(1).cpu()
            correct += (predictions == labels).sum().item()
            total += labels.numel()
            for target, pred in zip(labels.tolist(), predictions.tolist()):
                per_class[vision.classes[target]][1] += 1
                per_class[vision.classes[target]][0] += int(target == pred)
            if index == 1 or index % 100 == 0 or index == len(loader):
                print(f"Evaluated {min(index*args.batch, len(samples))}/{len(samples)}", flush=True)
    print(f"\nHeld-out accuracy: {correct/max(1,total):.2%} ({correct}/{total})")
    for name, (right, count) in per_class.items():
        if count:
            print(f"{name}: {right/count:.1%} ({right}/{count})")
    print("PlantVillage validation is not an independent real-world field trial.")


if __name__ == "__main__":
    main()
