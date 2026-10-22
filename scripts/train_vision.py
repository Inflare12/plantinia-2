from __future__ import annotations
import argparse, random, time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets.folder import default_loader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "datasets" / "PlantVillage-Dataset" / "raw" / "color"
OUT = ROOT / "models" / "plantinia_vision.pt"
LATEST = ROOT / "models" / "plantinia_vision_latest.pt"
IMAGE_SIZE, SEED = 160, 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class PlantDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(default_loader(path)), label


def save_checkpoint(payload, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(destination)


def main():
    parser = argparse.ArgumentParser(description="Train Plantinia on the full PlantVillage dataset.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-classes", type=int, default=0, help="0 = all classes.")
    parser.add_argument("--max-images-per-class", type=int, default=0, help="0 = all images.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        raise SystemExit("--epochs and --batch must be positive.")
    if not DATA.is_dir():
        raise SystemExit(f"Dataset not found: {DATA}. Run scripts/bootstrap_models.ps1 first.")

    random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    folders = sorted(p for p in DATA.iterdir() if p.is_dir())
    if args.max_classes > 0:
        folders = folders[:args.max_classes]
    if len(folders) < 2:
        raise SystemExit("Expected at least two class folders in PlantVillage.")
    classes = [folder.name for folder in folders]
    train_samples, val_samples = [], []
    for label, folder in enumerate(folders):
        paths = sorted(str(p) for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in EXTENSIONS)
        rng.shuffle(paths)
        if args.max_images_per_class > 0:
            paths = paths[:args.max_images_per_class]
        if len(paths) < 2:
            raise SystemExit(f"Not enough images in class {folder.name}.")
        split = max(1, min(int(len(paths) * 0.9), len(paths) - 1))
        train_samples.extend((path, label) for path in paths[:split])
        val_samples.extend((path, label) for path in paths[split:])

    print(f"Using {len(classes)} classes: {', '.join(classes)}", flush=True)
    print(f"Training images: {len(train_samples):,}; held-out validation images: {len(val_samples):,}", flush=True)
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}; batch={args.batch}; epochs={args.epochs}", flush=True)

    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_loader = DataLoader(PlantDataset(train_samples, train_tf), batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(PlantDataset(val_samples, eval_tf), batch_size=args.batch, shuffle=False, num_workers=0)

    print("Loading pretrained MobileNetV3-Small weights...", flush=True)
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception as exc:
        raise SystemExit("Could not load pretrained torchvision weights. Connect to the internet once to download them, then retry. Details: " + str(exc))
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        model.train()
        total = correct = 0
        loss_sum = 0.0
        for batch_index, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            count = labels.size(0)
            total += count
            correct += (logits.argmax(1) == labels).sum().item()
            loss_sum += loss.item() * count
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(train_loader):
                print(f"Epoch {epoch}/{args.epochs} train {batch_index}/{len(train_loader)} "
                      f"loss={loss_sum/max(1,total):.4f} accuracy={correct/max(1,total):.2%}", flush=True)

        model.eval()
        val_total = val_correct = 0
        with torch.inference_mode():
            for batch_index, (images, labels) in enumerate(val_loader, 1):
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(1)
                val_total += labels.numel()
                val_correct += (predictions == labels).sum().item()
                if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(val_loader):
                    print(f"Epoch {epoch}/{args.epochs} validation {batch_index}/{len(val_loader)}", flush=True)

        accuracy = val_correct / max(1, val_total)
        payload = {
            "state_dict": model.state_dict(), "classes": classes,
            "architecture": "mobilenet_v3_small", "image_size": IMAGE_SIZE,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "epoch": epoch, "epochs_trained": args.epochs,
            "validation_accuracy": accuracy, "train_images": len(train_samples),
            "validation_images": len(val_samples), "seed": SEED,
        }
        save_checkpoint(payload, LATEST)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(payload, OUT)
            print(f"New best model saved: {OUT}", flush=True)
        print(f"Epoch {epoch} complete: train_accuracy={correct/max(1,total):.2%}, "
              f"validation_accuracy={accuracy:.2%}, time={(time.time()-epoch_started)/60:.1f} min", flush=True)

    print(f"TRAINING COMPLETE: {len(classes)} classes; best validation accuracy={best_accuracy:.2%}; "
          f"total time={(time.time()-started)/60:.1f} min", flush=True)
    print(f"Best model: {OUT}", flush=True)
    print("Validation is a held-out PlantVillage split, not proof of field accuracy.", flush=True)


if __name__ == "__main__":
    main()
