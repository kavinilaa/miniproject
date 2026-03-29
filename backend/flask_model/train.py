"""
Training Script — Multimodal Polyp Detector
─────────────────────────────────────────────
Uses synthetic data since no real labelled dataset is available.
Replace SyntheticPolypDataset with your real dataset loader.

Run:
    python train.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from model import MultimodalPolypDetector

# ── Config ────────────────────────────────────────────────────────────────
EPOCHS      = 20
BATCH_SIZE  = 16
LR          = 1e-4
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH   = "multimodal_polyp_model.pth"
NUM_SAMPLES = 1000   # synthetic samples — replace with real data size

print(f"Training on: {DEVICE}")


# ── Synthetic Dataset ─────────────────────────────────────────────────────
class SyntheticPolypDataset(Dataset):
    """
    Generates synthetic (image, clinical, label) triplets for demonstration.
    Replace this class with your real colonoscopy dataset.

    Real dataset options:
      - Kvasir-SEG  : https://datasets.simula.no/kvasir-seg/
      - CVC-ClinicDB: https://polyp.grand-challenge.org/
      - ETIS-Larib  : https://polyp.grand-challenge.org/
    """

    def __init__(self, n_samples: int, transform=None):
        self.n       = n_samples
        self.transform = transform
        # Pre-generate labels: 50% polyp
        self.labels  = torch.randint(0, 2, (n_samples,)).float()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        label = self.labels[idx].item()

        # Synthetic image: polyp images have slightly higher red channel
        img_array = np.random.randint(50, 200, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        if label == 1:
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] + 40, 0, 255)
        img = Image.fromarray(img_array)
        if self.transform:
            img = self.transform(img)

        # Synthetic clinical vector: polyp patients have higher risk factors
        clinical = torch.zeros(MultimodalPolypDetector.CLINICAL_DIM)
        if label == 1:
            clinical[0] = torch.rand(1) * 0.4 + 0.5   # older age
            clinical[6] = torch.rand(1) * 0.5 + 0.5   # smoking
            clinical[7] = torch.rand(1) * 0.5 + 0.5   # family history
            clinical[4] = torch.rand(1) * 0.3 + 0.3   # CRP elevated
        else:
            clinical = torch.rand(MultimodalPolypDetector.CLINICAL_DIM) * 0.4

        return img, clinical, torch.tensor(label)


# ── Transforms ────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset & Loaders ─────────────────────────────────────────────────────
full_ds  = SyntheticPolypDataset(NUM_SAMPLES, transform=train_tf)
val_size = int(0.2 * NUM_SAMPLES)
train_ds, val_ds = random_split(full_ds, [NUM_SAMPLES - val_size, val_size])
val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ── Model, Loss, Optimizer ────────────────────────────────────────────────
model     = MultimodalPolypDetector(freeze_backbone=True).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ── Training Loop ─────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, clinical, labels in loader:
            imgs     = imgs.to(DEVICE)
            clinical = clinical.to(DEVICE)
            labels   = labels.to(DEVICE).unsqueeze(1)

            logits = model(imgs, clinical)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds       = (torch.sigmoid(logits) > 0.5).float()
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(val_loader,   train=False)
    scheduler.step()

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✔ Model saved (val_acc={val_acc:.3f})")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
print(f"Model saved to: {SAVE_PATH}")
