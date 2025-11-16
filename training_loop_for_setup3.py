import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models.mobilenet_setup3 import MobileNetV3_Setup3
from training.train_utils_setup3 import train_one_epoch, validate, create_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Data transforms
# -----------------------------
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder("DATASETS/raf/train", transform=train_tf)
test_dataset = datasets.ImageFolder("DATASETS/raf/test", transform=test_tf)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# -----------------------------
# Model
# -----------------------------
model = MobileNetV3_Setup3().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = create_scheduler(optimizer, train_loader, num_epochs=5)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(5):
    print(f"\nEpoch {epoch+1}/5")

    t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    v_loss, v_acc = validate(model, val_loader, criterion, device)

    scheduler.step()

    history["train_loss"].append(t_loss)
    history["train_acc"].append(t_acc)
    history["val_loss"].append(v_loss)
    history["val_acc"].append(v_acc)

    print(f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f}")
    print(f"Val Loss:   {v_loss:.4f} | Val Acc:   {v_acc:.4f}")

torch.save(model.state_dict(), "results/models/mobilenet_setup3.pth")
print("Model saved!")
