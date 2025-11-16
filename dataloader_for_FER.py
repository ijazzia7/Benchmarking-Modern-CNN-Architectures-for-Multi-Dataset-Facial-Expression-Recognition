from torch.utils.data import Dataset

# -----------------------------
# Data augmentations and normalization
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Randomly flip images for augmentation
    transforms.RandomRotation(20),      # Add rotation variability
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # Standard ImageNet normalization
])

# Validation transform — no augmentation, only normalization
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# Load FER2013 dataset using ImageFolder
# -----------------------------
train_base = datasets.ImageFolder("DATASETS_FOR_TRAINING/fer2013/train", transform=transform)
val_base   = datasets.ImageFolder("DATASETS_FOR_TRAINING/fer2013/test", transform=val_transforms)

# Remap original class indices to your emotion label order
label_map = {train_base.class_to_idx[k]: v for k, v in label_to_emotion.items()}

# -----------------------------
# Custom Dataset to apply remapped labels
# -----------------------------
class RemappedFERDataset(Dataset):
    def __init__(self, base_dataset, label_map):
        self.base = base_dataset
        self.label_map = label_map

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, self.label_map[y]   # Replace old label with remapped one

    def __len__(self):
        return len(self.base)

# -----------------------------
# Final datasets and dataloaders
# -----------------------------
train_dataset = RemappedFERDataset(train_base, label_map)
val_dataset   = RemappedFERDataset(val_base, label_map)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
