import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SimpleYOLOClassDataset(Dataset):
    def __init__(self, root_dir, transform=None, skip_class=None, class_map=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.transform = transform
        self.skip_class = [skip_class] if isinstance(skip_class, int) else skip_class
        self.class_map = class_map

        self.samples = []

        for f in os.listdir(self.img_dir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            label_path = os.path.join(self.label_dir, os.path.splitext(f)[0] + ".txt")
            if not os.path.exists(label_path):
                continue

            with open(label_path, "r") as file:
                lines = file.readlines()
                if not lines:
                    continue

                cls = int(lines[0].split()[0])

                # Apply remapping if provided
                if self.class_map and cls in self.class_map:
                    cls = self.class_map[cls]

                # Skip unwanted class
                if self.skip_class and cls in self.skip_class:
                    continue

                self.samples.append((os.path.join(self.img_dir, f), cls))

        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# Define transforms (same as before)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_mapping = {
    0: 5,
    1: 99,
    2: 2,
    3: 1,
    4: 3,
    5: 6,
    6: 4,
    7: 0
}

# Create datasets
train_dataset = SimpleYOLOClassDataset(
    "DATASETS_FOR_TRAINING/affectnet-yolo-format/YOLO_format/train",
    transform=train_transform,
    #skip_class=1,  # skip 'Contempt'
    class_map=class_mapping
 
)

val_dataset = SimpleYOLOClassDataset(
    "DATASETS_FOR_TRAINING/affectnet-yolo-format/YOLO_format/test",
    transform=val_transform,
    #skip_class=1,
    class_map=class_mapping
)



# DataLoaders
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
