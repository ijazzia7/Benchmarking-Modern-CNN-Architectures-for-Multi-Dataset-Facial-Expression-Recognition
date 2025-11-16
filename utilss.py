import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Random seed set to {seed}")



import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def show_tensor_images_orig(image_tensor, num_images=25, nrow=5, show=True):
    plt.figure(figsize=(8, 8))
    image_tensor = (image_tensor + 1) / 2
    img = make_grid(image_tensor[:num_images], nrow=nrow).permute(1, 2, 0).cpu()
    plt.imshow(img)
    plt.axis("off")
    if show:
        plt.show()


def show(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.clip(std * img + mean, 0, 1)
    plt.imshow(img)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()



import matplotlib.pyplot as plt

def plot_metrics(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'o-', label='Train Loss')
    plt.plot(epochs, history["val_loss"], 'o-', label='Val Loss')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], 'o-', label='Train Acc')
    plt.plot(epochs, history["val_acc"], 'o-', label='Val Acc')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

data_dir = "DATASETS_FOR_TRAINING/raf-db-dataset/DATASET"
train_dir = f"{data_dir}/train"
test_dir = f"{data_dir}/test"


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)



train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)


images, labels = next(iter(train_loader))
print("Train batch shape:", images.shape)
print("Labels:", labels[:8])



from collections import Counter
import torch

# Count how many samples per class
label_counts = Counter(train_dataset.targets)
num_samples = len(train_dataset)
num_classes = len(train_dataset.classes)
class_counts = torch.tensor([label_counts[i] for i in range(num_classes)], dtype=torch.float)

class_weights = num_samples / (num_classes * class_counts)
class_weights = class_weights / class_weights.sum() * num_classes  # optional normalization

print("Class weights:", class_weights)


label_to_emotion = {
'surprise':0,
'fear':1,
'disgust':2,
'happy':3,
'sad':4,
'angry':5,
'neutral':6
}
label_to_emotion
