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

