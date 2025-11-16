import torch.nn as nn
import torchvision.models as models

class ResNet50_Setup3(nn.Module):
    """
    Setup 3:
    - Full fine-tuning for AffectNet YOLO dataset.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        in_f = self.model.fc.in_features
        self.model.fc = nn.Linear(in_f, num_classes)

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)




import torch.nn as nn
import torchvision.models as models

class EfficientNetB0_Setup3(nn.Module):
    """
    Setup 3: Full fine-tuning on AffectNet.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        in_f = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_f, num_classes)

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)





import torch.nn as nn
import torchvision.models as models

class MobileNetV3_Setup3(nn.Module):
    """
    Setup 3: Full fine-tuning on AffectNet.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        in_f = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_f, num_classes)

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)
