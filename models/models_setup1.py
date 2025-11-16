import torch.nn as nn
import torchvision.models as models

class ResNet50_Setup1(nn.Module):
    """
    Setup 1:
    - Freeze pretrained backbone
    - Train classifier head only
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze all backbone weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace final classifier
        in_f = self.model.fc.in_features
        self.model.fc = nn.Linear(in_f, num_classes)

        # Unfreeze classifier head
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)




import torch.nn as nn
import torchvision.models as models

class EfficientNetB0_Setup1(nn.Module):
    """
    Setup 1: EfficientNet-B0 with frozen backbone.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Freeze backbone
        for p in self.model.parameters():
            p.requires_grad = False

        # Replace classifier head
        in_f = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_f, num_classes)

        # Unfreeze head only
        for p in self.model.classifier[1].parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)




import torch.nn as nn
import torchvision.models as models

class MobileNetV3_Setup1(nn.Module):
    """
    Setup 1: MobileNetV3-Large, frozen backbone.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )

        # Freeze backbone
        for p in self.model.parameters():
            p.requires_grad = False

        # Replace classifier
        in_f = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_f, num_classes)

        # Unfreeze head
        for p in self.model.classifier[3].parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)
