from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

def resnet18_model(num_classes=7, dropout_p=0.6):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(512, num_classes)
    )
    return model