from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
import torch

model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)

print(F.softmax(model(torch.rand(1, 3, 50, 50))).argmax())