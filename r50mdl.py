from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models

model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
print(model)

model = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT')
print(model)