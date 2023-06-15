from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models

# class CelebAModel(torch.nn.Module):
#     """
#     CelebA attributes (Attractive, High_Cheekbones, Mouth_Slightly_Open, Smiling) predictions
#     (binary perdiction) [0:4]
#     """
    
#     def __init__(self, out_feature=4, weights='MobileNet_V3_Small_Weights.DEFAULT'):
#         super(CelebAModel, self).__init__()
#         self.model = models.mobilenet_v3_small(weights=weights)
#         in_feature = self.model.classifier[0].in_features
#         self.model.classifier = nn.Sequential(OrderedDict([
#             ('fc', nn.Linear(in_feature, out_feature)),
#             ('sigm', nn.Sigmoid()),
#         ]))

#     def forward(self, x):
#         """
#         Input: 
#             x: Image of faces        (N, C, H, W)
#         Output:
#             z: attribute predictions (N, 4)
#         """
#         z = self.model(x)
#         return z

# class UTKFaceModel(torch.nn.Module):
#     """
#     UTKface attributes (race, gender, age) predictions
#     race: white, black, asian, indian, others -> [0:5]
#     gender: male, female -> [5:7]
#     age: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, more than 70 -> [7:16]
#     """

#     def __init__(self, out_feature=16, weights='MobileNet_V3_Small_Weights.DEFAULT'):
#         super(UTKFaceModel, self).__init__()
#         self.model = models.mobilenet_v3_small(weights=weights)
#         in_feature = self.model.classifier[0].in_features
#         self.model.classifier = nn.Sequential(OrderedDict([
#             ('fc', nn.Linear(in_feature, out_feature)),
#         ]))
    
#     def forward(self, x):
#         """
#         Input:
#             x: Image of faces        (N, C, H, W)
#         Output:
#             z: attribute predictions (N, 16)
#         """
#         z = self.model(x)
#         return z

# class FairFaceModel(torch.nn.Module):
#     """
#     FairFace attributes (race, gender, age) predictions
#     race: White, Black, Latino_Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern -> [0:7]
#     gender: male, female -> [7:9]
#     age: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, more than 70 -> [9:18]
#     """

#     def __init__(self, out_feature=18, weights='MobileNet_V3_Small_Weights.DEFAULT'):
#         super(FairFaceModel, self).__init__()
#         self.model = models.mobilenet_v3_small(weights=weights)
#         in_feature = self.model.classifier[0].in_features
#         self.model.classifier = nn.Sequential(OrderedDict([
#             ('fc', nn.Linear(in_feature, out_feature)),
#         ]))

#     def forward(self, x):
#         """
#         Input:
#             x: Image of faces        (N, C, H, W)
#         Output:
#             z: attribute predictions (N, 18)
#         """
#         z = self.model(x)
#         return z

# class AgeModel(torch.nn.Module):
#     """
#     Age attribute predictions
#     (binary perdiction) [0:1], 0-29 : 0, 30+ : 1
#     """

#     def __init__(self, out_feature=1, weights='MobileNet_V3_Small_Weights.DEFAULT'):
#         super(AgeModel, self).__init__()
#         self.model = models.mobilenet_v3_small(weights=weights)
#         in_feature = self.model.classifier[0].in_features
#         self.model.classifier = nn.Sequential(OrderedDict([
#             ('fc', nn.Linear(in_feature, out_feature)),
#             ('sigm', nn.Sigmoid()),
#         ]))
    
#     def forward(self, x):
#         """
#         Input:
#             x: Image of faces        (N, C, H, W)
#         Output:
#             z: attribute predictions (N, 1)
#         """
#         z = self.model(x)
#         return z

class CelebAModel(torch.nn.Module):
    """
    CelebA attributes (Attractive, High_Cheekbones, Mouth_Slightly_Open, Smiling) predictions
    (binary perdiction) [0:4]
    """
    
    def __init__(self, out_feature=4, weights='ResNet50_Weights.DEFAULT'):
        super(CelebAModel, self).__init__()
        self.model = models.resnet50(weights=weights)
        in_feature = self.model.fc.in_features
        self.model.fc = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_feature, out_feature)),
            ('sigm', nn.Sigmoid()),
        ]))

    def forward(self, x):
        """
        Input: 
            x: Image of faces        (N, C, H, W)
        Output:
            z: attribute predictions (N, 4)
        """
        z = self.model(x)
        return z

class UTKFaceModel(torch.nn.Module):
    """
    UTKface attributes (race, gender, age) predictions
    race: white, black, asian, indian, others -> [0:5]
    gender: male, female -> [5:7]
    age: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, more than 70 -> [7:16]
    """

    def __init__(self, out_feature=16, weights='ResNet50_Weights.DEFAULT'):
        super(UTKFaceModel, self).__init__()
        self.model = models.resnet50(weights=weights)
        in_feature = self.model.fc.in_features
        self.model.fc = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_feature, out_feature)),
        ]))
    
    def forward(self, x):
        """
        Input:
            x: Image of faces        (N, C, H, W)
        Output:
            z: attribute predictions (N, 16)
        """
        z = self.model(x)
        return z

class FairFaceModel(torch.nn.Module):
    """
    FairFace attributes (race, gender, age) predictions
    race: White, Black, Latino_Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern -> [0:7]
    gender: male, female -> [7:9]
    age: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, more than 70 -> [9:18]
    """

    def __init__(self, out_feature=18, weights='ResNet50_Weights.DEFAULT'):
        super(FairFaceModel, self).__init__()
        self.model = models.resnet50(weights=weights)
        in_feature = self.model.fc.in_features
        self.model.fc = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_feature, out_feature)),
        ]))

    def forward(self, x):
        """
        Input:
            x: Image of faces        (N, C, H, W)
        Output:
            z: attribute predictions (N, 18)
        """
        z = self.model(x)
        return z
    
class MAADFaceHQModel(torch.nn.Module):
    """
    Entire MAADFaceHQ attribute predictions (without Male)
    """

    def __init__(self, out_feature=46, weights='ResNet50_Weights.DEFAULT'):
        super(MAADFaceHQModel, self).__init__()
        self.model = models.resnet50(weights=weights)
        in_feature = self.model.fc.in_features
        self.model.fc = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_feature, out_feature)),
            ('sigm', nn.Sigmoid()),
        ]))

    def forward(self, x):
        """
        Input:
            x: Image of faces        (N, C, H, W)
        Output:
            z: attribute predictions (N, 47)
        """
        z = self.model(x)
        return z

class AgeModel(torch.nn.Module):
    """
    Age attribute predictions
    (binary perdiction) [0:1], 0-29 : 0, 30+ : 1
    """

    def __init__(self, out_feature=1, weights='ResNet50_Weights.DEFAULT'):
        super(AgeModel, self).__init__()
        self.model = models.resnet50(weights=weights)
        in_feature = self.model.fc.in_features
        self.model.fc = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_feature, out_feature)),
            ('sigm', nn.Sigmoid()),
        ]))
    
    def forward(self, x):
        """
        Input:
            x: Image of faces        (N, C, H, W)
        Output:
            z: attribute predictions (N, 1)
        """
        z = self.model(x)
        return z