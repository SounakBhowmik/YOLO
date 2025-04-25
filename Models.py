# -*- coding: utf-8 -*-

from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights # These are the models, can be found at the given link above
from torchsummary import summary
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mobilenet_v3 = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)

#print(mobilenet_v3)



# Define models 

import torch
import torch.nn as nn

'''
class SimpleCNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(SimpleCNN, self).__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Conv2d(3, 25, 32, 69)
        )
    def forward(self, x):
        x = self.model(x).permute(0, 2, 3, 1)
        return x


class FastYOLO_mobile(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(FastYOLO_mobile, self).__init__(*args, **kwargs)
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        for p in self.model.features[:-4].parameters():
            p.requires_grad = False

        for p in self.model.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(p=.02, inplace=True),
            nn.Linear(1280, 25*7*7)
        )

        self.head = nn.Sequential(
            nn.Conv2d(960, 1280, 3, padding=1),
            nn.Hardswish(),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.3),
            nn.Conv2d(1280, 25, 1, 1)  # Directly produce (7,7,25)
        )

    def forward(self, x):
        x = self.model.features(x)
        x = self.head(x)
        x = torch.reshape(x, (-1, 7, 7, 25))

        return x
'''

class FastYOLO_mobile01(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(FastYOLO_mobile01, self).__init__(*args, **kwargs)
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).features
        self.l_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)
        
        
        for p in self.model[:13].parameters():
            p.requires_grad = False
            
            
        self.head = nn.Sequential(
            nn.Conv2d(960, 1024, 1, 1),  # 1024, 14, 14
            nn.BatchNorm2d(1024),
            self.l_relu,
            
            nn.Conv2d(1024, 1024, 1,1), # 1024, 12, 12
            nn.BatchNorm2d(1024),
            self.l_relu,
            
            nn.Conv2d(1024, 1024, 1,1), # 1024, 10, 10
            nn.BatchNorm2d(1024),
            self.l_relu,
            self.dropout,
            
            nn.Conv2d(1024, 1024, 2,2), # 1024, 6, 6
            nn.BatchNorm2d(1024),
            self.l_relu,
            self.dropout,
            
            nn.Flatten(),  # Convert (B, 1280, 1, 1) → (B, 1280)
            nn.Linear(1024*7*7, 4096),
            self.l_relu,
            self.dropout,
            
            nn.Linear(4096, 25*7*7),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        x = torch.reshape(x, (-1, 7, 7, 25))

        return x

'''
class FastYOLO_mobile(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(FastYOLO_mobile, self).__init__(*args, **kwargs)
        pretrained_module_list = [m for m in mobilenet_v3.modules()]
        self.features = nn.Sequential(*list(mobilenet_v3.features.children()))
        for p in self.features.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Conv2d(1280, 512, 2, 2),
            nn.ReLU(),
            nn.Conv2d(512, 128, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*7*7, 64*7*7),
            nn.ReLU(),
            nn.Linear(64*7*7, 25*7*7)
        )
    def forward(self, x):
        x = self.features(x)
        #x = self.head(x)
        #x = torch.reshape(x, (-1, 7, 7, 25))

        return x

class FastYOLO_vit(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(FastYOLO_vit, self).__init__(*args, **kwargs)
        pretrained_module_list = [m for m in vit.modules()]
        self.features = nn.Sequential(*list(vit.encoder.children()))

        for p in self.features.parameters():
            p.requires_grad = False

        self.tail = nn.Sequential(nn.Conv2d(3,1280,16,12), nn.ReLU())
        self.head = nn.Sequential(
            nn.Conv2d(1280, 512, 2, 2), #512, 18, 18
            nn.ReLU(),
            nn.Conv2d(512, 64, 2, 2), #64, 9, 9
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*9*9, 25*7*7)
        )
    def forward(self, x):
        x = self.tail(x)
        x = self.features(x)
        x = self.head(x)
        x = torch.reshape(x, (-1, 7, 7, 25))

        return x


class FastYOLO_resnet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(FastYOLO_resnet, self).__init__(*args, **kwargs)
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.fc = nn.Linear(2048, 25*7*7)
    def forward(self, x):
        x = self.model(x)
        x = torch.reshape(x, (-1, 7, 7, 25))

        return x
'''

'''
# Test with dummy tensors to check if you get the expected dimensions
input_dim = (1, 3, 448, 448)
output_dim = (1, 7, 7, 25)

model = FastYOLO_mobile01()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

op = model(torch.rand(1, 3, 448, 448).to(device))
print(f'o/p shape is {op.shape} and the expected shape is {output_dim}')
'''



