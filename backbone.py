import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4 , 
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(2048 , 1024 ,kernel_size=(1, 1 ) ), 
            nn.Flatten()
        )
        

    def forward(self , x):
        x = self.backbone(x)
       
        return  x

if __name__ == "__main__":
    backbone = Backbone().to('cuda')
    print(summary(backbone   ,  (3, 448,448),  device='cuda'))

