import torch
import torch.nn as nn
import torchvision.models as models
#from torchsummary import summary

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = torch.nn.Identity()

    def forward(self , x):
        x = self.backbone(x)
        return  x

if __name__ == "__main__":
    backbone = Backbone().to('cuda')
    #print(summary(backbone   ,  (3, 448,448),  device='cuda'))