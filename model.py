import torch
import torch.nn as nn
#from torchsummary import summary
from backbone import Backbone
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.backbone = Backbone().to('cuda')
        self.dense = nn.Sequential(nn.Linear(7*7*1024,  2048)  , nn.LeakyReLU(0.1) , nn.Dropout(0.5) ,   nn.Linear(2048,  7*7*30)  ).to('cuda')
    def forward(self , x ):
        x = self.backbone(x )
        
        x = self.dense(x)
        x = x.view((-1 , 7,7,30))
        return torch.sigmoid(x )

if __name__ == "__main__":
    model = Model()
    