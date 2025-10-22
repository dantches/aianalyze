import torch.nn as nn
from .encoders import CodeBERTEncoder
from .fusion import Fusion
from .heads import Heads
class VulnOptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt=CodeBERTEncoder()
        self.gdim=256
        self.fusion=Fusion(self.txt.dim,self.gdim,256)
        self.heads=Heads(256)
    def forward(self,batch):
        h_txt=self.txt(batch['code'])
        h_g=batch['gfeat']
        h=self.fusion(h_txt,h_g)
        return self.heads(h)
