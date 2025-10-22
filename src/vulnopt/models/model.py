import torch.nn as nn
from .encoders import CodeBERTEncoder
from .fusion import Fusion
from .heads import Heads
class VulnOptModel(nn.Module):
    def __init__(self, graph_dim:int=256, fusion_hidden:int=256, encoder_name:str='microsoft/codebert-base'):
        super().__init__()
        self.txt=CodeBERTEncoder(name=encoder_name)
        self.gdim=graph_dim
        self.fusion=Fusion(self.txt.dim,self.gdim,fusion_hidden)
        self.heads=Heads(fusion_hidden)
    def forward(self,batch):
        h_txt=self.txt(batch['code'])
        h_g=batch['gfeat']
        h=self.fusion(h_txt,h_g)
        return self.heads(h)
