import torch.nn as nn
class Fusion(nn.Module):
    def __init__(self,d_txt,d_g,d_h=256):
        super().__init__()
        self.proj_txt=nn.Linear(d_txt,d_h)
        self.proj_g=nn.Linear(d_g,d_h)
        self.ff=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_h,nhead=8,batch_first=True),num_layers=2)
    def forward(self,h_txt,h_g):
        import torch
        x=torch.stack([self.proj_txt(h_txt),self.proj_g(h_g)],dim=1)
        return self.ff(x)[:,0]
