import torch.nn as nn
class Heads(nn.Module):
    def __init__(self,d,n_cwe=50):
        super().__init__()
        self.vuln=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Linear(d,1))
    def forward(self,h):
        import torch
        return {'vuln': self.vuln(h)}
