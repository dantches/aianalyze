from pathlib import Path
import json, random
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .models.model import VulnOptModel
from .features.ast_graph import python_ast_graph
class JsonlDataset(Dataset):
    def __init__(self,path:Path):
        self.rows=[json.loads(l) for l in path.read_text(encoding='utf8').splitlines() if l.strip()]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; code=r.get('code','')
        import collections, torch
        G=python_ast_graph(code)
        cnt=collections.Counter(n for _,n in [(None,d.get('type')) for _,d in G.nodes(data=True)] if n)
        vec=[cnt.get(k,0) for k in list(cnt.keys())[:256]]
        vec += [0]*(256-len(vec))
        return {'code':code,'gfeat':torch.tensor(vec,dtype=torch.float32)}, torch.tensor([r.get('label',0)],dtype=torch.float32)
def train_main(data:Path,out:Path):
    ds=JsonlDataset(data)
    loader=DataLoader(ds,batch_size=2,shuffle=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=VulnOptModel().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5)
    bce=nn.BCEWithLogitsLoss()
    for epoch in range(1):
        model.train()
        for batch,y in loader:
            batch={k:(v.to(device) if hasattr(v,'to') else v) for k,v in batch.items()}
            y=y.to(device)
            outp=model(batch)
            loss=bce(outp['vuln'].squeeze(-1),y.squeeze(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(model.state_dict(), out / 'best.ckpt')
