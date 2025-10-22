from pathlib import Path
import torch, json
from .train import JsonlDataset
from .models.model import VulnOptModel
def eval_main(ckpt:Path,data:Path):
    ds=JsonlDataset(data)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=VulnOptModel().to(device)
    model.load_state_dict(torch.load(ckpt,map_location=device))
    model.eval()
    tp=fp=tn=fn=0
    for i in range(len(ds)):
        batch,y=ds[i]
        batch={k:(v.to(device) if hasattr(v,'to') else v) for k,v in batch.items()}
        y=y.to(device)
        p=(model(batch)['vuln'].sigmoid()>0.5).float()
        if p.item()==1 and y.item()==1: tp+=1
        elif p.item()==1 and y.item()==0: fp+=1
        elif p.item()==0 and y.item()==0: tn+=1
        else: fn+=1
    print({'tp':tp,'fp':fp,'tn':tn,'fn':fn})
