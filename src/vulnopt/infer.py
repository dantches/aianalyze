from pathlib import Path
import json, numpy as np, torch
from .models.model import VulnOptModel
from .features.ast_graph import python_ast_graph
def iter_py_files(root:Path):
    for p in root.rglob('*.py'):
        if p.is_file(): yield p
def infer_main(ckpt:Path,path:Path,out:Path,sarif:Path|None):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=VulnOptModel().to(device)
    model.load_state_dict(torch.load(ckpt,map_location=device))
    model.eval()
    results=[]
    for f in iter_py_files(path):
        code=f.read_text(errors='ignore')
        import collections
        G=python_ast_graph(code)
        cnt=collections.Counter(n for _,n in [(None,d.get('type')) for _,d in G.nodes(data=True)] if n)
        keys=list(cnt.keys())[:128]
        vec=[cnt.get(k,0) for k in keys]+[0]*(256-len(keys))
        batch={'code':[code],'gfeat':torch.tensor([vec],device=device,dtype=torch.float32)}
        with torch.no_grad():
            outp=model(batch)
            prob=float(outp['vuln'].sigmoid().cpu().numpy().ravel()[0])
        if prob>0.5:
            results.append({'file':str(f),'prob':prob})
    out.write_text(json.dumps(results,ensure_ascii=False,indent=2))
