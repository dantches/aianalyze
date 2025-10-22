from __future__ import annotations

from pathlib import Path
import json

import torch

from .train import JsonlDataset
from .models.model import VulnOptModel


def _load_metadata(ckpt: Path) -> dict:
    meta_path = ckpt.with_name('meta.json')
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def eval_main(ckpt: Path, data: Path):
    meta = _load_metadata(ckpt)
    vocab = meta.get('ast_vocab')
    ds = JsonlDataset(data, vocab=vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VulnOptModel(graph_dim=meta.get('graph_dim', ds.g_dim)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    tp = fp = tn = fn = 0
    for i in range(len(ds)):
        batch, y = ds[i]
        if isinstance(batch.get('code'), str):
            batch['code'] = [batch['code']]
        if hasattr(batch.get('gfeat'), 'dim') and batch['gfeat'].dim() == 1:
            batch['gfeat'] = batch['gfeat'].unsqueeze(0)
        batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
        y = y.to(device)
        p = (model(batch)['vuln'].sigmoid() > 0.5).float()
        if p.item() == 1 and y.item() == 1:
            tp += 1
        elif p.item() == 1 and y.item() == 0:
            fp += 1
        elif p.item() == 0 and y.item() == 0:
            tn += 1
        else:
            fn += 1
    print({'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn})
