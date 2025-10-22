from __future__ import annotations

from pathlib import Path
import json
from collections import Counter
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .models.model import VulnOptModel
from .features.ast_features import ast_node_counter, build_vocab, vectorize


class JsonlDataset(Dataset):
    def __init__(self, path: Path, vocab: Optional[List[str]] = None, limit: int = 256):
        self.rows = [json.loads(l) for l in path.read_text(encoding='utf8').splitlines() if l.strip()]
        self.limit = limit
        self._counters: List[Counter[str]] = [Counter() for _ in self.rows]

        if vocab is None:
            counters = [ast_node_counter(r.get('code', '')) for r in self.rows]
            self._counters = counters
            self.vocab = build_vocab(counters, limit=limit)
        else:
            self.vocab = list(vocab)
            if not self.vocab:
                self.vocab = ['Module']
        self.g_dim = len(self.vocab)

    def __len__(self):
        return len(self.rows)

    def _counter(self, idx: int) -> Counter[str]:
        if not self._counters[idx]:
            self._counters[idx] = ast_node_counter(self.rows[idx].get('code', ''))
        return self._counters[idx]

    def __getitem__(self, i):
        r = self.rows[i]
        code = r.get('code', '')
        counter = self._counter(i)
        vec = vectorize(counter, self.vocab)
        gfeat = torch.tensor(vec, dtype=torch.float32)
        label = torch.tensor(float(r.get('label', 0)), dtype=torch.float32)
        return {'code': code, 'gfeat': gfeat}, label


def train_main(data: Path, out: Path, batch_size: int = 4, epochs: int = 1, lr: float = 2e-5):
    ds = JsonlDataset(data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VulnOptModel(graph_dim=ds.g_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        for batch, y in loader:
            batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
            y = y.to(device)
            outp = model(batch)
            logits = outp['vuln'].squeeze(-1)
            loss = bce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / 'best.ckpt')
    metadata = {'graph_dim': ds.g_dim, 'ast_vocab': ds.vocab}
    (out / 'meta.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
