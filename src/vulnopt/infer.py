from __future__ import annotations

from pathlib import Path
import json
from typing import List
import torch

from .models.model import VulnOptModel
from .features.ast_features import ast_node_counter, vectorize
from .utils.suggestions import suggest_remediations
def iter_py_files(root:Path):
    for p in root.rglob('*.py'):
        if p.is_file(): yield p
def _load_metadata(ckpt: Path) -> dict:
    meta_path = ckpt.with_name('meta.json')
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _vector_for_code(code: str, vocab: List[str]) -> torch.Tensor:
    counter = ast_node_counter(code)
    vec = vectorize(counter, vocab)
    return torch.tensor([vec], dtype=torch.float32)


def infer_main(ckpt: Path, path: Path, out: Path, sarif: Path | None):
    meta = _load_metadata(ckpt)
    vocab = meta.get('ast_vocab', [])
    if not vocab:
        vocab = ['Module']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VulnOptModel(graph_dim=len(vocab)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    results = []
    for f in iter_py_files(path):
        code = f.read_text(errors='ignore')
        gfeat = _vector_for_code(code, vocab).to(device)
        batch = {'code': [code], 'gfeat': gfeat}
        with torch.no_grad():
            outp = model(batch)
            prob = float(outp['vuln'].sigmoid().cpu().item())
        if prob > 0.5:
            suggestions = suggest_remediations(code)
            results.append({'file': str(f), 'prob': prob, 'suggestions': suggestions})
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
