import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
class CodeBERTEncoder(nn.Module):
    def __init__(self, name='microsoft/codebert-base'):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.dim = self.model.config.hidden_size
    def forward(self,texts):
        batch = self.tok(texts, padding=True, truncation=True, max_length=384, return_tensors='pt')
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        out = self.model(**batch)
        return out.last_hidden_state[:,0]
