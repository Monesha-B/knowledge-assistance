import os
import torch
from transformers import AutoModel, AutoTokenizer
from app.config import load_local_env

_model = None
_tokenizer = None

def embed(texts: list[str]) -> list[list[float]]:
    global _model, _tokenizer
    if not texts:
        return []
    if _model is None:
        name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        _tokenizer = AutoTokenizer.from_pretrained(name)
        _model = AutoModel.from_pretrained(name).eval()
    inputs = _tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
    with torch.no_grad():
        states = _model(**inputs).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        vectors = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return vectors.cpu().tolist()
