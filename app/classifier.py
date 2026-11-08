import os
from pathlib import Path
from functools import lru_cache

LABELS = ['order', 'refund', 'payment', 'other']

@lru_cache(maxsize=1)
def model_bundle():
    path = Path(os.getenv('CLASSIFIER_DIR', 'training/classifier_output'))
    if not (path/'config.json').exists():
        return None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    return tokenizer, model

def classify(question):
    bundle = model_bundle()
    if bundle is None:
        # Safe onboarding fallback until training completes; clearly not a trained classifier.
        q = question.lower()
        if any(word in q for word in ['refund', 'return', 'money back']): return 'refund'
        if any(word in q for word in ['order', 'delivery', 'package', 'shipping']): return 'order'
        if any(word in q for word in ['payment', 'card', 'charged', 'invoice']): return 'payment'
        return 'other'
    import torch
    tokenizer, model = bundle
    tokens = tokenizer(question, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        index = model(**tokens).logits.argmax(dim=-1).item()
    return model.config.id2label[index]
