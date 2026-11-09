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
    q = question.lower()

    # Give explicit business terms priority over the small ML model.
    rules = {
        "refund": ["refund", "return", "money back", "reimbursement"],
        "order": ["order", "delivery", "package", "shipping", "tracking"],
        "payment": ["payment", "card", "charged", "invoice", "billing"],
    }

    for category, keywords in rules.items():
        if any(keyword in q for keyword in keywords):
            return category

    # Use fine-tuned DistilBERT for less obvious questions.
    bundle = model_bundle()
    if bundle is None:
        return "other"

    import torch

    tokenizer, model = bundle
    tokens = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        logits = model(**tokens).logits
        index = logits.argmax(dim=-1).item()

    return model.config.id2label[index]
