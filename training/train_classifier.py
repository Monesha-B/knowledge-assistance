from pathlib import Path
import mlflow
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader

# Small demonstration data. Add many realistic labeled examples before judging accuracy.
EXAMPLES = {
    'order': ['Where is my order?', 'Track my package', 'When will delivery arrive?', 'My parcel is delayed', 'Can I change the shipping address?', 'Order has not arrived'],
    'refund': ['I want my money back', 'How do I return an item?', 'When will my refund arrive?', 'What is the refund policy?', 'I returned the shoes', 'I need a refund'],
    'payment': ['My card was charged twice', 'What payment methods do you accept?', 'Invoice is incorrect', 'Payment failed', 'How can I update my card?', 'Why was I charged?'],
    'other': ['What are your office hours?', 'Can I contact support?', 'Tell me about your company', 'Do you have stores?', 'How can I give feedback?', 'Hello']
}

def main():
    labels = list(EXAMPLES)
    rows = [(text, index) for index, label in enumerate(labels) for text in EXAMPLES[label]]
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4, id2label=dict(enumerate(labels)), label2id={v:k for k,v in enumerate(labels)})
    dataset = Dataset.from_dict({'text':[x[0] for x in rows], 'label':[x[1] for x in rows]})
    def collate(items):
        batch = tokenizer([x['text'] for x in items], padding=True, truncation=True, max_length=128, return_tensors='pt')
        batch['labels'] = torch.tensor([x['label'] for x in items])
        return batch
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    mlflow.set_experiment('rag-query-classifier')
    with mlflow.start_run():
        mlflow.log_params({'model':'distilbert-base-uncased','epochs':2,'samples':len(rows)})
        model.train()
        for epoch in range(2):
            total = 0.0
            for batch in loader:
                optimizer.zero_grad()
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                total += loss.item()
            mlflow.log_metric('training_loss',total/len(loader),step=epoch)
            print(f'Epoch {epoch+1}: loss={total/len(loader):.4f}')
        output = Path('training/classifier_output')
        output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        mlflow.log_artifacts(str(output), artifact_path='classifier')
        print('Saved classifier:', output)
        print('Warning: 24 example texts are demonstration-only; do not treat model as production-accurate.')

if __name__ == '__main__':
    main()
