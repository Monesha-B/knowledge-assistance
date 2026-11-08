from app.documents import load_chunks
from app.classifier import classify

def test_chunks():
    chunks = load_chunks()
    assert len(chunks) >= 3
    assert all(c['source'].endswith('.txt') and c['text'] for c in chunks)

def test_classifier_onboarding_fallback():
    # Only evaluates fallback before the user trains the optional model.
    assert classify('I need a refund') in {'refund','order','payment','other'}
