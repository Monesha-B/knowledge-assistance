from fastapi.testclient import TestClient
from app.api import app
from app.schemas import Answer, Citation

client = TestClient(app)

def test_health():
    assert client.get('/health').json() == {'status': 'ok'}

def test_invalid_question():
    assert client.post('/ask', json={'question': ''}).status_code == 422

def test_answer_schema():
    a = Answer(question='Where is my order?', category='order', answer='Check tracking.', citations=[Citation(source='orders.txt', excerpt='Tracking number')],model='test')
    assert a.citations[0].source == 'orders.txt'
