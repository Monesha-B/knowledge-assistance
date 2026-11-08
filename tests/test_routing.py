from unittest.mock import patch
from app.schemas import Answer


def test_general_question_skips_retrieval(monkeypatch):
    from app import service
    monkeypatch.setenv('LLM_PROVIDER', 'ollama')
    with patch('app.service.classify', return_value='other'), \
         patch('app.service.retrieve') as retrieve_mock, \
         patch('app.service.generate_general', return_value=('Hello! How can I help?', 'ollama:gemma3:1b')), \
         patch('app.service.redis.Redis.from_url', side_effect=service.redis.RedisError()):
        result = service.answer_question('hi')
        assert result.answer == 'Hello! How can I help?'
        assert result.category == 'other'
        assert result.citations == []
        retrieve_mock.assert_not_called()


def test_refund_question_uses_filtered_retrieval():
    from app import service
    docs = [{'source': 'refunds.txt', 'text': 'Refunds may be requested within 30 days.'}]
    with patch('app.service.classify', return_value='refund'), \
         patch('app.service.retrieve', return_value=docs) as retrieve_mock, \
         patch('app.service.generate', return_value=('Within 30 days.', 'ollama:gemma3:1b')), \
         patch('app.service.redis.Redis.from_url', side_effect=service.redis.RedisError()):
        result = service.answer_question('What is the refund policy?')
        retrieve_mock.assert_called_once_with('What is the refund policy?', 'refund')
        assert result.citations[0].source == 'refunds.txt'
