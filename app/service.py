import os
import json
import redis
from app.config import load_local_env
from app.schemas import Answer, Citation
from app.classifier import classify
from app.store import retrieve
from app.llm import generate, generate_general


def answer_question(question: str) -> Answer:
    question = question.strip()
    category = classify(question)
    provider = os.getenv('LLM_PROVIDER', 'ollama').lower()
    model_name = (
        os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite') if provider == 'gemini'
        else os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-5') if provider == 'claude'
        else os.getenv('OLLAMA_MODEL', 'gemma3:1b')
    )
    # v3 invalidates older cached answers generated without intelligent routing.
    key = f'rag:v3:{provider}:{model_name}:{category}:{question.casefold()}'
    cache = None
    try:
        cache = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
                                     socket_connect_timeout=1, socket_timeout=1)
        hit = cache.get(key)
        if hit:
            return Answer.model_validate_json(hit).model_copy(update={'cached': True})
    except redis.RedisError:
        cache = None

    if category == 'other':
        # General greetings and knowledge never need policy-document retrieval.
        reply, model = generate_general(question)
        docs = []
    else:
        docs = retrieve(question, category)
        reply, model = generate(question, docs)

    answer = Answer(question=question, category=category, answer=reply,
                    citations=[Citation(source=d['source'], excerpt=d['text'][:240]) for d in docs],
                    model=model)
    if cache is not None:
        try:
            cache.setex(key, 3600, answer.model_dump_json())
        except redis.RedisError:
            pass
    return answer


def answer_uploaded_question(question: str, chunks: list[dict], vectors: list[list[float]]) -> Answer:
    """Answer from the active visitor's uploads. Do not cache private documents in Redis."""
    from app.uploads import search_uploads
    question = question.strip()
    if not question:
        raise ValueError('Please enter a question')
    # Direct short greetings to the language model without retrieving irrelevant files.
    if question.casefold().rstrip('!?. ') in {'hi', 'hello', 'hey', 'good morning', 'good evening'}:
        reply, model = generate_general(question)
        return Answer(question=question, category='general', answer=reply,
                      citations=[], model=model, cached=False)
    docs = search_uploads(question, chunks, vectors)
    reply, model = generate(question, docs)
    return Answer(question=question, category='uploaded documents', answer=reply,
                  citations=[Citation(source=d['source'], excerpt=d['text'][:240]) for d in docs],
                  model=model, cached=False)
