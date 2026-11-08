from fastapi import FastAPI, HTTPException
from app.schemas import Question, Answer
from app.service import answer_question

app = FastAPI(title='Company Knowledge Assistant')

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/ask', response_model=Answer)
def ask(payload: Question):
    try:
        return answer_question(payload.question)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'Backend unavailable: {type(exc).__name__}. Check service logs and README setup.') from exc
