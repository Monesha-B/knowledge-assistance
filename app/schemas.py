from pydantic import BaseModel, Field

class Question(BaseModel):
    question: str = Field(min_length=2)

class Citation(BaseModel):
    source: str
    excerpt: str

class Answer(BaseModel):
    question: str
    category: str
    answer: str
    citations: list[Citation]
    model: str
    cached: bool = False
