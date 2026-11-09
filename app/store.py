import os
from qdrant_client import QdrantClient, models
from app.embeddings import embed

COLLECTION = 'company_knowledge'
VALID_CATEGORIES = {'order', 'refund', 'payment'}


def client():
    return QdrantClient(
        url=os.getenv('QDRANT_URL', 'http://localhost:6333'),
        api_key=os.getenv('QDRANT_API_KEY') or None,
    )


def index(chunks):
    if not chunks:
        raise ValueError('No text documents found in data/')
    vectors = embed([c['text'] for c in chunks])
    db = client()
    if db.collection_exists(COLLECTION):
        db.delete_collection(COLLECTION)
    db.create_collection(COLLECTION, vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE))
    db.create_payload_index(
        collection_name=COLLECTION,
        field_name='category',
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )
    db.upsert(collection_name=COLLECTION, points=[
        models.PointStruct(id=c['id'], vector=v, payload={
            'text': c['text'], 'source': c['source'],
            'category': ('payment' if c['source'].startswith('payments') else
                         'order' if c['source'].startswith('orders') else
                         'refund' if c['source'].startswith('refunds') else 'other')
        }) for c, v in zip(chunks, vectors)
    ])
    return len(chunks)


def retrieve(question, category=None, limit=4):
    vector = embed([question])[0]
    db = client()
    category_filter = None
    if category in VALID_CATEGORIES:
        category_filter = models.Filter(must=[models.FieldCondition(key='category', match=models.MatchValue(value=category))])
    result = db.query_points(collection_name=COLLECTION, query=vector, query_filter=category_filter,
                             limit=limit, with_payload=True)
    return [{'text': p.payload['text'], 'source': p.payload['source'], 'score': p.score} for p in result.points]
