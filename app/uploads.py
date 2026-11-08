"""Session-isolated user document indexing. Text never touches the project data folder."""
from __future__ import annotations

import io
import re
import uuid
import zipfile
from pathlib import PurePath
from xml.etree import ElementTree as ET

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from app.embeddings import embed
from app.store import client

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_FILES = 5
MAX_CHARS = 200_000
ALLOWED = {'.txt', '.md', '.pdf', '.docx'}


def safe_filename(name: str) -> str:
    # Drop directories and control characters from untrusted filenames.
    name = str(name).replace('\\', '/').split('/')[-1]
    return re.sub(r'[^\w.() -]', '_', name)[:120] or 'document'


def extract_document(name: str, content: bytes) -> list[tuple[str, str]]:
    name = safe_filename(name)
    suffix = PurePath(name).suffix.lower()
    if suffix not in ALLOWED:
        raise ValueError(f'{name}: supported files are PDF, DOCX, TXT and MD')
    if not content or len(content) > MAX_FILE_BYTES:
        raise ValueError(f'{name}: file is empty or exceeds 10 MB')
    if suffix in {'.txt', '.md'}:
        return [(name, content.decode('utf-8-sig', errors='replace'))]
    if suffix == '.pdf':
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise ValueError(f'{name}: encrypted PDFs are not supported')
            return [(f'{name}, page {i}', page.extract_text() or '')
                    for i, page in enumerate(reader.pages, start=1)]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f'{name}: unable to read PDF ({type(exc).__name__})') from exc
    # DOCX is a ZIP of XML: parse using only the Python standard library.
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            xml = archive.read('word/document.xml')
        root = ET.fromstring(xml)
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        paragraphs = []
        for p in root.findall('.//w:p', ns):
            text = ''.join(n.text or '' for n in p.findall('.//w:t', ns))
            if text.strip():
                paragraphs.append(text)
        return [(name, '\n'.join(paragraphs))]
    except Exception as exc:
        raise ValueError(f'{name}: unable to read DOCX ({type(exc).__name__})') from exc


def make_chunks(files: list[tuple[str, bytes]]) -> list[dict]:
    if not files or len(files) > MAX_FILES:
        raise ValueError('Upload between 1 and 5 documents at a time')
    splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=90)
    chunks = []
    total_chars = 0
    for name, content in files:
        for source, text in extract_document(name, content):
            total_chars += len(text)
            if total_chars > MAX_CHARS:
                raise ValueError('Combined extracted document text must be under 200,000 characters')
            for snippet in splitter.split_text(text):
                if snippet.strip():
                    chunks.append({'id': len(chunks) + 1, 'text': snippet, 'source': source})
    if not chunks:
        raise ValueError('No readable text was found. Scanned/image-only PDFs need OCR and are not supported.')
    return chunks


def new_collection_name() -> str:
    return f'upload_{uuid.uuid4().hex}'


def index_uploads(files: list[tuple[str, bytes]], collection: str) -> int:
    if not re.fullmatch(r'upload_[0-9a-f]{32}', collection):
        raise ValueError('Invalid session collection')
    chunks = make_chunks(files)
    vectors = embed([chunk['text'] for chunk in chunks])
    db = client()
    # New random collection per upload batch. Never modify the shared sample collection.
    db.create_collection(collection, vectors_config=models.VectorParams(
        size=len(vectors[0]), distance=models.Distance.COSINE))
    try:
        db.upsert(collection_name=collection, points=[
            models.PointStruct(id=chunk['id'], vector=vector, payload={
                'text': chunk['text'], 'source': chunk['source']})
            for chunk, vector in zip(chunks, vectors)
        ])
    except Exception:
        db.delete_collection(collection)
        raise
    return len(chunks)


def search_uploads(question: str, collection: str, limit: int = 4) -> list[dict]:
    if not re.fullmatch(r'upload_[0-9a-f]{32}', collection):
        raise ValueError('Invalid session collection')
    db = client()
    result = db.query_points(collection_name=collection, query=embed([question])[0],
                             limit=limit, with_payload=True)
    return [{'source': p.payload['source'], 'text': p.payload['text'], 'score': p.score}
            for p in result.points]


def delete_uploads(collection: str) -> None:
    if not re.fullmatch(r'upload_[0-9a-f]{32}', collection):
        raise ValueError('Invalid session collection')
    db = client()
    if db.collection_exists(collection):
        db.delete_collection(collection)
