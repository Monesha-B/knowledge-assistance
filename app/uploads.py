"""Session-isolated user document indexing. Text never touches the project data folder."""
from __future__ import annotations

import io
import numpy as np
import re
import zipfile
from pathlib import PurePath
from xml.etree import ElementTree as ET

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.embeddings import embed

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_FILES = 3
MAX_CHARS = 40_000
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
        raise ValueError(f'{name}: file is empty or exceeds 2 MB')
    if suffix in {'.txt', '.md'}:
        return [(name, content.decode('utf-8-sig', errors='replace'))]
    if suffix == '.pdf':
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(content))
            if len(reader.pages) > 20:
                raise ValueError(f'{name}: maximum 20 PDF pages')
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
            info = archive.getinfo('word/document.xml')
            if info.file_size > 1_000_000:
                raise ValueError(f'{name}: Word XML is too large')
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
        raise ValueError('Upload between 1 and 3 documents at a time')
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
    if len(chunks) > 120:
        raise ValueError('Too many document sections; upload shorter documents')
    return chunks




def index_uploads(files: list[tuple[str, bytes]]) -> tuple[list[dict], list[list[float]]]:
    """Return chunks and vectors for caller-owned Streamlit session state.

    No cloud database, disk writes, global caches, or shared session registry.
    """
    chunks = make_chunks(files)
    vectors = embed([chunk['text'] for chunk in chunks])
    return chunks, vectors


def search_uploads(question: str, chunks: list[dict], vectors: list[list[float]], limit: int = 4) -> list[dict]:
    if not chunks or len(chunks) != len(vectors):
        raise ValueError('No active document index for this session')
    query_vector = np.asarray(embed([question])[0], dtype=np.float32)
    document_vectors = np.asarray(vectors, dtype=np.float32)
    similarities = document_vectors @ query_vector
    matches = np.argsort(similarities)[::-1][:limit]
    return [
        {'source': chunks[i]['source'], 'text': chunks[i]['text'], 'score': float(similarities[i])}
        for i in matches
    ]
