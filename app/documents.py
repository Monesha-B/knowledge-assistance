from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'

def load_chunks(folder: Path = DATA_DIR):
    splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=90)
    chunks = []
    for path in sorted(folder.glob('*.txt')):
        for index, text in enumerate(splitter.split_text(path.read_text(encoding='utf-8'))):
            chunks.append({'id': len(chunks)+1, 'text': text, 'source': path.name, 'chunk': index})
    return chunks
