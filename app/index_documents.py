from app.config import load_local_env
from app.documents import load_chunks
from app.store import index

if __name__ == '__main__':
    print(f'Indexed {index(load_chunks())} document chunks')
