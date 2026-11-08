import os
from pathlib import Path

# Optional local .env reader using Python stdlib (no extra dependency).
def load_local_env():
    f = Path(__file__).resolve().parents[1] / '.env'
    if f.exists():
        for line in f.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith('#') or '=' not in s:
                continue
            key, value = s.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())

load_local_env()
