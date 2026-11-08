# Which accounts are needed?

| Service | Need an account? | What you connect |
|---|---|---|
| Anthropic Claude Console | **Yes**, if using cloud Claude | Create a Claude API key; set `ANTHROPIC_API_KEY` in `.env`; set `LLM_PROVIDER=claude`; API credits may be needed |
| GitHub | **Yes**, for GitHub Actions | Create repository, push project; workflows run on pushes |
| Hugging Face | **No** for public embedding/DistilBERT models | Public downloads work without token; account is optional for gated models or sharing |
| Ollama | **No** for local models | Install application; `ollama pull gemma3:1b` |
| Qdrant | **No** locally | `docker compose up -d qdrant`; optional Qdrant Cloud account later |
| Redis | **No** locally | `docker compose up -d redis`; optional Redis Cloud account later |
| Docker | **No** for local Docker Desktop setup in most cases | Install Desktop; check terms for your circumstances |
| MLflow / PyTorch / LangChain / FastAPI / Streamlit / Pydantic / Pytest | **No** | Installed as Python packages |

Never share API keys here or push `.env` to GitHub. The included `.gitignore` excludes `.env`.

Links:
- https://platform.claude.com/settings/keys
- https://github.com/signup
- https://huggingface.co/join
- https://ollama.com/download
- https://www.docker.com/products/docker-desktop/
- https://cloud.qdrant.io/
- https://redis.io/cloud/
