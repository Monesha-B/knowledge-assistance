# Knowledge Assistance — Project 2

A **learning-oriented, runnable starter** for RAG + DistilBERT query classification. No external API keys are bundled. Sample company policies are fictional. The project code covers the 15 requested technologies; model accuracy, citations, service availability and CI/deployment still need hands-on verification.

## Accounts / installs

- **GitHub account**: required only to push a repository and activate GitHub Actions. https://github.com/signup
- **Anthropic Console**: required for Claude API mode; this may require buying API credits. https://platform.claude.com/settings/keys
- **Docker Desktop**: install locally; free personal usage subject to Docker terms. https://www.docker.com/products/docker-desktop/
- **Ollama**: install locally; no account needed for local models. https://ollama.com/download
- **Hugging Face account**: optional for public models used here; useful later for publishing or gated models. https://huggingface.co/join
- **Qdrant Cloud**: optional; local Qdrant in Docker needs no account. https://cloud.qdrant.io/
- **Redis Cloud**: optional; Redis runs locally in Docker. https://redis.io/cloud/
- **MLflow, Python, PyTorch, Streamlit, FastAPI, LangChain, Pydantic, Pytest**: no accounts needed locally.

## Run locally on Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Install and start Docker Desktop first
docker compose up -d qdrant redis
# Install Ollama from https://ollama.com/download, then:
ollama pull gemma3:1b
python -m app.index_documents
uvicorn app.api:app --reload
# In a second activated terminal:
streamlit run app/ui.py
```

First launch downloads public embedding model weights from Hugging Face. Visit http://localhost:8501 for the chat UI and http://localhost:8000/docs for FastAPI docs. Qdrant is at http://localhost:6333/dashboard. Try asking: *When will my refund arrive?* The default LLM is local Ollama. **Ollama must be running** before generating answers. If your hardware cannot run the model, use Claude with a valid API key.

### Run Claude instead of local Ollama

Set `LLM_PROVIDER=claude` and `ANTHROPIC_API_KEY=...` in your own `.env`. Restart the API/Streamlit terminal. The Anthropic API is not necessarily free and the Claude model identifier may need updating based on your account's available models.

### Fine-tune query classifier (optional but in project scope)

```bash
python training/train_classifier.py
```

The training job downloads DistilBERT, trains on a **tiny synthetic dataset**, records epochs/loss in MLflow, and saves a model. Before training, a keyword-based fallback is used for quick onboarding. After training, restart the app to use DistilBERT. Expand dataset and evaluate on held-out questions before claiming performance.

### Tests

```bash
pytest -q
mlflow ui
```

Pytest unit tests require no external services. MLflow UI defaults to http://localhost:5000. On GitHub, pushes trigger the included workflow.

### Docker app container (advanced)

`docker compose up --build` starts Qdrant, Redis and API. **Index the data into Qdrant after Qdrant is running** by running `python -m app.index_documents` from your local Python environment. If Ollama runs on your Mac and the API runs in Docker, configure Ollama to listen on an interface accessible to the container (do not expose it publicly), or use Claude mode. The simplest development mode is running API and Streamlit natively, and Qdrant/Redis in Docker.

### Architecture

Documents -> text chunks -> Hugging Face embeddings -> Qdrant. Question -> DistilBERT classifier (trained model or onboarding fallback) -> Qdrant retrieval -> LangChain + Claude/Ollama -> Pydantic Answer with sources -> Streamlit. Redis caches previous questions. FastAPI exposes `/ask`. MLflow records classifier training. Docker and GitHub Actions package and test.

### Important limitations / next improvements

- Retrieved citations are **source excerpts**, not verified sentence-level attribution. Verify answer claims against source text before a public demo.
- The synthetic classifier dataset is intentionally tiny; add held-out validation, uncertainty threshold, route-specific retrieval and accuracy metrics. Currently classifier labels are returned but the Qdrant search is global, not filtered by category.
- Claude/Ollama comparisons must be measured separately on the same test questions; no fabricated speed/cost claims.
- CI verifies basic unit tests, not integration with external running services. Add integration tests later.
- Do not upload private company documents, API keys, or sensitive customer information.
- Avoid pointing the public internet at an unauthenticated local Qdrant, Redis or API instance.


## Final integration update

- `app/ui.py` now keeps prior messages in Streamlit session state (not across fresh sessions).
- `other` questions bypass Qdrant and get an ordinary Ollama/Claude reply without fabricated policy citations.
- `order`, `refund`, and `payment` queries use Qdrant category filters, sourced from the bundled `orders.txt`, `refunds.txt`, and `payments.txt` files. Additional files need an adapted category mapping.
- New cache namespace `rag:v3` prevents the previous broken `hi` answer being reused.
- **Required after updating:** `python -m app.index_documents` again. Old indexed vectors lack category payloads.
- Restart Streamlit and FastAPI, or rely on file watching when it works.
- DistilBERT is not trained until `python training/train_classifier.py` runs successfully. Its 24 tiny example questions are for learning, not production evaluation.
- The responses contain retrieved source excerpts, not guaranteed fact-checked citations. Claude comparison requires paid credits and separate evaluation.

### Mac quick update

```bash
cd ~/Downloads/company-knowledge-assistant
conda activate rag-project
docker compose up -d qdrant redis
python -m app.index_documents
python -m pytest -q
# If API/Streamlit were running, restart them in separate terminal tabs:
python -m uvicorn app.api:app --reload --port 8000
# New terminal: python -m streamlit run app/ui.py --server.port 8502
```

### MLflow classifier training and inspection

```bash
python training/train_classifier.py
python -m mlflow ui
```

Visit http://localhost:5000 after starting MLflow UI. Restart the chat application after fine-tuning to load the trained classifier. A full end-to-end test still requires running this on your Mac with the Ollama/Qdrant/Redis services and evaluating actual model output.

## Upload your own files (PDF, DOCX, TXT, MD)

**New:** The Streamlit UI now defaults to **My uploaded documents**. Choose 1–5 files, click **Index selected documents**, and ask questions. Switching to **Sample company policies** preserves the previous demonstration workflow. Uploaded text is chunked with LangChain, embedded using the same Hugging Face model, and stored in a **separate randomly named Qdrant collection for each Streamlit session**. Document citations display filenames and PDF page numbers where available. DOCX extraction uses Python's standard-library ZIP/XML support. PDF extraction uses the `pypdf` parser, installed through `requirements.txt`.

**After updating an existing project**, from the activated Conda environment run:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
# Restart Streamlit (and FastAPI if you changed backend files):
python -m streamlit run app/ui.py --server.port 8502
```

- No new API account or API key is needed for local Ollama.
- Uploaded documents are **not** added to `data/` or the shared sample collection; they're held in a session-scoped Qdrant collection. Queries to your uploads do **not** use Redis caching, avoiding persistent cached text from personal uploads.
- Use **Delete my indexed documents** to remove your collection from Qdrant. **Session expiry does not automatically delete Qdrant data**; for public hosting add authentication, a retention/cleanup policy, and access controls before accepting private files. Running Qdrant without authentication is suitable only on a trusted local machine.
- Session chat history and the uploaded collection identifier are not durable across a lost Streamlit session; documents can remain in Qdrant until deleted. Do not upload secrets, medical records, or sensitive company material to an unprotected demo. Claude sends retrieved excerpts to Anthropic; Ollama runs locally.
- The upload feature handles **text-based PDF**, DOCX, TXT and Markdown files. Scanned image-only PDFs and password-protected PDFs require extra processing not included in the requested project stack. Max 5 files, 10 MB/file, 200,000 extracted characters per batch. PDF page numbers are included in source labels. This is a local learning demo, not a multi-tenant secure document service.
- The current application retrieves the closest chunks; it has no confidence threshold, so **verify answers against source excerpts**. It cannot guarantee absence of hallucinations. Documents are used as evidence, never as executable instructions.


## Updated user interface

The Streamlit application is branded **Knowledge Assistance** with a monochrome Material book icon. The sidebar explains how to use the app, while document selection, preparation, conversation, sources, and technical information have separate clear areas. The underlying package and folder remain `company-knowledge-assistant` to preserve existing commands and imports.
