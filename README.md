# Knowledge Assistance

**AI document assistant built with RAG, Google Gemini, Streamlit, and Qdrant.**

- **Live demo:** https://knowledge-assistance-demo.streamlit.app
- **Source code:** https://github.com/Monesha-B/knowledge-assistance
- **CI:** https://github.com/Monesha-B/knowledge-assistance/actions

## What it does

Knowledge Assistance answers questions using retrieved document excerpts and displays the supporting sources. Users can ask about bundled fictional company policies or upload small text-based documents to search during their current Streamlit session.

**Public-demo warning:** Do not upload sensitive, private, regulated or confidential information. The app sends relevant excerpts to its configured LLM provider (Gemini in the hosted demo). Temporary session storage is not a guarantee of immediate or complete deletion from server memory or third-party systems.

## Architecture

**Example documents:** text files → LangChain chunking → Hugging Face transformer embeddings → Qdrant Cloud → category-filtered semantic search → Gemini → answer and source excerpts.

**Visitor uploads:** PDF/DOCX/TXT/MD → validated extraction → chunks and embeddings held in `st.session_state` on the Streamlit server → NumPy similarity search → Gemini → answer and source excerpts. **The current visitor upload flow does not write to Qdrant or Redis.**

A keyword-first classifier routes obvious order, refund and payment questions. An optional locally fine-tuned DistilBERT model handles less explicit requests when its weights are available. The small demonstration training dataset is not evidence of production-grade accuracy.

## Stack

| Area | Technologies |
| --- | --- |
| Frontend | Streamlit |
| Backend/API | Python, FastAPI, Uvicorn, Pydantic |
| Generative AI | Gemini API, LangChain; optional local Ollama/Claude modes |
| NLP and embeddings | Hugging Face Transformers, PyTorch, DistilBERT, all-MiniLM-L6-v2 |
| Retrieval | Qdrant Cloud (examples); NumPy in-memory similarity (uploads) |
| Processing | LangChain text splitters, pypdf, ZIP/XML DOCX parsing |
| Caching | Redis for example queries locally (optional) |
| MLOps/testing | MLflow, Pytest, GitHub Actions |
| Local infrastructure | Docker, Docker Compose (Qdrant and Redis) |
| Public hosting | Streamlit Community Cloud |

The public demo runs Streamlit, **not** a separately hosted FastAPI or Redis instance. No Docker Hub publication is verified.

## Run locally

Use Python 3.11 and install dependencies:

```bash
cd ~/Downloads/company-knowledge-assistant
conda activate rag-project
python -m pip install -r requirements.txt
```

Set credentials in your local, gitignored `.env` (never commit credentials). Example:

```dotenv
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.5-flash-lite
GOOGLE_API_KEY=YOUR_PRIVATE_KEY
QDRANT_URL=YOUR_QDRANT_URL
QDRANT_API_KEY=YOUR_PRIVATE_DATABASE_KEY
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Run the Streamlit application:

```bash
python -m streamlit run app/ui.py
```

For a local development Qdrant/Redis stack, use `docker compose up -d qdrant redis` and point `QDRANT_URL` to the matching service. If example documents have not been indexed into the selected Qdrant database:

```bash
python -m app.index_documents
```

**Warning:** The example-document indexing command deletes/recreates the `company_knowledge` collection. Never use it on a collection containing irreplaceable data.

Optional API: `python -m uvicorn app.api:app --reload --port 8000` (docs at http://localhost:8000/docs).

## Visitor upload protections and limits

- Up to **3 documents**, **2 MB each**, **20 PDF pages**, **40,000 total extracted characters**, and **120 text chunks**.
- Supports text-based PDF, DOCX, TXT and Markdown; scanned PDFs requiring OCR are not supported.
- Uploaded chunks and vectors live in a visitor's Streamlit server-side session state; they do not go to shared Qdrant collections.
- Upload questions bypass Redis caching.
- **Remove uploaded documents** clears that session's documents, embeddings, and upload chat. Disconnect/reboot can also discard session state but does not guarantee immediate memory erasure.
- Uploaded excerpts still leave the server when sent to Gemini. Do not use this public demo for personal, sensitive, or confidential files.
- There is no authentication or production-grade anti-abuse/rate-limit system.

### Hosted setup

Deploy `app/ui.py` from the `main` branch to Streamlit Community Cloud. Enter provider keys only in **App settings → Secrets**, not in source files or GitHub.

## Tests

```bash
python -m pytest -q
```

The GitHub Actions workflow runs tests on pushes and pull requests using Python 3.11. CI passing verifies unit tests, not the hosted document-upload experience.

### Live smoke test

1. Open https://knowledge-assistance-demo.streamlit.app and confirm both **My documents** and **Example documents**.
2. Under **Example documents**, ask `What is the refund policy?`; look for a relevant answer and `refunds.txt`.
3. Under **My documents**, create a fictional TXT file containing `The demo return period is 17 days.` Upload and select **Prepare documents**.
4. Ask `What is the demo return period?` and check the response and cited filename.
5. Select **Remove uploaded documents** and confirm the session's files/chat clear.
6. Confirm no *new* `upload_*` Qdrant Cloud collection was created.
7. Check Streamlit app logs for errors without disclosing credentials.

## Machine learning training

`training/train_classifier.py` trains a demonstration DistilBERT classifier on 24 synthetic questions and records loss in MLflow. The classifier needs a larger dataset, held-out evaluation and confidence calibration before any accuracy claims.

```bash
python training/train_classifier.py
python -m mlflow ui
```

## Limitations / next steps

- Excerpts are not independently verified, sentence-level fact checks.
- Session-only uploads reduce shared-storage exposure but are not a privacy or deletion guarantee.
- Add stronger parser and resource-abuse protections, rate limits, monitoring, and authentication for more than a small public demo.
- Rotate any previously exposed API keys.
- Clean up historical `upload_*` collections created by the *old* upload implementation; **preserve `company_knowledge`**.
