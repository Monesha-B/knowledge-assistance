"""Simple, document-first interface for Knowledge Assistance."""
import hashlib
import os

import streamlit as st

from pathlib import Path
import sys

# Make the project root available to Streamlit Cloud.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.service import answer_question, answer_uploaded_question
from app.uploads import ALLOWED, MAX_FILE_BYTES, delete_uploads, index_uploads, new_collection_name

st.set_page_config(
    page_title="Knowledge Assistance",
    page_icon=":material/menu_book:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 840px;}
[data-testid="stChatMessage"] {border: 1px solid rgba(130,130,130,.14); border-radius: 12px;}
[data-testid="stSidebar"] hr {margin: 0.75rem 0;}
</style>
""", unsafe_allow_html=True)

for key, default in (
    ("messages", []), ("sample_messages", []),
    ("upload_collection", None), ("upload_fingerprint", None),
    ("upload_file_names", []),
):
    if key not in st.session_state:
        st.session_state[key] = default

st.title("Knowledge Assistance")
st.caption("Ask questions about your documents and get answers with supporting sources.")

with st.sidebar:
    st.subheader("Workspace")
    mode = st.radio(
        "Choose a knowledge source",
        ["Example documents"] if os.getenv("PUBLIC_DEMO") == "1" else ["My documents", "Example documents"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**How it works**")
    st.markdown("1. Upload documents.\n2. Select **Prepare documents**.\n3. Ask questions in the chat.")
    st.caption("Example documents are fictional order, refund, and payment policies.")
    st.divider()
    st.caption("Answers are AI-generated. Check source excerpts for important details.")
    st.caption("Do not upload sensitive information. Uploaded text is kept in Qdrant until you delete it; this demo does not have account-based access controls.")

is_upload = mode == "My documents"
history_key = "messages" if is_upload else "sample_messages"

if is_upload:
    st.subheader("1. Add your documents")
    files_input = st.file_uploader(
        "Choose up to 5 files (PDF, Word, TXT or Markdown)",
        type=sorted(extension.lstrip(".") for extension in ALLOWED),
        accept_multiple_files=True,
        help="Maximum 10 MB per file. Scanned PDFs without selectable text are not supported.",
    )
    files = [(f.name, f.getvalue()) for f in files_input] if files_input else []
    fingerprint = None
    if files:
        digest = hashlib.sha256()
        for name, content in files:
            digest.update(name.encode("utf-8", errors="replace"))
            digest.update(content)
        fingerprint = digest.hexdigest()
        st.caption(f"{len(files)} selected · Up to {MAX_FILE_BYTES // (1024 * 1024)} MB per file")

    needs_indexing = bool(files) and fingerprint != st.session_state.upload_fingerprint
    if needs_indexing and st.session_state.upload_collection:
        st.info("You selected different files. Select **Prepare documents** to replace your current set.")

    if st.button("Prepare documents", type="primary", disabled=not needs_indexing):
        if len(files) > 5:
            st.error("Select no more than five files.")
        else:
            new_collection = new_collection_name()
            try:
                with st.spinner("Preparing your documents for search..."):
                    count = index_uploads(files, new_collection)
                previous = st.session_state.upload_collection
                st.session_state.upload_collection = new_collection
                st.session_state.upload_fingerprint = fingerprint
                st.session_state.upload_file_names = [name for name, _ in files]
                st.session_state.messages = []
                if previous:
                    try:
                        delete_uploads(previous)
                    except Exception:
                        st.warning("The previous document set could not be deleted. Please review stored collections before sharing this deployment.")
                st.success(f"Ready! Processed {len(files)} file(s) into {count} searchable sections.")
            except Exception as exc:
                st.error(f"Could not prepare documents: {exc}")

    if st.session_state.upload_collection:
        names = st.session_state.upload_file_names
        st.success("Your documents are ready for questions.")
        with st.expander(f"Current documents ({len(names)})"):
            for name in names:
                st.write(f"• {name}")
        if st.button("Remove uploaded documents", help="Delete the current documents from Qdrant"):
            try:
                delete_uploads(st.session_state.upload_collection)
            except Exception as exc:
                st.error(f"Could not delete documents: {exc}")
            else:
                st.session_state.upload_collection = None
                st.session_state.upload_fingerprint = None
                st.session_state.upload_file_names = []
                st.session_state.messages = []
                st.rerun()
    else:
        st.info("Add a document and select **Prepare documents** to start chatting.")
else:
    st.info("Try the example policies without uploading anything. These are fictional documents for demonstration.")

st.subheader("2. Ask a question" if is_upload else "Ask a question")
if not st.session_state[history_key]:
    st.caption(
        "Try: ‘What are the main points?’ or ‘What does the document say about deadlines?’"
        if is_upload else
        "Try: ‘What is the refund policy?’ or ‘How can I track an order?’"
    )

for message in st.session_state[history_key]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            if message.get("citations"):
                with st.expander(f"View sources ({len(message['citations'])})"):
                    for c in message["citations"]:
                        st.markdown(f"**{c['source']}**")
                        st.write(c["excerpt"])
            if "model" in message:
                with st.expander("Technical details"):
                    st.write(f"Category: {message['category']}")
                    st.write(f"Model: {message['model']}")
                    st.write(f"Retrieved from cache: {message['cached']}")

if st.session_state[history_key]:
    if st.button("Clear chat", key=f"clear_{history_key}"):
        st.session_state[history_key] = []
        st.rerun()

question = st.chat_input(
    "Ask about the documents..." if is_upload else "Ask a question...",
    disabled=is_upload and not st.session_state.upload_collection,
)
if question:
    st.session_state[history_key].append({"role": "user", "content": question})
    try:
        with st.spinner("Looking for relevant information..."):
            if is_upload:
                result = answer_uploaded_question(question, st.session_state.upload_collection)
            else:
                result = answer_question(question)
        st.session_state[history_key].append({
            "role": "assistant", "content": result.answer,
            "category": result.category, "model": result.model, "cached": result.cached,
            "citations": [{"source": c.source, "excerpt": c.excerpt} for c in result.citations],
        })
    except Exception as exc:
        st.session_state[history_key].append({
            "role": "assistant", "content": f"Sorry, I couldn't complete that request. Details: {exc}"
        })
    st.rerun()


# Keep chat input aligned with the document upload section
st.markdown("""
<style>
    [data-testid="stBottomBlockContainer"] {
        max-width: 816px !important;
        width: calc(100% - 2rem) !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    [data-testid="stChatInput"] {
        width: 100% !important;
        max-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)
