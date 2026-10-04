import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


def api_error_message(response):
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"Request failed with status {response.status_code}."
    if isinstance(payload, dict):
        return payload.get("detail", response.text)
    return response.text


st.set_page_config(page_title="Document Intelligence Assistant", page_icon="📄")
st.title("📄 AI Research & Document Intelligence Assistant")

st.session_state.setdefault("history", [])
st.session_state.setdefault("active_document_id", None)
st.session_state.setdefault("active_document_name", None)
st.session_state.setdefault("answer_scope", "All documents")

with st.sidebar:
    st.header("1. Upload a document or paper")
    f = st.file_uploader(
        "Choose a file from your device (PDF, DOCX, TXT or MD)",
        type=["pdf", "docx", "txt", "md"],
    )
    if f and st.button("Upload and index"):
        try:
            with st.spinner("Chunking, embedding and indexing..."):
                r = requests.post(
                    f"{API_URL}/upload", files={"file": (f.name, f.getvalue())}, timeout=120
                )
            r.raise_for_status()
            uploaded = r.json()
            st.session_state.active_document_id = uploaded["document_id"]
            st.session_state.active_document_name = f.name
            st.session_state.answer_scope = "This paper"
            st.session_state.history = []
            st.success(f"Indexed {uploaded['chunks_indexed']} chunks from {f.name}")
        except requests.HTTPError as e:
            st.error(f"Upload failed: {api_error_message(e.response)}")
        except requests.RequestException as e:
            st.error(f"Could not reach the API: {e}")
    if st.button("Clear all documents"):
        requests.post(f"{API_URL}/reset", timeout=120)
        st.session_state.history = []
        st.session_state.active_document_id = None
        st.session_state.active_document_name = None
        st.session_state.answer_scope = "All documents"
        st.info("Vector store cleared.")

st.header("2. Ask questions")
if st.session_state.active_document_id:
    scope_options = ["This paper", "All documents"]
    if st.session_state.answer_scope not in scope_options:
        st.session_state.answer_scope = "This paper"
else:
    scope_options = ["All documents"]
    st.session_state.answer_scope = "All documents"

selected_scope = st.radio(
    "Search in",
    scope_options,
    key="answer_scope",
    horizontal=True,
)
if selected_scope != st.session_state.get("previous_answer_scope"):
    if st.session_state.get("previous_answer_scope") is not None:
        st.session_state.history = []
    st.session_state.previous_answer_scope = selected_scope

if selected_scope == "This paper" and st.session_state.active_document_name:
    st.caption(f"Questions are grounded only in **{st.session_state.active_document_name}**.")

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["q"])
    with st.chat_message("assistant"):
        st.write(item["a"]["answer"])
        if item["a"].get("fallback"):
            st.warning(item["a"].get("reason", "No supporting information was found."))
        else:
            st.caption("✅ Grounded" if item["a"]["grounded"] else "⚠️ Could not be verified")
        if item["a"]["sources"]:
            with st.expander("Sources"):
                for s in item["a"]["sources"]:
                    page = f", page {s['page'] + 1}" if isinstance(s.get("page"), int) else ""
                    st.markdown(f"**{s['source']}{page}**: {s['snippet']}...")

q = st.chat_input("Ask a question about your uploaded document or paper")
if q:
    with st.spinner("Thinking..."):
        try:
            document_id = (
                st.session_state.active_document_id if selected_scope == "This paper" else None
            )
            r = requests.post(
                f"{API_URL}/ask",
                json={"question": q, "document_id": document_id},
                timeout=120,
            )
            r.raise_for_status()
            st.session_state.history.append({"q": q, "a": r.json()})
            st.rerun()
        except requests.HTTPError as e:
            st.error(f"Could not get an answer: {api_error_message(e.response)}")
        except requests.RequestException as e:
            st.error(f"Could not reach the API: {e}")
