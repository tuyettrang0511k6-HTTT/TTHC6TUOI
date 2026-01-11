import streamlit as st
import json
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ================== CẤU HÌNH ==================
JSON_FILE = "data/all_procedures_normalized.json"
COLLECTION_NAME = "dichvucong_rag"
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "BAAI/bge-m3"

# ================== KIỂM TRA API KEY ==================
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Chưa cấu hình GOOGLE_API_KEY trong Streamlit Secrets")
    st.stop()

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ================== LOAD + CHUNK DATA ==================
def load_and_chunk_data():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs, metas, ids = [], [], []

    for i, item in enumerate(data):
        content = item.get("content", "").strip()
        if not content:
            continue

        docs.append(content)
        metas.append({
            "hierarchy": item.get("hierarchy", "N/A"),
            "url": item.get("url", "N/A"),
            "source_file": item.get("source_file", "json")
        })
        ids.append(str(i))

    return docs, metas, ids

# ================== LOAD + INGEST VECTOR DB ==================
@st.cache_resource
def load_collection():
    client = chromadb.Client()

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    if collection.count() == 0:
        docs, metas, ids = load_and_chunk_data()
        collection.add(
            documents=docs,
            metadatas=metas,
            ids=ids
        )

    return collection

collection = load_collection()

# ================== QUERY RAG ==================
def query_rag(query: str, top_k: int):
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(
            f"[{meta.get('hierarchy')}]\n{doc}\n(Nguồn: {meta.get('url')})"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
Bạn là trợ lý tư vấn thủ tục hành chính công của Việt Nam.

CHỈ sử dụng thông tin trong CONTEXT.
KHÔNG dùng kiến thức bên ngoài.

Nếu CONTEXT không liên quan, trả lời đúng câu:
"Xin lỗi! Câu hỏi của bạn không nằm trong phạm vi hỗ trợ của tôi."

CONTEXT:
{context}

Câu hỏi: {query}
"""

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt, stream=True)
    return response

# ================== GIAO DIỆN ==================
st.set_page_config(
    page_title="Chatbot tư vấn thủ tục hành chính trẻ em dưới 6 tuổi",
    page_icon="🤖"
)

st.title("🤖 Chatbot tư vấn thủ tục hành chính trẻ em dưới 6 tuổi")

with st.sidebar:
    top_k = st.slider("Top-k retrieval", 1, 10, 3)
    st.write(f"📦 Vector DB: {COLLECTION_NAME}")
    st.write(f"🧩 Số chunk: {collection.count()}")
    st.write(f"📐 Embedding: {EMBEDDING_MODEL}")
    st.write(f"🤖 LLM: {GEMINI_MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Nhập câu hỏi của bạn...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = ""
        placeholder = st.empty()

        try:
            response = query_rag(prompt, top_k)
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Lỗi: {e}"
            placeholder.error(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
