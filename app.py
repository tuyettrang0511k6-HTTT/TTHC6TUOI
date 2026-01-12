import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ================== CẤU HÌNH ==================
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "tthc_collection"

# 🔑 LẤY ĐƯỜNG DẪN TUYỆT ĐỐI THEO FILE app.py (KHÔNG BAO GIỜ LỖI)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.join(BASE_DIR, "data", "all_procedures_normalized.json")

EMBEDDING_MODEL = "BAAI/bge-m3"
GEMINI_MODEL = "gemini-1.5-flash"

# ================== KIỂM TRA API KEY ==================
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Chưa cấu hình GOOGLE_API_KEY trong Streamlit Secrets")
    st.stop()

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ================== LOAD CHROMA COLLECTION ==================
@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )
    return collection

# ================== LOAD JSON → ADD VÀO CHROMA (CHẠY 1 LẦN) ==================
def load_json_to_chroma(collection, json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents, metadatas, ids = [], [], []

    for i, item in enumerate(data):
        documents.append(item["content"])
        metadatas.append({
            "hierarchy": item.get("hierarchy", ""),
            "url": item.get("url", ""),
            "source_file": item.get("source_file", "")
        })
        ids.append(f"doc_{i}")

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

# ================== KHỞI TẠO DB ==================
collection = load_collection()

# DEBUG an toàn (có thể xoá sau)
st.sidebar.write("📄 JSON exists:", os.path.exists(JSON_FILE))

if collection.count() == 0:
    st.warning("📥 Đang nạp dữ liệu vào Vector DB...")
    load_json_to_chroma(collection, JSON_FILE)
    st.success(f"✅ Đã nạp {collection.count()} chunks")

# ================== HÀM RAG QUERY ==================
def query_rag(query: str, top_k: int):
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    if not results["documents"][0]:
        return None

    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(
            f"[BLOCK: {meta['hierarchy']}]\n"
            f"{doc}\n"
            f"NGUỒN: {meta['url']}"
        )

    return "\n\n".join(context_parts)

# ================== GỌI GEMINI ==================
def call_gemini(context, question):
    prompt = f"""
Bạn là trợ lý tư vấn thủ tục hành chính công của Việt Nam.
Chỉ sử dụng thông tin trong CONTEXT.
Không sử dụng kiến thức bên ngoài.
Không nhắc lại câu hỏi.

Nếu CONTEXT không liên quan, chỉ trả lời đúng câu:
"Xin lỗi! Câu hỏi của bạn không nằm trong phạm vi hỗ trợ của tôi."

CONTEXT:
{context}

Câu hỏi:
{question}

Trả lời bằng tiếng Việt, ngắn gọn, có đánh số nếu cần.
Giữ nguyên dòng NGUỒN.
"""

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text

# ================== GIAO DIỆN STREAMLIT ==================
st.set_page_config(
    page_title="Chatbot TTHC trẻ em dưới 6 tuổi",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Chatbot tư vấn thủ tục hành chính")
st.markdown(
    "Hỗ trợ **đăng ký khai sinh – thường trú – BHYT** "
    "cho **trẻ em dưới 6 tuổi** từ dữ liệu chính thống."
)

with st.sidebar:
    st.markdown("## ⚙️ Cấu hình")
    top_k = st.slider("Top-k retrieval", 1, 10, 3)
    st.divider()
    st.write(f"📦 Vector DB: {COLLECTION_NAME}")
    st.write(f"🧩 Số chunk: {collection.count()}")
    st.write(f"📐 Embedding: {EMBEDDING_MODEL}")
    st.write(f"🤖 LLM: {GEMINI_MODEL}")

# ================== SESSION CHAT ==================
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
        context = query_rag(prompt, top_k)

        if context is None:
            answer = "Xin lỗi! Câu hỏi của bạn không nằm trong phạm vi hỗ trợ của tôi."
        else:
            answer = call_gemini(context, prompt)

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
