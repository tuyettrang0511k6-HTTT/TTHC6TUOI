import os
import json
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ======================
# CONFIG
# ======================
JSON_FILE = "data.json"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "tthc_collection"

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# ======================
# GEMINI
# ======================
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ======================
# LOAD CHROMA COLLECTION
# ======================
@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3"   # ✅ 1024 chiều
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    return collection

# ======================
# LOAD JSON TO CHROMA
# ======================
def load_json_to_chroma(collection, json_path):
    if not os.path.exists(json_path):
        st.error(f"❌ Không tìm thấy file: {json_path}")
        st.stop()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if collection.count() > 0:
        return

    documents = []
    metadatas = []
    ids = []

    for i, item in enumerate(data):
        if "content" not in item:
            continue

        documents.append(item["content"])
        metadatas.append({
            "title": item.get("title", f"Tài liệu {i+1}")
        })
        ids.append(str(i))

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

# ======================
# QUERY + GEMINI
# ======================
def ask_gemini(question, context):
    prompt = f"""
Bạn là trợ lý tư vấn thủ tục hành chính Việt Nam.
Chỉ trả lời dựa trên CONTEXT bên dưới.
Nếu không có thông tin thì nói rõ là không tìm thấy.

CONTEXT:
{context}

CÂU HỎI:
{question}
"""
    response = model.generate_content(prompt)
    return response.text

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="TTHC RAG", layout="wide")
st.title("📄 Tra cứu thủ tục hành chính")

collection = load_collection()
load_json_to_chroma(collection, JSON_FILE)

st.sidebar.markdown("### 📊 Trạng thái hệ thống")
st.sidebar.write("🧩 Số chunk:", collection.count())
st.sidebar.write("📐 Embedding: BAAI/bge-m3 (1024)")

question = st.text_input("❓ Nhập câu hỏi:")

if question:
    results = collection.query(
        query_texts=[question],
        n_results=5
    )

    docs = results["documents"][0]

    if not docs:
        st.warning("⚠️ Không tìm thấy thông tin liên quan.")
    else:
        context = "\n\n".join(docs)
        answer = ask_gemini(question, context)
        st.markdown("### ✅ Trả lời")
        st.write(answer)
