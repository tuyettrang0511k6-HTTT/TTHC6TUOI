import streamlit as st
import os
import uuid
import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions

# ================== CẤU HÌNH HỆ THỐNG ==================
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "dichvucong_rag"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3" 
GEMINI_MODEL_NAME = "gemini-1.5-flash" # Model phổ biến và ổn định nhất

# ================== CẤU HÌNH API GEMINI ==================
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Chưa cấu hình GOOGLE_API_KEY trong Streamlit Secrets")
    st.stop()

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ================== HÀM KHỞI TẠO DATABASE ==================
@st.cache_resource
def get_vector_db():
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )
    return collection

collection = get_vector_db()

# ================== GIAO DIỆN CHÍNH (UI) ==================
st.set_page_config(page_title="Chatbot Thủ tục Trẻ em", page_icon="👶")

st.markdown("""
<style>
    .stApp { background: #fff0f5; }
    h1 { color: #b91c5c; }
    div[data-testid="stChatMessageAssistant"] { background-color: #ffffff; border: 1px solid #f3c6d3; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Chatbot Thủ tục Hành chính Trẻ em")

# ================== XỬ LÝ SIDEBAR ==================
with st.sidebar:
    st.header("⚙️ Cấu hình")
    top_k = st.slider("Số lượng tài liệu (Top-k)", 1, 10, 3)
    
    if st.button("📥 Nạp dữ liệu mẫu"):
        texts = [
            "Thủ tục đăng ký khai sinh cho trẻ em dưới 6 tuổi thực hiện tại UBND cấp xã.",
            "Hồ sơ gồm: Giấy chứng sinh, CCCD của cha mẹ, Giấy kết hôn.",
            "Trẻ dưới 6 tuổi được cấp thẻ BHYT miễn phí."
        ]
        metadatas = [
            {"hierarchy": "Khai sinh", "url": "https://dichvucong.gov.vn"},
            {"hierarchy": "Hồ sơ", "url": "https://dichvucong.gov.vn"},
            {"hierarchy": "BHYT", "url": "https://baohiemxahoi.gov.vn"}
        ]
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=[str(uuid.uuid4()) for _ in texts]
        )
        st.success("✅ Đã nạp dữ liệu!")

    if st.button("🗑️ Xóa sạch dữ liệu"):
        ids = collection.get()['ids']
        if ids: collection.delete(ids=ids)
        st.rerun()

# ================== HÀM TRUY VẤN RAG ==================
def query_rag(query_text):
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    context_list = []
    if results["documents"][0]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            context_list.append(f"[{meta.get('hierarchy', 'N/A')}] {doc} (Nguồn: {meta.get('url', 'Internet')})")
    
    context = "\n\n".join(context_list)

    # LƯU Ý: Thêm tiền tố 'models/' để tránh lỗi 404
    model = genai.GenerativeModel(model_name=f"models/{GEMINI_MODEL_NAME}")
    
    prompt = f"Sử dụng context sau để trả lời câu hỏi. Context: {context}\n\nCâu hỏi: {query_text}"
    return model.generate_content(prompt, stream=True)

# ================== LOGIC CHAT ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            response_stream = query_rag(prompt)
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Lỗi: {str(e)}")
            full_response = "Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi."

    st.session_state.messages.append({"role": "assistant", "content": full_response})
