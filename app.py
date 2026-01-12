import streamlit as st
import os
import uuid
import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions

# ================== CẤU HÌNH HỆ THỐNG ==================
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "dichvucong_rag"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3" # Model 1024 chiều chuyên cho tiếng Việt
GEMINI_MODEL_NAME = "gemini-1.5-flash" # Bạn có thể đổi thành gemini-2.0-flash nếu API hỗ trợ

# ================== CẤU HÌNH API GEMINI ==================
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Chưa cấu hình GOOGLE_API_KEY trong Streamlit Secrets")
    st.stop()

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ================== HÀM KHỞI TẠO DATABASE ==================
@st.cache_resource
def get_vector_db():
    # Khởi tạo hàm embedding (Thống nhất dùng BGE-M3)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    
    # Khởi tạo Chroma Client
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Lấy hoặc tạo Collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )
    return collection

collection = get_vector_db()

# ================== GIAO DIỆN CHÍNH (UI) ==================
st.set_page_config(
    page_title="Chatbot Thủ tục Trẻ em dưới 6 tuổi",
    page_icon="👶",
    layout="centered"
)

# Thêm hiệu ứng CSS và hoa rơi
st.markdown("""
<style>
    .stApp { background: #fff0f5; font-family: "Segoe UI", sans-serif; }
    h1, h2, h3 { color: #b91c5c; font-weight: 700; }
    div[data-testid="stChatMessageUser"] { background-color: #ffe4ec; border-radius: 14px; }
    div[data-testid="stChatMessageAssistant"] { background-color: #ffffff; border-radius: 14px; border: 1px solid #f3c6d3; }
    @keyframes fall {
        0% { transform: translateY(-50px) rotate(0deg); opacity: 0; }
        10% { opacity: 1; }
        100% { transform: translateY(110vh) rotate(360deg); opacity: 0; }
    }
    .flower { position: fixed; top: -40px; font-size: 22px; animation: fall linear infinite; z-index: 0; pointer-events: none; }
</style>
<div class="flower" style="left:10%; animation-duration:7s;">🌸</div>
<div class="flower" style="left:30%; animation-duration:10s;">✨</div>
<div class="flower" style="left:50%; animation-duration:6s;">🌷</div>
<div class="flower" style="left:70%; animation-duration:9s;">🌸</div>
<div class="flower" style="left:90%; animation-duration:8s;">✨</div>
""", unsafe_allow_html=True)

st.title("🤖 Chatbot Thủ tục Hành chính Trẻ em")
st.info("Hỗ trợ: Khai sinh, Thường trú, Thẻ BHYT cho trẻ em dưới 6 tuổi.")

# ================== XỬ LÝ SIDEBAR ==================
with st.sidebar:
    st.header("⚙️ Cấu hình")
    top_k = st.slider("Số lượng tài liệu tham chiếu (Top-k)", 1, 10, 3)
    
    st.divider()
    if st.button("📥 Nạp dữ liệu mẫu vào DB"):
        texts = [
            "Thủ tục đăng ký khai sinh cho trẻ em dưới 6 tuổi được thực hiện tại UBND cấp xã nơi cư trú của cha hoặc mẹ.",
            "Hồ sơ đăng ký khai sinh gồm: Giấy chứng sinh, Giấy tờ tùy thân của cha/mẹ, Giấy chứng nhận kết hôn (nếu có).",
            "Trẻ em dưới 6 tuổi được ngân sách nhà nước đóng bảo hiểm y tế và cấp thẻ BHYT miễn phí.",
            "Thủ tục liên thông: Hiện nay người dân có thể đăng ký đồng thời Khai sinh, Thường trú và cấp thẻ BHYT trên Cổng dịch vụ công."
        ]
        metadatas = [
            {"hierarchy": "Khai sinh", "url": "https://dichvucong.gov.vn"},
            {"hierarchy": "Hồ sơ", "url": "https://dichvucong.gov.vn"},
            {"hierarchy": "BHYT", "url": "https://baohiemxahoi.gov.vn"},
            {"hierarchy": "Liên thông", "url": "https://dichvucong.gov.vn"}
        ]
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=[str(uuid.uuid4()) for _ in texts]
        )
        st.success("✅ Đã cập nhật dữ liệu vào Vector DB!")

    st.divider()
    st.write(f"📦 **DB:** {COLLECTION_NAME}")
    st.write(f"🧩 **Số chunk hiện tại:** {collection.count()}")
    if st.button("🗑️ Xóa sạch dữ liệu DB"):
        ids = collection.get()['ids']
        if ids:
            collection.delete(ids=ids)
            st.warning("Đã xóa toàn bộ dữ liệu.")
            st.rerun()

# ================== HÀM TRUY VẤN RAG ==================
def query_rag(query_text):
    # 1. Retrieval
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    # 2. Xây dựng Context
    context_list = []
    if results["documents"][0]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            context_list.append(f"[{meta.get('hierarchy', 'N/A')}] {doc} (Nguồn: {meta.get('url', 'Internet')})")
    
    context = "\n\n".join(context_list)

    # 3. Prompt Engineering
    prompt = f"""Bạn là trợ lý tư vấn thủ tục hành chính công Việt Nam chuyên về trẻ em dưới 6 tuổi.
Chỉ sử dụng thông tin từ CONTEXT để trả lời. Nếu không có thông tin, hãy nói đúng câu: "Xin lỗi! Câu hỏi của bạn không nằm trong phạm vi hỗ trợ của tôi."

CONTEXT:
{context}

CÂU HỎI: {query_text}

YÊU CẦU:
- Trình bày rõ ràng, đánh số thứ tự nếu có nhiều bước.
- Trích dẫn nguồn (URL) từ context ở cuối câu trả lời.
- Ngôn ngữ: Tiếng Việt.
"""
    
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    return model.generate_content(prompt, stream=True)

# ================== LOGIC CHAT ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhận câu hỏi
if prompt := st.chat_input("Hỏi về thủ tục làm giấy khai sinh..."):
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
            st.error(f"Lỗi hệ thống: {str(e)}")
            full_response = "Đã có lỗi xảy ra khi kết nối với AI."

    st.session_state.messages.append({"role": "assistant", "content": full_response})
