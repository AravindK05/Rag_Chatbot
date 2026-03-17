import os
import pdfplumber
import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough




# ===============================
# HUGGING FACE TOKEN
# ===============================
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""


# ===============================
# STREAMLIT PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)


# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #1a1a2e; }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        background: #4f46e5 !important;
        color: #ffffff !important;
        border: none !important;
    }
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #4f46e5;
        border-radius: 10px;
        padding: 10px;
    }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ===============================
# SESSION STATE
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.markdown("## 📄 PDF RAG Chatbot")
    st.markdown("*Powered by HuggingFace + LangChain + FAISS*")
    st.divider()

    st.markdown("### 📁 Upload PDFs")
    files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    process_btn = st.button("⚙️ Process PDFs", type="primary", use_container_width=True)

    if st.session_state.processed_files:
        st.success(f"✅ {len(st.session_state.processed_files)} file(s) loaded")
        for name in st.session_state.processed_files:
            st.markdown(f"- 📄 `{name}`")

    st.divider()

    if st.button("🗑️ Clear & Reset", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.chain = None
        st.session_state.processed_files = []
        st.rerun()

    st.divider()
    st.markdown("""
    **Models:**
    - 🤖 `Llama-3.1-8B-Instruct`
    - 🔢 `all-MiniLM-L6-v2`
    - 🗄️ `FAISS (in-memory)`
    - ☁️ Provider: `cerebras`
    """)


# ===============================
# FUNCTION: EXTRACT TEXT
# ===============================
def extract_text(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# ===============================
# FUNCTION: BUILD RAG CHAIN
# ===============================
def build_chain(files):

    # --- Extract text from all PDFs ---
    all_text = ""
    for file in files:
        all_text += extract_text(file) + "\n"

    # --- Text Chunking ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_text(all_text)

    # --- Embeddings (runs fully locally, no API needed) ---
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- FAISS Vector Store ---
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # --- LLM ---
    # Why nebius?
    # - hf-inference now focuses on CPU / small models only (no large LLMs)
    # - featherless-ai and novita only support "conversational" (OpenAI chat format)
    # - nebius supports Llama-3.1-8B-Instruct with full text-generation on GPU, free tier
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        provider="cerebras",
        max_new_tokens=512,
        temperature=0.3,
        do_sample=False,
    )
    chat_model = ChatHuggingFace(llm=llm)

    # --- Prompt ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions strictly based on the provided context.
If the answer is not in the context, say 'I don't have enough information to answer that.'
Do not make up or add any information beyond what is in the context.
Keep your answer concise and to the point."""),
        ("human", """Context:
{context}

Question:
{question}""")
    ])

    # --- Format documents ---
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # --- RAG Chain ---
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    return chain


# ===============================
# PROCESS PDFs ON BUTTON CLICK
# ===============================
if process_btn:
    if files:
        with st.spinner("⚙️ Processing PDFs — extracting, chunking & embedding..."):
            st.session_state.chain = build_chain(files)
            st.session_state.processed_files = [f.name for f in files]
            st.session_state.chat_history = []
        st.success("✅ PDFs processed! Ask your questions below.")
        st.rerun()
    else:
        st.warning("⚠️ Please upload at least one PDF first.")


# ===============================
# MAIN CHAT AREA
# ===============================
st.markdown("## 💬 Chat with your Documents")
st.divider()

if st.session_state.chain:

    # Display chat history
    for question, answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)

    # User input
    user_question = st.chat_input("Ask a question about your PDF(s)...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.chain.invoke(user_question)
            st.write(response)

        st.session_state.chat_history.append((user_question, response))

else:
    st.info("👈 Upload a PDF in the sidebar and click **⚙️ Process PDFs** to get started.")
