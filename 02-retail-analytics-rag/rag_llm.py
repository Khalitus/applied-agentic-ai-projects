from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# FILE PATH

BASE_DIR = Path(__file__).resolve().parent

POLICY_PATH = (
    BASE_DIR
    / "data"
    / "company_policies.txt"
)


# 1. LOAD DOCUMENT

def load_policy():

    loader = TextLoader(
        str(POLICY_PATH),
        encoding="utf-8"
    )

    documents = loader.load()

    return documents

# 2. SPLIT DOCUMENT

def split_policy(documents):

    splitter = CharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separator="\n"
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks


# 3. VECTOR STORE

def create_vector_store(chunks):

    # chunks
    # ↓
    # embeddings
    # ↓
    # Chroma

    pass

# 4. RETRIEVER

def create_retriever(vector_store):

    # vector store
    # ↓
    # retriever

    pass


# 5. LLM

def create_llm():

    # TASK LATER:
    #
    # Gemini model

    pass


# 6. ASK POLICY

def ask_policy(question):

    # question
    # ↓
    # retrieve context
    # ↓
    # prompt
    # ↓
    # LLM
    # ↓
    # answer

    return (
        "RAG answering is not implemented yet."
    )