from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent
POLICY_PATH = BASE_DIR / "data" / "company_policies.txt"
VECTOR_DB_PATH = BASE_DIR / "data" / "chroma_db"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

COLLECTION_NAME = "company_policies"

def load_policy():
    """Load the company policy document."""
    loader = TextLoader(
        str(POLICY_PATH),
        encoding="utf-8"
    )

    return loader.load()


def split_policy(
    documents,
    chunk_size=700,
    chunk_overlap=100
):
    """Split policy documents into smaller chunks."""
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(documents)

def inspect_chunks(chunks, preview_chars=250):
    """Display chunk information for debugging."""

    print("\n=== Chunk inspection ===")
    print("Total chunks:", len(chunks))

    for index, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {index}")
        print("Characters:", len(chunk.page_content))
        print(chunk.page_content[:preview_chars])

def add_chunk_metadata(chunks):
    """Attach useful metadata to each chunk."""

    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["source_name"] = POLICY_PATH.name

    return chunks

def create_embeddings():
    """Create the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )