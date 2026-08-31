import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
POLICY_PATH = BASE_DIR / "data" / "company_policies.txt"
VECTOR_DB_PATH = BASE_DIR / "data" / "chroma_db"
ENV_PATH = BASE_DIR / ".env.local"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

COLLECTION_NAME = "company_policies"
UNKNOWN_ANSWER = (
    "I don't know based on the available company policies."
)

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

def inspect_embedding(embeddings, text):
    """Display basic information about one embedding."""
    vector = embeddings.embed_query(text)

    print("\n=== Embedding inspection ===")
    print("Text:", text)
    print("Dimensions:", len(vector))
    print("First 5 values:", vector[:5])

    return vector

def create_vector_store(chunks):
    """Store policy chunks in Chroma."""

    embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_PATH)
    )

    ids = [
        f"policy-{chunk.metadata['chunk_id']}"
        for chunk in chunks
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )

    return vector_store

def inspect_vector_store(vector_store):
    """Display basic vector store information."""
    stored = vector_store.get()

    print("\n=== Vector store ===")
    print("Stored chunks:", len(stored["ids"]))
    print("IDs:", stored["ids"])

    return stored

def load_vector_store():
    """Load the existing Chroma vector store."""
    embeddings = create_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_PATH)
    )

def create_retriever(vector_store, k=3):
    """Create a semantic retriever."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

def retrieve_policy_context(retriever, question):
    """Retrieve policy chunks relevant to a question."""
    return retriever.invoke(question)

def inspect_retrieval(documents):
    """Display retrieved policy chunks."""

    print("\n=== Retrieved context ===")

    for index, document in enumerate(
        documents,
        start=1
    ):
        print(f"\nResult {index}")
        print(
            "Chunk:",
            document.metadata.get("chunk_id")
        )
        print(
            "Source:",
            document.metadata.get("source_name")
        )
        print(document.page_content)

def format_context(documents):
    """Combine retrieved documents into one context string."""
    return "\n\n".join(
        document.page_content
        for document in documents
    )

def create_llm():
    """Create the Gemini chat model."""
    load_dotenv(ENV_PATH)

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY is missing from .env.local"
        )

    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash"
    )

    return ChatGoogleGenerativeAI(
        model=model_name
    )

def build_prompt(context, question):
    """Build a grounded policy question-answering prompt."""
    return f"""
You are a company policy assistant.

Answer the question using only the provided policy context.

If the answer is not clearly supported by the context, respond exactly with:
"{UNKNOWN_ANSWER}"

Do not use outside knowledge.
Do not invent policies, requirements, approval rules, or timelines.

When the policy contains a specific number, percentage, approval level,
or time period, preserve that value in the answer.

Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
""".strip()

def ask_policy(question):
    """Answer a question using retrieved company policy context."""

    vector_store = load_vector_store()

    retriever = create_retriever(
        vector_store,
        k=3
    )

    documents = retrieve_policy_context(
        retriever,
        question
    )

    context = format_context(
        documents
    )

    prompt = build_prompt(
        context,
        question
    )

    llm = create_llm()

    response = llm.invoke(
        prompt
    )

    return response.text