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

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "company_policies"
UNKNOWN_ANSWER = "I don't know based on the available company policies."


def load_policy():
    """Load the company policy document."""
    loader = TextLoader(str(POLICY_PATH), encoding="utf-8")
    return loader.load()


def split_policy(documents, chunk_size=700, chunk_overlap=100):
    """Split policy documents into smaller chunks."""
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def add_chunk_metadata(chunks):
    """Attach source metadata to each chunk."""
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["source_name"] = POLICY_PATH.name

    return chunks


def create_embeddings():
    """Create the embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vector_store(chunks):
    """Store policy chunks in Chroma."""
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=create_embeddings(),
        persist_directory=str(VECTOR_DB_PATH),
    )

    ids = [f"policy-{chunk.metadata['chunk_id']}" for chunk in chunks]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    return vector_store


def load_vector_store():
    """Load the existing Chroma vector store."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=create_embeddings(),
        persist_directory=str(VECTOR_DB_PATH),
    )


def create_retriever(vector_store, k=3):
    """Create a semantic retriever."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def retrieve_policy_context(retriever, question):
    """Retrieve policy chunks relevant to a question."""
    return retriever.invoke(question)


def format_context(documents):
    """Combine retrieved documents into one context string."""
    return "\n\n".join(document.page_content for document in documents)


def create_llm():
    """Create the Gemini chat model."""
    load_dotenv(ENV_PATH)

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is missing from .env.local")

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    return ChatGoogleGenerativeAI(model=model_name)


def collect_sources(documents):
    """Collect unique sources from retrieved documents."""
    sources = []
    seen = set()

    for document in documents:
        source_name = document.metadata.get("source_name")
        chunk_id = document.metadata.get("chunk_id")
        key = (source_name, chunk_id)

        if key in seen:
            continue

        seen.add(key)
        sources.append({
            "source": source_name,
            "chunk_id": chunk_id,
        })

    return sources


def build_prompt(context, question):
    """Build a grounded policy question-answering prompt."""
    return f"""
You are a company policy assistant.

Answer the question using only the provided policy context.

If the answer is not clearly supported by the context, respond exactly with:
"{UNKNOWN_ANSWER}"

Do not use outside knowledge.
Do not invent policies, requirements, approval rules, or timelines.
Preserve any specific number, percentage, approval level, or time period stated in the policy.
Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def ask_policy(question):
    """Answer a question using retrieved company policy context."""
    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a policy question.",
            "sources": [],
        }

    vector_store = load_vector_store()
    retriever = create_retriever(vector_store, k=3)
    documents = retrieve_policy_context(retriever, question)
    context = format_context(documents)
    prompt = build_prompt(context, question)
    response = create_llm().invoke(prompt)
    answer = response.text.strip()

    sources = []
    if answer != UNKNOWN_ANSWER:
        sources = collect_sources(documents)

    return {
        "answer": answer,
        "sources": sources,
    }


def format_policy_result(result):
    """Format a policy answer and its sources for display."""
    answer = result["answer"]
    sources = result["sources"]

    if not sources:
        return answer

    source_lines = [
        f"- {source['source']} (chunk {source['chunk_id']})"
        for source in sources
    ]

    return f"{answer}\n\nSources:\n" + "\n".join(source_lines)