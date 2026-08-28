from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BASE_DIR / "data/knowledge"
CHROMA_DIR = BASE_DIR / "data/db/chroma"

PROMPT = PromptTemplate(
    template="""
You are a policy assistant for Northstar Retail.

Use ONLY the supplied context to answer the question.
If the answer is not supported by the context, say:
"I don't know based on the provided policy documents."

Do not invent policies, approval limits, time periods, or exceptions.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

def load_documents():
    documents = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        documents.extend(TextLoader(str(path), encoding="utf-8").load())
    return documents

def build_vectorstore():
    # INDEXING: Load -> Split -> Embed -> Store
    documents = load_documents()

    splitter = CharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separator="\n"
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

def build_rag_chain():
    # RUNTIME: Retrieve -> Augment -> Generate
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Copy .env.example to .env and add your key."
        )

    vectorstore = build_vectorstore()

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

def answer_question(chain, question: str):
    result = chain.invoke({"question": question})

    sources = []
    for doc in result.get("source_documents", []):
        source = Path(doc.metadata.get("source", "unknown")).name
        if source not in sources:
            sources.append(source)

    return {"answer": result["answer"], "sources": sources}
