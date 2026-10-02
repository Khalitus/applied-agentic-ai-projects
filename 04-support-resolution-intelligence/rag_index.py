import hashlib
from functools import lru_cache
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, KNOWLEDGE_DIR


class LocalSentenceEmbeddings(Embeddings):
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model = SentenceTransformer(
            model_name,
            local_files_only=True,
        )

    def embed_documents(self, texts):
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text):
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


def stable_id(value):
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def load_parent_documents():
    documents = []

    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        if path.name.startswith("._"):
            continue
        text = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "parent_id": stable_id(path.name),
                    "source": path.name,
                    "source_type": "policy",
                },
            )
        )

    cases = pd.read_csv(KNOWLEDGE_DIR / "resolved_cases.csv")
    for row in cases.to_dict("records"):
        priority = row["priority"] if pd.notna(row["priority"]) else "unknown"
        text = (
            f"Historical support case {row['case_id']}\n"
            f"Issue type: {row['issue_type']}\n"
            f"Priority: {priority}\n"
            f"Customer tier: {row['customer_tier']}\n"
            f"Product: {row['product_name']} ({row['category']})\n"
            f"Issue: {row['issue_text']}\n"
            f"Escalated: {row['escalated']}\n"
            f"Resolution: {row['resolution_text']}"
        )
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "parent_id": stable_id(row["case_id"]),
                    "source": row["case_id"],
                    "source_type": "historical_case",
                    "issue_type": row["issue_type"],
                },
            )
        )

    return documents


def split_documents(parent_documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=550,
        chunk_overlap=90,
    )

    children = []

    for parent in parent_documents:
        chunks = splitter.split_text(parent.page_content)

        for child_index, chunk in enumerate(chunks):
            metadata = {
                **parent.metadata,
                "child_index": child_index,
            }

            children.append(
                Document(
                    page_content=chunk,
                    metadata=metadata,
                )
            )

    return children


def build_index():
    parents = load_parent_documents()
    children = split_documents(parents)

    embeddings = LocalSentenceEmbeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    vector_store.reset_collection()

    ids = [
        f"{doc.metadata['parent_id']}-{doc.metadata['child_index']}"
        for doc in children
    ]

    vector_store.add_documents(
        documents=children,
        ids=ids,
    )

    print(
        f"Indexed {len(children)} child chunks "
        f"from {len(parents)} parent documents."
    )

    return vector_store


@lru_cache(maxsize=1)
def load_index():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=LocalSentenceEmbeddings(),
        persist_directory=str(CHROMA_DIR),
    )


if __name__ == "__main__":
    build_index()
