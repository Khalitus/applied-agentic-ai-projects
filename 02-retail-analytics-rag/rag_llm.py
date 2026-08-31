from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
POLICY_PATH = BASE_DIR / "data" / "company_policies.txt"


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