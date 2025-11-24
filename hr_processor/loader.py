# backend/hr_processor/loader.py

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_hr_document(file_path: str = "hr_document.txt"):
    """Load the HR text file and return full text."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HR document not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split long text into smaller chunks (documents)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.create_documents([text])
