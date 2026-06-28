from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader


BASE_DIR = Path(__file__).resolve().parent


def read_text_file(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf_file(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(page for page in pages if page).strip()


def load_chunks_from_files(file_paths: list[Path]) -> list[str]:
    chunks = []

    for file_path in file_paths:
        if not file_path.exists():
            continue

        if file_path.suffix.lower() == ".pdf":
            content = read_pdf_file(file_path)
        elif file_path.suffix.lower() in {".txt", ".md"}:
            content = read_text_file(file_path)
        else:
            continue

        if not content:
            continue

        parsed_chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
        chunks.extend(parsed_chunks)

    return chunks


# Load content from text files and PDF files in the workspace
files_to_load = [BASE_DIR / "profile.txt"]
files_to_load.extend(sorted(BASE_DIR.glob("*.pdf")))

chunks = load_chunks_from_files(files_to_load)

if not chunks:
    raise ValueError("No readable text content was found in the provided files.")

print(f"Loaded {len(chunks)} chunks from text/PDF files")

# Initialize ChromaDB - PersistentClient saves data to disk so it survives restarts
client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))

# Connect to Ollama's embedding model to convert text into vectors
ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://host.docker.internal:11434",  # Ollama's address (works in Docker on Windows/Mac)
)

# Create (or reuse) a collection - like a table in a database
collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,  # Tells ChromaDB how to convert text to vectors
)

# Add chunks to the collection - ChromaDB automatically generates embeddings
collection.add(
    ids=[f"chunk{i}" for i in range(len(chunks))],  # Unique ID for each chunk
    documents=chunks,  # The actual text content
    metadatas=[{"source": "profile", "chunk_index": i} for i in range(len(chunks))],
)

print(f"Added {len(chunks)} chunks to the 'personal_profile' collection.")
print("Knowledge base built successfully!")


