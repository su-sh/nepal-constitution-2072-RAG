import os
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from chat import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def load_documents():
    """Load documents from the data directory."""
    return PyPDFDirectoryLoader(DATA_PATH).load()


def split_documents(documents):
    """Split documents into chunks for processing."""
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    ).split_documents(documents)


def add_to_chroma(chunks):
    """Add chunks to the Chroma database."""
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_ids = set(db.get(include=[])["ids"])

    new_chunks = [
        chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids
    ]

    if new_chunks:
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    print(f"Added {len(new_chunks)} new chunks to the database.")


def calculate_chunk_ids(chunks):
    """Calculate unique IDs for each chunk based on the source and page number."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        current_chunk_index = (
            0 if current_page_id != last_page_id else current_chunk_index + 1
        )

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


def clear_database() -> None:
    """Clear the database."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def main() -> None:
    print("Populating database...")
    clear_database()
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    print("Database populated.")


if __name__ == "__main__":
    main()
