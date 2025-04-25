# dbcreate_gemini.py

import os
import json
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_PATH = "chunks.json"
DATA_DIR = "agents/data"

def load_documents():
    loader = DirectoryLoader(
        path=DATA_DIR,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader
    )
    return loader.load()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

def save_chunks(chunks):
    data = [{"content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]
    with open(CHUNK_PATH, "w") as f:
        json.dump(data, f, indent=2)

def main():
    documents = load_documents()
    chunks = split_documents(documents)
    save_chunks(chunks)
    print(f"Saved {len(chunks)} chunks to {CHUNK_PATH}.")

if __name__ == "__main__":
    main()