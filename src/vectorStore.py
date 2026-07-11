import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

from  src.embeddings import get_embeddings

load_dotenv()

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR")


def build_vectorstore(chunks: List[Document]) -> FAISS:
    embeddings = get_embeddings()
    store = FAISS.from_documents(chunks, embedding=embeddings)
    return store


def save_vectorstore(store: FAISS, path: str=VECTORSTORE_DIR):
    os.makedirs(path, exist_ok=True)
    store.save_local(path)


def load_vectorstore(path: str=VECTORSTORE_DIR):
    if not os.path.exists(path):
        return None
    embeddings = get_embeddings()
    
    return FAISS.load_local(path, embeddings=embeddings, allow_dangerous_deserialization=True)
    


def add_documents(store: FAISS, chunks: List[Document]):
    store.add_documents(chunks)
    return store


def get_retriever(store: FAISS, k: int = 4):
    return store.as_retriever(search_type = "similarity", search_kwargs={"k":k})
