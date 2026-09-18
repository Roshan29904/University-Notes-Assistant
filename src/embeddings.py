import os
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


def get_embeddings():
    model_name = (os.getenv("DEFAULT_EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2").strip()

    embeddings = HuggingFaceEmbeddings(
        model=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings