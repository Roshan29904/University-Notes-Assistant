
import os
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
 
load_dotenv()
 
# Default to a small, CPU-friendly sentence-transformers model. Running
# embeddings locally avoids depending on which third-party inference
# provider Hugging Face happens to route a given model to (some, like
# "novita", don't support the "feature-extraction" task at all, which is
# what caused the ValueError from HuggingFaceEndpointEmbeddings).
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
 
 
def get_embeddings():
    model_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings