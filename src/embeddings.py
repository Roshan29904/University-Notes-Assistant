import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_embeddings():
    embeddings = HuggingFaceEndpointEmbeddings(
        model=os.getenv("EMBEDDING_MODEL_NAME"),
        task="feature-extraction",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    return embeddings


