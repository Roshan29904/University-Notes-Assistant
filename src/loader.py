import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    file_name = os.path.basename(file_path)
    for doc in docs:
        doc.metadata["source"] = file_name
        
    return docs

