import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    file_name = os.path.basename(file_path)
    for doc in docs:
        doc.metadata["source"] = file_name
        
    return docs

def save_uploaded_file(uploaded_file, save_dir: str = "data/uploaded_pdfs") -> str:
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path