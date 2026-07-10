from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def splitter_doc(documents: List[Document], chunk_size: int=1000, chunk_overlap: int=150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        
    return chunks

