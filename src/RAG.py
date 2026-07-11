from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.llm import get_llm
from src.prompts import RAG_prompt, SUMMARIZE_PROMT, QUIZ_PROMPT, SHORT_NOTES_PROMPT, EXPLAIN_SIMPLY_PROMPT, EXTRACT_FORMULAS_PROMPT, EXAM_QUESTIONS_PROMPT, TOPIC_SEARCH_PROMPT

def context(docs: List[Document]) -> str:
    """Turns retrieved chunk into a single context string with source tags"""
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Chunk {i+1} | {source}, page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def question_answer(question: str, retriever, chat_history: List[Tuple[str,str]]=None) -> Dict:
    """Run RAG over the notes vectorstore.
    Returns {"answer":  str, "source": List[Document]}
    """
    chat_history = chat_history or []
    docs = retriever.invoke(question)
    context = context(docs)
    llm = get_llm()
    chain = RAG_prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": chat_history
        }
    )
    return {"answer": answer, "source": docs}


def text_prompt(prompt_template, text:str, **kwargs) -> str:
    llm = get_llm()
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"text": text, **kwargs})


def summarize_text(text: str)-> str:
    return text_prompt(SUMMARIZE_PROMT, text)


