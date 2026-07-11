from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm
from src.prompts import RAG_prompt, SUMMARIZE_PROMT, QUIZ_PROMPT, SHORT_NOTES_PROMPT, EXPLAIN_SIMPLY_PROMPT, EXTRACT_FORMULAS_PROMPT, EXAM_QUESTIONS_PROMPT, TOPIC_SEARCH_PROMPT

def format_docs(docs: List[Document]) -> str:
    """Turns retrieved chunk into a single context string with source tags"""
    return "\n\n".join(
        f"[Chunk {i+1} | {doc.metadata.get('source', 'unknown')}, page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


def question_answer(question: str, retriever, chat_history: List[Tuple[str,str]]=None) -> Dict:
    """Run RAG over the notes vectorstore.
    Returns {"answer":  str, "source": List[Document]}
    """
    chat_history = chat_history or []
    docs = retriever.invoke(question)
    context = format_docs(docs)
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


def text_prompt(prompt_template: ChatPromptTemplate, **kwargs) -> str:
    llm = get_llm()
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke(kwargs)


def summarize_text(text: str)-> str:
    return text_prompt(SUMMARIZE_PROMT, text=text)


def generate_quiz(text:str, num_questions: int=7)->str:
    return text_prompt(QUIZ_PROMPT, text=text, num_questions=num_questions)

def generate_short_notes(text: str) -> str:
    return text_prompt(SHORT_NOTES_PROMPT, text=text)


def explain_simply(text: str) -> str:
    return text_prompt(EXPLAIN_SIMPLY_PROMPT, text=text)


def extract_formulas(text: str) -> str:
    return text_prompt(EXTRACT_FORMULAS_PROMPT, text=text)


def generate_exam_questions(text: str, num_questions: int = 5) -> str:
    return text_prompt(EXAM_QUESTIONS_PROMPT, text=text, num_questions=num_questions)


def search_topic(topic: str, docs: List[Document]) -> str:
    text = format_docs(docs)
    llm = get_llm()
    chain = TOPIC_SEARCH_PROMPT | llm | StrOutputParser()
    return chain.invoke({"topic": topic, "text": text})