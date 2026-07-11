import os 
import streamlit as st
from dotenv import load_dotenv

from src.loader import load_pdf, save_uploaded_file
from src.splitter import splitter_doc
from src.vectorStore import build_vectorstore, save_vectorstore, load_vectorstore, add_documents, get_retriever
from src.RAG import question_answer, summarize_text, generate_quiz, generate_short_notes, explain_simply, extract_formulas, generate_exam_questions, search_topic
from src.agents import build_agent

load_dotenv()



st.set_page_config(page_title="University Notes Assistant", page_icon=":mortar_board:", layout="wide")