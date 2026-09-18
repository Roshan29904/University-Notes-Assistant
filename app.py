import os
import re
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk

from src.agents import build_agent
from src.loader import load_pdf, save_uploaded_file
from src.persistence import (
    load_chat_history,
    load_doc_texts,
    load_processed_files,
    save_chat_history,
    save_doc_texts,
    save_processed_files,
)
from src.RAG import (
    explain_simply,
    extract_formulas,
    generate_exam_questions,
    generate_quiz,
    generate_short_notes,
    question_answer,
    search_topic,
    summarize_text,
)
from src.splitter import splitter_doc
from src.vectorStore import add_documents, build_vectorstore, get_retriever, load_vectorstore, save_vectorstore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH, override=True)
load_dotenv(dotenv_path=ENV_PATH, override=True)

def get_hf_token():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        try:
            token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HF_TOKEN")
        except Exception:
            token = None
    if token:
        os.environ["HF_TOKEN"] = str(token)
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = str(token)
    return token


hf_token = get_hf_token()

DEFAULT_VECTORSTORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore")
DEFAULT_UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploaded_pdfs")
DEFAULT_HISTORY_PATH = os.path.join(BASE_DIR, "data", "chat_history.json")
DEFAULT_PROCESSED_FILES_PATH = os.path.join(BASE_DIR, "data", "processed_files.json")
DEFAULT_DOC_TEXTS_PATH = os.path.join(BASE_DIR, "data", "doc_texts.json")


def resolve_storage_dir(env_value: str | None, fallback: str) -> str:
    value = (env_value or "").strip()
    if value:
        return os.path.abspath(value)
    return os.path.abspath(fallback)


VECTORSTORE_DIR = resolve_storage_dir(os.getenv("VECTORSTORE_DIR"), DEFAULT_VECTORSTORE_DIR)
UPLOAD_DIR = resolve_storage_dir(os.getenv("UPLOAD_DIR"), DEFAULT_UPLOAD_DIR)

os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)



def render_math(text: str) -> str:
    """
    Streamlit's markdown renderer only understands $...$ / $$...$$ for math.
    Some models write LaTeX using \\(...\\) / \\[...\\] delimiters instead,
    which would otherwise show up as raw, unrendered text. Convert them here
    as a safety net (on top of asking the model to use $ delimiters directly).
    """
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)
    return text
 
st.set_page_config(page_title="University Notes AI Assistant", page_icon="📚", layout="wide")
 
 
# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "vectorstore": None,
        "retriever": None,
        "agent": None,
        "agent_messages": [],
        "display_messages": [],
        "doc_texts": {},
        "processed_files": set(),
        "enable_web_search": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "fresh_session_reset" not in st.session_state:
        st.session_state["fresh_session_reset"] = True
        # Ignore any persisted uploaded data for a brand-new deployment/browser session.
        st.session_state.doc_texts = {}
        st.session_state.processed_files = set()
        save_processed_files(DEFAULT_PROCESSED_FILES_PATH, [])
        save_doc_texts(DEFAULT_DOC_TEXTS_PATH, {})
        save_chat_history(DEFAULT_HISTORY_PATH, [])


init_state()

# Keep the app clean on each new site opening, while allowing the active session to work normally.
if st.session_state.vectorstore is None:
    store = load_vectorstore(VECTORSTORE_DIR)
    if store is not None:
        st.session_state.vectorstore = store
        st.session_state.retriever = get_retriever(store)
 
 
def get_or_build_agent():
    if st.session_state.agent is None or getattr(st.session_state.agent, "web_search_enabled", False) != st.session_state.get("enable_web_search", True):
        with st.spinner("Setting up the assistant..."):
            try:
                st.session_state.agent = build_agent(
                    st.session_state.retriever,
                    web_search_enabled=st.session_state.get("enable_web_search", True),
                )
                st.session_state.agent_error = ""
            except Exception as exc:
                st.session_state.agent = None
                st.session_state.agent_error = (
                    "The assistant could not start because the Hugging Face token is missing or invalid. "
                    "Add HUGGINGFACEHUB_API_TOKEN to your environment or .env file."
                )
                return None
    return st.session_state.agent
 
 
def stream_agent_reply(agent, messages, final_state_holder):
    """
    Yields plain-text chunks as the agent's final answer is generated, for use
    with st.write_stream. stream_mode=["messages", "values"] gives us both:
      - "messages": (message_chunk, metadata) pairs for token-level streaming
      - "values": the full graph state after each step, so we can capture the
        complete, updated message history once the run finishes (needed to
        keep multi-turn context for the next question).
    Tool-call chunks are skipped since they carry no user-facing text.
    """
    for mode, chunk in agent.stream({"messages": messages}, stream_mode=["messages", "values"]):
        if mode == "messages":
            message_chunk, _metadata = chunk
            if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
                yield message_chunk.content
        elif mode == "values":
            final_state_holder["messages"] = chunk.get("messages", messages)
 
 
def process_uploaded_files(uploaded_files):
    new_chunks = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.processed_files:
            continue

        save_path = save_uploaded_file(uploaded_file, save_dir=UPLOAD_DIR)
        docs = load_pdf(save_path)

        st.session_state.doc_texts[uploaded_file.name] = "\n\n".join(
            doc.page_content for doc in docs
        )

        chunks = splitter_doc(docs)
        new_chunks.extend(chunks)
        st.session_state.processed_files.add(uploaded_file.name)

    save_processed_files(DEFAULT_PROCESSED_FILES_PATH, sorted(st.session_state.processed_files))
    save_doc_texts(DEFAULT_DOC_TEXTS_PATH, st.session_state.doc_texts)

    if not new_chunks:
        return

    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = build_vectorstore(new_chunks)
    else:
        st.session_state.vectorstore = add_documents(st.session_state.vectorstore, new_chunks)

    save_vectorstore(st.session_state.vectorstore, VECTORSTORE_DIR)
    st.session_state.retriever = get_retriever(st.session_state.vectorstore)
    st.session_state.agent = None


with st.sidebar:
    st.title("📚 University Notes Assistant")
    st.caption("Upload lecture notes (PDF), then chat or generate study material from them.")

    uploaded_files = st.file_uploader(
        "Upload PDF notes", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process documents", disabled=not uploaded_files, use_container_width=True):
        with st.spinner("Reading, chunking, and indexing your notes..."):
            process_uploaded_files(uploaded_files)
        st.success("Notes indexed. You can now chat or use the study tools.")

    st.checkbox("Enable web search", key="enable_web_search")

    if st.session_state.processed_files:
        st.markdown("**Indexed documents**")
        for name in sorted(st.session_state.processed_files):
            st.markdown(f"- {name}")

    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.agent_messages = []
        st.session_state.display_messages = []
        st.rerun()


if st.session_state.retriever is None:
    st.info("Upload and process at least one PDF from the sidebar to get started.")
    st.stop()


tab_chat, tab_tools = st.tabs(["💬 Chat", "🛠️ Study Tools"])

# st.chat_input only auto-pins itself to the bottom of the page when called
# at the top level of the script - nested inside a tab/column/container, it
# renders inline instead. So it's called here, outside `with tab_chat:`, and
# the returned value is used inside the tab below.
query = st.chat_input("Ask a question about your notes...")


# ---------------------------------------------------------------------------
# Chat tab - talks to the tool-using agent (notes search, web search, calculator)
# ---------------------------------------------------------------------------
with tab_chat:
    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"]):
            st.markdown(render_math(msg["content"]))

    if query:
        st.session_state.display_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        agent = get_or_build_agent()
        if agent is None:
            error_msg = st.session_state.get(
                "agent_error",
                "The assistant could not start because the Hugging Face token is missing or invalid."
            )
            st.warning(error_msg)
            st.session_state.display_messages.append({"role": "assistant", "content": error_msg})
            st.rerun()
            st.stop()

        st.session_state.agent_messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            final_state = {}
            try:
                answer = st.write_stream(
                    stream_agent_reply(agent, st.session_state.agent_messages, final_state)
                )
                st.session_state.agent_messages = final_state.get(
                    "messages", st.session_state.agent_messages
                )
            except Exception as e:
                answer = f"Something went wrong while answering: {e}"
                st.markdown(answer)

        st.session_state.display_messages.append({"role": "assistant", "content": answer})
        st.rerun()
 
 
# ---------------------------------------------------------------------------
# Study tools tab - one-shot generation over a whole document
# ---------------------------------------------------------------------------
with tab_tools:
    st.subheader("Generate study material from your notes")
 
    if not st.session_state.doc_texts:
        st.info("Process at least one PDF to use the study tools.")
    else:
        doc_choice = st.selectbox("Choose a document", sorted(st.session_state.doc_texts.keys()))
        text = st.session_state.doc_texts[doc_choice]
 
        tool_choice = st.selectbox(
            "Choose a tool",
            [
                "Summarize",
                "Short revision notes",
                "Explain simply",
                "Extract formulas",
                "Generate quiz",
                "Predict exam questions",
                "Search a topic",
            ],
        )
 
        num_questions = None
        topic = None
        if tool_choice in ("Generate quiz", "Predict exam questions"):
            num_questions = st.slider("Number of questions", min_value=3, max_value=15, value=7)
        if tool_choice == "Search a topic":
            topic = st.text_input("Topic to search for")
 
        if st.button("Generate", use_container_width=True):
            output = None
            with st.spinner("Working on it..."):
                try:
                    if tool_choice == "Summarize":
                        output = summarize_text(text)
                    elif tool_choice == "Short revision notes":
                        output = generate_short_notes(text)
                    elif tool_choice == "Explain simply":
                        output = explain_simply(text)
                    elif tool_choice == "Extract formulas":
                        output = extract_formulas(text)
                    elif tool_choice == "Generate quiz":
                        output = generate_quiz(text, num_questions=num_questions)
                    elif tool_choice == "Predict exam questions":
                        output = generate_exam_questions(text, num_questions=num_questions)
                    elif tool_choice == "Search a topic":
                        if not topic:
                            st.warning("Please enter a topic to search for.")
                        else:
                            docs = st.session_state.retriever.invoke(topic)
                            output = search_topic(topic, docs)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
 
            if output:
                st.markdown(render_math(output))