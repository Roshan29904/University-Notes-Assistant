from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessageChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm


class NotesAssistant:
    """Simple notes-based assistant with optional web search support."""

    def __init__(self, retriever, web_search_enabled: bool = False):
        self.retriever = retriever
        self.web_search_enabled = web_search_enabled
        self.llm = None
        self.llm_error = ""
        try:
            self.llm = get_llm()
        except Exception as exc:
            self.llm_error = str(exc)
        self.search_tool = DuckDuckGoSearchRun() if web_search_enabled else None
        self.system_prompt = r"""You are an expert University Notes Assistant.

Answer using the uploaded notes first.

Rules:
1. Use the uploaded notes as the primary source.
2. If the notes do not contain enough information, say so clearly instead of guessing.
3. If web search is enabled and the notes do not contain enough detail, use the web results to supplement the answer.
4. Explain concepts clearly as if teaching a university student.
5. When information comes from the uploaded notes, mention the source and page number whenever available.
6. When writing any mathematical expression, equation, or formula, format it using dollar-sign delimiters: $...$ for inline math and $$...$$ for standalone/block equations. Never use \( \), \[ \], or other LaTeX delimiter styles."""

    def _format_docs(self, docs):
        if not docs:
            return "No relevant information was found in the uploaded notes."

        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            formatted.append(f"Source: {source}\nPage: {page}\n{doc.page_content}")
        return "\n\n".join(formatted)

    def _get_web_context(self, question: str) -> str:
        if not self.web_search_enabled or not question or self.search_tool is None:
            return ""
        try:
            results = self.search_tool.run(question)
            if isinstance(results, str):
                return f"\n\nWeb search results:\n{results[:3000]}"
            if isinstance(results, list):
                text = "\n".join(str(item) for item in results[:5])
                return f"\n\nWeb search results:\n{text[:3000]}"
        except Exception:
            pass
        return "\n\nWeb search is enabled, but the search provider did not return results."

    def _fallback_web_answer(self, question: str) -> str:
        if not self.web_search_enabled:
            return "Web search is disabled. The app cannot answer this without notes or an LLM configuration."
        try:
            result = self.search_tool.run(question)
            if isinstance(result, str):
                return result[:3000]
            if isinstance(result, list):
                return "\n".join(str(item) for item in result[:5])[:3000]
        except Exception:
            return "I could not fetch live results right now, but the app is configured to use web search when an LLM is available."
        return "I couldn't retrieve a live web result for this question."

    def stream(self, inputs, stream_mode=None):
        messages = inputs.get("messages", [])
        question = ""
        for message in reversed(messages):
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
            if content and isinstance(content, str):
                question = content
                break

        docs = []
        if self.retriever is not None and question:
            try:
                docs = self.retriever.invoke(question)
            except Exception:
                docs = []
        context = self._format_docs(docs)
        if self.web_search_enabled and (not docs or len(context) < 150):
            context += self._get_web_context(question)

        if self.llm is None:
            answer = self._fallback_web_answer(question) if question else "Web search is enabled, but no question was provided."
            final_messages = list(messages)
            final_messages.append({"role": "assistant", "content": answer})
            yield ("messages", (AIMessageChunk(content=answer), {}))
            yield ("values", {"messages": final_messages})
            return

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "Context from notes:\n{context}\n\nQuestion:\n{question}"),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        final_messages = list(messages)
        final_messages.append({"role": "assistant", "content": answer})

        yield ("messages", (AIMessageChunk(content=answer), {}))
        yield ("values", {"messages": final_messages})


def build_agent(retriever, web_search_enabled: bool = False):
    """Build a notes-focused assistant with optional web search support."""
    return NotesAssistant(retriever, web_search_enabled=web_search_enabled)