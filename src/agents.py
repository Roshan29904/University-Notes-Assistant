from langchain_core.messages import AIMessageChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm


class NotesAssistant:
    """Simple notes-based assistant that avoids unsupported tool-calling parameters."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = get_llm()
        self.system_prompt = r"""You are an expert University Notes Assistant.

Answer using the uploaded notes first.

Rules:
1. Use the uploaded notes as the primary source.
2. If the notes do not contain enough information, say so clearly instead of guessing.
3. Explain concepts clearly as if teaching a university student.
4. When information comes from the uploaded notes, mention the source and page number whenever available.
5. When writing any mathematical expression, equation, or formula, format it using dollar-sign delimiters: $...$ for inline math and $$...$$ for standalone/block equations. Never use \( \), \[ \], or other LaTeX delimiter styles."""

    def _format_docs(self, docs):
        if not docs:
            return "No relevant information was found in the uploaded notes."

        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            formatted.append(f"Source: {source}\nPage: {page}\n{doc.page_content}")
        return "\n\n".join(formatted)

    def stream(self, inputs, stream_mode=None):
        messages = inputs.get("messages", [])
        question = ""
        for message in reversed(messages):
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
            if content and isinstance(content, str):
                question = content
                break

        docs = self.retriever.invoke(question) if question else []
        context = self._format_docs(docs)

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


def build_agent(retriever):
    """Build a notes-focused assistant that does not send unsupported tools to the model."""
    return NotesAssistant(retriever)