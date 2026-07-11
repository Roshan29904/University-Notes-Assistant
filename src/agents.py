from langchain.agents import create_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm
from src.tools import get_all_tools


def create_notes_search_tool(retriever):
    """
    Creates a tool for searching the student's uploaded notes.
    """

    def search_notes(query: str):
        docs = retriever.invoke(query) or []

        if not docs:
            return (
                "No relevant information was found in the uploaded notes."
            )

        results = []

        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")

            results.append(
                f"""
                    Source : {source}
                    Page   : {page}
                    {doc.page_content}  
                """
            )

        return "\n\n".join(results)

    return Tool(
        name="notes_search",
        func=search_notes,
        description=(
            "Search the student's uploaded notes for information. "
            "Always use this tool first before using web_search."
        ),
    )


def build_agent(retriever):
    """
    Build the University Notes AI Agent.
    """

    llm = get_llm()

    notes_tool = create_notes_search_tool(retriever)

    tools = get_all_tools(notes_tool)

    # This prompt template is crucial for the agent to decide which tool to use.
    # It includes a placeholder for the "agent_scratchpad" where the agent's
    # intermediate steps (thoughts and tool calls) are stored.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert University Notes Assistant.

Your job is to answer student questions accurately.

Follow these rules:

1. ALWAYS search the uploaded notes first using notes_search.

2. If the uploaded notes do not contain enough information, use the web_search tool.

3. For arithmetic, mathematics, engineering calculations, numerical computation, unit conversions, or formulas, use the calculator tool.

4. Combine information from multiple tools whenever necessary.

5. If the notes contain the answer, prefer the notes over the web.

6. If neither the notes nor the web contain the answer, politely tell the student that you could not find enough reliable information.

7. Explain concepts clearly as if teaching a university student.

8. When information comes from the uploaded notes, mention the source and page number whenever available.""",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_agent(
        llm,
        tools,
        prompt,
    )

    return agent