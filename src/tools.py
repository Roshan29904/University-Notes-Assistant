import numexpr
from langchain_core.tools import BaseTool, tool
from langchain_community.tools import DuckDuckGoSearchRun


# web search
def get_web_search_tool() -> BaseTool:
    return DuckDuckGoSearchRun(
        name="web_search",
        description="""Use this when the answer cannot be found in the uploaded notes and,
                          when the context about a topic is not enough to answer.
                          Also it can be used when unable to solve a problem """,
    )

# calculate
@tool
def calculator(expression: str) -> str:
    """Evaluate a numeric/math expression, example '12*(3+4)/2' or 'sqrt(144) + 5**2'.
       Use this arithmetic, algebra or engineering calculations.
       Input must be valid maths expression string.
    """
    try:
        result = numexpr.evaluate(expression).item()
        return str(result)
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"


# get all tools
def get_all_tools(notes_tool: BaseTool) -> list:
    """Assemble the full toolset for the agent, including the notes search tool, web search tool, and calculator tool."""
    return [notes_tool, get_web_search_tool(), calculator]