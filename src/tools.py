import numexpr
from langchain.tools import Tool, tool
from langchain_community.tools import DuckDuckGoSearchRun


# web search
def get_web_search_tool() -> Tool:
    search = DuckDuckGoSearchRun()
    web_search = Tool(
        name = "web_search",
        func = search.run,
        description = ("""Use this when the answer cannot be found in the uploaded notes and,
                          when the context about a topic is not enough to answer.
                          Also it can be used when uable to slove a problem """ 
        )
    )
    return web_search
    
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
        return f"Erroe evaluating expression '{expression}' : {e}"
    
    
# get all tools
def get_all_tools(tools: Tool) -> list:
    """Assemble the full toolset for the agent, including the notes search tool, web search tool, and calculator tool."""
    return [tools, get_web_search_tool(), calculator]