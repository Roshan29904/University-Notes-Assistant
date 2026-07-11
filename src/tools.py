import numexpr
from langchain.tools import Tool, tool
from langchain_community.tools import DuckDuckGoSearchRun


# web search
def get_web_search_tool() -> Tool:
    search = DuckDuckGoSearchRun()
    web_search = Tool(
        name = "web_search",
        function = search.run,
        description = ("""Use this when the answer cannot be found in the uploaded notes and,
                          when the context about a topic is not enough to answer.""" 
        )
    )
    
    
