from langchain.tools import Tool
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate


from src.llm import get_llm
from src.tools import get_all_tools


Agent_system_instructions = """You are a helpful and an expert university notes assistant helping a student.
                                You have access to the following tools: {tools}
                                Always try the the notes_search tool first, as it searches the notes uploded bu the students.
                                Use the Web_serach if the notes do not contain the answer or if there is not much context about the question and answer in the notes.
                                Use the claculator for any math problem or engineering calculations, also you can use the web search tool be the problem is not sloved by the calculator tool
                                
                                Use the following format:

                                Question: the input question you must answer
                                Thought: think about what to do
                                Action: the action to take, must be one of [{tool_names}]
                                Action Input: the input to the action
                                Observation: the result of the action
                                ... (this Thought/Action/Action Input/Observation can repeat)
                                Thought: now i know the final answer
                                Final Answer: the final answer to the original question, written clearly
                                for a student

                                Question: {input}
                                Thought:{agent}
                            """
                            
   

AGENT_PROMT = PromptTemplate.from_template(Agent_system_instructions)

def notes_tool(retriever):
    def search_notes(query):
        docs = retriever.invoke(query)
        if not docs:
            return "No information found in the uploded notes" 
        parts = []
        for d in docs:
            source = d.metadata.get("source", "unknow")
            page = d.metadata.get("page", "?")
            parts.append(f"(Source:{source}, page:{page})\n{d.page_content}")
            return "\n\n".join(parts)
        tools = Tool(
            name = "notes_search",
            func = search_notes,
            description = (
                """Search the uploaded notes by the student for relevant information.
                   Always try this before web search"""
            )
        )                   
                    
                    


                            




