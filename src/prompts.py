from langchain_core.prompts import ChatPromptTemplate


# system prompt
RAG_system_prompt = """you are a helpful and expert university teaching assistant.
                Answer the student's questions using  ONLY the context (notes provided) provided below.
                Be clear, concise, and  use simple language so that the student could understand.
                If the context does not have enough information then, don't try to guess the answer or don't hallucinate instead use the web seaech tool.
                
                context:{context}"""
  
  
                
# RAG prompt 
RAG_rprompt = ChatPromptTemplate.from_messages([
    ("system", RAG_system_prompt),
    ("placeholder","{chat_history}"),
    ("human","{question}")
])



# summary prompt
SUMMARIZE_PROMT = ChatPromptTemplate.from_template(
    """Summarize the following notes or pdf into a clear, well structured summary with headings and bullet points.
       Keep key definations and important facts intact.
       
       Notes: {text} 
    """
)


# quiz prompt
QUIZ_PROMPT = ChatPromptTemplate.from_template(
    """Based on the lecture notes below, generate {num_questions} quiz
       questions to test a student's understanding. Include a mix of multiple
       choice and short answer questions. Provide the correct answer after each
       question, clearly labeled "Answer:".
       Notes: {text}
    """
) 


SHORT_NOTES_PROMPT = ChatPromptTemplate.from_template(
    """ Condense the following lecture notes into short, revision-friendly
        bullet-point notes. Focus on definitions, key terms, and core ideas only.

        Notes: {text}
    """
)

EXPLAIN_SIMPLY_PROMPT = ChatPromptTemplate.from_template(
    """ Explain the following concept in very simple language, as if
        teaching a beginner. Use an analogy if it helps.

        Concept / Notes: {text}
    """
)

EXTRACT_FORMULAS_PROMPT = ChatPromptTemplate.from_template(
    """ Extract every formula or equation mentioned in the notes below.
        For each formula, list: the formula itself, what each symbol means, and
        when/why it is used. Format as a numbered list.

        Notes: {text}
    """
)

EXAM_QUESTIONS_PROMPT = ChatPromptTemplate.from_template(
    """ Based on the lecture notes below, predict {num_questions} likely
    exam questions a professor might ask. Vary the difficulty (easy, medium,
    hard) and mark each question's likely difficulty.

    Notes: {text}
    """
)

TOPIC_SEARCH_PROMPT = ChatPromptTemplate.from_template(
    """ Search across the note excerpts below and answer where and how the
        topic "{topic}" is discussed. Cite which source/page each point comes from
        if that information is present in the excerpts.

        Excerpts: {text}
    """
)