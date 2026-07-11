import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id=os.getenv("HF_CHAT_MODEL"),
        task="text-generation",
        provider=os.getenv("HF_INFERENCE_PROVIDER"),
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature = 0.65,
        max_new_tokens = 1024,
        streaming = True,
    )

    model = ChatHuggingFace(llm=llm)
    
    return model