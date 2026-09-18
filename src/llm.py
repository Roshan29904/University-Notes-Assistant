import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    model_name = (os.getenv("HF_CHAT_MODEL") or "Qwen/Qwen2.5-Coder-32B-Instruct").strip()
    provider = (os.getenv("HF_INFERENCE_PROVIDER") or "auto").strip()
    if provider == "auto":
        provider = None

    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        provider=provider,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.65,
        max_new_tokens=1024,
        streaming=True,
    )

    model = ChatHuggingFace(llm=llm)
    return model