import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
    else:
        raise RuntimeError(
            "Hugging Face token is missing. Add HUGGINGFACEHUB_API_TOKEN to your .env file or environment."
        )

    model_name = (os.getenv("HF_CHAT_MODEL") or "Qwen/Qwen2.5-Coder-32B-Instruct").strip()
    provider = (os.getenv("HF_INFERENCE_PROVIDER") or "auto").strip()
    if provider == "auto":
        provider = None

    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        provider=provider,
        huggingfacehub_api_token=token,
        temperature=0.65,
        max_new_tokens=1024,
        streaming=True,
    )

    model = ChatHuggingFace(llm=llm)
    return model