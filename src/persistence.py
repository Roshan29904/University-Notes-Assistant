import json
import os
from typing import Any, Iterable


def _resolve_path(path: str | None, fallback_name: str) -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    if path:
        return os.path.abspath(path)
    return os.path.join(data_dir, fallback_name)


def load_chat_history(path: str | None = None) -> list[dict[str, Any]]:
    resolved = _resolve_path(path, "chat_history.json")
    if not os.path.exists(resolved):
        return []
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def save_chat_history(path: str | None, chats: Iterable[dict[str, Any]]) -> None:
    resolved = _resolve_path(path, "chat_history.json")
    directory = os.path.dirname(resolved)
    os.makedirs(directory, exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(list(chats), handle, ensure_ascii=False, indent=2)


def load_processed_files(path: str | None = None) -> list[str]:
    resolved = _resolve_path(path, "processed_files.json")
    if not os.path.exists(resolved):
        return []
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [str(item) for item in data]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def save_processed_files(path: str | None, files: Iterable[str]) -> None:
    resolved = _resolve_path(path, "processed_files.json")
    directory = os.path.dirname(resolved)
    os.makedirs(directory, exist_ok=True)
    ordered = list(dict.fromkeys(str(item) for item in files))
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(ordered, handle, ensure_ascii=False, indent=2)


def load_doc_texts(path: str | None = None) -> dict[str, str]:
    resolved = _resolve_path(path, "doc_texts.json")
    if not os.path.exists(resolved):
        return {}
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def save_doc_texts(path: str | None, doc_texts: dict[str, str]) -> None:
    resolved = _resolve_path(path, "doc_texts.json")
    directory = os.path.dirname(resolved)
    os.makedirs(directory, exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump({str(key): str(value) for key, value in doc_texts.items()}, handle, ensure_ascii=False, indent=2)
