
# llm/ollama_chat.py
import time
from typing import Generator, List, Optional, Tuple, Union
import ollama

def get_local_models() -> List[str]:
    try:
        result = ollama.list()

        result = ollama.list()
        names = sorted([m.model for m in result.get("models", [])])
        return sorted(set(names))
    except Exception as e:
        print(f"Could not list models from Ollama: {e}")
        return []

def build_messages_from_history(
    history: Union[List[Tuple[str, str]], List[dict]],
    user_message: str,
    system_prompt: str = "",
) -> List[dict]:
    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        first = history[0]
        if isinstance(first, dict) and "role" in first and "content" in first:
            for msg in history:
                role = msg.get("role")
                content = msg.get("content")[0]['text']
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        else:
            for (user, assistant) in history:
                if user:
                    messages.append({"role": "user", "content": user})
                if assistant:
                    messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": user_message})
    return messages

def stream_ollama_chat(
    messages: List[dict],
    model: str,
    temperature: float,
    seed: Optional[int],
) -> Generator[str, None, None]:
    try:
        options = {"temperature": float(temperature)}
        if seed is not None:
            options["seed"] = int(seed)

        stream = ollama.chat(model=model, messages=messages, options=options, stream=True)
        response_text = ""
        for chunk in stream:
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                response_text += delta
                yield response_text
        time.sleep(0.02)
    except Exception as e:
        yield f"[Error] {e}"
