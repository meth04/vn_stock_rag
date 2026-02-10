import numpy as np
from openai import AsyncOpenAI

from .config import OPENAI_API_KEY, OPENAI_BASE_URL


async def openai_complete_if_cache(
    prompt,
    system_prompt=None,
    history_messages=None,
    **kwargs,
) -> str:
    """Wrapper chuẩn cho LightRAG.

    Đối số đầu tiên BẮT BUỘC phải là prompt.
    """
    model = kwargs.get("model", "deepseek-v3")
    client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history_messages:
        messages.extend(history_messages)

    messages.append({"role": "user", "content": prompt})

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 4000),
            timeout=300,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ LLM Error (Model {model}): {exc}")
        return ""


async def text_embedding_func(texts: list[str]) -> np.ndarray:
    """Wrapper cho Embedding Local."""
    from .local_embedding import get_embedding_model

    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings)
