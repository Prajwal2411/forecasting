import json
import os
from typing import Any, Dict, List, Optional


class LLMClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    def available(self) -> bool:
        return bool(self.api_key)

    def _client(self):
        try:
            import openai  # type: ignore
        except Exception as e:
            raise RuntimeError("openai package not installed") from e
        openai.api_key = self.api_key
        return openai

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        if not self.available():
            raise RuntimeError("OPENAI_API_KEY not set")
        try:
            openai = self._client()
            resp = openai.ChatCompletion.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = resp["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            raise RuntimeError(f"LLM chat_json failed: {e}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.available():
            raise RuntimeError("OPENAI_API_KEY not set")
        try:
            openai = self._client()
            resp = openai.Embedding.create(model=self.embed_model, input=texts)
            return [d["embedding"] for d in resp["data"]]
        except Exception as e:
            raise RuntimeError(f"LLM embed failed: {e}")

