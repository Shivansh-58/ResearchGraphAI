from __future__ import annotations

import httpx


class ModelUnavailableError(RuntimeError):
    """A local model is missing, unreachable, or returned invalid output."""


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, path: str, payload: dict | None = None) -> dict:
        try:
            with httpx.Client(
                timeout=self.timeout, trust_env=False, follow_redirects=False
            ) as client:
                response = (
                    client.get(self.base_url + path)
                    if payload is None
                    else client.post(self.base_url + path, json=payload)
                )
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError("Expected an object.")
                return result
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                "Local model unavailable. Check the model service and installed weights."
            ) from exc

    def chat(
        self, model: str, messages: list[dict], schema: dict | None = None, output_tokens: int = 768
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": "0",
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": output_tokens},
        }
        if schema:
            payload["format"] = schema
        result = self.request("/api/chat", payload)
        message = result.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ModelUnavailableError("Local model returned an empty response.")
        return text

    def installed_models(self) -> set[str]:
        return {item["name"] for item in self.request("/api/tags").get("models", [])}
