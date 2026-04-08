"""OpenAI-compatible client for Gemma-backed inference."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib import error, request

from gridgulp.config import GemmaConfig

_PROBE_IMAGE_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
)


class GemmaAPIError(RuntimeError):
    """Raised when Gemma inference or health checks fail."""


class GemmaClient:
    """Minimal async wrapper around an OpenAI-compatible chat completions API."""

    def __init__(self, config: GemmaConfig) -> None:
        self.config = config
        self.config.ensure_ready()

    async def list_models(self) -> list[str]:
        """Return model ids exposed by the configured endpoint."""
        response = await asyncio.to_thread(self._request_json, "GET", "/models", None)
        models = response.get("data", [])
        if not isinstance(models, list):
            raise GemmaAPIError("Gemma /models response did not contain a data list.")
        return [str(item.get("id")) for item in models if isinstance(item, dict) and item.get("id")]

    async def chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Submit a chat completion request and return response text."""
        payload = self._build_payload(
            system_prompt=system_prompt,
            user_content=user_prompt,
        )
        response = await asyncio.to_thread(
            self._request_with_retries,
            "POST",
            "/chat/completions",
            payload,
        )
        return self._extract_message_text(response)

    async def multimodal_chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data_uris: list[str],
    ) -> str:
        """Submit a multimodal completion request with image data URIs."""
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": image_data_uri}}
            for image_data_uri in image_data_uris
        )
        payload = self._build_payload(
            system_prompt=system_prompt,
            user_content=content,
        )
        response = await asyncio.to_thread(
            self._request_with_retries,
            "POST",
            "/chat/completions",
            payload,
        )
        return self._extract_message_text(response)

    async def probe_multimodal_support(self) -> str:
        """Perform a tiny multimodal probe to confirm image support."""
        return await self.multimodal_chat_completion(
            system_prompt="Return only valid JSON.",
            user_prompt='Return {"ok": true, "kind": "vision_probe"}.',
            image_data_uris=[_PROBE_IMAGE_DATA_URI],
        )

    def _build_payload(
        self,
        *,
        system_prompt: str,
        user_content: str | list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "model": self.config.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "temperature": 0.1,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "reasoning_effort": "none",
        }

    def _request_with_retries(
        self, method: str, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for _attempt in range(self.config.max_retries + 1):
            try:
                return self._request_json(method, path, payload)
            except GemmaAPIError as exc:
                last_error = exc
        raise GemmaAPIError("Gemma request failed after retries.") from last_error

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key is not None:
            headers["Authorization"] = f"Bearer {self.config.api_key.get_secret_value()}"

        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            raise GemmaAPIError(f"Gemma endpoint request failed: {exc}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise GemmaAPIError("Gemma endpoint returned invalid JSON.") from exc

        if not isinstance(parsed, dict):
            raise GemmaAPIError("Gemma endpoint returned a non-object JSON payload.")
        return parsed

    def _extract_message_text(self, response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise GemmaAPIError("Gemma response did not contain choices.")

        first = choices[0]
        if not isinstance(first, dict):
            raise GemmaAPIError("Gemma response choice had an unexpected shape.")

        message = first.get("message")
        if not isinstance(message, dict):
            raise GemmaAPIError("Gemma response choice did not contain a message.")

        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts)

        raise GemmaAPIError("Gemma response did not contain text content.")
