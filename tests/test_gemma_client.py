"""Integration-style tests for the Gemma HTTP client."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from gridgulp.ai import GemmaClient
from gridgulp.config import GemmaConfig


class _MockGemmaHandler(BaseHTTPRequestHandler):
    seen_payload: dict[str, object] | None = None

    def do_GET(self) -> None:  # noqa: N802
        if self.path.endswith("/models"):
            self._write_json({"data": [{"id": "gemma-4-local"}]})
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return

        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length).decode("utf-8")
        type(self).seen_payload = json.loads(body)
        self._write_json(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"analysis_summary":{"sheet_complexity":"simple","reasoning_notes":"ok","confidence":0.9},"decisions":[]}'
                        }
                    }
                ]
            }
        )

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _write_json(self, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def test_gemma_client_uses_configured_model_id() -> None:
    """The Gemma client should use the exact configured model id."""
    server = HTTPServer(("127.0.0.1", 0), _MockGemmaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        client = GemmaClient(
            GemmaConfig(
                enabled=True,
                base_url=base_url,
                model_id="gemma-4-local",
            )
        )

        models = asyncio.run(client.list_models())
        content = asyncio.run(client.chat_completion("system", "user"))

        assert models == ["gemma-4-local"]
        assert "analysis_summary" in content
        assert _MockGemmaHandler.seen_payload is not None
        assert _MockGemmaHandler.seen_payload["model"] == "gemma-4-local"
        assert _MockGemmaHandler.seen_payload["stream"] is False
        assert _MockGemmaHandler.seen_payload["response_format"] == {"type": "json_object"}
        assert _MockGemmaHandler.seen_payload["reasoning_effort"] == "none"
    finally:
        server.shutdown()
        server.server_close()


def test_gemma_client_multimodal_payload_uses_image_url_blocks() -> None:
    """The multimodal client path should emit OpenAI-compatible image_url content."""
    server = HTTPServer(("127.0.0.1", 0), _MockGemmaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        client = GemmaClient(
            GemmaConfig(
                enabled=True,
                base_url=base_url,
                model_id="gemma-4-local",
            )
        )

        content = asyncio.run(
            client.multimodal_chat_completion(
                "system",
                "user",
                ["data:image/png;base64,AAAA"],
            )
        )

        assert "analysis_summary" in content
        assert _MockGemmaHandler.seen_payload is not None
        messages = _MockGemmaHandler.seen_payload["messages"]
        assert isinstance(messages, list)
        user_message = messages[1]
        content_blocks = user_message["content"]
        assert content_blocks[0] == {"type": "text", "text": "user"}
        assert content_blocks[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        }
    finally:
        server.shutdown()
        server.server_close()
