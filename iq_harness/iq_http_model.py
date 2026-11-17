"""First-party IQ chat inference backend for the agent harness.

Uses the existing IQ HTTP chat-completion endpoint, not another AI provider.
Current IQ inference supports text-only synchronous generation. Tool calling,
tool history and streaming are deliberately refused until the IQ server
actually implements and validates those features.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .types import ModelRequest, ModelTurn


class IQHTTPModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class IQHTTPModelBackend:
    base_url: str
    token: str | None = None
    max_tokens: int = 16
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("IQ endpoint must not embed credentials in the URL")
        if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("IQ endpoint must be a base URL, without a path or query")
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("IQ endpoint must be an http(s) URL")
        if parsed.scheme == "http" and parsed.hostname not in (
            "localhost", "127.0.0.1", "::1"
        ):
            raise ValueError("Plain HTTP IQ endpoints are restricted to localhost")
        if type(self.max_tokens) is not int or not (1 <= self.max_tokens <= 64):
            raise ValueError("IQ model max_tokens must be in 1..64")
        if not isinstance(self.timeout_seconds, (int, float)) or not (
            0 < self.timeout_seconds <= 600
        ):
            raise ValueError("IQ model timeout must be in (0,600] seconds")
        if self.token is not None and (
            not isinstance(self.token, str) or not self.token.strip()
        ):
            raise ValueError("IQ authorization token must be a nonempty string")

    def generate(self, request: ModelRequest) -> ModelTurn:
        if request.tools:
            raise IQHTTPModelError(
                "IQ inference endpoint does not yet implement native tool calling; "
                "do not silently ignore requested agent tools"
            )
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        for item in request.messages:
            if item.role not in ("system", "user", "assistant"):
                raise IQHTTPModelError(
                    f"IQ inference endpoint cannot encode role {item.role!r}"
                )
            if not isinstance(item.content, str) or not item.content:
                raise IQHTTPModelError("IQ message content must be nonempty text")
            messages.append({"role": item.role, "content": item.content})
        if not messages:
            raise IQHTTPModelError("IQ model request has no messages")
        if len(messages) > 24:
            raise IQHTTPModelError("IQ server accepts at most 24 messages per request")

        payload = {
            "model": "IQ",
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.token is not None:
            headers["Authorization"] = f"Bearer {self.token}"
        req = Request(
            self.base_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout_seconds) as response:
                result = json.load(response)
        except HTTPError as exc:
            # Do not leak bearer tokens, prompt text or arbitrary server
            # errors from inference hosts into user-visible responses.
            raise IQHTTPModelError(
                f"IQ inference endpoint returned HTTP {exc.code}"
            ) from exc
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            raise IQHTTPModelError("IQ inference endpoint unavailable or malformed") from exc

        if not isinstance(result, dict):
            raise IQHTTPModelError("IQ inference response is not a JSON object")
        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise IQHTTPModelError("IQ inference returned invalid choices")
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise IQHTTPModelError("IQ inference returned invalid message")
        message = choice["message"]
        if message.get("tool_calls"):
            raise IQHTTPModelError("IQ server returned unsupported tool calls")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise IQHTTPModelError("IQ returned empty/non-text generation")
        return ModelTurn(content=content)
