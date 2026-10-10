"""ASGI request-body size guard active before multipart parsing/spooling."""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable


class RequestBodyTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    """Limit actual streamed request bytes, including chunked bodies without Content-Length."""

    def __init__(self, app: Any, *, max_body_bytes: int) -> None:
        if isinstance(max_body_bytes, bool) or not isinstance(max_body_bytes, int) or max_body_bytes < 1:
            raise ValueError("max_body_bytes must be a positive integer")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: dict[str, Any], receive: Callable[[], Awaitable[dict[str, Any]]], send: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        received = 0
        response_started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if response_started:
                raise
            body = json.dumps({"detail": "Request body exceeds the configured aggregate upload limit."}).encode("utf-8")
            await send({"type": "http.response.start", "status": 413, "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
            ]})
            await send({"type": "http.response.body", "body": body})
