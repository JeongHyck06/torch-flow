"""인증 (기획서 §9 보안, Jupyter Server 모델).

불변식 하나: 모든 라우트가 인증을 지난다. 예외 목록을 두지 않고
미들웨어가 전부를 덮는다. marimo CVE-2026-39987, Gradio CVE-2024-47165가
모두 "인증 없는 라우트 하나"에서 시작했다. ``tests/test_hub.py``가 라우트
목록을 훑어 이 불변식을 강제한다.
"""

from __future__ import annotations

import hmac
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

COOKIE_NAME = "torchflow_token"
DEFAULT_HOSTS = ("127.0.0.1", "localhost", "[::1]")


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_matches(expected: str, given: str | None) -> bool:
    return bool(given) and hmac.compare_digest(expected, given)


def extract_token(request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("token "):
        return header[6:].strip()
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.query_params.get("token") or request.cookies.get(COOKIE_NAME)


class AuthMiddleware(BaseHTTPMiddleware):
    """토큰 · Host 허용목록 · Origin ``null`` 거부를 한자리에서 처리한다."""

    def __init__(self, app, *, token: str, allowed_hosts: tuple[str, ...] = DEFAULT_HOSTS):
        super().__init__(app)
        self.token = token
        self.allowed_hosts = tuple(allowed_hosts)

    def _host_allowed(self, host_header: str) -> bool:
        host = host_header.rsplit(":", 1)[0] if not host_header.startswith("[") else host_header.split("]")[0] + "]"
        return host in self.allowed_hosts

    async def dispatch(self, request, call_next):
        host = request.headers.get("host", "")
        if host and not self._host_allowed(host):
            return JSONResponse({"error": "host not allowed", "host": host}, status_code=403)

        # Origin: null은 sandboxed iframe·file:// 에서 온다. 거부한다(§9).
        origin = request.headers.get("origin")
        if origin == "null":
            return JSONResponse({"error": "origin null rejected"}, status_code=403)

        if not token_matches(self.token, extract_token(request)):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        response = await call_next(request)
        # 쿠키가 없을 때만 심으면, hub를 재시작해 토큰이 바뀐 뒤에도 브라우저가 낡은
        # 쿠키를 계속 보낸다 - 첫 페이지는 URL 토큰으로 통과하고 에셋은 전부 401이
        # 되어 화면이 통째로 빈다. 값이 다르면 항상 갱신한다.
        if request.cookies.get(COOKIE_NAME) != self.token:
            response.set_cookie(
                COOKIE_NAME, self.token, httponly=True, samesite="strict", path="/"
            )
        return response
