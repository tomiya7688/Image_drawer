"""OpenAI-compatibleの画像入力APIを標準ライブラリだけで呼び出す。"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request

from image_drawer.prompting._net import urlopen

from .core import PromptExtractionError, VisionResponse


@dataclass(frozen=True)
class OpenAICompatibleVisionProvider:
    """ローカルQwen・高精度クラウドVLMに共通の接続方式。

    モデル名・接続先・認証はアプリの設定。学習用の追加ファイルではない。
    """

    provider_id: str
    model_id: str
    endpoint: str
    api_key: str = ""
    timeout_seconds: int = 120
    json_mode: bool = False
    model_revision: str = ""

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("接続先はhttp(s)のURLを指定してください")
        if parsed.scheme == "http" and parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("localhost以外のVLM接続はHTTPSを使用してください")
        if not parsed.path.endswith("/chat/completions"):
            raise ValueError("OpenAI互換の/chat/completionsを指定してください")
        if not self.provider_id.strip() or not self.model_id.strip():
            raise ValueError("モデルIDと接続先識別子を指定してください")
        if not 1 <= self.timeout_seconds <= 3600:
            raise ValueError("タイムアウトは1〜3600秒です")

    @property
    def config_fingerprint(self) -> str:
        """認証秘密を含めず、結果に影響するモデル設定をキャッシュ識別する。"""
        settings = {
            "endpoint": self.endpoint,
            "model": self.model_id,
            "revision": self.model_revision,
            "json_mode": self.json_mode,
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        body = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(body).hexdigest()

    def describe(self, image: bytes, mime_type: str, instruction: str) -> VisionResponse:
        if not image:
            raise PromptExtractionError("invalid_input", "画像が空です")
        url = "data:" + mime_type + ";base64," + base64.b64encode(image).decode("ascii")
        payload: dict = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read(2_000_001)
        except HTTPError as exc:
            code, message = "", ""
            try:
                error = json.loads(exc.read(8192)).get("error", {})
                if isinstance(error, dict):
                    code = str(error.get("code", "")).casefold()
                    message = str(error.get("message", "")).casefold()
            except (TypeError, ValueError):
                pass
            if code in {
                "content_filter",
                "content_policy_violation",
                "safety_refusal",
            } or (
                exc.code in {400, 403}
                and ("content policy" in message or "safety filter" in message)
            ):
                return VisionResponse(refusal=True, finish_reason="content_filter")
            # 画像そのものが不正な場合(400/413/415/422)だけ打ち切る。認証・URL・権限の不備は、後続のモデルを試す。
            kind = "invalid_input" if exc.code in {400, 413, 415, 422} else "unavailable"
            raise PromptExtractionError(kind, f"VLM HTTP {exc.code}") from exc
        except (URLError, OSError, TimeoutError) as exc:
            raise PromptExtractionError("unavailable", "VLMの接続に失敗") from exc
        if len(raw) > 2_000_000:
            raise PromptExtractionError("invalid_format", "VLMの応答が長すぎます")
        try:
            body = json.loads(raw)
            choice = body["choices"][0]
            message = choice["message"]
            content = message.get("content", "")
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            raise PromptExtractionError("invalid_format", "VLMレスポンスが不正") from exc
        if isinstance(content, list):
            content = "\n".join(item.get("text", "") for item in content if isinstance(item, dict))
        if not isinstance(content, str):
            raise PromptExtractionError("invalid_format", "VLM文章が不正")
        return VisionResponse(
            content=content,
            refusal=bool(message.get("refusal")),
            finish_reason=str(choice.get("finish_reason") or ""),
        )
