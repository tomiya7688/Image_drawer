"""ComfyUI Support Tools系・PixAI系と互換なTagger HTTP Adapter。"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request

from image_drawer.prompting._net import urlopen

from .core import PromptExtractionError


@dataclass(frozen=True)
class HTTPTagProvider:
    """ユーザーが登録したTagger Service/PixAI互換HTTPエンドポイント。"""

    provider_id: str
    model_id: str
    endpoint: str
    threshold: float = 0.35
    character_threshold: float = 0.85
    timeout_seconds: int = 120
    backend: str = "tagger_service_http"

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Tagger URLはhttp(s)を指定する")
        if parsed.scheme == "http" and parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("localhost以外のTaggerはHTTPSが必要")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Tagger URLに認証情報・クエリを入れない")
        if self.backend not in {"tagger_service_http", "pixai_http", "custom_http"}:
            raise ValueError("未対応のTagger backend")
        if not self.provider_id or not self.model_id:
            raise ValueError("TaggerのID・モデルを指定する")
        for value in (self.threshold, self.character_threshold):
            if type(value) not in {int, float} or not 0 <= value <= 1:
                raise ValueError("Tagger閾値は0〜1")
        if not 1 <= self.timeout_seconds <= 600:
            raise ValueError("Taggerのtimeoutは1〜600秒")

    @property
    def config_fingerprint(self) -> str:
        payload = {
            "endpoint": self.endpoint,
            "model_id": self.model_id,
            "threshold": self.threshold,
            "character_threshold": self.character_threshold,
            "backend": self.backend,
        }
        text = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(text).hexdigest()

    def tag(self, image: bytes, mime_type: str) -> object:
        if not image or mime_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise PromptExtractionError("invalid_input", "Taggerへ送る画像が不正")
        payload: dict = {
            "image": base64.b64encode(image).decode("ascii"),
            "model": self.model_id,
            "threshold": self.threshold,
        }
        if self.backend == "pixai_http":
            payload["character_threshold"] = self.character_threshold
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read(1024 * 1024 + 1)
        except HTTPError as exc:
            code = exc.code
            body = exc.read(4096)
            if code in {400, 403} and b"content_filter" in body:
                raise PromptExtractionError("refused", "Taggerが説明を拒否") from exc
            # 画像そのものが不正な場合(400/413/415/422)だけ打ち切る。認証・URL・権限の不備は、後続のモデルを試す。
            kind = "invalid_input" if code in {400, 413, 415, 422} else "unavailable"
            raise PromptExtractionError(kind, f"Tagger HTTP {code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise PromptExtractionError("unavailable", "Taggerが応答しない") from exc
        if len(raw) > 1024 * 1024:
            raise PromptExtractionError("invalid_format", "Tagger JSONが上限1MiBを超えた")
        try:
            return json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise PromptExtractionError("invalid_format", "Tagger JSONが不正") from exc
