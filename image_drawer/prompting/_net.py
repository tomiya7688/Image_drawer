"""外部へHTTPリクエストを送る共通の入口。リダイレクトを追従しない。"""

from __future__ import annotations

from typing import Any
from urllib.request import HTTPRedirectHandler, build_opener


# {
#   責務: [
#     _NoRedirect: HTTPのリダイレクトを追従せず、そのままHTTPErrorとして扱う
#   ]
#   処理: [
#     画像やAPIキーを、設定で許可していない別ホストへ再送しないため、3xxを追従しない
#   ]
# }
class _NoRedirect(HTTPRedirectHandler):
    # {
    #   責務: [
    #     redirect_request: リダイレクト先のリクエストを作らない
    #   ]
    #   戻り値: [
    #     request: 常にNone。呼び出し元はHTTPErrorを受け取る
    #   ]
    # }
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = build_opener(_NoRedirect)


# {
#   責務: [
#     urlopen: リダイレクトを追従せずにリクエストを開く
#   ]
#   引数: [
#     request: urllib.request.Request
#     timeout: 秒
#   ]
#   戻り値: [
#     response: HTTPレスポンス(コンテキストマネージャー)
#   ]
#   エラー: [
#     HTTPError: 4xx・5xx、および追従しない3xx
#   ]
# }
def urlopen(request: Any, timeout: float) -> Any:
    return _OPENER.open(request, timeout=timeout)
