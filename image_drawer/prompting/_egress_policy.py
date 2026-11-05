"""外部のホストへ画像を送る設定の検査。内部IPへの接続と、宣言のない外部接続を設定の段階で止める。"""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Any, Mapping
from urllib.parse import urlsplit

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_INTERNAL_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain")


# {
#   責務: [
#     is_local_host: 画像を自分のPCの外へ出さないホスト(localhost・127.0.0.1・::1)か判定する
#   ]
# }
def is_local_host(host: str) -> bool:
    return host.casefold().strip("[]") in LOCAL_HOSTS


# {
#   責務: [
#     _literal_address: ホスト名がIPアドレスの書き方なら、そのアドレスを返す
#   ]
#   処理: [
#     通常の書き方に加え、10進整数・16進・省略形(inet_aton互換)も数値として扱う(名前解決はしない)
#   ]
#   戻り値: [
#     address: IPv4Address / IPv6Address。IPアドレスでなければNone
#   ]
# }
def _literal_address(host: str) -> Any:
    text = host.strip("[]")
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        pass
    try:
        return ipaddress.IPv4Address(socket.inet_aton(text))
    except (OSError, ValueError):
        return None


# {
#   責務: [
#     is_internal_host: 内部ネットワーク宛て(SSRFの標的になるアドレス・名前)か判定する
#   ]
#   処理: [
#     プライベート・ループバック・リンクローカル・予約・マルチキャスト・未指定のIPアドレス、
#     IPv4射影IPv6、.local/.internal等の名前を内部とする。localhost・127.0.0.1・::1は
#     ローカル(is_local_host)として別に扱うため、ここでは内部としない
#   ]
#   戻り値: [
#     internal: 内部ならTrue
#   ]
# }
def is_internal_host(host: str) -> bool:
    if is_local_host(host):
        return False
    name = host.casefold().rstrip(".")
    if name.endswith(_INTERNAL_SUFFIXES):
        return True
    address = _literal_address(name)
    if address is None:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    address = mapped or address
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


# {
#   責務: [
#     check_provider_endpoint: プロバイダのendpointを検査する(各プロバイダの生成時に呼ぶ)
#   ]
#   処理: [
#     1: ローカル以外のホストへのhttpを拒否する
#     2: 内部IP・内部名宛てを拒否する
#     3: data_egressはlocal/externalのみ。allowed_hostsを持つ場合、ホストがその中に無ければ拒否する
#   ]
#   引数: [
#     endpoint: URL, data_egress: 宣言, allowed_hosts: 許可するホスト名
#   ]
#   エラー: [
#     ValueError: 条件を満たさない
#   ]
# }
def check_provider_endpoint(
    endpoint: str, data_egress: str, allowed_hosts: tuple[str, ...]
) -> None:
    parsed = urlsplit(endpoint)
    host = (parsed.hostname or "").casefold()
    if data_egress not in {"local", "external"}:
        raise ValueError('data_egressは"local"か"external"を指定してください')
    if is_internal_host(host):
        raise ValueError("内部ネットワーク宛てのendpointは使用できません")
    if allowed_hosts and host not in {h.casefold() for h in allowed_hosts}:
        raise ValueError("endpointのホストがallowed_hostsにありません")


# {
#   責務: [
#     validate_provider_entry: 設定JSONのプロバイダ1件について、外部へ送る場合の宣言と認証を起動時に検査する
#   ]
#   処理: [
#     ローカル以外のホスト、またはdata_egress="external"の宣言があるものを外部とする。外部は
#     1: data_egress="external"の宣言と、ホストを含む空でないallowed_hostsが必要
#     2: api_key_envで指定した環境変数が設定済み・空でないことが必要(匿名では送らない)。
#        "auth": "none"はローカルのホストだけに許す
#   ]
#   引数: [
#     entry: プロバイダ設定, environ: 環境変数(テスト用に差し替え可能)
#   ]
#   エラー: [
#     ValueError: 宣言・許可ホスト・認証の不足。鍵の値は文面に出さない
#   ]
# }
def validate_provider_entry(
    entry: Mapping[str, Any], environ: Mapping[str, str] | None = None
) -> None:
    endpoint = entry.get("endpoint")
    if not isinstance(endpoint, str):
        return
    host = (urlsplit(endpoint).hostname or "").casefold()
    declared = entry.get("data_egress", "local")
    if declared not in {"local", "external"}:
        raise ValueError('data_egressは"local"か"external"を指定してください')
    if declared == "local" and is_local_host(host):
        return
    if declared != "external":
        raise ValueError(
            f'外部ホスト{host}へ送るには、"data_egress": "external" と allowed_hosts の指定が必要です'
        )
    allowed = entry.get("allowed_hosts")
    if (
        not isinstance(allowed, list)
        or not allowed
        or not all(isinstance(item, str) and item.strip() for item in allowed)
        or host not in {item.casefold() for item in allowed}
    ):
        raise ValueError(f"allowed_hostsに送信先のホスト{host}を含めてください")
    if entry.get("auth") == "none":
        if not is_local_host(host):
            raise ValueError('"auth": "none"はlocalhostのプロバイダだけに指定できます')
        return
    env_name = entry.get("api_key_env")
    if not isinstance(env_name, str) or not env_name:
        raise ValueError("外部へ送るプロバイダにはapi_key_envの指定が必要です")
    if not (environ if environ is not None else os.environ).get(env_name):
        raise ValueError(f"環境変数{env_name}が未設定または空です(匿名では送信しません)")
