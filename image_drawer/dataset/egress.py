"""画像を外部のホストへ送る経路の許可・記録。既定では何も送らない。

許可は三つすべてが必要: Datasetの設定(external_upload)、プロバイダの宣言(data_egress・allowed_hosts)、
実行時の許可(--allow-external-upload)。プロバイダの宣言は設定の読込時にprompting側で検査する。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from image_drawer.prompting._egress_policy import is_local_host

DATASET_CONFIG_NAME = "dataset_config.json"
LEDGER_NAME = "egress.jsonl"


# {
#   責務: [
#     ExternalUploadDenied: 外部への画像送信の許可が足りないことを示す
#   ]
# }
class ExternalUploadDenied(ValueError):
    pass


# {
#   責務: [
#     read_dataset_external_upload: Datasetの設定ファイルで外部送信が明示的に許可されているか読む
#   ]
#   処理: [
#     <dataset>/dataset_config.json の {"external_upload": "allowed"} だけを許可とする。
#     ファイルが無い・壊れている・他の値なら禁止("forbidden")。manifestとdataset_hashには含めない
#   ]
#   戻り値: [
#     allowed: 許可されていればTrue
#   ]
# }
def read_dataset_external_upload(dataset_dir: Path) -> bool:
    try:
        data = json.loads((dataset_dir / DATASET_CONFIG_NAME).read_text(encoding="utf-8"))
        return isinstance(data, dict) and data.get("external_upload") == "allowed"
    except (OSError, ValueError):
        return False


# {
#   責務: [
#     EgressLedger: 外部への送信を追記専用のegress.jsonlへ1行ずつ記録する
#   ]
#   処理: [
#     日時・provider_id・ホスト名・model_id・image_sha256・送信バイト数・状態を書く。
#     画像本体、応答本文、APIキーは受け取らず、記録しない。既存の行は書き換えない
#   ]
# }
class EgressLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    # {
    #   責務: [append: 1件の送信を追記する]
    #   引数: [
    #     provider_id, host, model_id: 送信先, image: 送ったbytes(ハッシュと大きさだけを記録)
    #     status: "ok" または失敗の種類
    #   ]
    # }
    def append(self, provider_id: str, host: str, model_id: str, image: bytes, status: str) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provider_id": provider_id,
            "host": host,
            "model_id": model_id,
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "bytes": len(image),
            "status": status,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


# {
#   責務: [
#     EgressGate: 抽出器ごとの送信許可と台帳を保持し、送信の直前に確認する共有の入れ物
#   ]
#   フィールド: [
#     dataset_allows: Datasetの設定, runtime_allows: 実行時の許可, ledger: 記録先(未設定なら記録しない)
#   ]
# }
@dataclass
class EgressGate:
    dataset_allows: bool = False
    runtime_allows: bool = False
    ledger: EgressLedger | None = None
    hosts: list[str] = field(default_factory=list)  # 直近の抽出で実際に送った先(記録用)

    # {
    #   責務: [missing: 足りない許可の説明を返す]
    #   戻り値: [reasons: 足りない許可。揃っていれば空]
    # }
    def missing(self) -> list[str]:
        reasons = []
        if not self.dataset_allows:
            reasons.append('Datasetの設定(dataset_config.jsonの"external_upload": "allowed")')
        if not self.runtime_allows:
            reasons.append("実行時の許可(--allow-external-upload)")
        return reasons

    # {
    #   責務: [check: 許可が揃っていなければ、送信せずにExternalUploadDeniedにする]
    # }
    def check(self, destinations: list[str]) -> None:
        reasons = self.missing()
        if reasons:
            raise ExternalUploadDenied(
                f"外部ホスト{sorted(set(destinations))}へ画像を送る許可が足りません: "
                + "、".join(reasons)
            )


def host_of(endpoint: str) -> str:
    return (urlsplit(str(endpoint)).hostname or "").casefold()


# {
#   責務: [
#     RecordingProvider: 外部へ送るプロバイダを包み、送信前の許可確認と送信の記録を行う
#   ]
#   処理: [
#     describe/tagを呼ぶ直前に許可を確認し、許可がなければ内側のプロバイダを呼ばない。
#     呼んだ後は結果の成否にかかわらず台帳に追記する。その他の属性は内側のプロバイダへ委ねる
#     (config_fingerprintなどが変わらないようにする)
#   ]
# }
class RecordingProvider:
    def __init__(self, inner: Any, gate: EgressGate) -> None:
        self._inner = inner
        self._gate = gate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _send(self, image: bytes, call: Any) -> Any:
        host = host_of(self._inner.endpoint)
        self._gate.check([host])
        status = "ok"
        try:
            return call()
        except BaseException as exc:
            status = str(getattr(exc, "kind", type(exc).__name__))
            code = re.search(
                r"HTTP (\d{3})", str(exc)
            )  # プロバイダが付ける固定の文面。本文は見ない
            if code:
                status += ":" + code.group(1)
            raise
        finally:
            self._gate.hosts.append(host)
            if self._gate.ledger is not None:
                inner = self._inner
                self._gate.ledger.append(inner.provider_id, host, inner.model_id, image, status)

    def describe(self, image: bytes, mime_type: str, instruction: str) -> Any:
        return self._send(image, lambda: self._inner.describe(image, mime_type, instruction))

    def tag(self, image: bytes, mime_type: str) -> Any:
        return self._send(image, lambda: self._inner.tag(image, mime_type))


# {
#   責務: [
#     is_external_provider: 画像が外部のホストへ送られうるプロバイダか判定する
#   ]
#   処理: [
#     data_egress="external"の宣言、またはendpointのホストがlocalhost・127.0.0.1・::1以外なら外部。
#     endpointを持たない(ネットワークを使わない)プロバイダは外部ではない
#   ]
#   戻り値: [
#     external: 外部ならTrue
#   ]
# }
def is_external_provider(model: Any) -> bool:
    if getattr(model, "data_egress", "local") == "external":
        return True
    host = host_of(getattr(model, "endpoint", ""))
    return bool(host) and not is_local_host(host)


# {
#   責務: [
#     EgressMixin: PromptingのCaption/Tag抽出器に、外部送信の許可確認・台帳記録・同一画像の結果再利用を足す
#   ]
#   処理: [
#     外部のプロバイダをRecordingProviderで包み、許可が揃わなければ送信せず、送れば台帳へ記録する。
#     失敗した結果は再利用しない
#   ]
# }
class EgressMixin:
    pipeline: Any

    def _setup_egress(self) -> None:
        self.gate = EgressGate()
        self._memo: tuple[str, Any, list[str]] | None = None
        self.pipeline.providers = tuple(self._wrap(m) for m in self.pipeline.providers)
        verifier = getattr(self.pipeline, "verifier", None)
        if verifier is not None:
            self.pipeline.verifier = self._wrap(verifier)

    def _wrap(self, model: Any) -> Any:
        if is_external_provider(model) and hasattr(model, "endpoint"):
            return RecordingProvider(model, self.gate)
        return model

    def _models(self) -> list[Any]:
        verifier = getattr(self.pipeline, "verifier", None)
        return list(self.pipeline.providers) + ([verifier] if verifier is not None else [])

    # {
    #   責務: [external_destinations: 画像が送られうる外部の送信先を返す]
    #   戻り値: [destinations: [{"provider_id", "host", "model_id"}]]
    # }
    def external_destinations(self) -> list[dict[str, str]]:
        return [
            {"provider_id": m.provider_id, "host": host_of(m.endpoint), "model_id": m.model_id}
            for m in self._models()
            if is_external_provider(m) and hasattr(m, "endpoint")
        ]

    # {
    #   責務: [bind_egress: 許可と台帳を結びつける(register_imagesが呼ぶ)]
    # }
    def bind_egress(
        self, *, dataset_allows: bool, runtime_allows: bool, ledger: EgressLedger | None
    ) -> None:
        self.gate.dataset_allows = dataset_allows
        self.gate.runtime_allows = runtime_allows
        self.gate.ledger = ledger

    # {
    #   責務: [ensure_upload_permitted: 外部へ送る抽出器なのに許可が足りなければ、開始前に止める]
    #   エラー: [ExternalUploadDenied: 足りない許可を文面に含める]
    # }
    def ensure_upload_permitted(self) -> None:
        hosts = [d["host"] for d in self.external_destinations()]
        if hosts:
            self.gate.check(hosts)

    def _run_cached(self, image: Any, call: Any) -> tuple[Any, list[str]]:
        from image_drawer.dataset.png_encode import encode_png

        png = encode_png(image)
        digest = hashlib.sha256(png).hexdigest()
        if self._memo is not None and self._memo[0] == digest:
            return self._memo[1], self._memo[2]
        self.gate.hosts.clear()
        result = call(png)
        hosts = sorted(set(self.gate.hosts))
        self._memo = (digest, result, hosts)
        return result, hosts


# {
#   責務: [
#     prepare_egress: 選んだ抽出器に許可と台帳を結びつけ、許可が足りなければ何も読まずに止める
#   ]
#   引数: [
#     extractors: 選んだTagger・Captioner, dataset_dir: Datasetの保存先
#     allow_external_upload: 実行時の許可(--allow-external-upload)
#   ]
#   エラー: [
#     ExternalUploadDenied: 外部へ送る抽出器に対して、Datasetの設定か実行時の許可が足りない
#   ]
# }
def prepare_egress(extractors: list[Any], dataset_dir: Path, allow_external_upload: bool) -> None:
    dataset_allows = read_dataset_external_upload(dataset_dir)
    ledger = EgressLedger(dataset_dir / LEDGER_NAME)
    for extractor in extractors:
        if hasattr(extractor, "bind_egress"):
            extractor.bind_egress(
                dataset_allows=dataset_allows,
                runtime_allows=allow_external_upload,
                ledger=ledger,
            )
            extractor.ensure_upload_permitted()
