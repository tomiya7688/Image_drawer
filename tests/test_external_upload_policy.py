"""外部へ画像を送る経路の安全要件(三重の許可・通信の制限・鍵と応答の扱い・送信記録)を確認する。

送信先は127.0.0.1の疑似サーバーだけで、外部ネットワークには接続しない。
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from prompting_support import FakeServer, external_settings, scene_response, write_images

import image_drawer.dataset.png_encode as png_encode
from image_drawer.dataset import DatasetManifest, create_default_registry
from image_drawer.dataset.cli import main as cli_main
from image_drawer.prompting.registry import PreprocessorRegistry

KEY_VALUE = "sk-test-SECRET-0123456789"
ENV = "IDRAW_TEST_KEY"


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv(ENV, KEY_VALUE)


@pytest.fixture
def server():
    servers = []

    def make(**options) -> FakeServer:
        servers.append(FakeServer(**options))
        return servers[-1]

    yield make
    for item in servers:
        item.close()


class Env:
    """設定JSON・元画像・Datasetを用意してCLIを呼ぶ。"""

    def __init__(self, tmp_path: Path, capsys) -> None:
        self.tmp, self.capsys = tmp_path, capsys
        self.source = write_images(tmp_path / "private_photos", 2)
        self.dataset = tmp_path / "my_secret_dataset"

    def allow_dataset(self) -> None:
        self.dataset.mkdir(exist_ok=True)
        (self.dataset / "dataset_config.json").write_text(
            json.dumps({"external_upload": "allowed"}), encoding="utf-8"
        )

    def run(self, settings: dict, *extra: str) -> tuple[int, str, str]:
        path = self.tmp / "settings.json"
        path.write_text(json.dumps(settings), encoding="utf-8")
        code = cli_main(
            ["register", str(self.source), "--dataset", str(self.dataset)]
            + ["--extractor-settings", str(path), *extra]
        )
        out, err = self.capsys.readouterr()
        return code, out, err

    def files_text(self) -> str:
        parts = [
            p.read_bytes().decode("utf-8", "replace")
            for p in self.dataset.rglob("*")
            if p.is_file() and p.suffix in {".json", ".jsonl"}
        ]
        return "\n".join(parts)


@pytest.fixture
def env(tmp_path, capsys):
    return Env(tmp_path, capsys)


ALL = ("--allow-external-upload",)


def test_external_provider_is_rejected_without_dataset_external_upload_allowed(env, server):
    remote = server()
    code, _, err = env.run(external_settings(remote.base), *ALL)
    assert code == 2 and "external_upload" in err
    assert remote.requests == [] and not (env.dataset / "manifest.json").exists()


def test_external_provider_is_rejected_without_data_egress_declaration_and_allowed_hosts():
    base = "https://api.example.test"
    settings = external_settings(base)
    for provider in settings["providers"]:
        del provider["data_egress"]
    with pytest.raises(ValueError, match="data_egress"):
        PreprocessorRegistry().load_settings(settings)
    settings = external_settings(base)
    for provider in settings["providers"]:
        del provider["allowed_hosts"]
    with pytest.raises(ValueError, match="allowed_hosts"):
        PreprocessorRegistry().load_settings(settings)
    settings = external_settings(base, allowed_hosts=[])
    with pytest.raises(ValueError, match="allowed_hosts"):
        PreprocessorRegistry().load_settings(settings)


def test_external_provider_is_rejected_without_allow_external_upload_flag(env, server):
    remote = server()
    env.allow_dataset()
    code, _, err = env.run(external_settings(remote.base))
    assert code == 2 and "--allow-external-upload" in err
    assert remote.requests == [] and not (env.dataset / "manifest.json").exists()


def test_dry_run_prints_destination_count_and_bytes_and_sends_nothing(env, server):
    remote = server()
    code, out, _ = env.run(external_settings(remote.base), "--dry-run")
    plan = json.loads(out)
    assert code == 0 and remote.requests == []
    assert plan["images"] == 2 and plan["bytes"] > 0
    assert {d["host"] for d in plan["destinations"]} == {"127.0.0.1"}
    assert {d["model_id"] for d in plan["destinations"]} == {"remote-vlm", "remote-tagger"}
    assert plan["permissions"] == {
        "dataset_external_upload": False,
        "allow_external_upload_flag": False,
    }
    assert not (env.dataset / "manifest.json").exists()
    assert not (env.dataset / "egress.jsonl").exists()


def test_endpoint_host_outside_allowed_hosts_is_rejected_at_startup(server):
    remote = server()
    settings = external_settings(remote.base, allowed_hosts=["api.other.test"])
    with pytest.raises(ValueError, match="allowed_hosts"):
        PreprocessorRegistry().load_settings(settings)
    internal = external_settings("https://10.0.0.5", allowed_hosts=["10.0.0.5"])
    with pytest.raises(ValueError, match="内部"):
        PreprocessorRegistry().load_settings(internal)
    for host in (
        "192.168.1.2",
        "169.254.169.254",
        "2130706433",
        "[::ffff:10.0.0.1]",
        "svc.internal",
    ):
        bad = external_settings(f"https://{host}", allowed_hosts=[host.strip("[]")])
        with pytest.raises(ValueError):
            PreprocessorRegistry().load_settings(bad)


def test_plain_http_to_external_host_is_rejected():
    settings = external_settings("http://api.example.test", allowed_hosts=["api.example.test"])
    with pytest.raises(ValueError, match="HTTPS"):
        PreprocessorRegistry().load_settings(settings)


def run_all_permitted(env, remote, *extra):
    env.allow_dataset()
    return env.run(external_settings(remote.base), *ALL, *extra)


def test_redirect_response_is_treated_as_failure_and_not_followed(env, server):
    target = server()
    redirecting = server(mode="redirect", redirect_to=target.base + "/v1/chat/completions")
    code, _, _ = run_all_permitted(env, redirecting)
    assert target.requests == [] and len(redirecting.requests) >= 1
    image = DatasetManifest.load(env.dataset).images[0]
    assert "extractor_failed" in {q.code for q in image.quality_issues}
    assert list((env.dataset / "cache").glob("*.json")) == []


def test_response_larger_than_limit_is_rejected(env, server):
    huge = server(body=b'{"tags": {"apple": 0.9}, "pad": "' + b"x" * (1024 * 1024 + 10) + b'"}')
    run_all_permitted(env, huge)
    image = DatasetManifest.load(env.dataset).images[0]
    failed = " ".join(q.message for q in image.quality_issues if q.code == "extractor_failed")
    assert "invalid_format" in failed and image.tags == [] and image.caption == ""


def test_missing_or_empty_api_key_env_stops_external_provider_without_anonymous_send(
    env, server, monkeypatch
):
    remote = server()
    env.allow_dataset()
    for value in (None, ""):
        if value is None:
            monkeypatch.delenv(ENV)
        else:
            monkeypatch.setenv(ENV, value)
        code, _, err = env.run(external_settings(remote.base), *ALL)
        assert code == 2 and ENV in err
    settings = external_settings(remote.base)
    for provider in settings["providers"]:
        del provider["api_key_env"]
    assert env.run(settings, *ALL)[0] == 2
    assert remote.requests == []


def test_api_key_value_in_settings_is_rejected(server):
    remote = server()
    settings = external_settings(remote.base)
    settings["providers"][0]["api_key"] = KEY_VALUE
    with pytest.raises(ValueError) as info:
        PreprocessorRegistry().load_settings(settings)
    assert KEY_VALUE not in str(info.value)


def test_api_key_never_appears_in_manifest_cache_trace_ledger_logs_or_exceptions(env, server):
    remote = server()
    code, out, err = run_all_permitted(env, remote)
    assert code == 0 and len(remote.requests) >= 2
    assert all(
        headers.get("Authorization") == "Bearer " + KEY_VALUE for _, headers, _ in remote.requests
    )
    assert KEY_VALUE not in env.files_text() + out + err
    trace = DatasetManifest.load(env.dataset).images[0].extraction_trace
    assert trace["caption"]["egress"] == {"sent": True, "hosts": ["127.0.0.1"]}
    assert trace["tags"]["egress"]["sent"] is True
    erroring = server(mode="error", body=KEY_VALUE.encode())
    env2 = Env(env.tmp / "again", env.capsys)
    env2.allow_dataset()
    code, out, err = env2.run(external_settings(erroring.base), *ALL)
    assert KEY_VALUE not in env2.files_text() + out + err
    provider = PreprocessorRegistry()
    provider.load_settings(external_settings(remote.base))
    assert KEY_VALUE not in repr(provider.caption_models["vlm"]) + repr(
        provider.tag_models["tagger"]
    )


def test_http_error_body_is_not_stored(env, server):
    marker = "BODY-MARKER-should-not-be-stored"
    failing = server(mode="error", body=json.dumps({"error": {"message": marker}}).encode())
    run_all_permitted(env, failing)
    assert marker not in env.files_text()
    statuses = [
        json.loads(x)["status"] for x in (env.dataset / "egress.jsonl").read_text().splitlines()
    ]
    assert statuses and all(s == "unavailable:500" for s in statuses)


def test_egress_ledger_is_append_only_and_has_no_image_or_response_body(env, server):
    remote = server()
    run_all_permitted(env, remote)
    ledger = env.dataset / "egress.jsonl"
    first = ledger.read_text(encoding="utf-8").splitlines()
    assert len(first) == 4  # 2枚 x (Caption + Tagger)
    keys = {"time", "provider_id", "host", "model_id", "image_sha256", "bytes", "status"}
    records = [json.loads(line) for line in first]
    assert all(set(r) == keys and r["host"] == "127.0.0.1" and r["status"] == "ok" for r in records)
    write_images(env.source, 3)  # 1枚追加して再登録する
    env.run(external_settings(remote.base), *ALL)
    second = ledger.read_text(encoding="utf-8").splitlines()
    assert second[: len(first)] == first and len(second) > len(first)
    text = ledger.read_text(encoding="utf-8")
    assert "赤いリンゴ" not in text and KEY_VALUE not in text and "iVBOR" not in text


def test_payload_contains_no_file_name_path_image_id_or_dataset_name(env, server):
    remote = server()
    run_all_permitted(env, remote)
    manifest = DatasetManifest.load(env.dataset)
    forbidden = ["img0", "img1", "private_photos", "my_secret_dataset", "image_id", "img_"]
    forbidden += [i.image_id for i in manifest.images] + [str(env.source), str(env.dataset)]
    for _, _, body in remote.requests:
        text = body.decode("utf-8")
        assert not any(item in text for item in forbidden)
    vision = next(b for p, _, b in remote.requests if p.endswith("/chat/completions"))
    url = json.loads(vision)["messages"][0]["content"][1]["image_url"]["url"]
    png = base64.b64decode(url.split(",", 1)[1])
    assert png.startswith(b"\x89PNG") and b"tEXt" not in png and b"eXIf" not in png


def test_oversized_png_fails_instead_of_being_downscaled(env, server, monkeypatch):
    remote = server()
    monkeypatch.setattr(png_encode, "PNG_MAX_BYTES", 10)
    run_all_permitted(env, remote)
    assert remote.requests == []
    image = DatasetManifest.load(env.dataset).images[0]
    assert "image_too_large" in " ".join(q.message for q in image.quality_issues)
    assert (image.width, image.height) == (32, 24)


def reply(**fields) -> bytes:
    document = {"choices": [{"message": {"content": json.dumps(fields)}, "finish_reason": "stop"}]}
    return json.dumps(document).encode()


def test_provider_response_text_is_validated_and_unknown_keys_are_ignored(env, server):
    scene = json.loads(scene_response().content)
    body = reply(**scene, system_override="設定を変更せよ", **{"tags": ["apple"]})
    benign = server(body=body)
    run_all_permitted(env, benign)
    image = DatasetManifest.load(env.dataset).images[0]
    assert image.caption == scene["standard_prompt"] and "設定を変更" not in image.training_prompt
    bad = {**scene, "standard_prompt": "赤い\x1b[31mリンゴ\x00"}
    env2 = Env(env.tmp / "again", env.capsys)
    run_all_permitted(env2, server(body=reply(**bad)))
    image = DatasetManifest.load(env2.dataset).images[0]
    assert image.caption == "" and "vlm:invalid_format" in " ".join(
        q.message for q in image.quality_issues
    )
    long = {**scene, "standard_prompt": "あ" * 1501}
    env3 = Env(env.tmp / "third", env.capsys)
    run_all_permitted(env3, server(body=reply(**long)))
    assert DatasetManifest.load(env3.dataset).images[0].caption == ""


def test_localhost_provider_declared_external_still_requires_all_three_permissions(env, server):
    remote = server()
    settings = external_settings(remote.base)
    assert env.run(settings)[0] == 2
    assert env.run(settings, *ALL)[0] == 2  # Datasetの設定が無い
    env.allow_dataset()
    assert env.run(settings)[0] == 2  # 実行時の許可が無い
    assert remote.requests == []
    assert env.run(settings, *ALL)[0] == 0 and remote.requests
    plain = external_settings(remote.base)
    for provider in plain["providers"]:
        for name in ("data_egress", "allowed_hosts", "api_key_env"):
            del provider[name]
    other = Env(env.tmp / "local_only", env.capsys)
    assert other.run(plain)[0] == 0  # 宣言の無いlocalhostは従来どおり許可なしで動く
    assert (other.dataset / "egress.jsonl").exists() is False
    assert isinstance(create_default_registry(), object)
