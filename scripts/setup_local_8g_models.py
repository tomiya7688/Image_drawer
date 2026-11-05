"""8GB VRAM向けGGUFと画像用mmprojを個別に取得・検証する。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEFAULT_MODELS_DIR = Path.home() / ".cache" / "image-drawer" / "local8g"


# {
#   責務: [DownloadFile: 取得元とローカル配置名を固定する]
#   フィールド: [repo_id: Hugging Face上のID, filename: 配布ファイル名, dest_name: 配置名]
#   処理: [モデルの重みと画像エンコーダを別々の配布元から取得できるようにする]
# }
@dataclass(frozen=True)
class DownloadFile:
    repo_id: str
    filename: str
    dest_name: str


# {
#   責務: [ModelFiles: 1モデルに必要な重みと画像投影器をまとめる]
#   フィールド: [model_id: llama.cppルーターが認識するサブフォルダ名, files: 重みとmmproj]
#   処理: [モデルとmmprojの同一ディレクトリ配置を保証する]
# }
@dataclass(frozen=True)
class ModelFiles:
    model_id: str
    files: tuple[DownloadFile, DownloadFile]


MODEL_FILES = (
    ModelFiles(
        "qwen3vl4b-q4",
        (
            DownloadFile(
                "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
                "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
            ),
            DownloadFile(
                "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
                "mmproj-Q8_0.gguf",
            ),
        ),
    ),
    ModelFiles(
        "joycaption-beta-one-q3",
        (
            DownloadFile(
                "mradermacher/llama-joycaption-beta-one-hf-llava-GGUF",
                "llama-joycaption-beta-one-hf-llava.Q3_K_M.gguf",
                "llama-joycaption-beta-one-hf-llava.Q3_K_M.gguf",
            ),
            DownloadFile(
                "concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf",
                "llama-joycaption-beta-one-llava-mmproj-model-f16.gguf",
                "mmproj-joycaption-f16.gguf",
            ),
        ),
    ),
    ModelFiles(
        "qwen3vl4b-uncensored-q4",
        (
            DownloadFile(
                "mradermacher/Qwen3-VL-4B-Instruct-Uncensored-GGUF",
                "Qwen3-VL-4B-Instruct-Uncensored.Q4_K_M.gguf",
                "Qwen3-VL-4B-Instruct-Uncensored.Q4_K_M.gguf",
            ),
            DownloadFile(
                "mradermacher/Qwen3-VL-4B-Instruct-Uncensored-GGUF",
                "Qwen3-VL-4B-Instruct-Uncensored.mmproj-Q8_0.gguf",
                "mmproj-Q8_0.gguf",
            ),
        ),
    ),
)

FetchModel = Callable[..., str]


# {
#   責務: [missing_model_files: 画像用の重み・投影器が揃っているかを調べる]
#   処理: [サイズ0も未取得扱いとして、空のモデル配置を成功とみなさない]
#   引数: [models_dir: ローカルのモデルルート]
#   戻り値: [list[Path]: 存在しないか空の必須ファイル]
# }
def missing_model_files(models_dir: Path) -> list[Path]:
    missing: list[Path] = []
    for model in MODEL_FILES:
        for component in model.files:
            dest = models_dir / model.model_id / component.dest_name
            if not dest.is_file() or dest.stat().st_size == 0:
                missing.append(dest)
    return missing


# {
#   責務: [download_model_files: 1台に必要な3モデルのみを取得する]
#   処理: [保存済みのモデルは上書きせず、mmprojをllama.cppが認識する名前にする]
#   引数: [models_dir: 出力先, fetch: huggingface_hub.hf_hub_download互換関数]
#   戻り値: [list[Path]: 取得したファイル]
# }
def download_model_files(models_dir: Path, fetch: FetchModel) -> list[Path]:
    downloaded: list[Path] = []
    for model in MODEL_FILES:
        folder = models_dir / model.model_id
        folder.mkdir(parents=True, exist_ok=True)
        for component in model.files:
            target = folder / component.dest_name
            if target.is_file() and target.stat().st_size > 0:
                continue
            source = Path(
                fetch(
                    repo_id=component.repo_id,
                    filename=component.filename,
                    local_dir=folder,
                )
            )
            if not source.is_file() or source.stat().st_size == 0:
                raise RuntimeError("ダウンロードが空です: " + component.filename)
            if source != target:
                source.replace(target)
            downloaded.append(target)
    return downloaded


# {
#   責務: [main: ダウンロード不要のdry-run・検証・実際の取得を切り替える]
#   処理: [取得前に全ファイルを表示し、モデル重みをGit管理下へ書き込まない]
#   引数: [argv: CLI引数]
#   戻り値: [int: 0成功、1不備、2依存関係不備]
# }
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="8GB向けGGUFとmmprojをモデル別に取得")
    parser.add_argument(
        "--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="モデルの保存先"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="必要なファイルを表示する")
    mode.add_argument("--check", action="store_true", help="保存済みのファイルを確認する")
    args = parser.parse_args(argv)
    root = args.models_dir.expanduser().resolve()
    if args.dry_run:
        for model in MODEL_FILES:
            for component in model.files:
                print(f"{model.model_id}: {component.repo_id}/{component.filename}")
        return 0
    if not args.check:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print('huggingface_hubが必要です: python -m pip install "huggingface_hub>=0.34,<2"')
            return 2
        download_model_files(root, hf_hub_download)
    missing = missing_model_files(root)
    if missing:
        for name in missing:
            print(f"不足: {name}")
        return 1
    print("画像の重みとmmprojを確認しました。以下でルーターを起動してください。")
    print(
        f'llama-server --models-dir "{root}" --models-max 1 '
        "--models-memory-margin 1024 -c 4096 -np 1 "
        "--host 127.0.0.1 --port 8080"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
