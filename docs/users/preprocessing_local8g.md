> **対象読者：** VRAM 8GB・メインメモリ32GBのPCで画像学習前処理を動かす利用者
> **目的：** 追加の大容量GPUを買わず、画像1枚からCaptionとタグを独立生成する
> **状態：** GGUF・mmprojを組み合わせる実行手順。モックCI以外のGPU実機確認は未実施

# 通常運用：8GB VRAM / 32GB RAMの画像前処理

[標準設定JSON](preprocessor_settings.example.json) はこの環境を優先します。
8Bや全精度JoyCaptionを使う比較・研究用の構成は
[高設定JSON](preprocessor_settings.high.example.json)に分離しました。

## 採用した軽量モデル

| 役割 | 量子化重みの配布元とモデル | 重み単体の目安 |
| --- | --- | ---: |
| Caption初段・Tagger初段 | [Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) Q4_K_M | 約2.5GB |
| Captionの第2段 | [JoyCaption Beta One GGUF](https://huggingface.co/mradermacher/llama-joycaption-beta-one-hf-llava-GGUF) Q3_K_M | 約4.02GB |
| Caption第3段・Tagger第2段 | [Felldude 4B Uncensored GGUF](https://huggingface.co/mradermacher/Qwen3-VL-4B-Instruct-Uncensored-GGUF) Q4_K_M | 約2.5GB |

JoyCaptionの画像投影器（mmproj）は本体のGGUFリポジトリと**別の**
[concedoの配布先](https://huggingface.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf)から
取得します。画像が見えないテキスト専用実行にならないよう、各モデルのmmprojを必ず配置します。
重みに加えて画像投影器、コンテキスト、実行時バッファのメモリが必要です。JoyCaptionは
Q3_K_Mで約5GB前後と案内されますが、**Image_drawerから8GB GPUで実測済みではありません**。

## 1. 準備（Windows PowerShell例）

リポジトリのルート `E:\projects\Image_drawer` で実行してください。
Python 3.11以上、動作可能なllama.cpp（GPUの場合は対応するCUDA/Vulkan等のビルド）が必要です。

```powershell
winget install llama.cpp
python -m pip install -e .
python -m pip install "huggingface_hub>=0.34,<2"
```

GPUが見えない場合は、llama.cppのGPU対応ビルドを確認してください。CPUのみでも
32GB RAMでのオフロードは検討できますが、推論はかなり遅くなり得ます。

## 2. 必要な量子化とmmprojだけを取得する

3モデルすべての全精度重みを取得する必要はありません。
このコマンドは事前に指定した**各モデル1種類の量子化 + mmproj**だけを
PC内の `~/.cache/image-drawer/local8g` に保存します。Git管理の画像やコードは変更しません。

```powershell
python scripts/setup_local_8g_models.py --dry-run
python scripts/setup_local_8g_models.py
python scripts/setup_local_8g_models.py --check
```

`--dry-run`はネットワークを使いません。取得処理はHugging Faceへの通信が必要で、
利用条件に同意していることが前提です。ダウンロード先を変えたい場合は
`--models-dir E:\models\image-drawer-local8g` を3コマンドすべてに追加してください。

## 3. VRAMに1モデルしか載せないルーターを起動

手順2の最後に表示された `llama-server` コマンドを実行します。既定の例：

```powershell
llama-server --models-dir "$HOME/.cache/image-drawer/local8g" --models-max 1 --models-memory-margin 1024 -c 4096 -np 1 --host 127.0.0.1 --port 8080
```

`--models-max 1`で同時ロード1モデル、`-np 1`で同時推論1件、
`-c 4096`で小さなコンテキストを指定します。llama.cppのルーターは
リクエストの`model`名に合わせてサブフォルダのモデルを必要に応じてロード・アンロードします。
3つのVLMを同時常駐させないことが8GB運用の重要条件です。

```powershell
Invoke-RestMethod http://127.0.0.1:8080/models
```

返されたmodel IDに`qwen3vl4b-q4`、`joycaption-beta-one-q3`、
`qwen3vl4b-uncensored-q4`があるか確認してください。
サーバーのバージョンによりIDの命名が違う場合は、実際の`/models`の値に合わせて
[設定JSON](preprocessor_settings.example.json)の`model_id`を修正してください。

## 4. 画像を1枚処理する

```powershell
image-drawer-preprocess --settings docs/users/preprocessor_settings.example.json --image E:\images\sample.png
```

ソフトは**Captionを3段まで順番に試行**し、タグ抽出では同じ画像に対し
**別の推論リクエスト**を発行します。Captionの結果をTaggerとして流用しません。
タグはQwen3-VL-4Bから抽出し、失敗時にFelldude 4Bへ切り替えます。
モデルの重みを共有できるため、タグ専用の別GPUサービスを必要としません。
構造化JSONの破損・拒否があれば次へ進みます。最後まで成功しなければ失敗理由を残します。

## 8GBでメモリ不足になる場合

1. 既存のGPUを使う別アプリ（ComfyUIやゲームなど）を停止し、ルーターが1モデルだけ
   ロードしているか確認してください。最初のモデルの切替は読み込み時間が発生します。
2. 画像投影器やGPUバッファでOOMになる場合はルーターの設定を調整し、
   `-ngl 20`などで一部レイヤーをCPUへオフロードする（GPU計算を減らす）方法を試してください。
   必要に応じて`--no-mmproj-offload`を使い、画像投影器をCPU上で実行してください。
3. それでも不足する場合は、Qwen3-VL-2B Q4_K_M（[公式GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF)）への差し替えが候補です。
   **自動的に精度が同じになるわけではない**ため、標準の4Bと別の実験として測定してください。

VRAM 8GBの上限とメインメモリ32GBは**推論用VLMのモデル選定基準**です。
DrawingPolicy本体の学習収束や学習画像1,000枚を使った実行時間は未検証であり、
現状のCLIは1画像の前処理を試す機能です。
モデルや個人画像、ベンチマーク用実データをリポジトリへコミットしないでください。

参考：[llama.cppのモデルルーター仕様](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) /
[量子化JoyCaptionの運用例](https://github.com/1038lab/ComfyUI-JoyCaption)
