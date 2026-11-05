> **対象読者：** 画像だけを登録してCaptionとタグを生成したい利用者
> **目的：** 画像説明モデル・Tagger・分類・整形を自由に登録する
> **状態：** 前処理レジストリと単一画像CLIを実装。実GPU/Trainer連携・GUI設定は未確認

# 画像学習の前処理設定

学習素材として指定するのは**画像だけ**です。Caption（自然言語の画像説明）と
Tagger（カテゴリ別のタグ抽出）は同じ画像から**独立して両方実行**します。
タグ、TXT、補助説明、手描きの制作履歴を追加で用意する必要はありません。

## 標準は8GB VRAM・32GB RAM向け

- **通常設定：** [preprocessor_settings.example.json](preprocessor_settings.example.json)
- **Windowsでのセットアップ・起動：** [8GB環境の手順](preprocessing_local8g.md)
- **高設定（大型8Bモデル向け）：** [preprocessor_settings.high.example.json](preprocessor_settings.high.example.json)

通常設定は次の順番です。

| 処理 | モデル |
| --- | --- |
| Caption 第1段 | Qwen3-VL-4B-Instruct **Q4_K_M**（構造化Caption） |
| Caption 第2段 | JoyCaption Beta One **Q3_K_M**（低拒否率の代替） |
| Caption 第3段 | Felldude Qwen3-VL-4B Uncensored **Q4_K_M**（追加の低拒否率候補） |
| Tagger 第1段 | Qwen3-VL-4B Q4_K_Mで**Captionとは別に画像を推論** |
| Tagger 第2段 | Felldude 4B Q4_K_Mで再度タグ専用推論 |

5つの処理を指定していますが、**読み込むモデル重みは3種類だけ**です。
単一のllama.cppルーターが必要なモデルだけを読み込み、`--fit on --gpu-layers auto`でGPU/CPU配置を自動調整し、同時ロードは1モデルに
制限します。別の大型Taggerを常時VRAMに載せる必要はありません。
VRAM 8GB/32GB RAM実機での性能と対応率の実測は未実施です。

以前の`Qwen3-VL-8B → 非量子化JoyCaption Beta One → Felldude 8B V2`
は**高設定用に分離**しました。通常設定では選択しません。

## Dataset全体の3パス前処理

登録済みDatasetでは**①タグを全画像から抽出→②キャプションを全画像から生成→③両者を統合してプロンプト生成**の順番で処理します。各パスの成果物を別々にキャッシュし、成功済み処理を再開時に再利用します。

```powershell
image-drawer-preprocess --settings docs/users/preprocessor_settings.example.json --dataset E:\datasets\my-drawings --seed 11
```

タグが失敗した画像もCaptionの実行を試みます。どちらか失敗すれば、その画像にはダミープロンプトを作らず、理由を保存します。タグの重複・明白な矛盾を調整する既定のPromptComposerは暫定ルールであり、自由登録したFormatterに差し替えできます。

## TextEncoderの選択・CPUフォールバック

通常設定では`paraphrase-multilingual-MiniLM-L12-v2`の384次元埋め込みを使用し、まずGPU、その後は**同じモデルをCPUで実行**します。異なる埋め込みモデルへの切替は学習済みcheckpointとの互換性が保証されないため、自動で混ぜません。

```powershell
python -m pip install -e ".[text-encoder]"
image-drawer-text-encode --settings docs/users/preprocessor_settings.example.json --prompt "赤いリンゴが1つ"
```

このCLIはTextEncoder単体の検査用で、現時点では描画PolicyのTrainerへは接続されていません。

## 画像1枚から結果を出す

[8GB環境のセットアップ](preprocessing_local8g.md)後に、リポジトリのルートで実行します。

```powershell
image-drawer-preprocess --settings docs/users/preprocessor_settings.example.json --image E:\images\sample.png
```

PNG/JPEG/WebPを自動判定します。JSONに短文・標準文・詳細文、Captionの採用モデルと
各モデルの拒否理由、独立Taggerのモデル・カテゴリ・confidence・整形結果を表示します。

## ユーザー定義のモデルと前処理

設定JSONで次の処理を登録・変更できます。

- `providers`：Caption・Tagger・Verifierを自由に登録します。通常の例は同じ
  `http://127.0.0.1:8080/v1/chat/completions`へ接続します。
- `caption_chain`と`tag_chain`：それぞれの優先順位です。
  **Qwen固定でも3段固定でもありません**。失敗したときは次に進みます。
- `adapter`：構造化Captionの`openai_compatible`、自然文の
  `joycaption`/`prose_caption`、タグ専用の`vision_tagger`、
  別サービスの`tagger_http`を使用できます。
- `tag_normalizer`：confidence閾値、カテゴリの上書き、独自カテゴリ。
- `tag_formatter`：重複排除とタグの出力設定。既定では
  rating/copyright/style/metaを学習用の肯定タグへ暗黙に連結しません。

ComfyUI Support Tools互換のTagger HTTPも引き続き登録できます。
タグの元スコアと分類元を保存し、モデルが間違える可能性も考慮します。
学習に使う文章とTaggerの結果が矛盾しても無条件で連結しません。

Pythonからは`PreprocessorRegistry`の`register_caption()`、
`register_tagger()`、`register_verifier()`、`register_normalizer()`、
`register_tag_formatter()`、`register_prompt_formatter()`、
`register_adapter()`で独自処理を登録できます。
JSONから任意のPythonコードをimport・実行しません。

## 限界

- どのモデルもNSFWを含むあらゆる画像を拒否しないとは保証できません。
  全モデル失敗なら、ダミー教師Captionを作らず失敗を記録します。
- 元画像の内容を理解したことと、人間の自然なプロンプトに忠実に描けることは
  別の検証項目です。
- 実モデルによるGPU推論とSFW/成人NSFWごとの拒否率・精度ベンチマーク、
  Dataset・TrainerとのEnd-to-End連携、GUI設定画面は未完成です。
- モデルの重みはImage_drawer本体に含めません。モデルごとの
  ライセンスと配布条件は各配布元で確認してください。

開発者向け正本：[画像Caption・Tagger抽出とN段ルーティング](../developers/prompt_extraction_v1.md)。
