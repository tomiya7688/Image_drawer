> **対象読者:** Datasetと抽出器を実装・拡張する開発者
>
> **目的:** 画像のみのDatasetの実装範囲、データの流れ、抽出器の契約、target隔離を説明する。
>
> **状態:** S2（Issue #69）の実装。設計の正本は[画像のみで描いて学ぶ仕様](image_only_learning_v1.md)と[データ品質](drawing_data_quality_v1.md)。

# 画像のみのDataset（実装）

## 実装した範囲

`image_drawer/dataset/` が次を提供します。

| モジュール | 内容 |
| --- | --- |
| `normalize.py` | サイズ検査、アニメーション拒否、EXIF回転、透明部分の白背景合成、近重複の署名とグループ化 |
| `registry.py` | `TagExtractor`・`CaptionExtractor`のProtocolと`ExtractorRegistry` |
| `baseline_extractors.py` | モデル不要の基本の抽出器（主な色・向き・明るさ） |
| `ingest.py` | 登録、`cache_key`、キャッシュ、品質警告、グループ単位の分割 |
| `manifest.py` | `DatasetManifest`の保存・読込・ハッシュ検証、`iter_samples` |
| `models.py` | `TrainingImage`、`ExtractorInfo`、`PolicyInput`、`RewardTarget` |
| `cli.py` | `image-drawer-dataset`（`register`、`extractors`、`show`。`register`は`--extractor-settings`、`--allow-external-upload`、`--dry-run`） |
| `extraction.py` | 抽出器の呼び出しと結果のキャッシュ。`extractor_failed`を含む結果は保存しない |
| `prompting_adapters.py` | `image_drawer.prompting`のPipelineを抽出器として包む（`PromptingCaptionExtractor`・`PromptingTagExtractor`・`register_prompting_extractors`） |
| `png_encode.py` | 外部へ送る正規化後の画像を、メタデータなしPNG（8MiB以下）にする |
| `egress.py` / `upload_plan.py` | 外部送信の許可（Datasetの`dataset_config.json`・実行時の許可）、`egress.jsonl`の追記、`--dry-run`の集計 |

学習済みモデルを使う抽出器は、`image_drawer.prompting`をアダプタ経由で登録して使います（[統合設計](design_data_pipeline_integration.md)）。基本の抽出器は画素の統計だけで、画像の内容を理解しません。

### 任意の拡張（後方互換）

抽出器は次のメソッドを**持っていれば**使われます。持たない抽出器は従来どおりです。

- `extract_with_trace(image)`: `(結果, trace)`を返す。`trace`は`TrainingImage.extraction_trace["caption" / "tags"]`に保存されます（選ばれたモデル、`attempts`、`verifier_status`、`egress`＝送信の有無とホスト名）。
- `extract_variants(image)`: `{"short", "standard", "detailed"}`を返す。`TrainingImage.caption_variants`に保存され、PromptBuilderは**v2**（タグを文へ追記しない）になります。v2のcache_keyはv1と別です。
- `iter_samples(dataset_dir, split, *, prompt_variant="standard", variant_seed=None)`: `variant_seed`を指定すると、`SHA256("{seed}:{image_id}")`から短文40%・標準文40%・詳細文20%を決定的に選びます。
- タグの`in_prompt`が`false`のもの（rating・作品名・画風・meta）は、v1のPromptBuilderでも文に使いません。
- 新しい項目が空のときは`dataset_hash`の計算に含めないため、古いmanifestはそのまま読めます。

## 抽出器の契約

```python
class TagExtractor(Protocol):
    info: ExtractorInfo        # capabilitiesに "tag"
    def extract(self, image) -> list[dict]: ...   # {"label": str, "confidence": float}

class CaptionExtractor(Protocol):
    info: ExtractorInfo        # capabilitiesに "caption"
    def extract(self, image) -> str: ...
```

`image`は正規化済みのRGB `PIL.Image`です。抽出器は`ExtractorRegistry.register_tagger`／`register_captioner`で登録します。モデル名をコードに埋め込まず、CLI・設定・GUIは同じregistryを参照します。

`cache_key = SHA256(画像のSHA-256, 正規化version, PromptBuilderのversion, 両抽出器の id・version・model_hash・config_hash)`。抽出器や設定が変わると鍵が変わり、古いキャッシュは使われません。抽出が失敗した場合は`QualityIssue`として記録し、画像は除外しません。

## 近重複とsplit

SHA-256が同じ画像は1件だけ登録します。構造（9x8の差分ハッシュ）と色（4x4のRGB）の署名が両方近い画像は、同じ`group_id`にまとめます。**差分ハッシュだけでは、単色や滑らかなグラデーションの画像がすべて同じハッシュになる**ため、色の署名を併用しています（テスト`test_flat_color_images_with_the_same_structure_are_not_grouped`）。

近重複の連結は**推移的**です。A–BとB–Cが近ければ、A–Cが離れていても同じグループにします（train・testへの漏洩を避けるため、グループを広く取る方針です）。

splitは`SHA256(seed:group_id)`から決定的に割り当て、train 8割・validation 1割・test 1割を目安にします。同じグループは必ず同じsplitになります。ただし、画像が数件のように少ない場合は、割合どおりには分かれません。

## Policy入力と目標の分離

`iter_samples(dataset_dir, split)`は、`(PolicyInput, RewardTarget)`の組を返します。

- `PolicyInput`は`image_id`と`prompt`だけを持ちます。画像のパス、ハッシュ、画素を持ちません。
- `RewardTarget`は目標画像のパスとハッシュを持ち、**Trainerの評価器だけ**が読みます。

型が分かれていることは、Policyが目標画像を見ないことの**必要条件**であって、十分条件ではありません。Policy側のコードが`RewardTarget`を受け取らないことは、S3のTrainerのテストで確認します。

## 制限

- `dataset_hash`と`manifest.json`の検証は、誤編集や破損を検出する**完全性の検査**です。悪意のある改ざんの検出には、署名や外部に保管した期待値が必要です。
- 抽出器の実装を更新したのに`version`・`model_hash`・`config_hash`を変えないと、古いキャッシュが使われます。実装を変えたら識別子も更新してください。
- `TagExtractor`は無効にできます（id `none`、何も出力せず警告も出しません）。`CaptionExtractor`は無効にできません。
- 学習用promptが空の画像は、`quality_issues`に`empty_prompt`を記録し、`iter_samples`の対象から外します（[入力契約](training_input_v1.json)の`missing_condition`）。外部テキストの入力は要求しません。
- キャッシュが壊れている、または形式が正しくない場合は、使わずに再抽出します。抽出器が不正な形式を返した場合は、`extractor_invalid_output`の品質警告にします。
- 近重複の判定は、わずかな明るさの変更や再保存を想定したものです。大きなクロップや回転は検出しません。
- 新しい画像を追加すると、近重複のグループが統合され、既存画像の`group_id`と`split`が変わることがあります。固定のtestを何度も学習調整に使わないため、追加後は`show`でsplitの件数を確認してください。
- `image_dataset_hash`の再現性は、同じ入力・同じ抽出器・同じPillowで確認しています。Pillowの縮小処理が変わると署名が変わる可能性があります。
