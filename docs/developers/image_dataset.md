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
| `cli.py` | `image-drawer-dataset`（`register`、`extractors`、`show`） |

**未実装**: 学習済みモデルを使うタグ・説明文の抽出器。基本の抽出器は画素の統計だけで、画像の内容を理解しません。

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
- 権利が未確認の画像があるsplitは、既定で`ValueError`にします（`require_rights_confirmed=False`で外せます）。

型が分かれていることは、Policyが目標画像を見ないことの**必要条件**であって、十分条件ではありません。Policy側のコードが`RewardTarget`を受け取らないことは、S3のTrainerのテストで確認します。

## 権利の状態

`--confirm-rights`（`rights_status="confirmed"`）は、**その登録で走査した画像にだけ**適用します。以前の登録分の状態は変わりません。確認済みの状態を、再登録で未確認へ戻すことはありません。

## 制限

- `dataset_hash`と`manifest.json`の検証は、誤編集や破損を検出する**完全性の検査**です。悪意のある改ざんの検出には、署名や外部に保管した期待値が必要です。
- 抽出器の実装を更新したのに`version`・`model_hash`・`config_hash`を変えないと、古いキャッシュが使われます。実装を変えたら識別子も更新してください。
- キャッシュが壊れている、または形式が正しくない場合は、使わずに再抽出します。抽出器が不正な形式を返した場合は、`extractor_invalid_output`の品質警告にします。
- 近重複の判定は、わずかな明るさの変更や再保存を想定したものです。大きなクロップや回転は検出しません。
- 新しい画像を追加すると、近重複のグループが統合され、既存画像の`group_id`と`split`が変わることがあります。固定のtestを何度も学習調整に使わないため、追加後は`show`でsplitの件数を確認してください。
- `image_dataset_hash`の再現性は、同じ入力・同じ抽出器・同じPillowで確認しています。Pillowの縮小処理が変わると署名が変わる可能性があります。
