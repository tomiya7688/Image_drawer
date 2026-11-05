> **対象読者:** Datasetと抽出器（S2、PR #74）、プロンプト前処理（PR #78）を実装・レビューする開発者
>
> **目的:** 重複している二つの実装を、どちらも捨てずに最小の境界でつなぐ方針を決める。外部のモデルAPIへ画像を送る場合の安全要件も定める。
>
> **状態:** 設計書。統合用コード（`image_drawer/dataset/prompting_adapters.py`など）は Issue #83 でI-1〜I-4として実装済み。実装時の補足は9.1節に記す。調査時点（2026-11-05）では#74・#78は未マージだった。外部モデルの精度や、抽出文が画像に忠実であることは、この文書では示さない。

# 画像のみのDatasetとプロンプト前処理の統合設計

## 1. 結論

- **Dataset（S2）を、永続化の唯一の所有者にします。** 画像の正規化、近重複とsplit、manifest、キャッシュ、`iter_samples`、`image-drawer-dataset`は`image_drawer/dataset/`だけが持ちます。
- **プロンプト前処理（#78）は、抽出ロジックの提供者にします。** 複数のVLM・Taggerの段階的な切り替え、独立VLMによる検証、タグの分類と整形は`image_drawer/prompting/`が持ち続けます。Datasetの`ExtractorRegistry`へ**アダプタ**を通して登録します。
- 統合の境界は、`image_drawer/dataset/prompting_adapters.py`の2つのクラスと1つの登録関数です（4節）。どちらの既存の公開名も変更しません。
- 外部のAPIへ画像を送る経路は、**既定でオフ**とし、Dataset・設定・実行時の3か所で明示的に許可された場合だけ使います（6節）。現行の#78は、`localhost`以外のhttpsの任意のホストへ接続できるため、この要件に合わせて制限を追加します。

## 2. 二つの実装の比較

調査した対象は、S2が`E:/projects/worktrees/s2-dataset/Image_drawer`のブランチ、#78が`git fetch origin pull/78/head:pr78`で取得したコミットです。

### 2.1 範囲と契約

| 観点 | S2（`image_drawer/dataset/`、PR #74） | #78（`image_drawer/prompting/`） |
| --- | --- | --- |
| 担当 | 画像の登録、正規化、近重複、split、manifest、キャッシュ、`iter_samples` | 画像から文章とタグを得る処理（モデルの切り替え、検証、分類、整形） |
| 入力 | ディレクトリの画像。正規化済みのRGB `PIL.Image`を抽出器へ渡す | `bytes`と`mime_type`（PNG/JPEG/WebP、32MiB以内）。元ファイルをそのまま読む |
| 抽出器の契約 | `TagExtractor.extract(image) -> list[{"label", "confidence"}]`、`CaptionExtractor.extract(image) -> str`、識別は`ExtractorInfo(id, version, capabilities, model_hash, config_hash, license, languages, runtime)` | `VisionProvider.describe(image: bytes, mime_type, instruction) -> VisionResponse`、`TagProvider.tag(image: bytes, mime_type) -> object`。識別は`provider_id`、`model_id`、`config_fingerprint` |
| 選択の単位 | 利用者がTaggerとCaptionerを**1つずつ**選ぶ | 利用者が**順序つきの列**（`caption_chain`、`tag_chain`）を設定する |
| フォールバック | なし（抽出器が1つ） | 任意の段数。拒否・空・不正JSON・障害で次のモデルへ |
| 検証 | なし | 独立VLMによる支持／不一致の再照合（`verified_by_model`） |
| タグ | ラベルとconfidence | カテゴリ分類（general、character、copyright、rating、style、meta、服・姿勢・背景など）、正規化、整形 |
| 文章 | 1つの`caption`と、`build_prompt(caption, tags)`による`学習用prompt`（`caption（上位5タグ）`） | 短文・標準文・詳細文を構造化JSONから作り、`training_prompt(seed)`で40:40:20の確率で選ぶ |
| 登録簿 | `ExtractorRegistry.register_tagger(extractor)`、`register_captioner(extractor)`、`tagger(id)`、`captioner(id)`、`describe()` | `PreprocessorRegistry.register_caption(name, model)`、`register_tagger(name, model)`、`register_verifier`、`register_normalizer`、`register_tag_formatter`、`register_prompt_formatter`、`register_adapter`、`build()` |
| 失敗 | 抽出器の例外は`QualityIssue`にして**画像を残す**。空promptの画像は`iter_samples`から除外 | どちらかが失敗すると`PromptExtractionError("preprocessing_failed")`。ダミーのpromptは作らない |
| ネットワーク | なし | OpenAI互換のVision APIと、Tagger Service/PixAI互換HTTP |

### 2.2 キャッシュ、manifest、CLI、文書

| 観点 | S2 | #78 |
| --- | --- | --- |
| キャッシュの鍵 | `SHA256(画像SHA-256, 正規化version, PromptBuilder version, 両抽出器のid・version・model_hash・config_hash)` | `ImagePromptPipeline._cache_key`（画像SHA-256、MIME、profile、指示文のハッシュ、各モデルの`config_fingerprint`）。結果に`cache_key`を持つが、**ファイルには保存しない** |
| キャッシュの保存 | `<dataset>/…/<cache_key>.json`。壊れていれば再抽出。**失敗した結果も保存する** | なし |
| manifest | `DatasetManifest`と`TrainingImage`（抽出器の`ExtractorInfo`、`cache_key`、`quality_issues`、`split`） | なし。[training_input_v1.json](training_input_v1.json)の`internally_generated`に`prompt_variants`、`fallback_reason`、`selected_caption_model`などの名前を追加 |
| CLI | `image-drawer-dataset register`/`extractors`/`show`（終了コード 0/1/2） | `image-drawer-preprocess --settings --image`（1枚を処理して結果を標準出力へ。Datasetを更新しない） |
| 設定 | CLI引数で抽出器idを指定 | JSON設定（`idraw-preprocessor-registry/v1`）。`api_key_env`で環境変数名を指定 |
| 文書 | `image_dataset.md`、`docs/users/image_dataset.md` | `prompt_extraction_v1.md`/`.json`、`docs/users/preprocessing.md`、設定例 |
| 変更する既存ファイル | `docs/developers/README.md`、`drawing_data_quality_v1.md`、`image_only_learning_v1.md`、`docs/users/README.md`、`pyproject.toml` | `README.md`、`docs/developers/README.md`、`architecture.md`、`drawing_data_quality_v1.md`、`drawing_policy_accuracy_v1.md`、`drawing_process_acceptance.md`、`image_only_learning_v1.md`、`start_here.md`、`training_input_v1.json`、`docs/kadoka/implementation_plan.md`、`docs/users/README.md`、`pyproject.toml` |

### 2.3 重複と衝突

| 重複している機能 | 内容 |
| --- | --- |
| 抽出器の登録簿 | 2つの登録簿が、Tagger/Captionerの登録を別々に持つ |
| `register_tagger` | 同じ名前で、シグネチャが異なる（S2は`(extractor)`、#78は`(name, model)`） |
| `cache_key` | 2つの定義があり、保存するのはS2だけ |
| 学習用promptの組み立て | S2の`build_prompt`と、#78の`training_prompt(seed)`、`prompt_formatter`の3経路 |
| CLI | 抽出器を指定する入口が`image-drawer-dataset`と`image-drawer-preprocess`の2つ |
| 文書 | 利用者向けの説明が`image_dataset.md`と`preprocessing.md`に分かれる |

衝突になるもの: `pyproject.toml`の`[project.scripts]`（双方が行を追加）、`docs/developers/README.md`、`docs/users/README.md`、`drawing_data_quality_v1.md`、`image_only_learning_v1.md`。パッケージ名（`dataset`と`prompting`）とモジュール名は重ならないため、importの衝突はありません。

## 3. 役割分担

```text
利用者: 画像ディレクトリだけを選ぶ（抽出器の選択は「設定」）
  │
  ▼
image-drawer-dataset register / GUI（S2）
  │  正規化・近重複・split・キャッシュ・manifest      ← Dataset（永続化の所有者）
  │
  ├─ ExtractorRegistry ─ ColorLayout…（モデル不要、既定・オフライン）
  │                    └ prompting:caption:<profile> / prompting:tags:<profile>   ← アダプタ（本書）
  │                              │
  │                              ▼
  │                        image_drawer/prompting/（#78）
  │                          VLM・Taggerの段階的な切り替え、検証、タグの分類・整形
  │                          └─ OpenAI互換API / Tagger HTTP（6節の制限つき）
  ▼
iter_samples(dataset_dir, split) → (PolicyInput, RewardTarget)
```

- Datasetは、抽出器が内部でどのモデルを何段使うかを知りません。`ExtractorInfo`の`model_hash`と`config_hash`だけが変化を伝えます。
- `prompting`は、Datasetのmanifestやsplitを知りません。`bytes`と`mime_type`を受け取って結果を返します。
- GUI（[推論・GUI設計](design_inference_gui.md)の5.2(c)）は、Datasetの`ExtractorRegistry.describe()`だけを参照します。`PreprocessorRegistry`の`register_*`をGUIから呼びません。

## 4. 最小の統合境界

### 4.1 追加するもの（`image_drawer/dataset/prompting_adapters.py`）

```python
class PromptingCaptionExtractor:
    """S2のCaptionExtractorを満たす。#78のImagePromptPipelineを1つの抽出器として包む。"""
    info: ExtractorInfo                       # capabilities=["caption"]
    def __init__(self, pipeline: ImagePromptPipeline, *, extractor_id: str, settings_sha256: str) -> None: ...
    def extract(self, image: Any) -> str: ...                              # 標準文（scene.standard）
    def extract_variants(self, image: Any) -> dict[str, str]: ...          # {"short", "standard", "detailed"}（任意の拡張）
    def extract_with_trace(self, image: Any) -> tuple[str, dict[str, Any]]: ...   # 任意の拡張

class PromptingTagExtractor:
    """S2のTagExtractorを満たす。#78のTaggingPipelineを包む。"""
    info: ExtractorInfo                       # capabilities=["tag"]
    def __init__(self, pipeline: TaggingPipeline, *, extractor_id: str, settings_sha256: str,
                 include_categories: frozenset[str] = DEFAULT_PROMPT_CATEGORIES) -> None: ...
    def extract(self, image: Any) -> list[dict[str, Any]]: ...             # [{"label", "confidence", "category"}]
    def extract_with_trace(self, image: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...

def register_prompting_extractors(
    registry: ExtractorRegistry,
    preprocessors: PreprocessorRegistry,
    *,
    settings_sha256: str,
) -> list[str]: ...                           # 登録したextractor idの一覧を返す
```

- 登録されるidは、`prompting:caption:<profile_id>`と`prompting:tags:<profile_id>`です。Caption列とTagger列を、それぞれ**1つの抽出器**として扱います。フォールバックの段数は抽出器の内側に隠れます。
- `include_categories`の既定は`general`、`clothes`、`pose`、`expression`、`background`、`situation`です。`rating`、`copyright`、`character`、`style`、`meta`は、タグの`category`として保存しますが、学習用promptの文には使いません（#78の「ratingを本文へ黙って混ぜない」に従う）。

### 4.2 画像の渡し方

S2の抽出器は、正規化済みのRGB `PIL.Image`を受け取ります。アダプタは、これを**メタデータを含まない固定設定のPNG**（`bytes`、MIME `image/png`）に変換して#78へ渡します。

- 元のファイルの`bytes`ではなく正規化後を送ることで、EXIF、位置情報、アルファ（透明）の扱い、画素数の上限がDatasetの規則にそろいます。外部へ送る場合は、メタデータが含まれない点も安全要件になります（6節）。
- #78の`preprocess_file`（元ファイルを直接読む）は、Datasetの経路では使いません。
- 画像1枚につきPNGの符号化が1回増えます。キャッシュがある場合は抽出自体を行いません。

### 4.3 識別とキャッシュ

`ExtractorInfo`は次のとおり決めます。S2の`cache_key`が#78の`_cache_key`と同じ変化に反応するようにします。

| フィールド | 値 |
| --- | --- |
| `id` | `prompting:caption:<profile_id>`、`prompting:tags:<profile_id>` |
| `version` | `prompting-adapter-v1`と、#78の`profile_id`を連結した文字列 |
| `model_hash` | 列の全モデル（検証器を含む）の`config_fingerprint`を**順序つきで**並べたもののSHA-256 |
| `config_hash` | 指示文（`CAPTION_INSTRUCTION`、`VERIFY_INSTRUCTION`）のSHA-256、正規化器・整形器のid、しきい値、`include_categories`、`settings_sha256`のSHA-256 |
| `capabilities`、`languages`、`runtime` | 実際の値（外部APIを使う列は`runtime="remote"`） |

`config_fingerprint`は#78の`OpenAICompatibleVisionProvider`/`HTTPTagProvider`にあるものの、`VisionProvider`と`TagProvider`のProtocolには**含まれていません**。アダプタが必要とするため、Protocolへ`config_fingerprint: str`を追加します（変更はPR I-1）。順序が変わる、段が増える、モデルが変わる、のいずれでもS2の`cache_key`が変わります（`test_cache_key_changes_when_fallback_order_changes`）。

### 4.4 Datasetへの追加（後方互換）

S2の既存の契約は変更しません。次を**任意の追加**として入れます（PR I-2）。

| 追加 | 内容 |
| --- | --- |
| 任意メソッド`extract_with_trace` | 抽出器が実装していれば`ingest`が使い、返された`trace`をキャッシュと`TrainingImage.extraction_trace`（既定`{}`）に保存する。`trace`は`selected_model`、`fallback_used`、`attempts[]`（`provider_id`、`model_id`、`status`）、`verifier_status`、`egress`（6.6節） |
| 任意メソッド`extract_variants` | 実装していれば`TrainingImage.caption_variants`（既定`{}`）に保存する |
| `PromptBuilder` v2 | `caption_variants`がある画像は、タグを文に追記しない。自然文とタグの矛盾を避けるため（#78の「単純に結合しない」）。タグは別に保存する。baseline抽出器は従来の`prompt-builder-v1`のまま。`PROMPT_BUILDER_VERSION`が変わるため、`cache_key`も変わる |
| `iter_samples`の追加キーワード | `iter_samples(dataset_dir, split, *, prompt_variant="standard", variant_seed=None)`。`variant_seed`を指定すると、`SHA256("{variant_seed}:{image_id}")`から決まる乱数で短文40%・標準文40%・詳細文20%を選ぶ（比率は#78の仮の分布）。**位置引数の契約`iter_samples(dataset_dir, split)`と戻り値の型は変わらない** |
| 失敗のキャッシュ | `extractor_failed`の`QualityIssue`を含む結果は**キャッシュに保存しない**。現行の`extract_with_cache`は失敗した結果も保存するため、外部APIの一時的な障害や拒否が、抽出器が復旧した後も残る。再実行は失敗した画像だけが再抽出される |
| 例外の写し方 | `PromptExtractionError`は、`QualityIssue("extractor_failed", "<kind>: <message>")`にする。`kind`は#78が持つ分類（拒否、空、不正JSONなど） |
| manifestの版 | 調査した範囲でS2のmanifestに版を示すキーは確認できなかった。新しい任意フィールドは、古いmanifestを読めるよう既定値つきで追加し、`dataset_hash`の計算に含める範囲をPR I-2で明記する |

### 4.5 置き換えない・並行して残すもの

| 残すもの | 理由 |
| --- | --- |
| S2のbaseline抽出器（`color-layout-*`、`none`） | モデル不要・ネットワーク不要の既定。テストとオフライン環境の土台 |
| #78の`image-drawer-preprocess` | 1枚の診断用。設定とモデルの動作確認に使う。Datasetを更新しない |
| #78の`ImagePromptPipeline`、`TaggingPipeline`、検証、正規化、整形 | 抽出ロジックの本体 |

## 5. 廃止するもの・変更するもの

| 対象 | 扱い | 時期 |
| --- | --- | --- |
| #78の`ExtractionResult.cache_key`、`ImagePromptPipeline._cache_key`による識別 | キャッシュの鍵はS2の`cache_key`だけとする。フィールドは診断用に残すが、永続化と再利用の判断に使わない。のちに削除 | I-4 |
| #78の`PreprocessorRegistry.register_prompt_formatter`と`prompt_formatter`設定 | **廃止**。学習用promptの組み立てはDatasetの`PromptBuilder`のversionに一本化する。理由は、任意のformatterの出力が`cache_key`に反映されず、再現性を保証できないため。設定JSONに指定された場合は、I-3で警告、I-4でエラー | I-3〜I-4 |
| #78の`TrainingPreprocessingResult.training_prompt(seed)` | 診断コマンド用に残す。学習用の選択は`iter_samples(variant_seed)`に移す | I-2 |
| #78の「CaptionとTagを必ず両方実行する」 | [training_input_v1.json](training_input_v1.json)は`TagExtractor.may_be_disabled=true`。S2の無効なTagger（id `none`）を維持し、**Taggerは無効にできる**。#78の列は、Taggerを有効にしたときだけ使う | I-3 |
| #78の「どちらかが失敗したら例全体を失敗」 | Datasetの経路では、S2の方針に従い、失敗を`QualityIssue`として画像を残す。空promptの画像は`iter_samples`から除外し、件数を報告する。`image-drawer-preprocess`の診断では従来どおり失敗を返す | I-2 |
| `docs/users/preprocessing.md` | 独立した入口として残さず、`docs/users/image_dataset.md`に「外部モデルを使う抽出器の設定」の節として統合し、`preprocessing.md`からは参照だけにする | I-4 |
| `training_input_v1.json`の`prompt_extraction.pipeline: image_drawer.prompting.ImageTrainingPreprocessor` | `image_drawer.dataset.register_images`（アダプタ経由）に更新する | I-4 |

`training_input_v1.json`の`internally_generated`に#78が加えた名前は、次のDatasetのフィールドへ対応づけます。

| #78の名前 | Datasetでの置き場所 |
| --- | --- |
| `prompt_variants` | `TrainingImage.caption_variants` |
| `fallback_reason`、`selected_caption_model`、`chosen_caption_model`、`chosen_tag_model`、`verification_status`、`tagger_fallback_attempts` | `TrainingImage.extraction_trace` |
| `normalized_tag_groups`、`formatted_tags` | `TrainingImage.tags[].category`と`extraction_trace` |
| `prompt_formatter_version` | `PROMPT_BUILDER_VERSION`と`ExtractorInfo.config_hash` |

## 6. 外部APIへ画像を送る経路の安全要件

#78の標準アダプタは、OpenAI互換のVision APIとTagger Service/PixAI互換HTTPです。現行の実装は、`http`は`localhost`/`127.0.0.1`/`::1`に限定していますが、`https`は任意のホストへ接続でき、リダイレクトを追います（`urlopen`の既定）。画像は利用者の非公開データであり得るため、次を必須要件とします。

### 6.1 既定でオフ（三重の許可）

外部のホスト（`localhost`、`127.0.0.1`、`::1`以外）へ画像を送る抽出器は、次の**すべて**が揃ったときだけ動きます。

1. **Datasetの設定**: Datasetの設定に`"external_upload": "allowed"`が明示されている。既定は`"forbidden"`。
2. **プロバイダの設定**: 設定JSONのそのプロバイダに`"data_egress": "external"`と`"allowed_hosts": ["<ホスト名>"]`がある。エンドポイントのホストが`allowed_hosts`に無ければ、起動時にエラー。
3. **実行時の許可**: CLIは`--allow-external-upload`、GUIは確認ダイアログ（Datasetごと・実行ごと）。設定ファイルだけでは送信しない。

どれかが欠けた場合は、画像を送らず、理由を明示して停止します（未登録と同じ扱いのエラー。ダミーの抽出結果を作らない）。`localhost`宛であっても、別のホストへ転送するプロキシの場合は`"data_egress": "external"`を宣言する必要があります。転送の有無はコードで検出できないため、設定の責任とし、利用者向け文書に明記します。

### 6.2 送信前の表示

`register`は、送信前に**送信先のホスト名、モデルID、送信する画像の枚数、総バイト数**を表示します。`--dry-run`は、これらを表示するだけで何も送りません。GUIの確認ダイアログも同じ内容を表示します。

### 6.3 送るデータの最小化

- 送るのは、4.2節の正規化後のPNGだけです。EXIF、ファイル名、元のパス、`image_id`、Dataset名、利用者名を含めません。
- 指示文は定型文（`CAPTION_INSTRUCTION`など）だけです。利用者の他のデータを指示文に加えません。
- 1回の送信の画像サイズに上限を設け、既定は正規化後のPNGが8MiB以下とします（超える場合は縮小せずに失敗とし、勝手に品質を下げた画像を送らない）。

### 6.4 キーの管理

- キーは環境変数名（`api_key_env`）だけで指定します。設定JSONへの値の直書き（`api_key`）はエラーです（#78で実装済み）。
- 外部のプロバイダは、環境変数が未設定または空の場合に**匿名では送信しません**。エラーで停止します（`auth: none`の宣言は`localhost`のプロバイダだけに許す）。
- キーの値は、manifest、キャッシュ、`extraction_trace`、`egress.jsonl`、ログ、例外メッセージ、`config_fingerprint`のいずれにも含めません。`Authorization`ヘッダを含む要求オブジェクトを`repr`や例外に出しません。
- HTTPエラーの応答本文は保存せず、状態コードだけを記録します（応答にキーの断片や画像の内容が入り得るため）。

### 6.5 通信の制限

- 外部のホストは`https`のみです。TLS検証を無効にする設定を持ちません。
- リダイレクトを追いません。`3xx`は失敗として扱います（別ホストへの転送を防ぐ）。
- 応答のサイズ上限（1MiB）と、タイムアウト（既存の1〜3600秒）を適用します。
- 同じプロバイダへの自動再試行は行いません。失敗は、列の次のプロバイダへ進むか、`extractor_failed`として記録します。
- 応答は**データ**として扱います。JSONのスキーマを検証し、未知のキーを無視し、文字列の長さと制御文字を制限します。画像内の文字による指示（プロンプトインジェクション）の結果として、設定や挙動が変わらないようにします。

### 6.6 記録

Datasetに追記専用の`egress.jsonl`を置きます。1行に、日時、`provider_id`、ホスト名、`model_id`、`image_sha256`、送信バイト数、状態コード（または失敗の種類）を記録します。**画像本体、応答本文、キーは記録しません**。`TrainingImage.extraction_trace.egress`には、送信の有無（真偽）とホスト名だけを残します。

評価（[評価プロトコル](design_evaluation_protocol.md)）でも同じ規則を適用します。`test`の画像を外部へ送った場合は、その事実が`egress.jsonl`と報告書の`limitations`に残ります。

### 6.7 利用者向けの説明

外部のモデルを使うかどうかは、画像の公開範囲、権利、プライバシーに関わる判断です。利用者向けの説明（`docs/users/image_dataset.md`）に、「既定では画像は外部へ送られません」「許可の三段階」「送られるもの・送られないもの」を、設定例とともに書きます。

## 7. 移行手順

### 7.1 マージの順序

1. **PR #74（S2）を先にマージします。** S3以降（Trainer、推論、評価）が`iter_samples`に依存しており、契約が小さいためです。
2. PR #78を`main`へ更新し、衝突を**両方を残す**方針で解消します（`pyproject.toml`の`[project.scripts]`は両方の行、README類は両方の項目を残す）。`training_input_v1.json`は#78の追加キーを維持します（S2は変更していません）。
3. その後、統合のPR I-1からI-4を、次の順で入れます。

### 7.2 PRの分割

1PRは1つの目的で、変更は500行以下を目安にします。

| PR | 目的 | 内容 | 前提 |
| --- | --- | --- | --- |
| I-1 | アダプタ | `dataset/prompting_adapters.py`、`VisionProvider`/`TagProvider`のProtocolへ`config_fingerprint`を追加、`ExtractorInfo`の決め方（4.3節）、`test_prompting_adapters.py` | #74と#78のマージ |
| I-2 | Datasetの任意拡張 | `extract_with_trace`/`extract_variants`の対応、`TrainingImage.caption_variants`と`extraction_trace`、`PromptBuilder` v2、`iter_samples`の追加キーワード、失敗をキャッシュしない変更、`PromptExtractionError`の写し方 | I-1 |
| I-3 | 入口と安全 | `image-drawer-dataset register --extractor-settings --allow-external-upload --dry-run`、Datasetの`external_upload`設定、`allowed_hosts`、リダイレクト禁止、応答サイズ上限、`egress.jsonl`、`prompt_formatter`の警告 | I-2 |
| I-4 | 整理 | 廃止（5節）、`training_input_v1.json`の更新、`docs/users/`の統合、`docs/developers/`の索引 | I-3 |

I-3を入れる前は、外部のホストを使う設定を文書で案内しません。I-1・I-2は、`localhost`のプロバイダと、テスト用の擬似プロバイダだけで検証します。

## 8. 受け入れテスト一覧

### `tests/test_prompting_adapters.py`

- `test_caption_adapter_satisfies_caption_extractor_protocol`
- `test_tag_adapter_satisfies_tag_extractor_protocol`
- `test_adapter_sends_png_bytes_without_metadata_to_the_provider`
- `test_cache_key_changes_when_fallback_order_changes`
- `test_cache_key_changes_when_a_provider_model_or_endpoint_changes`
- `test_cache_key_changes_when_instruction_or_verifier_changes`
- `test_cache_key_is_unchanged_by_api_key_value`
- `test_chain_is_registered_as_one_extractor_id_per_profile`
- `test_registry_describe_lists_prompting_extractors_with_hashes`
- `test_tag_adapter_keeps_rating_and_copyright_out_of_prompt_categories`
- `test_prompting_error_becomes_extractor_failed_quality_issue_and_image_is_kept`
- `test_all_providers_failing_leaves_empty_prompt_and_image_is_skipped_by_iter_samples`

### `tests/test_dataset_prompting_integration.py`

- `test_register_images_with_prompting_extractors_writes_trace_and_variants`
- `test_prompt_builder_v2_does_not_append_tags_when_variants_exist`
- `test_baseline_extractors_still_use_prompt_builder_v1`
- `test_iter_samples_default_contract_is_unchanged`
- `test_iter_samples_variant_seed_selects_prompt_variants_deterministically`
- `test_iter_samples_variant_distribution_is_close_to_40_40_20`
- `test_failed_extraction_is_not_cached_and_is_retried_on_next_register`
- `test_successful_extraction_is_cached_and_providers_are_not_called_again`
- `test_old_manifest_without_new_fields_still_loads`
- `test_target_image_stays_out_of_policy_input_with_prompting_extractors`
- `test_tagger_can_be_disabled_with_none_while_caption_chain_is_used`

### `tests/test_external_upload_policy.py`

- `test_external_provider_is_rejected_without_dataset_external_upload_allowed`
- `test_external_provider_is_rejected_without_data_egress_declaration_and_allowed_hosts`
- `test_external_provider_is_rejected_without_allow_external_upload_flag`
- `test_dry_run_prints_destination_count_and_bytes_and_sends_nothing`
- `test_endpoint_host_outside_allowed_hosts_is_rejected_at_startup`
- `test_plain_http_to_external_host_is_rejected`
- `test_redirect_response_is_treated_as_failure_and_not_followed`
- `test_response_larger_than_limit_is_rejected`
- `test_missing_or_empty_api_key_env_stops_external_provider_without_anonymous_send`
- `test_api_key_value_in_settings_is_rejected`
- `test_api_key_never_appears_in_manifest_cache_trace_ledger_logs_or_exceptions`
- `test_http_error_body_is_not_stored`
- `test_egress_ledger_is_append_only_and_has_no_image_or_response_body`
- `test_payload_contains_no_file_name_path_image_id_or_dataset_name`
- `test_oversized_png_fails_instead_of_being_downscaled`
- `test_provider_response_text_is_validated_and_unknown_keys_are_ignored`
- `test_localhost_provider_declared_external_still_requires_all_three_permissions`

### 既存テストの維持

`tests/test_dataset.py`（S2）、`tests/test_prompting_pipeline.py`、`tests/test_tagging_preprocessing.py`、`tests/test_tagger_http_cli.py`、`tests/test_prompt_extraction_contract.py`（#78）は、削除・弱体化せず、5節の廃止に合わせて必要な箇所だけを更新します。

## 9. 正本との差分と判断

既存の正本は変更しません。

| # | 正本 | 本書の判断 | 理由 |
| ---: | --- | --- | --- |
| 1 | [training_input_v1.json](training_input_v1.json)（`main`）は`TagExtractor.may_be_disabled=true`。#78の`prompt_extraction_v1.md`は「Caption生成とTag抽出は必ず両方実行」 | Taggerは無効にできる。#78の列はTaggerを有効にしたときの動作 | `main`の入力契約に従う。S2にも無効なTagger（id `none`）がある |
| 2 | #78は、CaptionとTagのどちらかが失敗すると例全体を失敗にする | Datasetの経路では画像を残し、`QualityIssue`と`iter_samples`からの除外で扱う | 画像を黙って捨てない、失敗を分母に残す、というS2・[データ品質](drawing_data_quality_v1.md)の方針 |
| 3 | #78は`prompt_formatter`で学習用promptを組み立てられる | 廃止し、Datasetの`PromptBuilder`のversionに一本化 | formatterの出力が`cache_key`と再現性に反映されない |
| 4 | #78は、元ファイルの`bytes`をそのまま送る | 正規化後のPNGを送る | メタデータの除去、規則の統一、送信量の最小化 |
| 5 | #78の`https`は任意のホストへ接続できる | 三重の許可、`allowed_hosts`、リダイレクト禁止などを追加 | 画像は非公開のデータであり得る。要件は6節 |
| 6 | [画像のみ学習](image_only_learning_v1.md)は、キャッシュ鍵を`SHA256(image_bytes_hash, tagger…, captioner…, prompt_builder_version)`とする | S2の実装に従い、正規化versionを加える。#78の列は`model_hash`/`config_hash`に畳み込む | S2が実装しているキーが正本の意図を満たし、列の変更も検出できる |

### 9.1 実装時の補足（Issue #83、設計の範囲内の具体化）

| # | 内容 |
| ---: | --- |
| 1 | タグの`prompt`用カテゴリ（4.1節）は、`tags[]`の各要素の`in_prompt`（真偽）で表す。`PromptBuilder`は`in_prompt`が`false`のタグを文に使わない。ratingは`tags`ではなく`extraction_trace.tags.rating`に保存する |
| 2 | `extraction_trace`は`{"caption": {...}, "tags": {...}}`の形。各`trace`は4.4節の項目（`egress`は`{"sent": bool, "hosts": [...]}`）を持つ |
| 3 | `PromptBuilder` v2は、抽出器が`extract_variants`を持つ場合にmanifest全体で使う。変種が得られなかった画像は、v1と同じ組み立てになる |
| 4 | 設定JSONに任意の`profile_id`（既定`human-prompt-aligned-v1`）を追加した。抽出器idの`<profile_id>`になる |
| 5 | `register_images(..., allow_external_upload=False)`と、CLIの`--extractor-settings`（指定時の`--tagger`/`--captioner`の既定は`prompting:*`）を追加した。Datasetの設定は`dataset_config.json`の`{"external_upload": "allowed"}` |
| 6 | 外部判定は「ホストがlocalhost・127.0.0.1・::1以外、または`data_egress`が`external`」。内部IP（プライベート・リンクローカル・予約・マルチキャスト、10進/16進表記、IPv4射影IPv6）と`.local`/`.internal`等の名前は常に拒否する。DNSで内部IPへ解決される名前は検出していない |
| 7 | `egress.jsonl`の`status`は`ok`、または失敗の種類（HTTPエラーは`unavailable:500`のように状態コード付き）。`image_sha256`は送ったPNGのSHA-256 |
| 8 | 応答の制限は1MiB（VLM・Tagger共通）。説明文の制御文字は不正として次のモデルへ進む |
| 9 | `prompt_formatter`設定は、5節の「I-3で警告、I-4でエラー」に従い、I-4で`ValueError`にした（PR #78の「設定で選べるようにする」指摘は、この廃止方針が優先される）。`register_prompt_formatter`は廃止予定の警告を出して残す |

## 10. 未決事項

| ID | 内容 | 決める場所 |
| --- | --- | --- |
| D1 | Datasetの`external_upload`設定を、manifestに入れるか、別の設定ファイルに置くか（`dataset_hash`の範囲に含めるか） | **I-3で決定: `<dataset>/dataset_config.json`に置き、manifestと`dataset_hash`には含めない**（9.1節） |
| D2 | 外部APIの利用料金と上限の扱い（本書は金額の管理を決めない） | 運用ルール |
| D3 | 8MiBの上限と、縮小の禁止が、実用に合うか | I-3の実測 |
| D4 | S2のmanifestに版キーを追加するか | I-2。S2のレビュー |
| D5 | 抽出文が画像に忠実であることの評価（Completeness/Correctness）をDatasetの品質警告に反映するか | 実画像の測定後 |
| D6 | #78が`main`に入った時点の実コードとの再照合（本書は`pr78`のコミットを読んだ内容に基づく） | I-1の着手前 |
