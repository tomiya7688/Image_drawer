> **対象読者:** Episode v2の保存・読込、Replayer、受け入れテストを実装する開発者
> **目的:** S1c（Issue #68）で実装するEpisode v2の保存形式、厳密な読込、製品コードのReplayer、fixture、受け入れテスト、PR分割を実装前に確定する。
> **状態:** 実装設計書。Episode v2の保存・読込、構造検証、製品Replayer、7工程fixtureはS1c（Issue #68、ブランチ `claude/68-episode-replay`）で**実装済み（マージ前）**です。実装との差分は§12の末尾の表にあります。Journal・ACKを使う実行中の永続化とクラッシュ注入（S4）は未実施です。[StageController設計書](design_stage_controller.md)（S1b）の後続で、その公開名を前提にします。正本の[Controller契約](stage_controller_contract_v2.md)、[Stage JSON](drawing_stage_workflow_v2.json)、[v1仕様](drawing_process_v1.md)は変更しません。

# Episode v2の保存とReplayer 実装設計（S1c）

## 1. 目的・範囲・範囲外

現状の受け入れテスト（`tests/test_drawing_design_contract.py`）は、**テスト内のオラクル**がfixtureを再生します。製品コードの再生ではないため、描画器やControllerを実装しても検証になりません。S1cは、製品コードの`replay()`で、v1 goldenとv2の工程例を画素一致で再生します。

| 区分 | 内容 |
| --- | --- |
| 範囲 | Episode v2のJSON Schema（[drawing_episode_v2.schema.json](schemas/drawing_episode_v2.schema.json)）、strict loader、semantic validator（グループの原子性・工程連鎖）、1ファイルの原子的保存、末尾の未確定グループの切り詰め回復、製品コードの`replay()`、v1 goldenの製品Replayerによる検証、7工程の実行例fixture、`Episode`のArtifact型名とメタデータ形式 |
| 範囲外 | Controllerを通さないStage規則の再監査（有効筆跡の再判定など）、Journal・ACK・manifestを使う実行中の永続化（S4）、`DRAW_EPISODE` Step・Trajectory連携（S4）、PNG出力、Dataset化、画質評価 |
| 保証しないこと | 再生できることは、絵が良いことも、モデルが学習したことも意味しません。 |

Replayerは`StageController`を**通しません**。Controllerが受理したイベントを`Engine`へ再適用し、同じRGBA・レイヤー・表示状態へ戻せることを示します。これにより、Controllerの記録が自己完結していること（暗黙の状態に依存しないこと）を検証します。

## 2. 前提（S1aの現状）

- `DrawingEpisode`と`loads_strict`（小数・指数・NaN・重複キーの拒否）はS1aにあります。`DrawingEpisode.from_dict`はv1とv2（brush-raster-v2）のheaderを検証しますが、v2の**JSON Schemaは未作成**で、`DrawingAction.from_dict`は5キーのeventだけを受け付けます。
- 保存・読込・Replayerはありません。v2 headerは`seed`・`brush_manifest_sha256`・`control_set_id`・`control_sha256`を持ち、`schema_version`は`drawing-episode/v2`です。
- `DrawingEpisode._check_events`は「systemのみのepisode」を拒否し、`finish`が末尾に1回あることを要求します。v2では、受理modelイベントが0件のdraftと、`finish`のないdraftを許す変更が必要です（§3.3）。
- S1bのPR S1b-1が`DrawingAction.stage_id`・`DrawingAction.group`・`EventGroup`・`DrawingState.stage_id`と工程マーカーを`Engine`に追加し、S1b-5/6が`DrawingEpisode`へ`workflow_id`・`workflow_sha256`・`style_intent`・`budget`・`status`・`termination_reason`を追加します。

## 3. モジュール構成

### 3.1 新規・変更ファイル

| ファイル | 種別 | 責務 | 目安行数 |
| --- | --- | --- | --- |
| `image_drawer/drawing/episode_groups.py` | 新規 | グループ・工程連鎖・レイヤー表のsemantic validator（`validate_episode_structure`）と、末尾の未確定グループの切り詰め（`truncate_to_last_complete_group`） | 260 |
| `image_drawer/drawing/episode_io.py` | 新規 | canonicalな`dumps_episode`、strictな`loads_episode`、原子的な`save_episode`、`load_episode`、`episode_sha256`、Artifactメタデータ | 250 |
| `image_drawer/drawing/replay.py` | 新規 | `replay`・`iter_replay`・`ReplayResult`・`ReplayedLayer`・`ReplayFrame` | 160 |
| `image_drawer/drawing/models.py` | 変更 | `DrawingAction.to_dict/from_dict`をv2の7キーeventへ拡張（`stage_id`・`group`はv2のみ） | +40 |
| `image_drawer/drawing/episode.py` | 変更 | v2 headerの追加キーの出力と検証。v2に限りsystemのみ・finishなしのdraftを許可。v1の動作は不変 | +60 |
| `image_drawer/drawing/validation.py` | 変更 | 例外`EpisodeFormatError`・`EpisodeStorageError`・`ReplayError`の追加 | +15 |
| `tests/fixtures/drawing_episode_v2/` | 新規 | 7工程の実行例（`*.episode.json`）と期待frames（`*.frames.json`） | データ |
| `tests/test_episode_v2_*.py`、`tests/test_replay_*.py` | 新規 | §10 | — |
| `pyproject.toml` | 変更 | 開発用依存に`jsonschema`（Draft 2020-12対応の版）を追加 | +1 |

### 3.2 既存コードとの関係

- 例外は`DrawingError`を直接継承します（具象A→B→Cの継承を避ける）。
- `replay.py`と`episode_groups.py`は、`image_drawer.drawing.stage.controller`・`rules`・`groups`・`progress`を**importしません**。グループの文法はvalidatorが独立して実装します。Controllerの出力をこの独立実装で検査することで、Controllerとvalidatorが同じ誤りを共有する事態を避けます。許可するのは`stage.table`（工程表の読取）と`stage.errors`だけです（テストで固定、§10.6）。
- v1 Episodeは`drawing-episode/v1`のまま、S1aの実装で読みます。`loads_episode`は`schema_version`で分岐し、v1とv2を混ぜません。
- 画像（`RenderedImage`）はS1aの型をそのまま返します。PNG化はS4です。

### 3.3 S1a・S1bの変更点の整理

| 変更 | 所有PR |
| --- | --- |
| `DrawingAction.stage_id/group`の**フィールド**追加 | S1b-1 |
| `DrawingAction`の**JSON入出力**（v2は7キー、v1は5キー） | S1c-1 |
| `DrawingEpisode`の追加フィールド | S1b-5/6 |
| `DrawingEpisode.to_dict/from_dict`のv2拡張、systemのみ・finishなしのdraft許可 | S1c-1 |

S1aにはv2のEpisodeを保存・読込する実装がなく、利用者もいないため、v2 headerの必須キーを増やしても既存の保存データを壊しません。

## 4. 公開API

### 4.1 保存・読込（`episode_io.py`）

```python
EPISODE_ARTIFACT_TYPE: Final[str] = "Episode"
MAX_EPISODE_BYTES: Final[int] = 16 * 1024 * 1024

def dumps_episode(episode: DrawingEpisode) -> str: ...
def loads_episode(text: str | bytes, *, recover: bool = False) -> DrawingEpisode: ...
def save_episode(episode: DrawingEpisode, path: str | os.PathLike[str]) -> str: ...   # 保存したバイト列のSHA-256（16進）
def load_episode(path: str | os.PathLike[str], *, recover: bool = False) -> DrawingEpisode: ...
def episode_sha256(episode: DrawingEpisode) -> str: ...
def episode_artifact_metadata(episode: DrawingEpisode, sha256: str) -> dict[str, Any]: ...
```

- `dumps_episode`は§5.1のcanonical形式を返します。検証に失敗するEpisodeは`EpisodeFormatError`で拒否し、書き出しません。
- `loads_episode`は§7の順序で検証します。`recover=True`のときだけ、末尾の未確定グループを切り詰めてdraftとして返します。
- `save_episode`は検証→canonical化→一時ファイルへ書込→`fsync`→`os.replace`→（POSIXのみ）ディレクトリ`fsync`の順で、**1ファイルを原子的に置換**します。保存先のディレクトリは作成しません。
- `episode_artifact_metadata`は、`Episode` Artifactの`metadata`に入れる値を返します。`schema_version`、`episode_sha256`、`status`、`termination_reason`、`event_count`、`group_count`、`renderer_version`、`workflow_id`、`canvas`です。`Artifact(artifact_type="Episode", uri=<ファイルパス>)`の組み立てと`DRAW_EPISODE`の`StepResult`（`Episode`と`Image`の2つ）は、S4の設計書が定めます。

### 4.2 構造検証（`episode_groups.py`）

```python
def validate_episode_structure(episode: DrawingEpisode) -> None: ...
def truncate_to_last_complete_group(episode: DrawingEpisode) -> DrawingEpisode: ...
```

`validate_episode_structure`は§6の規則R1〜R12を検査します。`truncate_to_last_complete_group`は、末尾の不完全なグループのイベントを取り除き、`status="draft"`・`termination_reason="recovered_truncated"`にします。切り詰める対象がなくても、`finish`を含まないEpisodeは同様にdraftとして返し、完成Episodeは変更しません。

### 4.3 Replayer（`replay.py`）

```python
@dataclass(frozen=True)
class ReplayedLayer:
    layer_id: str
    name: str
    role: str | None
    visible: bool
    rgba: bytes                         # 行優先のRGBA8。幅×高さ×4バイト

@dataclass(frozen=True)
class ReplayResult:
    image: RenderedImage                # 可視レイヤーを白背景へ合成した画像
    layers: tuple[ReplayedLayer, ...]   # 作成順
    state: DrawingState                 # 再生後のEngine状態（stage_id・active_layer_id・history）

@dataclass(frozen=True)
class ReplayFrame:
    event: DrawingAction                # 直前に適用したイベント
    is_group_end: bool
    def capture(self) -> ReplayResult: ...

def replay(episode: DrawingEpisode, *, until_seq: int | None = None) -> ReplayResult: ...
def iter_replay(episode: DrawingEpisode) -> Iterator[ReplayFrame]: ...
```

- `replay`は`Engine(width, height, profile=episode.renderer_version, seed=episode.seed)`を作り、v2ではグループ単位で`Engine.apply_group`（1件のグループは`Engine.apply`）、v1ではイベントごとに`Engine.apply`を呼びます。
- `until_seq`は、そのseqまでを適用します。v2では、そのseqがグループの最後のイベントでなければ`ReplayError`です（グループの途中状態は公開しません）。
- `iter_replay`は適用のたびに`ReplayFrame`を返します。`capture()`を呼んだときだけ合成と複製を行うため、反復自体は軽量です。
- Replayerは入力の`DrawingEpisode`を変更しません。

### 4.4 例外

| 例外 | 継承 | 発生 |
| --- | --- | --- |
| `EpisodeFormatError` | `DrawingError` | 不正なJSON・schema・header・構造（グループ・工程連鎖）、サイズ超過 |
| `EpisodeStorageError` | `DrawingError` | ファイルの読み書き・置換の失敗（`OSError`を`__cause__`に保持） |
| `ReplayError` | `DrawingError` | 再生中のEngine拒否、`stage_id`注釈の不一致、`until_seq`の不正。`seq`を含む |

## 5. データ形式

### 5.1 canonical形式

保存するファイルは次の形式のUTF-8（BOMなし）・LF改行で、同じEpisodeは常に同じバイト列になります。

1. 最上位のキーを辞書順に並べる。
2. `events`以外の値は、`json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`で出力する。
3. `events`は、`"events":[`の後に改行し、各eventをキー辞書順・空白なしの1行にして`,\n`で連結し、改行の後に`]`を置く。
4. 文書の末尾に`\n`を1つ付ける。

整数だけを使い、小数・指数表記・`NaN`は出力も受理もしません。画像バイト列は含めません。

### 5.2 headerと識別

v1とv2は`schema_version`で識別します。v2のheaderは、v1の10キーに次を加えた**厳密なキー集合**です（[schema](schemas/drawing_episode_v2.schema.json)）。

| キー | 内容 |
| --- | --- |
| `schema_version` | `drawing-episode/v2`（v1は`drawing-episode/v1`。`renderer_version`との組合せも固定） |
| `origin` | 常に`mixed`（model + system）。人間の履歴ではない |
| `seed`, `brush_manifest_sha256`, `control_set_id`, `control_sha256` | S1aのv2 header。ブラシ・段階表の同一性 |
| `workflow_id`, `workflow_sha256` | `illustration-standard-v2`とStage JSONのcanonical SHA-256 |
| `style_intent` | `{version, monochrome_line_art_only, flat_style_no_shading}`。レビュー退出の根拠 |
| `budget` | `{max_actions, max_attempts}` |
| `status` | `complete`または`draft` |
| `termination_reason` | `complete`は`finished`。`draft`は`action_budget_exhausted`・`attempt_budget_exhausted`・`in_progress`・`recovered_truncated`・`internal_error` |

拒否された試行（attempt）はEpisodeに含めません。`StageController.attempts()`が返すログを、S3/S4のTrial記録へ保存します。

### 5.3 event

v2のeventは7キー（`seq`・`time_ms`・`actor`・`type`・`payload`・`stage_id`・`group`）です。

| キー | 内容 |
| --- | --- |
| `actor` | `model`または`system`。Controller専用の`enter_stage`・`exit_stage`・`set_layer_visibility`・`clear_active_layer`・`finish`は`system`、`advance_stage`と描画系は`model`、`create_layer`は両方 |
| `stage_id` | そのeventを適用した直後の工程。`enter_stage`は入場先、`finish`は`finish` |
| `group` | `{id, index, size}`。原子的な記録グループ |

`type`ごとの`payload`は、スキーマの`oneOf`で固定します。`stroke_v2`・`erase_v2`は`samples`（1〜4096件、各`x_q8`・`y_q8`・`time_us`・`pressure_level_id`・`tilt_x_level_id`・`tilt_y_level_id`）、`advance_stage`は`{}`または`{"review_reason": <3コードのいずれか>}`です。

### 5.4 完成Episodeの冒頭（抜粋）

```json
{
  "schema_version": "drawing-episode/v2",
  "episode_id": "ep-3f9a1c0b7d2e4a65",
  "origin": "mixed",
  "prompt": "赤いリンゴを描いて",
  "canvas": {"width": 16, "height": 12},
  "palette_id": "idraw-palette-256-v1",
  "brush_set_id": "brush-v2-core12",
  "renderer_version": "brush-raster-v2",
  "seed": 0,
  "brush_manifest_sha256": "<64桁の16進>",
  "control_set_id": "brush-controls-v2-levels-20261103",
  "control_sha256": "<64桁の16進>",
  "workflow_id": "illustration-standard-v2",
  "workflow_sha256": "<64桁の16進>",
  "style_intent": {"version": "style-intent-rules-v1", "monochrome_line_art_only": false, "flat_style_no_shading": false},
  "budget": {"max_actions": 1000, "max_attempts": 5000},
  "status": "complete",
  "termination_reason": "finished",
  "events": [
    {"seq": 0, "time_ms": 0, "actor": "system", "type": "enter_stage",        "payload": {"stage_id": "rough"}, "stage_id": "rough", "group": {"id": 0, "index": 0, "size": 3}},
    {"seq": 1, "time_ms": 0, "actor": "system", "type": "create_layer",       "payload": {"layer_id": "sys_rough_001", "name": "ラフ", "role": "rough"}, "stage_id": "rough", "group": {"id": 0, "index": 1, "size": 3}},
    {"seq": 2, "time_ms": 0, "actor": "system", "type": "clear_active_layer", "payload": {}, "stage_id": "rough", "group": {"id": 0, "index": 2, "size": 3}},
    {"seq": 3, "time_ms": 0, "actor": "model",  "type": "select_layer",       "payload": {"layer_id": "sys_rough_001"}, "stage_id": "rough", "group": {"id": 1, "index": 0, "size": 1}}
  ],
  "metadata": {}
}
```

（上記は構造を示す抜粋で、実際の保存形式は§5.1のcanonical形式です。工程移行グループの例は[S1b設計書§5.3](design_stage_controller.md)にあります。）

### 5.5 原子性の表現

- 受理したmodel Actionと、それに伴うsystemイベントは**同じ`group.id`**を持ち、`seq`が連続し、`index`が`0..size-1`、`size`が全イベントで一致します。
- 1ファイル内では、全グループが完結していることをloaderが要求します。したがって、公開される完成Episodeは、途中の工程状態を含みません。
- ファイルの置換は1回の`os.replace`なので、読者は置換前の完全なファイルか、置換後の完全なファイルのどちらかだけを見ます。グループ構造は、手編集・別ツール・将来の追記型writerによる**部分書込**を検出するためのものです。

## 6. 構造検証の規則（`validate_episode_structure`）

| 規則 | 内容 |
| --- | --- |
| R1 | `seq`が0から連続し、`time_ms`が単調非減少。総数は1〜20,000 |
| R2 | `group.id`が0から1ずつ増え、同一グループの`seq`が連続し、`index`が`0..size-1`、`size`が一致。末尾のグループが完結している |
| R3 | group 0は`initial`文法: `enter_stage(rough)`, `create_layer(sys_rough_001, role=rough)`, `clear_active_layer`（全てsystem） |
| R4 | group 1以降は、`action`（modelの1イベント。`advance_stage`以外）、`transition`、`finalize`のいずれかの文法に一致する（文法はS1b設計書§5.4と同じ。非表示の件数は、その時点の可視レイヤーのうち入場先で非表示にするroleのものと一致する） |
| R5 | 各eventの`stage_id`が、直前までの連鎖から導かれる値と一致する（`enter_stage`は入場先、`finish`は`finish`、他は直前の値） |
| R6 | `enter_stage`の順序が`rough → sketch → final_drawing → base_color → shading → refine`で、各`enter_stage`の前に現在工程の`exit_stage`がある。逆行・重複・飛ばしを含まない |
| R7 | `status="complete"`なら、最後のグループが`finalize`でその最後のイベントが`finish`、`finish`は1回だけ。`draft`なら`finish`を含まない |
| R8 | `status`と`termination_reason`の組合せがスキーマの条件と一致する |
| R9 | `create_layer`のIDが一意。`set_layer_visibility`の対象が存在する。`sys_`で始まるIDは`create_layer`のsystemイベントだけが使い、`sys_<role>_<連番>`の連番が通し番号と一致する。modelの`create_layer`は`sys_`で始まらず、現在工程の許可roleに属する |
| R10 | `actor`が§5.3の対応に一致する |
| R11 | `actor=model`のイベント数が`budget.max_actions`以下。`max_attempts ≥ max_actions` |
| R12 | `advance_stage`の`review_reason`が現在工程の許可コードに属し、`base_color`の`monochrome_line_art_only`は`style_intent.monochrome_line_art_only=true`、`shading`の`flat_style_no_shading`は`style_intent.flat_style_no_shading=true`を要する |

validatorは**画素・筆跡の有効性**を判定しません（有効筆跡の計数、レビュー退出前に描画がないことなどはControllerの責務です）。再監査が必要な場合は、別設計で扱います。

## 7. 読込の順序（`loads_episode`）

1. **サイズ**: バイト列が`MAX_EPISODE_BYTES`以下。超過は`EpisodeFormatError`。
2. **文字符号**: UTF-8として復号できる。BOMは拒否。
3. **厳密JSON**: `loads_strict`（小数・指数・`NaN`・重複キーを拒否）。最上位がオブジェクト。
4. **識別**: `schema_version`で分岐。`drawing-episode/v1`はS1aのv1経路、`drawing-episode/v2`はv2経路、それ以外は`EpisodeFormatError`。
5. **header**: キー集合が厳密に一致し、定数（`palette_id`・`brush_set_id`・`renderer_version`・`control_set_id`・`workflow_id`）と、製品のブラシ・段階表・Stage JSONのSHA-256が一致する。`seed`・`canvas`・`budget`・`style_intent`の型と範囲。
6. **event**: 各eventが7キーで、型・範囲がスキーマと同一。`payload`の厳密なキー集合。
7. **構造**: `validate_episode_structure`（v2）。v1は従来の`seq`連続・actor・origin・`finish`の検査。
8. `recover=True`のときは、手順7の前に末尾の不完全なグループだけを切り詰める。それ以外の不整合は回復せず、`EpisodeFormatError`にする。

Engineの規則（座標範囲・色・ブラシmode・累積予算）は、`replay`で初めて検査します。保存済みEpisodeを読むだけで、Engineの再実行は行いません。

## 8. 失敗時の動作

| 状況 | 動作 |
| --- | --- |
| 不正JSON・重複キー・未知キー・型/範囲違反・グループ不整合 | `EpisodeFormatError`。何も返さない |
| 末尾の不完全なグループ（`recover=False`） | `EpisodeFormatError`（不完全なグループがあることを示す） |
| 末尾の不完全なグループ（`recover=True`） | 最後の確定グループまでを`status="draft"`、`termination_reason="recovered_truncated"`で返す |
| ファイルが存在しない・読めない | `EpisodeStorageError` |
| 保存先ディレクトリがない・書込失敗・置換失敗 | `EpisodeStorageError`。既存の保存先ファイルは**元のまま**。一時ファイルは削除を試みる |
| 保存中のクラッシュ（プロセス終了） | 保存先は置換前の完全なファイルか置換後の完全なファイル。残った`.<名前>.*.tmp`はEpisodeとして読まれない |
| `replay`中にEngineが拒否 | `ReplayError`（`seq`・元の`DrawingError`を保持）。部分的な結果は返さない |
| `stage_id`注釈がEngineの`state.stage_id`と不一致 | `ReplayError` |
| 完成Episodeで再生後に`state.finished`が偽 | `ReplayError` |

## 9. 決定性の要件

- 同じ`DrawingEpisode`は常に同じcanonicalバイト列とSHA-256になります。`dumps_episode(loads_episode(text)) == text`が、canonicalな`text`で成立します。
- `replay`は乱数・時刻・環境に依存せず、同じEpisodeから同じRGBA・レイヤー・`stage_id`・選択状態を得ます。異なる`PYTHONHASHSEED`でも一致します。
- 一時ファイル名は結果に影響しません。保存結果はパス以外の入力だけで決まります。
- Controllerの実行で得たEngine状態と、`replay(controller.episode())`の状態が、画素・レイヤー（id・role・visible）・`active_layer_id`・選択色/ブラシ/段階・`stage_id`・`finished`で一致します。

## 10. 受け入れテスト一覧

テストは製品の`replay`・`loads_episode`・`save_episode`・`StageController`を通して行い、テスト内に再生器を持ちません。小さな画布（例: 16×12）を使います。

### 10.1 スキーマ（`tests/test_episode_v2_schema.py`）

- `test_schema_is_valid_draft_2020_12`
- `test_schema_accepts_every_event_type_example`（13種のeventを1件ずつ）
- `test_schema_accepts_all_v2_fixtures`
- `test_schema_rejects_v1_schema_version`
- `test_schema_rejects_event_without_stage_id_or_group`
- `test_schema_rejects_unknown_event_type_and_extra_keys`
- `test_schema_rejects_controller_only_event_with_model_actor`
- `test_schema_rejects_model_only_event_with_system_actor`
- `test_schema_rejects_complete_status_with_non_finished_reason`
- `test_schema_rejects_draft_status_with_finished_reason`
- `test_schema_rejects_float_and_bool_in_integer_fields`
- `test_schema_group_bounds`（`index`・`size`の境界、`size=65`の拒否）
- `test_schema_review_reason_enum_has_exactly_three_codes`
- `test_schema_stage_enum_has_no_skip_stage`
- `test_python_validator_and_schema_agree_on_header_key_set`
- `test_schema_events_max_items_equals_engine_limit`

### 10.2 保存・読込（`tests/test_episode_v2_io.py`）

- `test_dumps_is_canonical_and_ends_with_single_lf`
- `test_dumps_loads_roundtrip_is_byte_identical`
- `test_loads_rejects_float_exponent_nan_and_duplicate_keys`
- `test_loads_rejects_unknown_schema_version`
- `test_loads_rejects_v1_header_with_v2_renderer`
- `test_loads_rejects_wrong_brush_manifest_or_control_hash`
- `test_loads_rejects_wrong_workflow_hash`
- `test_loads_rejects_bom_and_invalid_utf8`
- `test_loads_rejects_missing_unknown_and_duplicate_header_keys`
- `test_loads_rejects_oversized_input`（`MAX_EPISODE_BYTES`の境界: ちょうどは受理、1バイト超は拒否）
- `test_loads_rejects_event_count_20001`
- `test_loads_rejects_event_with_missing_stage_id_or_group`
- `test_loads_rejects_canvas_out_of_range_and_bad_seed`
- `test_loads_dispatches_v1_episode_to_v1_path`
- `test_save_then_load_returns_equal_episode`
- `test_save_returns_sha256_of_written_bytes`
- `test_save_replaces_existing_file_atomically`（置換後は新内容のみ）
- `test_save_is_deterministic_for_same_episode`
- `test_save_rejects_invalid_episode_without_touching_destination`
- `test_save_fails_cleanly_when_directory_is_missing`
- `test_save_failure_before_replace_keeps_original_and_removes_temp`
- `test_save_crash_after_temp_write_leaves_original_intact`（`os.replace`前に注入）
- `test_save_crash_after_fsync_leaves_original_intact`
- `test_save_crash_after_replace_leaves_new_file_complete`
- `test_orphan_temp_file_is_not_loadable_as_episode`
- `test_load_missing_file_raises_storage_error`
- `test_episode_artifact_metadata_has_required_keys`

### 10.3 構造・原子性（`tests/test_episode_v2_structure.py`）

- `test_structure_accepts_controller_generated_episodes`（S1bのfull・review・draft）
- `test_r1_rejects_seq_gap_duplicate_and_nonzero_start`
- `test_r1_rejects_decreasing_time_ms`
- `test_r2_rejects_group_id_gap_or_reuse`
- `test_r2_rejects_group_with_noncontiguous_seq`
- `test_r2_rejects_group_index_out_of_order`
- `test_r2_rejects_inconsistent_group_size_within_group`
- `test_r2_rejects_truncated_trailing_group`
- `test_r3_rejects_missing_or_reordered_initial_group`
- `test_r4_rejects_transition_missing_exit_stage`
- `test_r4_rejects_transition_missing_clear_active_layer`
- `test_r4_rejects_transition_with_wrong_visibility_count`
- `test_r4_rejects_advance_stage_alone_in_its_own_group`
- `test_r4_rejects_system_event_inside_action_group`
- `test_r4_rejects_set_layer_visibility_outside_transition`
- `test_r5_rejects_wrong_stage_id_annotation`
- `test_r6_rejects_skipped_stage`
- `test_r6_rejects_backward_enter_stage`
- `test_r6_rejects_duplicate_enter_stage`
- `test_r7_rejects_events_after_finish`
- `test_r7_rejects_complete_without_finish_and_draft_with_finish`
- `test_r7_rejects_finish_outside_finalize_group`
- `test_r8_rejects_inconsistent_status_and_reason`
- `test_r9_rejects_duplicate_layer_id`
- `test_r9_rejects_visibility_event_for_unknown_layer`
- `test_r9_rejects_model_layer_id_with_sys_prefix`
- `test_r9_rejects_system_layer_id_with_wrong_sequence_number`
- `test_r10_rejects_wrong_actor_for_each_event_type`
- `test_r11_rejects_more_model_events_than_max_actions`
- `test_r12_rejects_review_reason_not_allowed_for_stage`
- `test_r12_rejects_monochrome_review_when_intent_false`
- `test_r12_rejects_flat_review_when_intent_false`
- `test_structure_accepts_draft_with_only_initial_group`
- `test_structure_accepts_review_exit_in_three_stages`

### 10.4 回復（`tests/test_episode_v2_recover.py`）

- `test_recover_truncates_trailing_incomplete_group_to_last_complete_group`
- `test_recover_result_is_draft_with_recovered_truncated_reason`
- `test_recover_result_replays_to_state_after_last_complete_group`
- `test_recover_off_by_default_rejects_incomplete_group`
- `test_recover_does_not_modify_complete_episode`
- `test_recover_cannot_repair_non_group_corruption`（hash不一致・R5違反は回復しない）
- `test_recover_with_only_initial_group_incomplete_is_rejected`
- `test_recovered_draft_can_be_saved_and_loaded`

### 10.5 Replayer（`tests/test_replay_core.py`）

- `test_replay_of_v2_episode_matches_controller_engine_pixels`
- `test_replay_matches_layers_visibility_selection_and_stage_of_controller`
- `test_replay_returns_layers_in_creation_order_with_roles`
- `test_replay_image_is_composited_by_role_z_order`
- `test_replay_until_seq_at_group_end_matches_intermediate_state`
- `test_replay_until_seq_inside_group_raises`
- `test_replay_until_seq_out_of_range_raises`
- `test_replay_of_draft_returns_state_before_unfinished_stage_end`
- `test_replay_does_not_mutate_episode`
- `test_replay_raises_replay_error_with_seq_on_engine_rejection`
- `test_replay_detects_stage_id_annotation_mismatch`
- `test_replay_detects_hash_or_header_mismatch_before_applying`
- `test_replay_is_deterministic_across_two_runs`
- `test_replay_is_independent_of_pythonhashseed`（サブプロセス2つのRGBA SHA-256が一致）
- `test_replay_uses_group_atomic_apply_for_transition_groups`（途中失敗でEngineが部分状態にならない）
- `test_iter_replay_yields_one_frame_per_event_and_marks_group_ends`
- `test_iter_replay_capture_matches_replay_until_seq`
- `test_replay_hidden_layers_are_excluded_from_image_but_kept_in_layers`

### 10.6 Controllerを通さない確認（`tests/test_replay_independence.py`）

- `test_replay_modules_do_not_import_stage_controller_rules_groups_or_progress`
- `test_replay_works_without_constructing_stage_controller`
- `test_validator_group_grammar_is_independent_implementation`（Controllerの出力が通り、文法を1か所壊した手作りEpisodeが落ちる）
- `test_replayer_does_not_judge_stage_rules`（有効筆跡のない工程退出を含む手作りEpisodeが構造的に正しければ再生される。Stage規則の再監査は範囲外と明示するテスト）

### 10.7 v1 goldenの製品Replayer検証（`tests/test_replay_v1_golden.py`）

- `test_product_replayer_matches_v1_golden_frames`（`golden_episode.json`と`golden_frames.json`の全checkpoint: 合成画素と`layer_pixels`）
- `test_product_replayer_keeps_sketch_layer_black_under_red_layer`（下書き→別レイヤー→修正: seq=10で`sketch[1,0]`が黒、合成が赤）
- `test_product_replayer_erase_on_upper_layer_restores_black_composite`（seq=12）
- `test_product_replayer_matches_v1_geometry_cases`（`golden_geometry.json`の全case・全checkpoint）
- `test_product_replayer_replays_mixed_actor_episode_with_human_and_model_actors`
- `test_product_replayer_rejects_events_after_finish_in_v1`
- `test_v1_episode_dispatch_does_not_require_stage_id_or_group`
- `test_save_episode_rejects_v1_episode`（保存はv2だけ。v1は読込と再生のみで、v1 fixtureを書き換えない）
- `test_legacy_oracle_tests_still_pass`（既存の`test_drawing_design_contract.py`は残す）

### 10.8 7工程のfixture（`tests/test_replay_stage_episodes.py`）

fixtureは`tests/fixtures/drawing_episode_v2/`に置き、`python tests/test_replay_stage_episodes.py --regenerate`で作り直します。生成はスクリプト化した提案をControllerへ与えて行い、再生成の差分をレビューします。frames fixtureには、各グループ終了後の合成画素、各レイヤーの`id`・`role`・`visible`・RGBA（16進）、`stage_id`を保存します。

| fixture | 内容 |
| --- | --- |
| `full_seven_stage` | 7工程すべてで描画。rough・sketchの後に非表示、base_colorが本描きの下、shadeが下・highlightが上、refineでcorrectionレイヤーに描画、既存レイヤーを消去 |
| `review_only_stages` | 「白黒の線画だけ、影なしで描いて」。base_color・shadingを理由付きレビューで退出し、refineは`no_correction_needed` |
| `draft_budget_exhausted` | `max_actions`切れの途中状態（`status="draft"`、`action_budget_exhausted`）。final_drawing途中 |

- `test_full_seven_stage_replays_to_golden_frames_at_every_group_end`
- `test_review_only_stages_replays_to_golden_frames_at_every_group_end`
- `test_draft_fixture_replays_to_golden_frames_and_is_not_finished`
- `test_fixtures_load_with_strict_loader_and_pass_structure_validation`
- `test_fixtures_are_canonical_bytes`
- `test_fixture_frames_match_controller_regeneration`（再生成と一致）
- `test_full_fixture_base_color_does_not_cover_final_drawing_line`（独立した確認: 本描きのalphaが255の画素で、合成色が`palette_rgb(0)`と一致）
- `test_full_fixture_rough_and_sketch_layers_are_hidden_but_present`
- `test_full_fixture_highlight_pixels_render_above_final_drawing`
- `test_review_fixture_has_enter_and_exit_events_for_reviewed_stages`
- `test_review_fixture_has_no_stroke_events_in_reviewed_stages`
- `test_every_fixture_contains_all_six_enter_stage_events_in_order_when_complete`
- `test_fixture_actors_are_model_and_system_only_and_origin_is_mixed`
- `test_fixture_metadata_marks_scripted_source`（`metadata.source="scripted-fixture"`。モデルの学習データと混同しない印）

### 10.9 Controllerとの結合（`tests/test_stage_controller_replay.py`）

- `test_controller_episode_replays_to_identical_state_for_full_run`
- `test_controller_episode_replays_to_identical_state_for_review_run`
- `test_controller_draft_replays_to_identical_state_at_every_attempt_limit`
- `test_controller_rejected_attempts_are_absent_from_replayed_episode`
- `test_save_and_load_controller_episode_replays_identically`
- `test_every_committed_group_boundary_replays_to_controller_checkpoint`（グループごとにControllerのEngine状態と`replay(until_seq=...)`が一致）
- `test_replay_after_recover_equals_controller_state_at_that_group`

## 11. 実装手順（PR分割）

1PR=1目的、目安500行以下（テストとfixtureを除くコードで）です。

| PR | 目的 | 内容 | 前提 |
| --- | --- | --- | --- |
| S1c-1 | v2 schemaと構造検証 | `drawing_episode_v2.schema.json`の反映確認、`DrawingAction`/`DrawingEpisode`のJSON拡張、`episode_groups.py`、§10.1・§10.3、`jsonschema`の開発依存追加 | S1b-1、S1b-6 |
| S1c-2 | strict loaderと原子的保存 | `episode_io.py`、`EpisodeFormatError`・`EpisodeStorageError`、§10.2・§10.4 | S1c-1 |
| S1c-3 | Replayerとv1 golden | `replay.py`、`ReplayError`、§10.5・§10.6の独立性テスト、§10.7 | S1c-1 |
| S1c-4 | 7工程fixture | `tests/fixtures/drawing_episode_v2/`、生成スクリプト、§10.8 | S1c-2、S1c-3、S1b-6 |
| S1c-5 | 結合と文書更新 | §10.9、`drawing_process_acceptance.md`・`start_here.md`・`README`の状態欄更新、`design_audit.md`の対応表更新 | S1c-4 |

各PRで`python scripts/check_local.py`を実行し、整形（Ruff）を適用します。fixtureの再生成差分は、画素の16進を目視でレビューします。

## 12. 既存設計書との差分と判断

正本は変更していません。

| 項目 | 正本・現状 | 判断 |
| --- | --- | --- |
| systemのみのepisode | S1aの`DrawingEpisode`は拒否、v1受け入れ試験も混在actorを要求 | v2の`status="draft"`で、受理modelイベントが0件のEpisodeを許す。v1は変更しない |
| `finish`のないEpisode | v1の読込は`finish`を必須とする | v2に限り`status="draft"`で許す。完成Episodeは`finish`が最後の1回であることを要求する |
| `origin` | v1は`human`/`model`/…/`mixed`。契約§5は`mixed`が正常と記載 | v2は常に`mixed`（model + system）とする |
| 画像の正本 | v1仕様は「`events`が正本」 | v2も同じ。画像バイトは含めず、再生で得る |
| 保存の原子性 | v1受け入れ試験はJournal・manifestを含むクラッシュ注入を要求 | S1cは**1ファイルの原子的置換**までとし、Journal・ACKは実行中の永続化としてS4で扱う。受け入れ試験の該当項目はS4まで未実施として残す |
| イベントの注釈 | 契約§5は「グループIDを含める」とのみ記載 | `stage_id`と`group={id,index,size}`を全eventに持たせる。グループ長が自己記述的で、部分書込を検出できる |
| `max_actions`上限 | Stage JSONは100000 | S1bに合わせ19,900。schemaの`budget.max_actions`も同じ上限 |
| 構造検証の範囲 | 契約§8は「Controllerが受け付けたAction列をReplayする」 | Replayerは構造と再適用だけを行い、Stage規則（有効筆跡など）を再判定しない。理由は、Controllerの内部状態（画素差分）に依存するため、Controllerを通さない原則と両立しないこと |
| `Episode` Artifact | 監査は`Episode`型の追加を計画 | 型名`Episode`とメタデータ形式だけをS1cで定め、Artifactの組み立てと`StepResult`はS4に委ねる |

### 実装時の差分（S1c）

| 項目 | 設計 | 実装 |
| --- | --- | --- |
| 構造検証の分割 | `episode_groups.py`に集約（目安260行） | 300行制限のため、文法を`group_grammar.py`、event 1件の型・値域検査を`episode_events.py`へ分けた（Controllerのgroups・rules・progressは依然としてimportしない） |
| `DrawingAction.to_dict/from_dict` | v2は7キー | `annotated`引数で切り替える（v1は5キーのまま） |
| `DrawingEpisode`のimport | `stage.table`を使ってworkflow hashを照合 | `stage`パッケージの初期化が`episode`をimportするため、循環を避けて`check_workflow_identity`内で遅延importする |
| `iter_replay`のframe | 適用のたびにframeを返す | グループ全体を適用した後に、そのグループの各eventのframeを返す。`capture()`できるのはグループ末尾のframeで、それ以外と次へ進んだ後はReplayError |
| `replay`のheader検査 | hash・header不一致を再生前に検出 | workflow識別（id・SHA-256）と画布・seed（Engine構築）を再生前に検査する。brush/controlのhashは`to_dict`/`loads_episode`が検査する |
| `loads_episode(recover=True)` | 末尾の未確定グループを切り詰める | 同じ。確定グループが残らない場合と、グループ以外の破損はEpisodeFormatError |
| クラッシュ注入 | `os.replace`前などに注入 | `os.fsync`・`os.replace`・`_sync_directory`をテストで差し替え、`BaseException`で打ち切るシミュレーション。実プロセスの強制終了は未実施（S4） |
| frames fixture | 合成画素・レイヤーRGBA・stage_id | 12×8の画布。ReplayerではなくControllerのEngine状態から記録する。`.gitattributes`でfixtureの改行変換を無効にする |
| S1aのv2 Episodeテスト | — | v2 headerにworkflow等が必須になったため、`test_drawing_engine_v2.py`の往復テストをController由来のEpisodeに変更 |

## 13. 未決事項

- Stage規則の再監査（保存済みEpisodeの有効筆跡を再判定する）を別機能として作るか。必要になった時点で別Issueにします（S3の評価でEpisodeを検証する需要が生じた場合）。
- `Episode`のArtifactを`ImageSet`と同じ実体ストアで管理するか、path参照だけにするか（S4）。
- 大きな画布での`replay`の速度は、S1aの純Python実装に依存します。実測後に別Issueで扱います。
