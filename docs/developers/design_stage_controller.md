> **対象読者:** StageController・Engine拡張・Policy・テストを実装する開発者
> **目的:** S1b（Issue #67）で実装する`StageController`の公開API、状態遷移、失敗処理、受け入れテスト、PR分割を実装前に確定する。
> **状態:** 実装設計書。S1b（Issue #67）のStageControllerは、本書の「実装手順（PR分割）」どおりに実装済み（ブランチ`claude/67-stage-controller`、マージ前）です。本書の記述との差分は§12に記録しています。正本は[Controller契約](stage_controller_contract_v2.md)、[固定工程](fixed_drawing_workflow_v2.md)、[Stage JSON](drawing_stage_workflow_v2.json)で、本書はこれらを変更しません。

# StageController 実装設計（S1b）

## 1. 目的・範囲・範囲外

目的は、`rough → sketch → final_drawing → base_color → shading → refine → finish`の固定順を強制する`StageController`を、S1aの`Engine`の外側に実装することです。Controllerはモデルが提案した`DrawingAction`を審査し、受理したActionと、工程移行に伴う`actor=system`のイベントを原子的なグループとして記録します。

| 区分 | 内容 |
| --- | --- |
| 範囲 | 工程順序の強制（skip・逆行の拒否）、有効`stroke_v2`の計数、理由付きレビューによる退出、工程別の書込レイヤー権限、固定z順の維持（Engineのrole順を利用）、工程入場時の自動レイヤー作成と非表示、原子的な遷移グループ、Action予算とattempt予算、`StyleIntent`の判定、受理済みイベントからの`DrawingEpisode`生成 |
| 範囲外 | Episode v2のJSON保存・読込・Replayer（S1c、[design_episode_replay.md](design_episode_replay.md)）、Policy・Trainer・Reward（S3）、`DRAW_EPISODE` Step・Runtime統合（S4）、画質評価、GUI |
| 範囲外の保証 | Controllerは**工程の合法性**だけを検査します。絵が上手いこと、`StyleIntent`が実際の作品に合うこと、「修正不要」の判断が正しいことは保証しません。 |

目標画像は、Controller・`StyleIntent`・`Budget`のいずれにも渡しません。Controllerが受け取る入力は、`Engine`、`StyleIntent`（promptだけから決まる）、`Budget`、モデルのActionに限ります。

## 2. 前提（S1a実装の確認結果）

`E:/projects/worktrees/s1a-engine/Image_drawer/image_drawer/drawing/`（未マージ、読取のみ）の確認結果です。

- `Engine.apply()`は検証後に状態を変更し、不正なActionでは状態が変わりません。`stroke_v2`・`erase_v2`・`create_layer(role)`・`select_layer`・`select_color`・`select_brush`・`set_brush_params`・`set_layer_visibility`・`finish`を受理します。
- **`advance_stage`・`enter_stage`・`exit_stage`・`clear_active_layer`は`Engine`が受理しません。** `Engine.apply()`が`seq`を履歴長から決めるため、これらのイベントを記録するには`Engine`の拡張が必要です（§3.3、PR S1b-1）。
- `Engine`にはグループ単位のrollbackがありません。1Actionは原子的ですが、複数イベントの途中失敗は戻せません。
- `DrawingState`に工程IDがありません。`DrawingAction`にも`stage_id`・`group`がありません。
- 上限: 履歴20,000件、レイヤー16枚、`幅×高さ×レイヤー数 ≤ 4,194,304`、累積点数200,000、累積候補面積50,000,000。

## 3. モジュール構成

### 3.1 新規ファイル

既存の`image_drawer/drawing/`に`stage/`サブパッケージを追加します。S1aのモジュールは、§3.3の最小変更を除いて変えません。

| ファイル | 責務 | 目安行数 |
| --- | --- | --- |
| `image_drawer/drawing/stage/__init__.py` | 公開名の再エクスポート | 30 |
| `image_drawer/drawing/stage/errors.py` | 例外型（§8） | 30 |
| `image_drawer/drawing/stage/table.py` | `StageSpec`・`StageTable`。Stage JSONの読込と検証 | 130 |
| `image_drawer/drawing/stage/intent.py` | `StyleIntent`と`judge_style_intent`（版付きの規則判定器） | 110 |
| `image_drawer/drawing/stage/budget.py` | `Budget`の検証と既定値の解決 | 60 |
| `image_drawer/drawing/stage/groups.py` | 初期・遷移・完了グループのイベント列を作る純関数 | 140 |
| `image_drawer/drawing/stage/rules.py` | 提案Actionの審査（拒否コードを返す純関数群） | 220 |
| `image_drawer/drawing/stage/progress.py` | 工程内の有効筆跡の計数（画素差分） | 90 |
| `image_drawer/drawing/stage/controller.py` | `StageController`・`ProposalResult`・`AttemptRecord` | 260 |
| `image_drawer/drawing/data/drawing_stage_workflow_v2.json` | Stage JSONの複製 | 原本と同一 |

`rules.py`は`Engine`の状態を読み取るだけで、変更しません。状態を変更する入口は`controller.py`の`propose`に一本化します。

### 3.2 既存コードとの関係

- `Engine`・`DrawingState`・`DrawingAction`・`DrawingEpisode`・`DrawingError`は`image_drawer.drawing`から使います。
- 新しい例外はすべて`DrawingError`を直接継承します。具象クラスA→B→Cの継承を避けるため、例外同士は継承しません。
- S1aがブラシ・段階表のJSONを`data/`へ複製して原本とバイト一致を検査したのと同じ方法で、Stage JSONも複製します。`StageTable`はこの複製から工程を読み込み、工程ID・role・上限を文字列としてコードに散在させません。
- `image_drawer.evaluation`・`image_drawer.part_bank`・画像ライブラリを`stage/`から**importしません**。目標画像を扱う型に到達できない構造にするためです（テストで固定、§10.9）。

### 3.3 S1aへの最小変更（PR S1b-1）

Controllerの前提となる変更を、先に別PRで入れます。

| 対象 | 変更 |
| --- | --- |
| `models.py` | `STAGE_IDS`（6工程）、`FINISH_STAGE_ID = "finish"`、`REVIEW_REASONS`（3コード）を定義。`EventGroup(id, index, size)`（frozen）を追加。`DrawingAction`へ`stage_id: str \| None = None`と`group: EventGroup \| None = None`を追加。`DrawingState`へ`stage_id: str \| None = None`を追加。`to_dict`/`from_dict`はこのPRでは変えません（S1c-1で拡張） |
| `handlers.py` | v2専用の工程マーカー4種を追加。`enter_stage{stage_id}`（`state.stage_id`を設定）、`exit_stage{stage_id}`（現在の工程と一致を要求。状態は変えない）、`advance_stage{}`または`{review_reason}`（工程設定済みを要求。状態は変えない）、`clear_active_layer{}`（`active_layer_id=None`）。`handle_finish`はv2で`stage_id`が設定済みなら`"finish"`にする |
| `engine.py` | `_dispatch`のv2表へ上記を追加。`apply_group(actions) -> list[DrawingAction]`を追加（§4.4） |

`Engine`は**工程の順序を判定しません**。マーカーの形式と整合（`exit_stage`が現在工程と一致するなど）だけを検査します。順序の強制はController、順序の事後検証はS1cの構造検証が担当します。

## 4. 公開API

### 4.1 StyleIntent

```python
@dataclass(frozen=True)
class StyleIntent:
    prompt: str
    version: str                      # 例: "style-intent-rules-v1"
    monochrome_line_art_only: bool
    flat_style_no_shading: bool

STYLE_INTENT_VERSION: Final[str] = "style-intent-rules-v1"

def judge_style_intent(prompt: str) -> StyleIntent: ...
```

`judge_style_intent`はpromptだけを受け取ります。判定規則は§7.4に示す初期表で、曖昧な場合は`False`にします。規則表を変えるときは`version`を更新し、Episodeのheaderにその版を残します。

### 4.2 Budget

```python
MAX_ACTIONS_LIMIT: Final[int] = 19_900   # Engineの履歴20,000件から、system用の余裕100件を引く

@dataclass(frozen=True)
class Budget:
    max_actions: int
    max_attempts: int | None = None      # 省略時は max(100, 5 * max_actions) へ解決される
```

`max_attempts`は生成後の属性として常に`int`です（`__post_init__`で解決）。検証は§8の`StageConfigError`で行います。

### 4.3 StageController

```python
@dataclass(frozen=True)
class ProposalResult:
    accepted: bool
    events: list[DrawingAction]       # 受理時: modelのActionとsystemイベントの原子的グループ。拒否時: []
    reason: str | None                # 拒否時: "<code>" または "<code>: <detail>"。受理時: None
    attempt: int                      # 1始まり。受理と拒否を通した通し番号

@dataclass(frozen=True)
class AttemptRecord:
    attempt: int
    action: DrawingAction             # 提案時点のコピー
    accepted: bool
    reason: str | None
    stage_id: str                     # 提案を審査した時点の工程
    seqs: tuple[int, ...]             # 受理時に割り当てたseq。拒否時は空

class StageController:
    def __init__(self, engine: Engine, intent: StyleIntent, budget: Budget) -> None: ...

    @property
    def stage_id(self) -> str: ...                  # "rough"…"refine" または "finish"
    @property
    def finished(self) -> bool: ...                 # finishイベントを記録済み
    @property
    def closed(self) -> bool: ...                   # これ以上proposeできない
    @property
    def termination_reason(self) -> str | None: ... # None | "finished" | "action_budget_exhausted"
                                                    #  | "attempt_budget_exhausted" | "internal_error"
    @property
    def accepted_action_count(self) -> int: ...     # 受理したactor=modelのイベント数
    @property
    def attempt_count(self) -> int: ...

    def propose(self, action: DrawingAction) -> ProposalResult: ...
    def episode(self, episode_id: str | None = None) -> DrawingEpisode: ...
    def attempts(self) -> tuple[AttemptRecord, ...]: ...
```

`episode()`は**受理済みイベントだけ**から`DrawingEpisode`を作ります。`finish`前に呼ぶと`status="draft"`、`termination_reason`は`closed`なら閉じた理由、閉じていなければ`"in_progress"`です。`episode_id`を省略すると、`"ep-"` + SHA-256（prompt・canvas・seed・workflow_idのcanonical JSON）の先頭16桁にします。

`DrawingEpisode`には次のフィールドを追加します（既定値付き。シリアライズはS1c）。

```python
workflow_id: str | None = None
workflow_sha256: str | None = None
style_intent: dict[str, Any] | None = None   # {version, monochrome_line_art_only, flat_style_no_shading}
budget: dict[str, int] | None = None         # {max_actions, max_attempts}
status: str = "complete"                     # "complete" | "draft"
termination_reason: str | None = "finished"
```

### 4.4 Engine拡張の公開API（PR S1b-1）

```python
class Engine:
    def apply_group(self, actions: Sequence[DrawingAction]) -> list[DrawingAction]: ...
```

`apply_group`は1〜64件のActionを順に適用し、**全件成功か全件不変**にします。画素を変更するActionの`stroke`・`stroke_v2`・`erase`・`erase_v2`は含められません（含むと`DrawingError`）。このため、巻き戻しに画素の複製は不要で、レイヤー一覧・各レイヤーの`visible`・選択状態・`stage_id`・`finished`・履歴長を保存して復元します。計算量はレイヤー数に比例します。

### 4.5 内部関数（`groups.py` / `rules.py`）

```python
def build_initial_group(table: StageTable, time_ms: int) -> list[DrawingAction]: ...
def build_transition_group(
    table: StageTable, state: DrawingState, advance: DrawingAction
) -> list[DrawingAction]: ...        # 最終工程（refine）では finalize グループを返す
def check_proposal(
    table: StageTable, state: DrawingState, progress: StageProgress,
    intent: StyleIntent, action: DrawingAction,
) -> str | None: ...                 # 拒否コード（"<code>: <detail>"）、受理可能ならNone
```

`check_proposal`は状態を変更せず、同じ入力から必ず同じ結果を返します。

## 5. データ形式

### 5.1 StageTable

`StageTable`は複製したStage JSONから次を読みます。

| 読む項目 | 用途 |
| --- | --- |
| `stages[].id/layer_roles/write_layer_roles/min_effectful_strokes/review_only_reason_codes` | 工程ごとの権限・退出条件 |
| `stage_entry_default_layer_roles` | 入場時に作る既定role（`refine`は`null`） |
| `layer_visibility_transitions` | 入場時に非表示にするrole |
| `allowed_model_actions` / `controller_only_actions` / `forbidden_action_types` | 提案可能な種別 |
| `layer_z_order_bottom_to_top` | Engineのrole順との一致検査 |
| `workflow_id` | `illustration-standard-v2`。Episodeのheaderに記録 |

読込時の検証: 工程IDが`STAGE_IDS`と同順であること、roleが`LAYER_ROLES_Z_ORDER`に含まれること、`layer_z_order_bottom_to_top`が`LAYER_ROLES_Z_ORDER`と一致すること。不一致は`StageTableError`にします。`workflow_sha256`はS1aの`canonical_sha256`で求めます。

### 5.2 レイヤーIDとイベントの注釈

- Controllerが作るレイヤーのIDは`sys_<role>_<連番3桁>`（連番はそのroleのsystem作成数、1始まり）です。例: `sys_rough_001`。
- modelの`create_layer`は、`layer_id`が`sys_`で始まる場合に拒否します。IDの衝突でsystemの`create_layer`が失敗する事態を防ぐためです。
- 記録する全イベントに`stage_id`（そのイベントを適用した**直後**の工程。`enter_stage`は入場先、`finish`は`"finish"`）と`group`（`{id, index, size}`）を付けます。`group.id`は0始まりの連番です。
- 同一グループのイベントは`seq`が連続し、`time_ms`は同じ値です。値は、modelの提案に`time_ms`があればそれ、なければ直前イベントの値（初期グループは0）です。

### 5.3 遷移グループの例（sketch → final_drawing、roughレイヤーが1枚の場合）

```json
[
  {"seq": 40, "time_ms": 900, "actor": "model",  "type": "advance_stage",        "payload": {},
   "stage_id": "sketch",        "group": {"id": 21, "index": 0, "size": 6}},
  {"seq": 41, "time_ms": 900, "actor": "system", "type": "exit_stage",           "payload": {"stage_id": "sketch"},
   "stage_id": "sketch",        "group": {"id": 21, "index": 1, "size": 6}},
  {"seq": 42, "time_ms": 900, "actor": "system", "type": "set_layer_visibility", "payload": {"layer_id": "sys_rough_001", "visible": false},
   "stage_id": "sketch",        "group": {"id": 21, "index": 2, "size": 6}},
  {"seq": 43, "time_ms": 900, "actor": "system", "type": "enter_stage",          "payload": {"stage_id": "final_drawing"},
   "stage_id": "final_drawing", "group": {"id": 21, "index": 3, "size": 6}},
  {"seq": 44, "time_ms": 900, "actor": "system", "type": "create_layer",         "payload": {"layer_id": "sys_final_drawing_001", "name": "本描き", "role": "final_drawing"},
   "stage_id": "final_drawing", "group": {"id": 21, "index": 4, "size": 6}},
  {"seq": 45, "time_ms": 900, "actor": "system", "type": "clear_active_layer",   "payload": {},
   "stage_id": "final_drawing", "group": {"id": 21, "index": 5, "size": 6}}
]
```

### 5.4 グループの種類とイベント列

| 種類 | 発生 | イベント列（順序固定） |
| --- | --- | --- |
| initial | 構築時（group 0） | `enter_stage(rough)`, `create_layer(sys_rough_001)`, `clear_active_layer`（3件） |
| action | modelの通常Action | そのActionの1件 |
| transition | `advance_stage`の受理（`refine`以外） | `advance_stage`(model), `exit_stage(現在)`, `set_layer_visibility(false)`×k（入場先で非表示にするroleの**可視**レイヤーを作成順に1件ずつ。0件もある）, `enter_stage(次)`, `create_layer`（既定roleがある工程のみ）, `clear_active_layer` |
| finalize | `refine`での`advance_stage`の受理 | `advance_stage`(model), `exit_stage(refine)`, `finish` |

| 遷移 | 件数 |
| --- | --- |
| rough → sketch | 5（`create_layer(sketch)`あり） |
| sketch → final_drawing | 5 + k（rough非表示k件） |
| final_drawing → base_color | 5 + k（sketch非表示k件） |
| base_color → shading | 5（`create_layer(shade)`） |
| shading → refine | 4（既定レイヤーなし） |
| refine → finish | 3 |

systemイベントの総数の上限は、初期3 + 遷移 + 非表示（最大でレイヤー枚数分）で約50件です。`MAX_ACTIONS_LIMIT`の余裕100件に収まります。

### 5.5 ProposalResultの例

```python
# 受理（工程移行）
ProposalResult(accepted=True, events=[<上記6件>], reason=None, attempt=57)
# 拒否（筆跡なしで基本色から退出しようとした）
ProposalResult(accepted=False, events=[], reason="stage_not_ready: no effectful stroke in base_color", attempt=58)
```

## 6. 状態遷移表

### 6.1 工程の遷移

`advance_stage`の移行先は常に次の工程で、モデルは選べません。通常の退出は`payload={}`、レビュー退出は`payload={"review_reason": <code>}`です。

| 現在 | 通常退出の条件 | レビュー退出の条件 | 次 | グループ |
| --- | --- | --- | --- | --- |
| rough | 有効`stroke_v2` ≥ 1、かつ`rough`roleの可視レイヤーにalpha > 0の画素が残る | 不可（許可コードなし） | sketch | transition |
| sketch | 同上（`sketch`role） | 不可 | final_drawing | transition |
| final_drawing | 同上（`final_drawing`role） | 不可 | base_color | transition |
| base_color | 有効`stroke_v2` ≥ 1（`base_color`role） | `monochrome_line_art_only`かつ`intent.monochrome_line_art_only`かつ有効筆跡/消去が0 | shading | transition |
| shading | 有効`stroke_v2` ≥ 1（`shade`または`highlight`role） | `flat_style_no_shading`かつ`intent.flat_style_no_shading`かつ有効筆跡/消去が0 | refine | transition |
| refine | 有効`stroke_v2`または有効`erase_v2` ≥ 1 | `no_correction_needed`かつ有効筆跡/消去が0 | finish | finalize |
| finish | — | — | （終端） | — |

「有効」は、受理された`stroke_v2`（消去は`erase_v2`）が、書込先レイヤーのRGBAを実際に変えたことを指します。同じRGBAのままの描画は受理しますが数えません。計数は工程入場時に0へ戻します。

### 6.2 提案Actionの許可（工程別）

| Action | rough | sketch | final_drawing | base_color | shading | refine |
| --- | --- | --- | --- | --- | --- | --- |
| `create_layer`の許可role | rough | sketch | final_drawing | base_color | shade, highlight | correction |
| `select_layer`の対象role（可視のみ） | rough | sketch | final_drawing | base_color | shade, highlight | base_color, shade, final_drawing, highlight, correction |
| `stroke_v2`/`erase_v2`の書込先role | 選択中レイヤーが左の`select_layer`の対象roleであること（全工程共通の規則） |||||
| `select_color`/`select_brush`/`set_brush_params` | 全工程で可（Engineが値を検査） |||||
| `advance_stage` | 上表 |||||
| 上記以外（`skip_stage`、`set_phase`、`finish`、`enter_stage`、`exit_stage`、`set_layer_visibility`、`clear_active_layer`、`stroke`、`erase`、未知の種別） | 全工程で`action_not_allowed` |||||

### 6.3 Controller全体の状態

| 状態 | 条件 | `propose`の動作 |
| --- | --- | --- |
| open | 閉じていない | 審査して受理または拒否 |
| closed(finished) | `finish`を記録した | `StageClosedError` |
| closed(action_budget_exhausted) | 受理した`actor=model`のイベント数が`max_actions`に達し、未完成 | `StageClosedError` |
| closed(attempt_budget_exhausted) | 試行数が`max_attempts`に達し、未完成 | `StageClosedError` |
| closed(internal_error) | 遷移グループの適用が失敗しrollbackした（不変条件違反） | `StageClosedError` |

閉じる判定は`propose`の最後に、`finished` → `action_budget_exhausted` → `attempt_budget_exhausted`の優先順で行います。したがって、最後の許可されたActionまたは試行で`finish`に至った場合は、完成として閉じます。予算切れで残りの工程を自動で進めたり、`finish`を発行したりしません。

## 7. 審査の手順と拒否コード

### 7.1 `propose`の手順

[契約§3](stage_controller_contract_v2.md)の順序に対応させます。

1. **閉じていないことを確認**する。閉じていれば`StageClosedError`（attemptは数えない）。
2. **試行番号を割り当てる**（`attempt_count += 1`）。
3. **基本の正当性**（`invalid_envelope`）: `DrawingAction`であること、`type`が`str`、`payload`が`dict`、`actor == "model"`、`seq is None`、`stage_id is None`、`group is None`。
4. **提案の複製**: `payload`を`copy.deepcopy`する。以後の審査・記録は複製を使い、呼び出し側が後から`payload`を変更してもEpisodeが変わらないようにする。
5. **工程による権限**（`check_proposal`）: §6.2の許可、role、可視性、予約ID、レイヤー余裕、`advance_stage`の条件。
6. **実行**: 通常Actionは`engine.apply`、`advance_stage`は`build_transition_group`の結果を`engine.apply_group`で適用する。通常Actionが`stroke_v2`/`erase_v2`のときは、適用前に書込先レイヤーの画素を`bytes`で保持し、適用後に比較して有効筆跡を判定する。
7. **記録**: 注釈（`stage_id`・`group`）付きのイベントを`_events`へ追加し、計数・`group.id`・受理数を更新する。
8. **閉じる判定**（§6.3）。

### 7.2 拒否コード

`reason`の形式は`"<code>"`または`"<code>: <detail>"`です。コードは安定した識別子で、ログと集計に使います。

| コード | 発生条件 |
| --- | --- |
| `invalid_envelope` | §7.1の3に違反 |
| `action_not_allowed` | `allowed_model_actions`にない種別（`skip_stage`・`set_phase`・`finish`・Controller専用種別・v1種別・未知の種別） |
| `invalid_payload` | `advance_stage`のpayloadが`{}`でも`{"review_reason": str}`でもない（移行先を指すキーを含む） |
| `reserved_layer_id` | modelの`create_layer`の`layer_id`が`sys_`で始まる |
| `role_not_creatable` | `create_layer`のroleが現在工程の`layer_roles`にない |
| `layer_capacity` | 作成すると、これから工程入場で作るsystemレイヤーの分を確保できない（枚数16・面積上限） |
| `layer_not_selectable` | `select_layer`の対象が非表示、またはroleが書込可能でない |
| `layer_not_writable` | `stroke_v2`/`erase_v2`の書込先（選択中レイヤー）のroleが書込可能でない |
| `stage_not_ready` | 通常退出の条件（§6.1）を満たさない |
| `review_reason_not_allowed` | レビュー理由が現在工程の許可コードにない、または文字列でない |
| `review_after_drawing` | 現在工程で有効な`stroke_v2`/`erase_v2`が1件以上あるのにレビュー退出を提案した |
| `review_intent_not_confirmed` | `StyleIntent`が理由を裏付けない（`base_color`/`shading`） |
| `engine_rejected` | Engineが拒否（未選択・色未選択・ブラシmode不一致・予算超過など）。`detail`にEngineのメッセージ |

### 7.3 レビュー退出の補足

- `refine`の`payload={}`は、有効な`stroke_v2`/`erase_v2`が0件なら`stage_not_ready`です。修正しない場合は`no_correction_needed`を使います（Stage JSONの`stage_exit_approval.normal_refine`に従う）。
- `review_after_drawing`は、**画素を変えた**筆跡/消去だけを根拠にします（Stage JSONの`no_prior_effectful_stroke_or_erase_in_current_stage`）。画素が変わらなかった描画はレビューを妨げません。
- 理由コードの形式違反（`null`・数値・空文字）は`review_reason_not_allowed`です。

### 7.4 StyleIntentの初期規則（`style-intent-rules-v1`）

promptをUnicode NFKCに正規化し、`casefold`し、空白と句読点（`、。,.!！?？・「」『』()（）`）を除去してから、部分文字列で判定します。

| フラグ | `True`にする肯定表現（いずれか） | `False`に倒す否定手がかり（いずれか） |
| --- | --- | --- |
| `monochrome_line_art_only` | 白黒の線画 / モノクロの線画 / モノクロ線画 / 白黒線画 / 線画のみ / 線画だけ / blackandwhitelineart / monochromelineart / lineartonly | カラー / 着色 / 彩色 / 色付き / フルカラー / 水彩 / fullcolor / colorful / colored |
| `flat_style_no_shading` | 影なし / 影無し / 影をつけない / 影を付けない / 影を入れない / 陰影なし / 陰影をつけない / 陰影を付けない / ノーシェーディング / ベタ塗り / フラットカラー / フラットな塗り / noshading / withoutshading / flatcolor | 立体感 / グラデーション / ハイライト / 厚塗り / リアル |

肯定表現が一つもない、または否定手がかりが一つでもあれば`False`です。空のprompt、非文字列（`StageConfigError`）を除き、任意の文字列を受け付けます。これは意味解析ではなく保守的な規則です。取りこぼしは`False`側（レビュー拒否）に倒れるため、Policyが描く方向の誤りになり、不正な通過にはなりません。

## 8. 失敗時の動作と例外型

| 例外 | 継承 | 発生 | Controller・Engineの状態 |
| --- | --- | --- | --- |
| `StageConfigError` | `DrawingError` | `Budget`の値不正、`StyleIntent`でない引数、`Engine`が`brush-raster-v2`でない/使用済み、画布が必須5レイヤーを保持できない（`幅×高さ×5 > 4,194,304`。正方形では915×915まで）、非文字列のprompt | 構築されない |
| `StageTableError` | `DrawingError` | Stage JSONの不整合 | 構築されない |
| `StageClosedError` | `DrawingError` | 閉じたControllerへの`propose` | 変化なし |
| `StageInvariantError` | `DrawingError` | 審査済みの遷移グループが`apply_group`で失敗（実装の不具合）。Engineはrollback済み。attemptを`internal_error`で記録し、Controllerを閉じる | 直前の確定グループの状態 |

通常の不正提案は**例外にせず**`ProposalResult(accepted=False)`で返します。拒否ではEngine・`_events`・`seq`・`group.id`・工程内計数が変化せず、`attempts()`にだけ残ります。呼び出し側は`closed`を見てループを終了します。

`episode()`は閉じる前でも呼べます（draft）。

## 9. 決定性の要件

- 乱数・時刻・環境変数・辞書/集合の走査順に依存しません（レイヤー走査は作成順、role走査は`StageTable`の宣言順）。
- 同じ`Engine`初期条件・`StyleIntent`・`Budget`・同じ提案列から、同じイベント列（`seq`・`time_ms`・`stage_id`・`group`を含む）と同じ画素を得ます。
- systemレイヤーのIDと名前は、roleと連番だけから決まります（名前は`StageSpec`の日本語名: ラフ・下書き・本描き・基本色・陰影（`shade`）・修正）。
- 提案の`payload`は受理前に複製し、受理後の呼び出し側の変更が記録へ波及しません。
- 有効筆跡の判定は画素のバイト比較で、近似を使いません。
- Controllerの出力は`Engine`の決定性（S1a）に依存します。S1aのgoldenが変わる変更（式の変更）は、本書の受け入れテストを再実行して確認します。

## 10. 受け入れテスト一覧

テストは`tests/test_stage_*.py`と`tests/test_drawing_engine_stage_markers.py`に置きます。製品コードのControllerを通して検証し、テスト内にControllerを再実装しません。テスト用のスクリプト化したActionは、modelの提案の代わりに使う入力です。

### 10.1 Engine拡張（`test_drawing_engine_stage_markers.py`）

- `test_enter_stage_sets_stage_id`
- `test_enter_stage_rejects_unknown_stage_id`
- `test_exit_stage_requires_current_stage`
- `test_exit_stage_without_entered_stage_is_rejected`
- `test_advance_stage_accepts_empty_and_review_payload`
- `test_advance_stage_rejects_unknown_review_reason_and_extra_keys`
- `test_advance_stage_requires_entered_stage`
- `test_clear_active_layer_unselects_and_keeps_pixels`
- `test_stage_markers_do_not_change_any_pixel`
- `test_stage_markers_are_rejected_in_raster_v1_profile`
- `test_finish_sets_stage_id_finish_in_v2`
- `test_apply_group_commits_all_events_with_consecutive_seq`
- `test_apply_group_rolls_back_layers_visibility_selection_and_stage_on_failure`
- `test_apply_group_rejects_painting_actions`
- `test_apply_group_rejects_empty_and_oversized_groups`
- `test_apply_group_respects_history_limit_boundary`
- `test_v1_golden_still_matches_after_engine_extension`

### 10.2 StageTable・StyleIntent・Budget（`test_stage_table.py`、`test_stage_intent.py`、`test_stage_budget.py`）

- `test_stage_table_has_six_stages_in_fixed_order`
- `test_stage_table_package_copy_is_byte_identical_to_docs_json`
- `test_stage_table_rejects_reordered_stages`
- `test_stage_table_rejects_role_missing_from_engine_z_order`
- `test_stage_table_workflow_sha256_uses_canonical_json`
- `test_style_intent_monochrome_positive_phrases`（表の全肯定表現）
- `test_style_intent_flat_positive_phrases`
- `test_style_intent_negative_cue_overrides_positive_phrase`
- `test_style_intent_ambiguous_prompt_is_false`（「線画を描いて」「絵を描いて」）
- `test_style_intent_empty_prompt_is_false_for_both_flags`
- `test_style_intent_normalizes_width_case_and_whitespace`（全角・大文字・空白）
- `test_style_intent_records_version_and_prompt`
- `test_style_intent_signature_takes_only_prompt`
- `test_style_intent_rejects_non_string_prompt`
- `test_budget_default_attempt_limit_floor_is_100`（`max_actions=10`→100）
- `test_budget_default_attempt_limit_is_five_times_actions`（`max_actions=1000`→5000、境界`max_actions=20`→100）
- `test_budget_explicit_attempt_limit_is_kept`
- `test_budget_rejects_zero_negative_bool_float_and_none_actions`
- `test_budget_accepts_boundary_max_actions_19900`
- `test_budget_rejects_max_actions_above_engine_limit`
- `test_budget_rejects_attempt_limit_below_action_limit`

### 10.3 構築と初期状態（`test_stage_controller_setup.py`）

- `test_controller_starts_in_rough_with_unselected_system_layer`
- `test_initial_group_is_enter_create_clear_with_group_zero`
- `test_initial_events_are_system_actor_and_time_zero`
- `test_controller_rejects_raster_v1_engine`
- `test_controller_rejects_engine_with_history`
- `test_controller_rejects_canvas_too_large_for_five_layers`
- `test_controller_accepts_canvas_at_five_layer_area_boundary`（915×915）
- `test_controller_rejects_non_intent_argument`
- `test_controller_constructor_has_no_target_image_parameter`（`inspect.signature`）

### 10.4 工程順序・原子性（`test_stage_controller_order.py`）

- `test_full_run_passes_seven_stages_in_fixed_order`
- `test_full_run_stage_id_property_follows_each_transition`
- `test_advance_without_stroke_is_rejected_in_rough_sketch_final_drawing`（3工程のparametrize）
- `test_advance_after_color_selection_only_is_rejected`
- `test_advance_after_noop_stroke_is_rejected_when_stroke_required`
- `test_advance_after_erasing_all_pixels_is_rejected_in_rough`
- `test_erase_only_does_not_satisfy_base_color`
- `test_effective_stroke_count_resets_on_stage_entry`
- `test_advance_payload_with_destination_is_rejected`
- `test_skip_stage_and_set_phase_are_action_not_allowed`
- `test_model_cannot_propose_controller_only_actions`（`finish`・`enter_stage`・`exit_stage`・`set_layer_visibility`・`clear_active_layer`のparametrize）
- `test_no_backward_transition_is_possible`（`enter_stage(rough)`提案の拒否と状態不変）
- `test_finish_event_is_emitted_exactly_once_and_last`
- `test_propose_after_finish_raises_stage_closed_error`
- `test_transition_events_form_one_group_with_consecutive_seq`
- `test_transition_group_has_exact_event_order_for_each_stage`（6遷移のparametrize、§5.4）
- `test_group_index_and_size_match_for_every_group`
- `test_group_ids_start_at_zero_and_increase_by_one`
- `test_group_events_share_time_ms`
- `test_event_stage_id_equals_engine_stage_id_after_apply`
- `test_active_layer_is_cleared_on_every_transition`
- `test_brush_color_size_and_opacity_are_preserved_across_transition`
- `test_stroke_without_reselecting_layer_after_transition_is_rejected`
- `test_injected_apply_group_failure_rolls_back_and_closes_controller`（`StageInvariantError`、Engine状態・`episode()`が直前の確定グループと同一）
- `test_rejected_proposal_does_not_consume_seq_or_group_id`
- `test_rejected_proposal_leaves_engine_state_identical`（`to_dict(include_pixels=True)`の比較）
- `test_result_events_are_the_recorded_events`
- `test_proposal_payload_is_copied_on_accept`（受理後に呼び出し側がpayloadを変更してもEpisodeが不変）
- `test_proposal_with_seq_group_or_stage_id_is_invalid_envelope`
- `test_proposal_with_non_model_actor_is_invalid_envelope`
- `test_non_action_object_and_non_dict_payload_are_invalid_envelope`

### 10.5 レイヤー権限・重ね順（`test_stage_controller_layers.py`）

- `test_select_layer_of_other_stage_role_is_rejected`
- `test_select_hidden_layer_is_rejected`
- `test_select_unknown_layer_is_engine_rejected`
- `test_create_layer_role_must_belong_to_current_stage`（6工程のparametrize、許可・不許可の双方）
- `test_shading_allows_shade_and_highlight_layers`
- `test_refine_create_layer_allows_only_correction`
- `test_refine_can_select_visible_prior_roles`
- `test_refine_cannot_select_hidden_rough_or_sketch`
- `test_stroke_on_layer_with_unwritable_role_is_rejected`（防御的検査。状態を直接設定して確認）
- `test_model_layer_id_with_sys_prefix_is_rejected`
- `test_model_layer_id_cannot_collide_with_future_system_layer`
- `test_create_layer_that_starves_mandatory_layers_is_rejected`（枚数16の境界）
- `test_create_layer_that_exceeds_area_budget_is_rejected`
- `test_system_layer_ids_follow_role_and_sequence`
- `test_rough_layers_are_hidden_on_entering_final_drawing`
- `test_sketch_layers_are_hidden_on_entering_base_color`
- `test_every_visible_layer_of_hidden_role_gets_one_visibility_event`（複数枚）
- `test_already_hidden_layers_get_no_visibility_event`
- `test_hidden_layers_are_not_deleted`
- `test_base_color_layer_does_not_cover_final_drawing_line_in_render`
- `test_shade_renders_below_and_highlight_above_final_drawing`
- `test_correction_layer_renders_on_top`
- `test_same_role_layers_stack_in_creation_order`

### 10.6 レビュー退出（`test_stage_controller_review.py`）

- `test_review_base_color_is_accepted_when_monochrome_intent_true`
- `test_review_base_color_is_rejected_when_intent_false`
- `test_review_shading_is_accepted_when_flat_intent_true`
- `test_review_shading_is_rejected_when_intent_false`
- `test_review_refine_no_correction_needed_is_accepted`
- `test_review_reason_of_other_stage_is_rejected`（`base_color`で`flat_style_no_shading`など）
- `test_review_is_rejected_in_rough_sketch_final_drawing`
- `test_review_is_rejected_after_effective_stroke_in_same_stage`
- `test_review_is_rejected_after_effective_erase_in_same_stage`
- `test_review_after_noop_stroke_is_accepted`
- `test_review_with_unknown_null_or_non_string_reason_is_rejected`
- `test_review_still_records_exit_and_enter_events`
- `test_review_exit_creates_next_stage_default_layer`
- `test_refine_empty_normal_advance_is_stage_not_ready`
- `test_refine_advance_after_effective_erase_is_accepted`
- `test_intent_true_does_not_waive_effective_stroke_requirement_in_rough`
- `test_all_three_reviews_in_one_run_reach_finish`（モノクロ線画・影なしのprompt）

### 10.7 予算とattempt（`test_stage_controller_budget.py`）

- `test_attempt_numbers_count_accepted_and_rejected_proposals`
- `test_attempts_log_keeps_reason_stage_and_seqs`
- `test_rejected_attempts_do_not_count_as_accepted_actions`
- `test_system_events_do_not_consume_action_budget`
- `test_action_budget_exhaustion_closes_controller_without_auto_finish`
- `test_action_budget_exhaustion_does_not_advance_stage`
- `test_attempt_limit_closes_controller_with_draft_episode`
- `test_attempt_limit_exact_boundary_last_attempt_is_still_judged`
- `test_final_advance_at_last_allowed_action_is_complete_not_draft`
- `test_final_advance_at_last_allowed_attempt_is_complete_not_draft`
- `test_propose_after_budget_exhaustion_raises_stage_closed_error`
- `test_infinite_invalid_proposals_terminate_at_attempt_limit`
- `test_default_attempt_limit_for_max_actions_1000_is_5000`
- `test_termination_reason_priority_finished_over_action_over_attempt`

### 10.8 Episode生成（`test_stage_controller_episode.py`）

- `test_episode_contains_only_accepted_events`
- `test_episode_origin_is_mixed_and_seq_is_contiguous`
- `test_episode_of_finished_run_has_status_complete_and_finished_reason`
- `test_episode_of_open_run_is_draft_in_progress`
- `test_episode_of_exhausted_run_records_exhaustion_reason`
- `test_episode_records_style_intent_budget_and_workflow_ids`
- `test_episode_default_id_is_stable_for_same_inputs`
- `test_episode_events_are_independent_copies`
- `test_episode_events_replay_to_same_pixels_on_fresh_engine`（`Engine.apply`の再適用。S1cのReplayer導入前の暫定確認）

### 10.9 決定性・目標画像の隔離（`test_stage_controller_determinism.py`、`test_stage_isolation.py`）

- `test_same_proposals_give_identical_events_and_pixels`
- `test_controller_does_not_use_time_or_random`（`time`・`random`・`os.urandom`を例外送出に差し替えて全工程を実行）
- `test_controller_result_is_independent_of_pythonhashseed`（異なる`PYTHONHASHSEED`のサブプロセスでイベント列のSHA-256が一致）
- `test_stage_package_does_not_import_evaluation_or_image_modules`（`stage/`のimportグラフに`image_drawer.evaluation`・`image_drawer.part_bank`・PIL・numpyがない）
- `test_stage_public_signatures_have_no_target_image_parameter`（`StageController`・`StyleIntent`・`Budget`・`propose`）
- `test_style_intent_result_depends_only_on_prompt`

## 11. 実装手順（PR分割）

1PR=1目的、目安500行以下（テストを含む）です。順に依存します。

| PR | 目的 | 内容 | 目安 |
| --- | --- | --- | --- |
| S1b-1 | Engineの工程マーカーとグループ適用 | §3.3、§10.1。`DrawingAction`/`DrawingState`の追加フィールド | 450 |
| S1b-2 | `StageTable`とStage JSONの複製 | `table.py`、`errors.py`、`data/`、§10.2のtable | 250 |
| S1b-3 | `StyleIntent`と`Budget` | `intent.py`、`budget.py`、§10.2のintent・budget | 350 |
| S1b-4 | グループ生成と提案審査（純関数） | `groups.py`、`rules.py`、`progress.py`。Controllerなしで§7.2の拒否コードを検証 | 480 |
| S1b-5 | `StageController`の本体 | `controller.py`、`DrawingEpisode`の追加フィールド、§10.3・§10.4・§10.5・§10.6 | 500 |
| S1b-6 | 予算・Episode生成・決定性・隔離 | §10.7・§10.8・§10.9、文書更新（`README`・[契約](stage_controller_contract_v2.md)の「現在できたもの」） | 400 |

S1b-5のテストが500行を超える場合は、§10.5と§10.6を独立PRにします。各PRで`python scripts/check_local.py`を実行し、整形（Ruff）を適用します。S1b-1はS1aのマージ後に着手します。S1c-1（[Replayer設計書](design_episode_replay.md)）はS1b-1の後に着手できます。

## 12. 既存設計書との差分と判断

正本は変更していません。次は実装で従う判断です。

| 項目 | 正本の記述 | 判断 |
| --- | --- | --- |
| 工程マーカーのEngine対応 | 契約§5は`enter_stage`などの記録を求めるが、S1a Engineは受理しない | S1b-1で`Engine`を最小拡張する。工程の順序判定は`Engine`に持たせない |
| 最大Action数 | Stage JSONの`max_actions_hard_limit_current_workflow`は100000 | Engineの履歴上限20,000件に合わせ、`MAX_ACTIONS_LIMIT=19_900`とする。JSONの値は変更しない。到達できない上限は設計書上の目安として扱う |
| 初期状態の履歴 | 契約§2は「空の行動履歴」かつ「工程入場を履歴へ記録」 | 構築時にgroup 0（`enter_stage`・`create_layer`・`clear_active_layer`）を記録する。modelのActionは0件のまま |
| レビュー退出の条件 | 契約§4は「筆跡を一本も描かなかった場合」、JSONは「effectful」 | 契約末尾の規則に従いJSON（有効な筆跡/消去が0件）を採用する |
| 画布の大きさ | 記載なし | 必須5レイヤー（rough・sketch・final_drawing・base_color・shade）を保持できない画布（`幅×高さ×5 > 4,194,304`）を構築時に拒否する。1024×1024では工程が完走できないため |
| `finish`後の`stage_id` | JSONは`terminal: "finish"` | `finish`後の`stage_id`を`"finish"`とする |
| 不正提案の例外 | 契約§3は拒否の記録のみを要求 | 通常の拒否は戻り値、閉じたControllerへの提案だけ例外とする |
| 工程名`shading`のrole | 契約表は「陰影・仕上げ: `shade`」 | 工程IDは`shading`、既定レイヤーのroleは`shade`、追加できるroleは`shade`と`highlight` |
| `refine`の`create_layer` | 契約§2は「新しいレイヤー不要」、表は`correction`を追加可 | 入場時は作成せず、model提案の`correction`だけ許可する |
| モジュール分割 | §3.1は`groups.py`・`rules.py`・`controller.py`など | 300行超のファイルを避けるため、`advance_stage`の審査を`exit_rules.py`、提案の基本検査と複製を`envelope.py`、Episode組立を`episode_build.py`へ分離した。公開APIは不変 |
| systemレイヤーの名前 | roleごとの日本語名（§9） | `table.py`の`ROLE_LAYER_NAMES`で定義（ラフ・下書き・本描き・基本色・陰影。`highlight`は「ハイライト」、`correction`は「修正」） |
| Actionでない提案の試行ログ | `AttemptRecord.action`は提案時点のコピー | `DrawingAction`でない値、またはコピーできない提案は`type="<invalid>"`の代用Actionで記録する |
| 不正な`time_ms` | 記載なし | model提案の`time_ms`が範囲外・逆行のときは、Controllerが事前に検査して`engine_rejected`で拒否する（遷移グループの適用失敗＝`StageInvariantError`と区別するため） |

## 13. 未決事項

- Policyが観測する「現在の合法Action種別」「現在工程」「残り予算」を返す読取専用ビューの形（S3）。`stage_id`・`closed`・`accepted_action_count`・`attempt_count`はその材料として公開済みの案です。
- 有効筆跡の判定は、筆跡ごとに書込先レイヤーの全画素を比較します（O(幅×高さ)）。大きな画布で遅い場合は、Engineが変更領域を返す拡張を別Issueで検討します。計測するまで最適化しません。
- `StyleIntent`の規則表の拡充と多言語対応は、`version`を更新して別PRで行います。
