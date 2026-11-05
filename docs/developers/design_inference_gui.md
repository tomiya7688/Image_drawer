> **対象読者:** txt2img推論（Step・CLI・GUI）を実装する開発者、レビュー担当
>
> **目的:** Issue #71（S4）の実装設計を定める。文章だけを入力し、AIが7工程を自律して描く推論と、その過程を確認するGUIを、既存のRuntime・DSL・CLIの仕組みに乗せて作る手順を示す。
>
> **状態:** 実装済み（Issue #71）。`DRAW_EPISODE`、`REPLAY_EPISODE`、`image-drawer-draw`、GUIの3画面（Replay・生成・Dataset）が`image_drawer/inference/`、`steps/draw_episode.py`、`gui/`にある。確認したのは**水準A（配線）まで**で、モデルが絵を描けること、学習できることはこの文書では示さない。画面の操作・見た目は自動テストでは未確認。実装との差分は9節の7〜12。

# txt2img推論・GUI・Workflow連携の実装設計（S4）

## 1. 範囲と前提

### 1.1 この文書が決めること

- Workflow Step `DRAW_EPISODE`、`REPLAY_EPISODE`、Artifact型`Episode`、Trajectoryへの保存方針
- CLI `image-drawer-draw`
- GUIの段階的な設計（Replay表示、生成、Dataset登録と抽出器の選択）
- 受け入れテストの一覧と、PRの分け方

### 1.2 この文書が決めないこと

| 項目 | 決める文書 |
| --- | --- |
| `DrawingEngine`、Action、Replayerの内部 | S1a/S1c（`image_drawer/drawing/`、未マージ） |
| `StageController`の工程規則 | [Controller契約](stage_controller_contract_v2.md)、S1b |
| `DrawingPolicy`の内部、`RewardEvaluator`、`Trainer`、checkpoint形式 | `design_trainer.md`（別の設計書） |
| 評価指標と「高精度」と公表できる条件 | [評価プロトコル](design_evaluation_protocol.md)（S5） |
| 抽出器の統合 | [データ前処理の統合](design_data_pipeline_integration.md) |

### 1.3 固定する共通契約

他の設計書と名前を合わせるため、次の名前と形を変えません。

| 契約 | 形 |
| --- | --- |
| 工程Controller | `StageController(engine, intent: StyleIntent, budget: Budget)` |
| 提案の審査 | `propose(action) -> ProposalResult(accepted, events, reason, attempt)` |
| Episodeの取り出し | `episode() -> DrawingEpisode` |
| 再生 | `replay(episode) -> ReplayResult(image, layers, state)` |
| Dataset | `iter_samples(dataset_dir, split) -> Iterator[(PolicyInput(image_id, prompt), RewardTarget(image_id, image_path, image_sha256))]` |
| 学習側 | `DrawingPolicy`（目標画像を受け取らない）、`RewardEvaluator`、`Trainer` |
| 7工程 | `rough, sketch, final_drawing, base_color, shading, refine, finish` |

この文書の推論は、`DrawingPolicy`が返したActionを`StageController.propose`へ渡すことだけでキャンバスを変更します。Engineや`DrawingState`を直接書き換えません。

### 1.4 現在の状態

| 部品 | 状態（2026-11-05時点） |
| --- | --- |
| `DrawingEngine`、`DrawingEpisode`（`image_drawer/drawing/`） | 実装済み（S1a） |
| `StageController`、Replayer | 実装済み（S1b、S1c） |
| `DrawingPolicy`、Trainer、checkpoint | 実装済み（S3）。学習の成功は未確認 |
| `DRAW_EPISODE`、`REPLAY_EPISODE`、`image-drawer-draw` | 実装済み（水準Aまで確認） |
| GUI | Replay・生成・Dataset画面を実装。`--check`は従来どおりheadlessで`{"component": "gui", "status": "ok"}`を出力する。クリック・ドラッグ・見た目は未確認 |
| Step登録の仕組み（`StepSchema`、`StepResult`、`StepRegistry`、`output_resolver`） | 実装済み。`image_drawer/steps/` |
| Runtime（追加Artifactの記録、系譜の検証） | 実装済み。`image_drawer/runtime/engine.py` |

## 2. 推論の流れ

### 2.1 一回の生成

```text
prompt（文章）, seed, max_actions, canvas
   │
   ├─ PolicyRegistry.resolve(policy) ──→ LoadedPolicy（目標画像を受け取らない）
   ├─ StyleIntent判定（prompt → 「線画のみ」「影なし」など。S1bが定義）
   └─ Engine(width, height, "brush-raster-v2", seed)
         │
         └─ StageController(engine, intent, Budget(max_actions, attempts=max(100, 5×max_actions)))
               ▲                                │
               │ ProposalResult                 │ episode()
   DrawingPolicy ←─ PolicyObservation           ▼
   （prompt＋自分の状態だけ）           DrawingEpisode ──→ Replayer ──→ Image
```

推論ループは`image_drawer/inference/session.py`の`generate_episode`が持ちます。擬似コードは次のとおりです（シグネチャは設計案）。

```python
def generate_episode(request: DrawRequest, *, cancel: Event | None = None) -> DrawOutcome:
    loaded = policy_registry.resolve(request.policy)            # 未登録は例外（CLIの終了コード2）
    engine = Engine(request.width, request.height, RENDERER_V2, seed=request.seed)
    controller = StageController(engine, judge_style_intent(request.prompt), budget_for(request))
    last: ProposalResult | None = None
    attempts: list[AttemptRecord] = []
    while True:
        if cancel is not None and cancel.is_set():
            return finish_as_draft("cancelled")
        observation = build_observation(controller, request.prompt, last)   # 学習と同じ構築関数
        action = loaded.policy.next_action(observation)                      # 仮名。S3が確定する
        last = controller.propose(action)
        attempts.append(AttemptRecord.from_result(action, last))
        if any(event.type == "finish" for event in last.events):
            return finish_as_complete(controller.episode())
        if last.reason in TERMINAL_DRAFT_REASONS:                            # 予算切れなど
            return finish_as_draft(last.reason)
```

- **観測の構築は学習と共通**にします。`build_observation`は`design_trainer.md`が定義する関数を呼び、`prompt`、現在の`stage_id`、**自分の**合成キャンバス・レイヤー・受理済みAction履歴、直前の`ProposalResult`だけを含めます。目標画像、Datasetのパス、画像のembeddingは含めません。
- **終了判定は`ProposalResult`だけ**で行います。`events`に`finish`があれば完成です。`budget_exhausted`、`attempt_limit_exceeded`などの理由が返れば未完成のDraftです。これらの理由コードは契約に未定義であり、S1bで確定します（未決事項U1）。
- 受理されなかった提案は`attempts.json`へ記録し、Episodeの`seq`を消費しません（[Controller契約](stage_controller_contract_v2.md)の3節）。
- 時刻は`DrawingAction.time_ms`をEngineが決定的に割り当てるため、実時間を保存しません。同じprompt・seed・checkpointなら、決定的な推論モードでAction列と画素が一致することを受け入れ条件にします。

### 2.2 結果の種類

| `status` | 条件 | 保存するもの |
| --- | --- | --- |
| `complete` | Controllerが`finish`を確定 | `episode.json`、`image.png`、`attempts.json` |
| `draft` | 予算切れ、試行上限、`cancelled`、`policy_error`、`event_limit_exceeded` | `draft.json`、`image.png`（その時点の合成画像）、`attempts.json` |
| 実行不能 | policy未登録、canvasの不一致、出力先の衝突 | なし（CLIは終了コード2） |

Draftを**完成Episodeとして保存・出力しません**。`draft.json`は`episode.json`と別名にし、次の形にします。

```json
{
  "schema_version": "idraw-draft/v1",
  "reason": "budget_exhausted",
  "stage_id": "shading",
  "episode_prefix": { "...": "Episodeと同じキー。eventsは受理済みActionの接頭辞でfinishを含まない" }
}
```

`DrawingEpisode.from_dict`は終端の`finish`を要求するため、Draftは通常のEpisodeとして読み込めません。GUIはDraftを`Engine.apply_event`で直接再生して表示します（5.4節）。Replayerがfinishなしの接頭辞を受け付けるかはS1cの設計に依存します（未決事項U2）。

`max_actions`は`finish`を除く受理済みモデルActionの上限です。Engineの`MAX_EVENTS`は20000件で、Controllerの`actor=system`イベントも含みます。このため、`max_actions=19999`では`event_limit_exceeded`のDraftになり得ます。Stepは`max_actions + SYSTEM_EVENT_RESERVE(64)`が20000を超える値を拒否せず、実行時にDraftとして扱います（正本の上限1..19999は変えません）。

## 3. Workflow Step

### 3.1 Artifact型

| 型 | 内容 | 保存先 |
| --- | --- | --- |
| `Text` | prompt | 既存。`INPUT`のArtifact |
| `Episode`（新設） | 完成した`DrawingEpisode`への参照 | `uri`が`episode.json`、`metadata`に要約 |
| `Image` | 完成画像 | `uri`が`image.png` |

Artifact型の名前は`Episode`です。[Workflow DSL](workflow_dsl.md)は`DrawingEpisode`、[設計監査](design_audit.md)は`Episode`と書いており、後者に合わせます（9節の差分1）。Pythonのクラス名は`DrawingEpisode`のままです。

### 3.2 `DRAW_EPISODE`

```text
StepSchema(
    step_type="DRAW_EPISODE",
    input_types=("Text",),
    output_type="Episode",
    parameters={
        "policy":      ParameterSpec(str, required=True),
        "max_actions": ParameterSpec(int, default=1000),
        "seed":        ParameterSpec(int, default=0),
        "width":       ParameterSpec(int, default=256),
        "height":      ParameterSpec(int, default=256),
    },
    capabilities=frozenset({"draws", "writes_files"}),
)
```

- **入力**は`Text`1個だけです。`Image`、`LineArt`、`Episode`を入力に取りません。`validate_workflow`が型の不一致として拒否します（参照画像を必須としない根拠）。
- **主出力**は`Episode`です。`REPLAY_EPISODE`が下流で受け取れます。出力型は固定のため`output_resolver`を使いません。Runtimeは主出力の型だけを`output_for(params)`と照合します。追加Artifactの型はRuntimeが検査しないため、Stepの責任とし、テストで固定します。
- **追加Artifact**として`Image`を`StepResult.additional_artifacts`で返します。IDは`context.artifact_id("image")`、親は`Episode`のIDです。下流のStepはこの`Image`を受け取れず（Runtimeは主出力だけを下流へ渡す）、`REPLAY_EPISODE`が再生した`Image`が主たるWorkflow出力になります。
- `ParameterSpec`には範囲指定がありません。`max_actions`（1..19999）、`seed`（0..4294967295）、`width`/`height`（1..1024、Engineの`MAX_CANVAS_SIDE`）の範囲検査は`run`の冒頭で行い、範囲外は`ValueError`にします。Runtimeの変更は不要です。
- `width`/`height`の既定256は暫定値です。`LoadedPolicy.canvas`が学習時のcanvasを持つ場合、異なる指定はエラーにします（黙って拡縮しません）。
- `seed`は[Workflow DSL](workflow_dsl.md)の契約案にないパラメータの追加です。画素の再現に必要なため、任意パラメータとして加えます（9節の差分3）。

#### Trajectoryと大きなEpisodeの扱い

1回の描画は数百から数千のActionを持ちます。現在のTrajectoryはStep単位で`Artifact`を1個ずつ保存するため、Actionを`Artifact.metadata`へ入れません。

| 場所 | 保存するもの |
| --- | --- |
| ファイル `<output_dir>/<run_id>/<step_id>/episode.json` | Episode全体。保存は一時ファイルへ書いてから改名し、途中の破損ファイルを`episode.json`の名前で残さない |
| ファイル `.../attempts.json` | 拒否されたものを含む全提案（試行番号、Action、受理か否か、理由） |
| ファイル `.../image.png` | 完成画像（RGBA8、標準ライブラリ`zlib`で符号化し、同じ画素から同じバイト列を得る） |
| `Artifact(Episode).uri` | `episode.json`への絶対パス（`file:`なし。既存の`PartBankComposeStep`の出力と同じ流儀） |
| `Artifact(Episode).metadata` | 下表のキーのみ |
| `Artifact(Image).metadata` | `width`、`height`、`rgba_sha256`、`png_sha256`、`episode_artifact_id` |
| `Trajectory.prompt`、`artifacts`、`executions` | 既存のまま。Episode本体を入れない |

`Episode`の`metadata`キー:

| キー | 内容 |
| --- | --- |
| `episode_id`、`renderer_version`、`event_count` | Episodeの識別と規模 |
| `accepted_actions`、`rejected_attempts`、`system_events` | 件数（評価の分母に使う） |
| `episode_sha256` | `episode.json`のバイト列のSHA-256 |
| `final_image_rgba_sha256` | 描画中の最終合成のRGBAバイト列のSHA-256 |
| `attempts_sha256` | `attempts.json`のSHA-256 |
| `policy_id`、`checkpoint_sha256`、`policy_trained` | 使ったPolicy。未学習やテスト用は`policy_trained=false` |
| `stage_summary` | 工程ごとの受理Action数とレビュー理由コード |
| `status` | `complete`（`Episode`のArtifactは完成時だけ作る） |
| `execution_identity` | Runtimeが`StepExecution`へ写す辞書（policy、checkpoint、renderer、palette、brush manifestのハッシュ） |

系譜は`Text → Episode → Image`です。`parent_artifact_ids`はRuntimeが既知のIDだけを許すため、`Image`は同じStepの`Episode`を親にできます。

#### 失敗時の記録

Draftのとき、`run`は`DrawIncomplete(reason, draft_path)`を送出します。Runtimeは`RuntimeExecutionError`に途中のTrajectoryを付けて再送出します。このとき`Episode` Artifactは作らず、`draft.json`の場所は例外メッセージと`StepExecution.error`に残ります。Runtimeは例外時に追加Artifactを記録しないため、現行のRuntimeではDraftの`Image`をArtifactにできません。変更が必要かは未決事項U3とし、S4では**Runtimeを変更しません**。

### 3.3 `REPLAY_EPISODE`

```text
StepSchema(step_type="REPLAY_EPISODE", input_types=("Episode",), output_type="Image", parameters={})
```

1. 入力`Episode`の`uri`を読み、`episode_sha256`と照合します。不一致は`ValueError`です。
2. `loads_strict`と`DrawingEpisode.from_dict`で検証します。
3. `Replayer.replay(episode)`で`ReplayResult.image`を得ます。
4. 再生画像のRGBAハッシュを`final_image_rgba_sha256`と照合します。不一致は`ValueError`（Replay不一致）で、画像を返しません。
5. PNGへ書き出し、`Image`を主出力として返します。親は入力`Episode`です。

履歴を再生せずに画像を返す実装は不合格です（[Workflow DSL](workflow_dsl.md)の2節）。

### 3.4 DSLの例（実装後に`test_dsl_text_to_image_workflow_parses_validates_and_roundtrips`で固定）

```text
INPUT prompt: Text
episode = DRAW_EPISODE(prompt, policy="drawing-policy-v1", max_actions=1000, seed=7)
image = REPLAY_EPISODE(episode)
OUTPUT image
```

次は拒否されるWorkflowです。

```text
INPUT sketch: Image
episode = DRAW_EPISODE(sketch, policy="drawing-policy-v1")   # 入力型がImage。型の不一致
```

```text
episode = DRAW_EPISODE(prompt, policy="p", strokes=[...])    # 未定義パラメータ。筆跡の直書きを許さない
```

### 3.5 registryへの登録

既存の`create_mock_registry`と`create_part_bank_registry`は変更しません。新しい関数を追加します。

```python
def register_drawing_steps(
    registry: StepRegistry,
    *,
    policies: PolicyRegistry,
    output_dir: str | Path,
) -> None:
    registry.register(DrawEpisodeStep(policies, output_dir), default=True)
    registry.register(ReplayEpisodeStep(output_dir), default=True)
```

呼び出し側は、既存のregistryを作ってから`register_drawing_steps`を呼びます。`StepRegistry.register`は同じ`(step_type, backend)`の二重登録を拒否します。`image_drawer/steps/__init__.py`の`__all__`へ`register_drawing_steps`を追加します。

### 3.6 Part Bank・COMPOSE・SELECTとの関係

- Part Bank（`RETRIEVE_PARTS`、`SELECT_PARTS`、`COMPOSE`）は**補助機能として残します**。`DRAW_EPISODE`はPart Bankを読まず、`COMPOSE`の出力（`Image`）を`DRAW_EPISODE`の入力にできません（型が合いません）。
- `COMPOSE`に描画機能を足しません。Part配置をActionの代わりにしません。
- `SELECT`は、同じpromptを異なる`seed`で複数回`DRAW_EPISODE`した候補の選択に使えます。これは**Best-of-Nの推論**であり、学習ではありません。評価では単発の結果と区別して報告します（[評価プロトコル](design_evaluation_protocol.md)）。
- Workflowの1つのStepが工程Controllerを呼ぶ上位単位であり、`DRAW_EPISODE`の内側で筆跡が消えてはいけません。`Episode`のArtifactが存在し、`REPLAY_EPISODE`で再生できることを必須にします。

## 4. CLI `image-drawer-draw`

### 4.1 使い方（実装後の仕様）

```powershell
# 文章だけから描く（実行ディレクトリはリポジトリのルート。要: python -m pip install -e .）
image-drawer-draw --prompt "赤いリンゴが白い背景に一つ" --policy drawing-policy-v1 --out-dir out/apple --seed 7

# 保存したEpisodeを再生する
image-drawer-draw --replay out/apple/episode.json --out-dir out/apple-replay --frames stages
```

### 4.2 引数

| 引数 | 内容 | 既定 |
| --- | --- | --- |
| `--prompt TEXT` | prompt。`--prompt-file`、`--replay`、`--list-policies`のどれか1つと排他 | なし |
| `--prompt-file PATH` | UTF-8のpromptファイル（シェルの引用で日本語が崩れる環境向け） | なし |
| `--policy ID` | `PolicyRegistry`に登録されたPolicyのID | 必須（描画時） |
| `--max-actions N` | 1..19999 | 1000 |
| `--seed N` | 0..4294967295 | 0 |
| `--width N`、`--height N` | 1..1024。Policyのcanvasと異なればエラー | 256 |
| `--out-dir DIR` | 出力先。空でない既存ディレクトリはエラー | 必須 |
| `--overwrite` | 出力先の既知の4ファイル（`episode.json`、`draft.json`、`image.png`、`attempts.json`）だけを上書きする。他のファイルは削除しない | 無効 |
| `--replay EPISODE` | 再生モード。Episodeを検証し、画像を再生する | なし |
| `--frames {none,stages}` | `stages`は工程の退出時点の画像を`frame_<stage_id>.png`として保存する | `none` |
| `--expect-image PNG` | 再生画像のRGBAがこのPNGと一致するか確認する（再生モードのみ） | なし |
| `--list-policies` | 登録済みPolicyのID・学習状態・canvasを一覧する | なし |

**画像や下書きを入力として受け取る引数を持ちません**（`test_draw_cli_has_no_image_input_option`）。

### 4.3 出力

標準出力に1行のJSONを書きます。

```json
{"status": "complete", "episode_path": "out/apple/episode.json", "image_path": "out/apple/image.png",
 "attempts_path": "out/apple/attempts.json", "accepted_actions": 412, "rejected_attempts": 37,
 "policy_id": "drawing-policy-v1", "checkpoint_sha256": "…", "policy_trained": false,
 "image_rgba_sha256": "…"}
```

`policy_trained`が`false`のときは、「未学習またはテスト用のPolicyです。絵の品質は評価されていません」という趣旨の文を`notice`に含めます。エラーは標準エラー出力に`{"error": "<種別>", "message": "<説明>"}`の1行で書きます（`image-drawer-dataset`と同じ形式）。

### 4.4 終了コード

| コード | 意味 |
| --- | --- |
| 0 | 描画が完成した、または再生が一致した |
| 1 | 未完成（Draft）。`draft.json`と`image.png`を保存済み。理由は`status`と`reason`に出力 |
| 2 | 実行できない（引数の誤り、未登録のPolicy、canvasの不一致、出力先の衝突、入出力エラー） |
| 3 | 再生モードのみ。Episodeの検証失敗、`--expect-image`との不一致、再生画素のハッシュ不一致 |

`argparse`の使い方エラーも2です。成功を示す0は、完成または一致のときだけ返します。

### 4.5 配線

`pyproject.toml`の`[project.scripts]`へ`image-drawer-draw = "image_drawer.inference.cli:main"`を追加します。CLIとStep、GUIは`image_drawer.inference.session.generate_episode`と`image_drawer.inference.replay.replay_episode_file`を共通に呼びます。CLIだけが持つ処理を作りません。

## 5. GUI

### 5.1 方針

- [GUI設計](gui.md)の「生成画面」と「学習画面のうちDataset登録と抽出器の選択」を、3段階で作ります。学習の開始・停止、Trialの報酬表示は**Trainer（S3）が実装された後の別PR**とし、S4の範囲に含めません（9節の差分4）。
- GUI部品は**表示（view）**と**状態・操作（viewmodel、controller）**に分けます。viewmodelとcontrollerは画面ライブラリをimportせず、CLIと同じサービス関数を呼びます。これにより、画面を開かずに状態遷移をテストできます。
- 画面ライブラリは、標準ライブラリの`tkinter`を暫定案とします。`pyproject.toml`の`dependencies`が空で、wheelに追加依存を持たせないためです。Linuxで`python3-tk`が無い環境があること、見た目の制約があることから、最終決定はPR S4-5の着手前に行います（未決事項U4）。決定までviewmodelにはtkinterをimportしません。

### 5.2 画面

#### (a) Replay画面（G1。読み取り専用）

```text
+--------------------------------+-----------------------------+
| Canvas（現在のseqの合成画像）   | 工程バー                     |
|   直近のストロークを重ねて強調  |  ①rough ②sketch ③final ...  |
|                                |  状態: 未到達/進行中/完了/    |
|                                |        レビュー(理由コード)   |
+--------------------------------+-----------------------------+
| Replayスライダー  |◀ ▶ ⏸| 速度 | レイヤー一覧（上が前面）      |
| seq 0 ─────●───── N  [工程境界へ]| role, 表示/非表示, サムネイル |
+--------------------------------+-----------------------------+
| Action履歴（seq, actor, type, 要約）| 筆・色・サイズ・不透明度・筆圧  |
| 拒否された提案（別タブ）            | （表示のみ。変更できない）     |
+--------------------------------+-----------------------------+
```

| 要素 | 内容 |
| --- | --- |
| Canvas | スライダーのseqまで再生した合成画像。`Engine.render`と同じz順 |
| 直近のストローク | 直前の`stroke_v2`の経路を半透明の線で重ねる（画素は変更しない） |
| 工程バー | 7工程。`enter_stage`/`exit_stage`の履歴から、未到達・進行中・完了・レビュー通過（理由コード付き）を表示。Draftは停止した工程を「中断」と表示 |
| レイヤー一覧 | roleのz順（[固定工程](fixed_drawing_workflow_v2.md)の3節）。`visible`と、非表示にした工程（`rough`は`final_drawing`入場時など）を表示。表示切替は閲覧用で、Episodeを変更しない |
| 筆・色・サイズ・不透明度・筆圧 | 現在のseq時点の選択状態の読み取り表示。固定256色は色見本と`color_id`。**操作できません**（AIの行動を黙って差し替えない） |
| Action履歴 | 受理済みActionの`seq`、`actor`（`model`と`system`を区別）、`type`、要約。選択でそのseqへ移動 |
| 拒否された提案 | `attempts.json`の拒否分。理由コードと、どの工程で起きたか |
| Policyのバッジ | `policy_trained=false`は「未学習／テスト用」、学習済みは「学習済み（評価は別途）」。S5の報告書で`claim_level`が`high_accuracy_claim_allowed`のときだけ、その報告書への参照つきで表示する。それ以外で「高精度」と表示しない |

読み込める入力は`episode.json`、または`draft.json`です。Draftには「未完成」の目立つ表示と理由を出します。

#### (b) 生成画面（G2）

(a)の画面に、左上の入力欄（prompt、seed、max_actions、canvas、Policy選択）と「生成」「停止」を加えます。

- 生成は**別スレッド**で`generate_episode`を呼びます。`cancel`イベントで停止し、結果は`draft`（理由`cancelled`）になります。
- 生成中は、受理されたActionごとに画面を更新します。更新はキューを介して画面スレッドが行い、ワーカーは画面部品に触れません。
- 完了後は同じ`episode.json`をReplay画面として開きます（生成時の表示と再生の表示が同じ経路になります）。

#### (c) Dataset画面（G3）

[GUI設計](gui.md)の学習画面の上半分にあたります。**ユーザーが入力するのは画像ディレクトリだけ**です。

| 要素 | 内容 |
| --- | --- |
| 画像の登録 | 画像ディレクトリとDatasetディレクトリの選択、seed（split用）、「登録」 |
| Tagger／Captionerの選択 | `image-drawer-dataset extractors`と同じ`ExtractorRegistry.describe()`の一覧から選ぶ2つのドロップダウン。id、version、capabilities、languages、runtime、`model_hash`を表示。Taggerには無効（`none`）を含める |
| 結果表示 | 登録数、重複、除外、`quality_issues`、split別件数、`dataset_hash`（`image-drawer-dataset show`と同じ内容） |
| 抽出結果の閲覧 | 画像ごとのタグ、Caption、学習用prompt、抽出器のid/version、失敗理由。**手入力・編集の欄を置きません** |
| 再抽出 | 抽出器を変えて再実行。cache_keyが変わり、manifestの記録が更新されることを表示 |
| 注意表示 | 現在の選択とmanifestに記録された抽出器が異なる場合は、差分を表示する。外部へ画像を送る抽出器を選んだ場合の警告は[統合設計](design_data_pipeline_integration.md)の6節に従う |

画面は`image_drawer.dataset.register_images`、`create_default_registry`、`DatasetManifest`をCLIと同じ引数で呼びます。画面側で抽出器のモデル名を持ちません。

### 5.3 状態

```text
GenerationState: Idle ─生成→ Generating ─完了→ Complete
                                  │ ├─停止/予算切れ/policy_error→ Draft
                                  │ └─実行不能→ Failed
RegistrationState: Idle ─登録→ Registering ─→ Done(report) | Failed
ViewerState: Empty ─読み込み→ Loaded(cursor) | LoadError(reason)
```

各状態は不変のデータクラスで表し、`controller`が遷移だけを行います。`Failed`は理由を保持し、Idleへ戻る操作を持ちます。

### 5.4 Replayカーソル

スライダー操作ごとに全Actionを先頭から再生すると、最大約2万件の再計算になります。`ReplayCursor`は次のように動きます。

1. 前進は`Engine.apply_event`を1件ずつ適用します。
2. 後退と任意のseekは、工程境界（`enter_stage`のseq。最大8か所）に保存した`DrawingState`のコピーから再生し直します。保存数を工程境界に限る理由は、1枚のレイヤーが最大約4MB（1024×1024のRGBA）で、一定間隔ごとに全レイヤーを保存するとメモリが過大になるためです。
3. 合成済みフレームは最近使った32枚だけLRUで保持します。
4. 最終seekの結果は`Replayer.replay`の`ReplayResult`と一致しなければなりません（`test_replay_cursor_end_state_equals_replayer_result`）。

S1cのReplayerが`upto_seq`の部分再生を提供する場合は、`ReplayCursor`はそれを使い、Engineの直接利用は不要です（未決事項U2）。

### 5.5 テスト方法と未確認の範囲

| 層 | 確認する方法 | 確認できる範囲 |
| --- | --- | --- |
| viewmodel・controller | 擬似のサービス関数を差して`pytest`で状態遷移を検査 | 状態遷移、CLIと同じサービスの呼び出し、表示データ（工程バー、レイヤー順、バッジ文言） |
| `ReplayCursor` | 小さなEpisodeで前進・後退・工程ジャンプ・終端一致を検査 | 画素の正しさ |
| `--check` | `image-drawer-gui --check`（headless）を維持 | 起動口の存在 |
| tkinterの画面生成 | 表示が使える環境でのみ`root.withdraw()`で組み立ててすぐ破棄する煙テスト。表示が無い環境は**SKIP**として表示し、成功にしない | ウィジェットの組み立てが例外にならないこと |
| クリック、スライダーのドラッグ、再生ボタン、実際の見た目、別スレッド生成中の画面更新 | **自動テストでは確認しない（未確認）**。PR本文の手動確認表に、操作・結果・環境・スクリーンショットの有無を記録する | なし |

GUIの自動テストが通ったことを、画面の使いやすさやモデルの品質の確認として報告しません。

## 6. txt2imgの受け入れ条件

### 6.1 参照画像を必須としないこと

次をすべて満たすことを必須とします。

1. `DRAW_EPISODE`の入力型は`Text`のみです。画像型の入力をDSL検証が拒否します。
2. `image-drawer-draw`に画像・下書き・Episodeを入力する引数がありません（再生モードの`--replay`は生成ではありません）。
3. `PolicyObservation`の型が許す項目は、prompt、stage、自分のキャンバス・レイヤー・履歴、直前の`ProposalResult`だけです（項目名の集合をテストで固定します）。
4. 生成中にDatasetディレクトリ、`RewardTarget`、`image_path`へアクセスしません（テストでは、checkpointとEngineの設定以外のファイルを開くと失敗するように`open`を差し替えて実行します）。
5. 学習用の`RewardEvaluator`を推論経路でimportしません（import先の確認テスト）。

### 6.2 txt2img（文章のみ）の合格水準

成功を3つの水準に分け、それぞれ別に報告します。上の水準が通っても下の水準が通ったことにはなりません。

| 水準 | 内容 | 使うPolicy | 判定するもの |
| --- | --- | --- | --- |
| A 配線 | 文章から7工程を通り、`Episode`と`Image`が得られ、全ActionをReplayで同じ画素に戻せる | テスト用の決定的Policy（製品の登録簿に載せない） | Step、CLI、Controller、Engine、Replayの接続 |
| B 学習済みの実行 | S3で学習したcheckpointで、水準Aの条件が成り立つ | 実際のcheckpoint | モデルを載せた実行ができること |
| C 品質 | 未学習より未見画像・未見promptで改善している | 実際のcheckpoint | [評価プロトコル](design_evaluation_protocol.md)の公表条件 |

S4の完了は水準Aです。水準Bは学習済みcheckpointが存在するまで**未検証**と表示します（`pytest.skip`の理由に書き、PASSにしない）。S4のPRで「モデルが絵を描ける」と報告しません。

### 6.3 Done（Issue #71）の対応

| Issueの条件 | 対応する検査 |
| --- | --- |
| 文章だけのtxt2imgが成立する | 6.1の全項目、`test_text_only_workflow_end_to_end_with_scripted_policy` |
| 全ActionのReplayが成立する | `test_replay_episode_returns_image_identical_to_drawn_image`、`test_draw_cli_replay_reproduces_image_and_returns_0` |
| モデルが参照画像を必須としない | 6.1の1から5 |
| 描画途中のレイヤー・筆跡・工程を見られる | `test_replay_cursor_*`、`test_stage_bar_*`、`test_layer_panel_*`（画面での見え方は手動確認） |
| Datasetの画像登録と抽出器の選択UI | `test_dataset_panel_*` |
| Stepで筆跡が消えるものは不合格 | `test_draw_episode_trajectory_stores_uri_and_hash_not_events`、`test_replay_episode_fails_when_replayed_pixels_differ_from_recorded_hash` |

## 7. 受け入れテスト一覧

テストは`tests/`の次のファイルに置きます。S1c以降のEngineとControllerが必要なテストは、それらがマージされるまで追加しません。

### `tests/test_inference_session.py`

- `test_generate_episode_completes_with_scripted_policy_and_finishes_once`
- `test_generate_episode_returns_draft_when_action_budget_is_exhausted`
- `test_generate_episode_returns_draft_when_attempt_limit_is_reached`
- `test_generate_episode_returns_draft_with_event_limit_exceeded_near_max_events`
- `test_generate_episode_policy_exception_becomes_draft_with_policy_error`
- `test_generate_episode_cancel_event_stops_with_reason_cancelled`
- `test_rejected_attempts_are_logged_and_do_not_consume_seq`
- `test_policy_observation_contains_only_allowed_fields`
- `test_policy_first_action_is_independent_of_dataset_directory`
- `test_same_seed_prompt_checkpoint_gives_identical_event_sequence`
- `test_inference_module_does_not_import_reward_evaluator`

### `tests/test_inference_artifacts.py`

- `test_episode_file_roundtrip_matches_sha256_in_metadata`
- `test_episode_is_written_atomically_and_draft_is_never_named_episode_json`
- `test_png_encoder_is_deterministic_and_decodes_to_same_rgba`
- `test_attempts_file_records_reason_and_action_for_each_rejection`
- `test_overwrite_touches_only_known_output_files`

### `tests/test_draw_episode_step.py`

- `test_draw_episode_schema_declares_text_input_and_episode_output`
- `test_draw_episode_returns_episode_primary_and_image_additional_artifact`
- `test_draw_episode_artifact_lineage_text_to_episode_to_image`
- `test_draw_episode_trajectory_stores_uri_and_hash_not_events`
- `test_draw_episode_rejects_image_typed_input_at_validation`
- `test_draw_episode_rejects_unknown_parameters_such_as_strokes`
- `test_draw_episode_rejects_values_outside_parameter_ranges`
- `test_draw_episode_draft_raises_runtime_error_with_draft_path`
- `test_replay_episode_returns_image_identical_to_drawn_image`
- `test_replay_episode_fails_when_episode_file_hash_mismatches`
- `test_replay_episode_fails_when_replayed_pixels_differ_from_recorded_hash`
- `test_dsl_text_to_image_workflow_parses_validates_and_roundtrips`
- `test_register_drawing_steps_keeps_part_bank_and_mock_steps`
- `test_compose_output_cannot_feed_draw_episode`

### `tests/test_draw_cli.py`

- `test_draw_cli_writes_episode_png_attempts_and_exit_0`
- `test_draw_cli_returns_1_and_writes_draft_when_budget_exhausted`
- `test_draw_cli_returns_2_for_unknown_policy`
- `test_draw_cli_returns_2_for_non_empty_out_dir_without_overwrite`
- `test_draw_cli_returns_2_when_canvas_differs_from_policy_canvas`
- `test_draw_cli_replay_reproduces_image_and_returns_0`
- `test_draw_cli_replay_returns_3_for_tampered_episode`
- `test_draw_cli_replay_returns_3_when_expected_image_differs`
- `test_draw_cli_has_no_image_input_option`
- `test_draw_cli_stdout_is_single_json_line_and_marks_untrained_policy`
- `test_draw_cli_and_step_produce_identical_episode_for_same_request`

### `tests/test_gui_viewmodel.py`

- `test_replay_cursor_end_state_equals_replayer_result`
- `test_replay_cursor_seek_backward_matches_forward_state`
- `test_replay_cursor_stage_jump_lands_on_stage_boundaries`
- `test_layer_panel_order_follows_role_z_order`
- `test_stage_bar_shows_review_reason_for_reviewed_stages`
- `test_draft_is_shown_with_draft_badge_and_reason`
- `test_brush_panel_is_read_only`
- `test_policy_badge_never_says_high_accuracy_without_report`
- `test_generation_state_machine_idle_generating_complete`
- `test_generation_stop_requests_cancel_and_shows_draft`
- `test_gui_generation_calls_the_same_service_as_cli`
- `test_dataset_panel_lists_registry_describe_entries`
- `test_dataset_panel_passes_selected_extractor_ids_to_register_images`
- `test_dataset_panel_has_no_input_for_prompt_tags_or_captions`
- `test_gui_check_flag_stays_headless_and_ok`（既存の`tests/test_gui.py`を維持）
- `test_tk_window_builds_when_display_is_available`（表示が無い環境はSKIP）

### `tests/test_inference_e2e.py`

- `test_text_only_workflow_end_to_end_with_scripted_policy`（水準A。出力に「モデル未検証」を明記）
- `test_text_only_workflow_end_to_end_with_trained_checkpoint`（水準B。checkpointが無い間はSKIP）

## 8. 実装手順（PRの分割）

1PRは1つの目的で、変更は500行以下を目安にします。前提はS1a/S1b/S1c（Engine、Controller、Replayer）のマージです。`DrawingPolicy`の実体はS3ですが、S4は`LoadedPolicy`の窓口とテスト用Policyで先に進めます。

| PR | 目的 | 主な変更 | 前提 |
| --- | --- | --- | --- |
| S4-1 | 推論ループ | `inference/session.py`、`DrawRequest`/`DrawOutcome`/`AttemptRecord`、`PolicyRegistry`の窓口、テスト用Policy（`tests/`配下）、`test_inference_session.py` | S1a-c |
| S4-2 | 成果物の保存 | `inference/artifacts.py`、`inference/png.py`、`draft.json`の形式、`test_inference_artifacts.py` | S4-1 |
| S4-3 | Workflow Step | `steps/draw_episode.py`、`register_drawing_steps`、DSLの検証、`test_draw_episode_step.py` | S4-2 |
| S4-4 | CLI | `inference/cli.py`、`inference/replay.py`、`pyproject.toml`のscript、`docs/users/`の手順、`test_draw_cli.py` | S4-3 |
| S4-5 | GUI G1（Replay表示） | viewmodel、`ReplayCursor`、tkinterの閲覧画面、`--check`の維持 | S4-4、U4の決定 |
| S4-6 | GUI G2（生成） | 生成パネル、ワーカー、キャンセル、状態機械 | S4-5 |
| S4-7 | GUI G3（Dataset登録・抽出器選択） | Datasetパネル。`image_drawer.dataset`を呼ぶ | S4-5、S2のマージ |
| S4-8 | E2Eと文書 | `test_inference_e2e.py`、`docs/users/getting_started.md`と`docs/developers/`索引の更新 | S4-1〜4 |

S4-7は、[統合設計](design_data_pipeline_integration.md)のI-1（アダプタ）より前に出せます。S2の`ExtractorRegistry`だけを使い、#78の抽出器は統合後に同じ一覧へ現れます。

各PRで`python scripts/check_local.py`を実行し、変更したPythonファイルを`--format`で整形してから再実行します。完了報告には、実行したコマンド、結果、水準（A/B/C）のどこまで確認したか、未確認の範囲を書きます。

## 9. 正本との差分と判断

既存の正本は変更しません。差分は実装PRで正本を更新するときに解消します。

| # | 正本 | 本書の判断 | 理由 |
| ---: | --- | --- | --- |
| 1 | [Workflow DSL](workflow_dsl.md)は出力を`DrawingEpisode`と書く。[設計監査](design_audit.md)は型`Episode`と書く | Artifact型の文字列は`Episode`、Pythonクラスは`DrawingEpisode` | 既存のArtifact型（`Image`、`ImageSet`、`ScoreSet`）と同じ名詞形にそろえる |
| 2 | DSLは`DRAW_EPISODE`の主出力を`DrawingEpisode`、監査は「`StepResult`で`Episode`と`Image`を返す」と書く | 主出力`Episode`、追加Artifact`Image`で両方を満たす | `REPLAY_EPISODE(episode)`を下流で書くため。Runtimeは主出力だけを下流へ渡す |
| 3 | DSLの契約案は`policy`と`max_actions`だけ | `seed`、`width`、`height`を任意パラメータとして追加 | 画素の再現と、canvasの指定に必要。後方互換（既定値あり） |
| 4 | [GUI設計](gui.md)の学習画面は、学習の開始・停止、Trialの評価・報酬を含む | S4では学習の開始・停止・Trial表示を作らず、Dataset登録と抽出器選択までとする | Issue #71の範囲が「Datasetの画像登録と抽出器の選択UI」であり、Trainerは未実装（S3）のため |
| 5 | DSLは`max_actions`を1..19999とするが、Engineの`MAX_EVENTS`は20000で`actor=system`イベントを含む | 正本の範囲は変えず、実行時に`event_limit_exceeded`のDraftとして扱う | 上限付近でsystemイベントが加わると収まらない。上限の再定義はS1bで決める |
| 6 | Draftは「完成Episodeと偽らない」（DSL、Controller契約） | Draftは`draft.json`と別名にし、`Episode` Artifactを作らない。Runtimeは変更しない | 完成Episodeと混同しないため。Draftの`Image`をArtifact化するRuntime変更は未決事項U3 |
| 7 | 2.2節はDraft理由に`budget_exhausted`等を例示し、`max_actions`を1..19999とする | 理由はS1bの`termination_reason`（`action_budget_exhausted`、`attempt_budget_exhausted`、`internal_error`）と、推論側の`cancelled`、`policy_error`、`event_limit_exceeded`、`engine_error`。`max_actions`の上限はS1bの`MAX_ACTIONS_LIMIT`（19900） | S1bが確定した値に合わせた（U1の解決） |
| 8 | 2.2節は「`DrawingEpisode.from_dict`は終端`finish`を要求し、Draftは読み込めない」 | S1c実装ではdraft状態のEpisodeを`loads_episode`が読め、`replay`も接頭辞を再生できる。`draft.json`は`episode_prefix`にDraftのEpisodeをそのまま入れ、GUIは`ReplayCursor`、CLIの`--replay`は`replay`で表示・再生する | U2の解決 |
| 9 | 4.2節の`--width`/`--height`の既定は256 | CLIの既定は「Policyのcanvas、無ければ256」。異なる値を指定したときのエラーは設計どおり。Stepの既定は設計どおり256のまま（学習時canvasと違えばエラー） | 唯一の登録Policyが小canvasで、既定値のままでは必ずエラーになるため。U7は未決 |
| 10 | 4.2節の引数一覧 | `--checkpoint ID=SIDECAR_JSON`と`--train-config`を追加。学習済みcheckpointをPolicyとして登録する（水準Bの実行口）。`PolicyRegistry`の既定には`random-legal-v1`（未学習・テスト用、numpyが必要）だけが載る | 登録簿を外から差せる口が無いと水準Bを実行できない |
| 11 | 6.1節の5「学習用の`RewardEvaluator`を推論経路でimportしない」 | `image_drawer/inference/`と`steps/draw_episode.py`は`RewardEvaluator`・`reward`・`dataset`をimportしない（ASTで検査）。ただしcheckpoint読込に使う`training.checkpoint`が`training.optimizer`経由で`training.reward`の型を間接的に読み込む | checkpointの形式知識を複製せず`load_checkpoint`を再利用したため |
| 12 | 5.2節(b)「生成中は、受理されたActionごとに画面を更新」 | 生成中に更新するのは受理数と現在の工程のラベルまで。生成中のキャンバスのライブ描画は未実装で、完了後に同じファイルをReplay画面で開く | Controllerの内部Engineを公開せずに実現するため（Controller契約の変更が要る） |

## 10. 未決事項

| ID | 内容 | 決める場所 |
| --- | --- | --- |
| U1 | 予算切れ・試行上限を示す`ProposalResult.reason`のコード名と、終端を判定する公式の方法 | 解決: `StageController.termination_reason`を使う（9節の7） |
| U2 | Replayerが`finish`のないDraftの接頭辞、または`upto_seq`の部分再生を受け付けるか | 解決: 両方を受け付ける（9節の8） |
| U3 | 例外時にもDraftの`Image`をTrajectoryのArtifactとして残すRuntime拡張が必要か | S4-3のレビュー。必要ならRuntime変更を別PRにする |
| U4 | GUIの画面ライブラリ（tkinter暫定）。Linuxの`python3-tk`、見た目、テスト容易性の比較 | 暫定のままtkinterで実装した。viewmodelはtkinterをimportしないため差し替え可能。最終決定は未了 |
| U5 | `DrawingPolicy.next_action`と`PolicyObservation`、`PolicyRegistry.resolve`の正式な名前とシグネチャ（本書の名前は仮名） | 解決: 学習側は`DrawingPolicy.act(Observation, rng)`。推論側は`inference.policy`の`PolicyRegistry.resolve`が`LoadedPolicy`を返し、`open_session(prompt, seed)`の`PolicySession.next_action(controller, engine, budget)`を呼ぶ。`training_policy`が学習と同じ`ObservationBuilder`で観測を作る |
| U6 | `StyleIntent`の判定器の名前と入出力（`judge_style_intent`は仮名） | 解決: `judge_style_intent(prompt)`（S1b） |
| U7 | 既定canvas（256）と、学習時のcanvasの関係 | S3の学習設定 |
