> **対象読者:** 学習したPolicyを評価し、結果を報告する利用者
>
> **目的:** 学習前後を同じ条件で比べ、未見の画像で改善したかを報告書にまとめる手順を示す。
>
> **状態:** コマンドは実装済みです。実checkpointでの評価は未実測で、学習が成功したことは確認していません。

# 汎化評価を実行して報告書を作る

## 背景と目的

学習で上達したと言えるのは、学習に使っていない画像（`test`）で、学習前（`before`）より改善した場合だけです。アプリは評価集合を凍結し、`test`を調整に使えないようにして、指標・信頼区間・失敗の件数を報告書にまとめます。

## 前提

- リポジトリのルートで、`python -m pip install -e ".[dev,train]"`を実行済みであること。
- 画像だけのDataset（[登録手順](image_dataset.md)）があり、`before`と`after`のcheckpoint（`iter_N.json`）があること。

## 手順

詳細な設計は[汎化評価プロトコル](../developers/design_evaluation_protocol.md)の12節にあります。

```powershell
# 1. 評価集合を凍結する（学習・調整の前に1回）
image-drawer-evaluate plan --dataset datasets/main --out reports/eval/eval_set_main_v1.json

# 2. validationで調整する（凍結宣言は不要）
image-drawer-evaluate run --dataset datasets/main --eval-set reports/eval/eval_set_main_v1.json `
  --split validation --condition before=checkpoints/init/iter_0.json --condition after=checkpoints/run1/iter_200.json `
  --condition random_valid=random --out reports/eval/validation_run1

# 3. 凍結宣言を作る（調整の終了後に1回）
image-drawer-evaluate freeze --eval-set reports/eval/eval_set_main_v1.json `
  --before checkpoints/init/iter_0.npz --after checkpoints/run1/iter_200.npz `
  --analysis-plan docs/developers/eval_analysis_plan_v1.json --out reports/eval/freeze_manifest.json

# 4. testで1回だけ評価する
image-drawer-evaluate run --dataset datasets/main --eval-set reports/eval/eval_set_main_v1.json `
  --split test --freeze-manifest reports/eval/freeze_manifest.json `
  --condition before=checkpoints/init/iter_0.json --condition after=checkpoints/run1/iter_200.json `
  --condition random_valid=random --out reports/eval/test_run

# 5. 比較と報告書
image-drawer-evaluate compare --run reports/eval/test_run --eval-set reports/eval/eval_set_main_v1.json --out reports/eval/compare.json
image-drawer-evaluate report --compare reports/eval/compare.json --run reports/eval/test_run `
  --eval-set reports/eval/eval_set_main_v1.json --out reports/eval/report.json --markdown reports/eval/report.md
```

人間評価は`human-pack`で盲検パックを作り、評価者の回答を`human-collect`で集計します。

## 出力と終了コード

- `run.json`（全生成と指標）、`compare.json`、`report.json`、`report.md`を保存します。
- 終了コード: 0 成功、1 失敗した生成を含む（結果は保存済み）、2 実行不能（`test`で凍結宣言が無い、`retest`など）、3 Replayまたは監査器の不一致。

## 制限

- 同じ評価集合に別の凍結宣言で`test`を再実行すると`retest`になり、その結果は公表条件を満たしません。新しい評価集合を作ってください。
- `report`コマンドは、学習したPolicyであることの確認・再現の確認・対照指標・独立指標の一部を未確認として扱い、`claim_level`を`none`にします。確認した事実は`GateInputs`で与えます（今後の拡張）。
- 報告書の数値を、記載したデータ・解像度・seed数の範囲を超えて一般化しないでください。
