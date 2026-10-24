# 繰り返す操作と確認範囲

必要な節だけを使ってください。コマンドはリポジトリルートを起点にします。テストやビルドの実行範囲は利用者の依頼に合わせ、以下を一律の必須手順にはしません。

## 変更前の状態

```powershell
git status --short
git branch --show-current
git diff --stat
```

共同作業の更新が必要ならfetchし、関係するファイル名と差分から確認します。fetchと作業ツリーへの取り込みは別操作です。未コミット変更を自動でstash・破棄したり、古い保全ツリーでmainを上書きしたりしません。

## Pythonの開発と配布

依存関係の正本は [pyproject.toml](../../../../pyproject.toml)です。必要な作業だけを実行してください。

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m build
```

配布後の確認方法は [ci.yml](../../../../.github/workflows/ci.yml)と、そこから呼ばれる二つのverifyスクリプトを参照してください。ソース上の成功と、新しい環境に入れたwheelの成功は別に記録します。

## Windowsでテストする場合

一時フォルダに制限がある場合は、今回専用の未使用パスを指定できます。

```powershell
$agentRun = Get-Date -Format 'yyyyMMdd-HHmmssfff'
python -m pytest -q --basetemp ".tmp/pytest-$agentRun"
```

`WinError 1314`がsymlink_toで出た場合は、シンボリックリンク作成権限の不足です。テストを削除・スキップして成功扱いにせず、実行環境の制約として報告します。
GUIの`--check`、実際の画面操作、画像生成、実モデル、Linux CIは別の確認です。一つの結果を他の成功として使いません。

## コンテキスト文書の保守

`python scripts/check_agent_context.py`で、短い入口と参照資料の行数・UTF-8容量・ローカルリンクを確認します。
上限超過時はまず不要・重複・古い説明を削り、違う作業だけで必要な詳細を分離します。空行を詰めるだけで通したり、上限を自動的に上げたりしません。
スキルを自動発見しないクライアントでも、AGENTS.mdから必要な資料を明示して同じ手順を利用できます。
