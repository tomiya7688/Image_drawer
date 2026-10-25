# 共有エージェントコンテキスト

Codex・Claude・ChatGPT・Copilot共通の短い入口です。手順や履歴の全文はここへ複製しません。

- 依頼の目的と完了条件に必要な範囲を扱い、既存の未コミット変更を保持してください。追加worktreeは作らず、必要なら通常のブランチで分けてください。
- ファイルが分かる修正は、その実装と関連テストから読んでください。不明なら `rg --files`、`rg -n` で候補を絞り、必要な定義と呼び出し元だけを確認してください。
- 場所の再探索・引き継ぎ・保全実装の統合では、[image-drawer-context](.agents/skills/image-drawer-context/SKILL.md)から今回に合う参照先だけを開いてください。
- コード変更では[コーディング規約](docs/kadoka-codex/coding_conventions.md)の対象となる節を参照してください。規約本文はこの入口へ複製しません。
- 作業前に更新を取り込み、オープンPRを先に処理してください。[共同作業ルール](docs/kadoka-codex/collaboration_workflow.md)に従い、PR前は `python scripts/check_local.py` を実行してください。
- 要約は入口です。現在のコード・Git差分・実行結果を根拠にし、確認済みと未確認を分けてください。
- モデル・個人画像・生成レポートはGit管理外に置きます。説明は日本語を正本にします。詳細は必要なときだけ[保存場所](data/README.md)と[言語方針](docs/developers/language_policy.md)を参照してください。

## 保守

一時的な進捗は引き継ぎメモへ移し、共通入口へ履歴を追記しないでください。繰り返した探索だけを参照表へ反映します。
入口・参照先を変更した場合は `python scripts/check_agent_context.py` で行数・容量・リンクを確認できます。
Claudeの入口は [CLAUDE.md](CLAUDE.md)、Copilotの入口は [.github/copilot-instructions.md](.github/copilot-instructions.md)です。
自動読込のない環境では、依頼文でこのファイルを指定してください。
ファイルにアクセスできない環境には、今回の参照行と必要な原文だけを渡してください。
