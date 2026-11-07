# 共有エージェントコンテキスト

Codex・Claude・ChatGPT・Copilot共通の**リンクポータル**です。全文は複製しません。今回の作業に合う1〜2文書だけを開いてください。

1. **開始時**: 最新を取り込み、オープンPRを先に処理し、Issueは優先度順に進めます。→[共同作業ルール](docs/kadoka-codex/collaboration_workflow.md)
2. **設計が先**: 設計書に該当する節が無い作業は、実装せず設計PRを先に出します。→[実装マップ](docs/developers/implementation_map.md)
3. **何を読むか**: 作業ごとの設計書・実装先・テストは、実装マップの経路表にあります。コードの探索は[参照表](.agents/skills/image-drawer-context/SKILL.md)から。
4. **書き方**: [コーディング規約](docs/kadoka-codex/coding_conventions.md)の対象の節だけを読みます。
5. **確認**: PR前に `python scripts/check_local.py` を実行します。→[ローカル確認](docs/developers/local_checks.md)
6. **方針**: [実装計画](docs/kadoka/implementation_plan.md)、入力契約は[training_input_v1.json](docs/developers/training_input_v1.json)。
7. **制約**: 既存の未コミット変更を保持します。モデル・個人画像・生成物はGit管理外です→[保存場所](data/README.md)。説明は日本語が正本です→[言語方針](docs/developers/language_policy.md)。
8. **報告**: 現在のコード・差分・実行結果を根拠にし、確認済みと未確認を分けます。

## 保守

入口・参照先を変えたら `python scripts/check_agent_context.py` で行数・容量・リンクを確認します。Claudeの入口は [CLAUDE.md](CLAUDE.md)、Copilotの入口は [.github/copilot-instructions.md](.github/copilot-instructions.md) です。自動読込のない環境では、依頼文でこのファイルを指定してください。
