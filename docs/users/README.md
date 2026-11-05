# 利用者向け文書

**対象読者:** Image Drawerを使いたい人  
**目的:** 内部実装を知らなくても、現在利用可能な機能を使えるようにする。

## 画像から自動でタグ・プロンプトを生成する

[8GB VRAM・32GB RAM向けの起動手順](preprocessing_local8g.md)／[通常設定JSON](preprocessor_settings.example.json)／[高設定JSON](preprocessor_settings.high.example.json)／[前処理モデルを自由に登録する方法](preprocessing.md)。画像以外の学習素材は必要ありません。

## まず読む

- [はじめに](getting_started.md)
- [画像だけのDatasetを登録する](image_dataset.md)
- [配布物の真正性を確認する](verify_release.md)

## このfolderに置く文書

- install方法
- 起動方法
- GUI / CLIの操作方法
- Workflowを使う方法
- modelやdatasetを利用するときの注意
- user-facing troubleshooting

architectureや実装仕様はここへ置きません。内部を変更したい場合は [開発者向け文書](../developers/) を参照してください。
