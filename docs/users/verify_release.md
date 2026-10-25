# 配布物の真正性を確認する

**対象読者:** Image DrawerのReleaseから配布物をダウンロードして使う人  
**目的:** ダウンロードした配布物が、このリポジトリのGitHub Actionsから生成されたものか確認する。

## 配布物の種類

現在、このリポジトリが配布する成果物は次の3種類です。

| 成果物 | 内容 |
| --- | --- |
| `image_drawer-<version>-py3-none-any.whl` | Pythonのwheel |
| `image_drawer-<version>.tar.gz` | Pythonのsdist |
| `SHA256SUMS` | 上の2ファイルのSHA-256一覧 |

Windowsの実行ファイル、macOSのアプリ、Android、コンテナは配布していません。したがって、Authenticode署名、macOSのnotarization、APK/AABの署名は対象外です。これらを配布する場合は、配布前に署名方式を決めてこの文書を更新します。

## 署名の方式

vタグをpushすると、`.github/workflows/release.yml` がwheelとsdistを作成します。GitHub ActionsのOIDCを使い、秘密鍵を使わずにArtifact Attestation(ビルド来歴の署名)を付けます。`SHA256SUMS` も同じ方法で署名します。検証に失敗した場合、Releaseは作成されません。

## 確認手順

前提として、[GitHub CLI](https://cli.github.com/)(`gh`)をインストールしてください。次のコマンドは、ダウンロードしたファイルがあるディレクトリで実行します。

1. SHA-256を確認します。

   ```bash
   sha256sum --check SHA256SUMS
   ```

   PowerShellでは、次のコマンドで値を表示し、`SHA256SUMS`の値と比較してください。

   ```powershell
   Get-FileHash .\image_drawer-<version>-py3-none-any.whl -Algorithm SHA256
   ```

2. 来歴の署名を確認します。`<version>`は実際のバージョンに置き換えてください。

   ```bash
   gh attestation verify image_drawer-<version>-py3-none-any.whl --repo tomiya7688/Image_drawer
   gh attestation verify image_drawer-<version>.tar.gz --repo tomiya7688/Image_drawer
   gh attestation verify SHA256SUMS --repo tomiya7688/Image_drawer
   ```

   成功すると、ファイルのダイジェスト、生成したリポジトリ、workflowが表示されます。失敗した場合は、そのファイルを使用しないでください。

## 過去の配布物

署名機能の導入前に配布された成果物はありません(Releaseとタグは存在しませんでした)。今後のReleaseは、署名と検証を通ったものだけを公開します。
