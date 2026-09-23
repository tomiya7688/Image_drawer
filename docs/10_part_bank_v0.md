# Part Bank v0（Issue #5）

このmilestoneではsemantic segmentationやvector retrievalではなく、画像ingestとprovenance管理を実装します。既存M1のSourceImage / Part recordは変更せず使用します。

## 実行

    python -m pip install -e ".[dev]"
    image-drawer-part-bank ingest /path/to/images       --bank ./data/part-bank --dataset my-dataset --split train

checkoutからnon-editable installする場合:

    python -m pip install ".[part-bank]"

PyPI公開済みであることは前提にしません。

同等のmodule entrypoint:

    python -m image_drawer.part_bank

M0の既存command image-drawer --input ... は変更しません。

Pillowはoptional dependencyであり、core packageとmock runtimeは引き続きruntime dependencyなしで動作します。

input directoryとbank directoryは分離する必要があります。commandはsymlinkをfollowせず、directory/nameのstable順で再帰走査します。

対応suffix:

- PNG
- JPEG
- WebP
- BMP
- TIFF
- PPM

大文字小文字は区別しません。その他fileはskippedとして数えます。extensionだけでなく実体formatも対応formatとして識別できる必要があります。

animated / multipage imageは暗黙に先頭pageだけを使わず、明示的にrejectします。

stdoutはJSON reportです。

- processed
- skipped
- sources_added
- parts_added
- duplicates
- errors

exit code:

- 0: 試行したfileがすべて成功
- 1: 一部file失敗。path/type/messageをreportする
- 2: invalid configuration、optional dependency不足、storage-level failure

fatal errorはstderrへ出します。空directoryはcount 0のreportを返します。upload処理はありません。

## Python API

    from image_drawer.part_bank import SQLitePartRepository, ingest_directory

    report = ingest_directory(
        "/path/to/images", "./data/part-bank",
        dataset="my-dataset", split="train",
        provenance={"license": "record the actual license", "source_group": "group-a"},
    )
    with SQLitePartRepository("./data/part-bank/metadata.sqlite3") as repository:
        for part in repository.list_parts():
            source = repository.get_source(part.source_image_id)
            origins = repository.origins(source.id)

record内のimage URIはすべてbank root基準の**bank-relative POSIX path**とします。process working directory基準ではありません。

元local file locationはOrigin URIとして別に記録します。bankを移動してもrecord IDとrelative pathは維持されます。

    part-bank/
      metadata.sqlite3
      source/<sha256>.<detected-format>
      parts/crops/part_<identity-hash>.png

元byte列をbankへcopyします。Source IDとexact checksumはfilenameではなくbyte列から生成します。

Part IDには次を含むidentityを使います。

- source ID
- extractor method/version
- category
- bbox
- versioned Pillow rasterizer identity

異なるextractor versionは共存可能です。

binary imageやembedding payloadをJSONへ格納しません。

## Extractionと座標

WholeImageExtractor v1はdecode済みimage全体を1 regionとして返します。

これはfixture/development用baselineであり、face/object/semantic detectorではありません。

custom extractorは次のinterfaceを実装します。

    extract(source, image) -> Iterable[Region]

明示的なmethodとversionを持たせます。0 regionも許可します。

invalidまたはduplicate bboxが返されたfileはrejectします。

extractorは渡されたimageをread-onlyとして扱い、挙動を変えた場合はversionを更新します。

v0のRegionはbboxとcategoryを持ち、maskは生成しません。Part.mask_uriはnullです。

bboxはstored-pixel coordinateの (x, y, width, height) とします。

Source dimensionとRGBA PNG cropも同じcoordinate frameを使います。

EXIF display orientationは適用しません。crop側EXIFは削除し、rotation metadataによって解釈が変わらないようにします。

RGBA alphaは保持します。

このmilestoneでは次を行いません。

- resizing
- ICC color management
- semantic quality filtering

decoder / rasterizer versionは記録します。

## Deduplication、split policy、recovery

1 bankにつきexact SHA-256 checksumごとに1 SourceImageとします。

renameされたcopyは新Source/Partではなくorigin recordを追加します。

同一fileの再ingestはidempotentです。不足cropがあれば再生成します。

元byte列が変われば新IDになります。

splitは次のいずれかです。

- train
- validation
- test
- unset

最初にacceptedされたdataset/split assignmentを保持します。

同じbyte列を別dataset/splitでimportしようとした場合、silent reassignmentせずrejectします。unsetとassignedの差もconflictとして扱います。

これによりexact duplicateが同一bank内でsplitを跨ぐことを防ぎます。

multi-dataset membershipとnear-duplicate groupingは将来課題です。

group-level splittingは引き続きextraction前に行う責務です。ingesterが個別Partをrandom splitすることはありません。

changing input pathによってchecksum対象とextracted imageが食い違わないよう、private temporary snapshotをhash/decodeします。

validationではfile structureとfull pixel decodeを確認します。

default limit:

- 40 million pixels
- 256 MiB / source file

Pillow decompression warningはerror扱いにします。

limitはPython APIから設定可能です。

fileをatomic publishした後、1 SQLite transactionでSource、Part、Origin recordをcommitします。

transaction失敗時にrowだけ一部insertされることはありません。

file publish後SQL commit前のcrashではunreferenced fileが残る可能性がありますが、retryは安全です。

automatic orphan deletionはv0では実装しません。

既存recordをsilent overwriteしません。

v0は1 bankにつき1 ingest writerを前提とし、concurrent ingest orchestrationは対象外です。

listing APIはresultをmaterializeしますが、ingest pathはdataset全体をmemoryへloadせず1 imageずつ処理します。

## TestsとCI

小さい手書きPPM fixtureと生成test imageで次をcoverします。

- exact pixel output
- alpha
- provenance
- persistence / reopening
- deterministic IDs
- duplicate imports
- split conflicts
- corrupt / truncated image
- resource limits
- EXIF policy
- multipage rejection
- extractor replacement / failure
- rollback
- command exit code

CIは既存のtest/build/core-entrypoint checkを維持します。

その後fresh venvへbuilt wheelのpart-bank extraをinstallし、checkout外のtemporary directoryからreal CLIとmodule entrypointをPYTHONPATHなしで実行します。

さらに次を検証します。

- import元がvenvである
- SQLite recordをopenできる
- PNG cropをopenできる
- pixelとchecksumが正しい
- repeat ingestでdeduplicationされる
- corrupt fileでnonzero exitと明示error reportになる

file validationで参照するAPI:

- [Pillow Image.open/load/verify](https://pillow.readthedocs.io/en/stable/reference/Image.html)
- [Pillow release notes](https://pillow.readthedocs.io/en/stable/releasenotes/index.html)

次: Issue #6でembedding、index abstraction、実際のRETRIEVE_PARTSを追加します。
