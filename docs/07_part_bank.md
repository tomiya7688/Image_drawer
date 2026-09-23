# Part Bank仕様

状態: 採用済み設計
日付: 2026-09-20

## 1. 目的

Part Bankは、Image Drawerで再利用可能な視覚assetを管理する層です。

このプロジェクトでは大量の画像datasetが利用できる前提とし、各source imageをfinal training targetとしてだけ扱うのではなく、再利用可能なPartや中間構造を抽出・index化・検索・評価します。

最初の実装目標は完璧なsemantic decompositionではありません。後から抽出手法を改善してもWorkflow/Runtime interfaceを変えずに済む、安定したdata modelとpipelineを作ることです。

## 2. 基本概念

1枚のSourceImageから0個以上のPartを生成できます。

例:

- face
- hair
- eye
- mouth
- hand
- body region
- clothes
- accessory
- background region
- line-art region
- palette
- texture
- composition / layout template
- silhouette
- shading pattern

taxonomyはdataset依存であり、拡張可能に保ちます。

## 3. Data model

### SourceImage

必須field:

    SourceImage
      id: string
      uri: string
      width: int
      height: int
      checksum: string
      dataset: string
      metadata: map

任意metadata:

- tags
- caption
- 法的に保持可能なartist/source情報
- style labels
- quality flags
- split: train / validation / test
- license / provenance情報

### Part

必須field:

    Part
      id: string
      source_image_id: string
      category: string
      crop_uri: string
      bbox: [x, y, width, height]
      metadata: map

推奨field:

    Part
      mask_uri: string | null
      embedding_refs: map
      tags: list[string]
      attributes: map
      quality: map
      extraction_method: string
      extraction_version: string

embedding_refsにより複数embedding modelを共存させます。

例:

    embedding_refs:
      clip_v1: embeddings/clip/abc.npy
      custom_part_v2: embeddings/custom/abc.npy

Part schemaへ特定のembedding modelをhard-codeしません。

### PartSet

検索結果:

    PartSet
      query_id
      category
      part_ids[]
      retrieval_scores[]
      metadata

### PartPlacement

canvas上のPart配置:

    PartPlacement
      part_id
      x
      y
      scale_x
      scale_y
      rotation
      z_index
      opacity
      transform_metadata

### Composition

    Composition
      canvas_size
      placements[]
      background
      metadata

Compositionはrendered Imageだけでなく、中間Artifactとして保持します。

## 4. Storage layout

初期local実装:

    data/
      source/
      parts/
        crops/
        masks/
      embeddings/
        <model_name>/
      indexes/
      manifests/
      runs/

MVPではmetadataにJSONLまたはSQLiteを利用します。

推奨MVP:

- image / crop / mask file: filesystem
- SourceImage / Part / embedding metadata: SQLite
- vector index: repository interfaceの背後へ抽象化

これにより最初は単純に保ちつつ、将来FAISS、Qdrant、Milvus等へ差し替えられます。

## 5. Extraction pipeline

推奨pipeline:

    INGEST
      -> VALIDATE
      -> EXTRACT_PARTS
      -> EXTRACT_METADATA
      -> EMBED
      -> INDEX
      -> QUALITY_FILTER

各stageは途中から再実行可能にします。

### INGEST

責務:

- source image id付与
- checksum
- dimension
- provenance metadata
- dataset split

### EXTRACT_PARTS

抽出algorithmは交換可能にします。

候補:

- bounding-box detector
- segmentation model
- semantic segmentation
- pose-based region extraction
- 手動mask / annotation
- dataset-specific parser

outputは常にPartへ正規化します。

### EMBED

1個以上の検索表現を作ります。

初期候補:

- global visual semantic embedding
- category-specific embedding
- color / style embedding

どのembedding modelがvectorを作ったかをrepository APIで識別可能にします。

### QUALITY_FILTER

利用しづらいPartを削除またはflagします。

- 極端に小さいregion
- 無効/空mask
- corrupt crop
- low-information region
- 必要に応じたobvious duplicate

原則として破壊的削除より記録・flagを優先します。

## 6. Retrieval API

基本interface:

    retrieve_parts(
        query,
        category=None,
        top_k=20,
        filters=None,
        embedding_model=None
    ) -> PartSet

queryには将来次を含められます。

- text
- current image / canvas
- reference part
- workflow state
- attributes

version 1ではtext + category filteringのみで構いません。

## 7. Workflow operation

### RETRIEVE_PARTS

input:

- prompt: Text
- optional current image/state

parameter:

- category
- top
- embedding_model
- filters

output:

- PartSet

例:

    faces = RETRIEVE_PARTS(
      prompt,
      category="face",
      top=20
    )

### SELECT_PARTS

input:

- PartSet
- optional canvas/context
- optional evaluator scores

parameter:

- top
- strategy

MVP strategy:

- retrieval_score
- evaluator_score
- weighted_score

output:

- selected PartSet または Part

### COMPOSE

input:

- parts
- layout / placements

output:

- Composition
- rendered Image

version 1では単純なaffine placementで構いません。

### HARMONIZE / REFINE

最初のreal-model pipelineでは任意です。

目的:

- seam除去
- color/style統一
- missing region inpaint
- coherence改善

COMPOSE内部へ隠さず独立operatorにします。

## 8. Layout表現

layoutをpixelだけで表現しません。

最小model:

    Layout
      canvas_width
      canvas_height
      slots[]

    Slot
      category
      x
      y
      width
      height
      rotation
      constraints

将来的にはpromptからLayoutを生成するmodelを追加できます。

MVPでは次で構いません。

- manual configuration
- fixed template
- simple rule-based placement

## 9. Retrieval score

retrieved Partは複数scoreを持てます。

    retrieval.semantic
    retrieval.style
    retrieval.color
    retrieval.pose
    quality
    compatibility

これらを永続的に1 scoreへ潰しません。

最終Selector側でweighted aggregateを計算できます。

## 10. Compatibility score

Part単体が良くても現在のCompositionと合わない場合があります。

そのため将来的に

    part_quality(part)

と

    compatibility(part, current_state)

を分離して扱います。

例:

- face angleとbody pose
- lighting direction
- line thickness
- style
- color palette
- perspective
- scale

これは重要な将来学習targetです。

## 11. Deduplication

大量datasetではnear-duplicateが一般的です。

Part Bankは次をsupportする方向です。

- exact checksum dedup
- perceptual duplicate grouping
- embedding-neighbor inspection

MVPではexact checksum dedupで十分です。

本格的なtrain/test評価前にはnear-duplicate groupingを追加してdata leakageを防ぎます。

## 12. Dataset split rule

near-duplicateや同一sequence由来のsource imageがsplitを跨がないようにします。

推奨順:

1. related imageをgroup化
2. group単位でsplit割当
3. Part抽出

関連source image由来のPartを個別にrandom splitしません。

## 13. Provenance

すべてのPartから次を追跡できるようにします。

- source image
- extraction method/version
- embedding model/version
- preprocessing configuration

debugとexperiment reproducibilityに必須です。

## 14. 初期Python interface

    class PartRepository:
        def get(self, part_id): ...
        def query(self, query, category=None, top_k=20, filters=None): ...
        def add(self, part): ...

    class PartExtractor:
        def extract(self, source_image): ...

    class PartEmbedder:
        def embed(self, part): ...

    class PartIndex:
        def add(self, part_id, vector): ...
        def search(self, vector, top_k, filters=None): ...

Workflow RuntimeをFAISS等の具体vector databaseへ直接coupleしません。

## 15. Package構成案

    image_drawer/
      part_bank/
        models.py
        repository.py
        extractor.py
        embedding.py
        index.py
        ingest.py
        quality.py

      steps/
        retrieve_parts.py
        select_parts.py
        compose.py
        refine.py

## 16. MVP受け入れ条件

Part Bank v0は次を満たせば完了とします。

1. source image folderをingestできる。
2. 各画像にstable SourceImage recordが作られる。
3. 最低1種類のextractorがPartを作る。
4. source provenance付きでPartを保存する。
5. embeddingを生成できる。
6. text/category queryからranked PartSetを返せる。
7. GUI/RuntimeからRETRIEVE_PARTSを実行できる。
8. selected Part IDをrun Trajectoryへ記録する。
9. 同じdataset/index versionで再実行した際にexperiment用途として十分な再現性がある。

v0では高品質なPart抽出は必須ではありません。

主目的はinterfaceの安定化です。
