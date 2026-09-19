# Part Bank Specification

Status: adopted design
Date: 2026-09-20

## 1. Purpose

Part Bank is the reusable visual-asset layer of Image Drawer.

The project assumes access to a large image dataset. Instead of treating every source image only as a final training target, the system extracts, indexes, retrieves and evaluates reusable visual parts and intermediate structures.

The first implementation goal is not perfect semantic decomposition. It is a stable data model and pipeline that allows later extraction methods to improve without changing the workflow/runtime interface.

## 2. Core idea

A source image can yield zero or more Parts.

Examples:

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
- composition/layout template
- silhouette
- shading pattern

The exact taxonomy is dataset-dependent and must remain extensible.

## 3. Data model

### SourceImage

Required fields:

    SourceImage
      id: string
      uri: string
      width: int
      height: int
      checksum: string
      dataset: string
      metadata: map

Optional metadata:

- tags
- caption
- artist/source information when legally available
- style labels
- quality flags
- split: train / validation / test
- license/provenance information

### Part

Required fields:

    Part
      id: string
      source_image_id: string
      category: string
      crop_uri: string
      bbox: [x, y, width, height]
      metadata: map

Recommended fields:

    Part
      mask_uri: string | null
      embedding_refs: map
      tags: list[string]
      attributes: map
      quality: map
      extraction_method: string
      extraction_version: string

`embedding_refs` allows several embedding models to coexist.

Example:

    embedding_refs:
      clip_v1: embeddings/clip/abc.npy
      custom_part_v2: embeddings/custom/abc.npy

Do not hard-code one embedding model into the Part schema.

### PartSet

A query result is represented as:

    PartSet
      query_id
      category
      part_ids[]
      retrieval_scores[]
      metadata

### PartPlacement

Represents a selected part on a canvas:

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

This should be an intermediate Artifact, not only a rendered image.

## 4. Storage layout

Initial local implementation:

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

Metadata should use JSONL or SQLite for the MVP.

Recommended MVP choice:

- image/crop/mask files on filesystem
- SQLite for SourceImage / Part / embedding metadata
- vector index abstraction behind a repository interface

This keeps the first implementation simple while allowing FAISS, Qdrant, Milvus or another backend later.

## 5. Extraction pipeline

Recommended pipeline:

    INGEST
      -> VALIDATE
      -> EXTRACT_PARTS
      -> EXTRACT_METADATA
      -> EMBED
      -> INDEX
      -> QUALITY_FILTER

Each stage must be restartable.

### INGEST

Responsibilities:

- assign source image id
- checksum
- dimensions
- provenance metadata
- dataset split

### EXTRACT_PARTS

The extraction algorithm is replaceable.

Possible implementations:

- bounding-box detector
- segmentation model
- semantic segmentation
- pose-based region extraction
- manually supplied masks/annotations
- dataset-specific parser

Output is always normalized to `Part`.

### EMBED

Generate one or more searchable representations.

Initial suggested embeddings:

- global visual semantic embedding
- optional category-specific embedding
- optional color/style embedding

The repository API must identify which embedding model produced a vector.

### QUALITY_FILTER

Remove or flag unusable parts:

- extremely small regions
- invalid/empty masks
- corrupt crops
- low-information regions
- obvious duplicates if desired

Filtering should be recorded, not destructive by default.

## 6. Retrieval API

Core conceptual interface:

    retrieve_parts(
        query,
        category=None,
        top_k=20,
        filters=None,
        embedding_model=None
    ) -> PartSet

The query may contain:

- text
- current image/canvas
- reference part
- workflow state
- attributes

Version 1 only needs text and category filtering.

## 7. Workflow operations

### RETRIEVE_PARTS

Inputs:

- prompt: Text
- optional current image/state

Parameters:

- category
- top
- embedding_model
- filters

Outputs:

- PartSet

Example:

    faces = RETRIEVE_PARTS(
      prompt,
      category="face",
      top=20
    )

### SELECT_PARTS

Inputs:

- PartSet
- optional canvas/context
- optional evaluator scores

Parameters:

- top
- strategy

MVP strategies:

- retrieval_score
- evaluator_score
- weighted_score

Outputs:

- selected PartSet or Part

### COMPOSE

Inputs:

- parts
- layout/placements

Outputs:

- Composition
- rendered Image

Version 1 may use simple affine placement.

### HARMONIZE / REFINE

Optional in the first real-model pipeline.

Purpose:

- remove seams
- unify color/style
- inpaint missing regions
- improve coherence

This should be implemented as a separate operator, not hidden inside COMPOSE.

## 8. Layout representation

Do not encode layout only as pixels.

Minimal layout model:

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

A future model may generate Layout from a prompt.

For MVP, layout can be:

- manually configured
- fixed templates
- simple rule-based placement

## 9. Retrieval scoring

A retrieved part can have several scores:

    retrieval.semantic
    retrieval.style
    retrieval.color
    retrieval.pose
    quality
    compatibility

Do not collapse these permanently into one score.

The final selector may compute a weighted aggregate.

## 10. Compatibility scoring

A part may be individually good but incompatible with the current composition.

Therefore selection should eventually distinguish:

    part_quality(part)

from:

    compatibility(part, current_state)

Examples:

- face angle vs body pose
- lighting direction
- line thickness
- style
- color palette
- perspective
- scale

This is an important future learning target.

## 11. Deduplication

Large image datasets commonly contain near-duplicates.

The Part Bank should support:

- exact checksum dedup
- perceptual duplicate grouping
- embedding-neighbor inspection

For MVP, exact checksum dedup is sufficient.

Near-duplicate grouping can be added before serious train/test evaluation to avoid leakage.

## 12. Dataset split rules

Avoid leakage between source images that are near-duplicates or part of the same sequence.

Preferred split order:

1. group related images
2. assign group to split
3. extract parts

Do not randomly split extracted Parts independently if their source images are related.

## 13. Provenance

Every Part must remain traceable to:

- source image
- extraction method/version
- embedding model/version
- preprocessing configuration

This is required for debugging and experiment reproducibility.

## 14. Initial Python interfaces

Suggested interfaces:

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

Avoid coupling the workflow runtime directly to FAISS or another concrete vector database.

## 15. Suggested package structure

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

## 16. MVP acceptance criteria

Part Bank v0 is complete when:

1. a folder of source images can be ingested;
2. each image receives a stable SourceImage record;
3. at least one extractor creates Parts;
4. Parts are stored with source provenance;
5. embeddings can be generated;
6. text/category query returns a ranked PartSet;
7. GUI/runtime can execute RETRIEVE_PARTS;
8. selected Part IDs are written into the run Trajectory;
9. rerunning with the same dataset/index version is reproducible enough for experiments.

High extraction quality is not required for v0.

The main goal is a stable interface.
