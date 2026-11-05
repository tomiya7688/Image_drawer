"""全Datasetへのタグ→Caption→Promptの3パス順と再開・隔離を検証する。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from image_drawer.dataset import DatasetManifest, ExtractorInfo, TrainingImage, iter_samples
from image_drawer.prompting import ImagePromptPipeline, ImageTrainingPreprocessor, VisionResponse
from image_drawer.prompting.dataset_pipeline import preprocess_dataset
from image_drawer.prompting.tagging import TaggingPipeline


@dataclass
class FakeTags:
    events: list[str]
    provider_id: str = "tag-provider"
    model_id: str = "tags-v1"

    def tag(self, image: bytes, mime_type: str) -> object:
        self.events.append("tag")
        return {"general": {"apple": 0.95, "red_apple": 0.94}}


@dataclass
class FakeCaption:
    events: list[str]
    provider_id: str = "caption-provider"
    model_id: str = "caption-v1"

    def describe(self, image: bytes, mime_type: str, instruction: str) -> VisionResponse:
        self.events.append("caption")
        return VisionResponse(
            json.dumps(
                {
                    "summary": "赤いリンゴ",
                    "short_prompt": "赤いリンゴ",
                    "standard_prompt": "机の上に赤いリンゴ",
                    "detailed_prompt": "白い背景の机の上に赤いリンゴ",
                    "subjects": [],
                },
                ensure_ascii=False,
            )
        )


def make_dataset(tmp_path):
    info = ExtractorInfo(
        id="legacy", version="1", capabilities=["caption", "tag"]
    )
    images = []
    for index in range(2):
        image = Image.new("RGB", (16, 16), (230, index * 30, 30))
        buf = BytesIO()
        image.save(buf, format="PNG")
        content = buf.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        relative = "images/" + digest + ".png"
        dest = tmp_path / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        images.append(
            TrainingImage(
                image_id="image_" + str(index),
                image_sha256=digest,
                group_id=str(index),
                split="train" if index == 0 else "test",
                image_path=relative,
                width=16,
                height=16,
                normalization_version="test",
                tags=[],
                caption="",
                training_prompt="",
                tagger=info,
                captioner=info,
                prompt_builder_version="legacy",
                cache_key="",
            )
        )
    DatasetManifest(
        normalization_version="test",
        prompt_builder_version="legacy",
        tagger=info,
        captioner=info,
        split_seed=0,
        images=images,
    ).save(tmp_path)


def test_dataset_runs_whole_tag_pass_before_caption_and_composes(tmp_path):
    make_dataset(tmp_path)
    events: list[str] = []
    pipeline = ImageTrainingPreprocessor(
        ImagePromptPipeline(providers=[FakeCaption(events)]),
        TaggingPipeline([FakeTags(events)]),
    )
    report = preprocess_dataset(tmp_path, pipeline)
    assert report.images == 2
    assert report.tags_ok == report.captions_ok == report.prompts_ok == 2
    assert report.failures == ()
    assert events == ["tag", "tag", "caption", "caption"]
    manifest = DatasetManifest.load(tmp_path)
    assert all(img.training_prompt for img in manifest.images)
    assert all(img.tags for img in manifest.images)
    assert manifest.prompt_builder_version == "prompt-composer-v2"
    samples = list(iter_samples(tmp_path, "train"))
    assert len(samples) == 1
    policy, target = samples[0]
    assert policy.prompt
    assert not hasattr(policy, "image_path")
    assert target.image_path.endswith(".png")
    # 再実行で成功した抽出結果を使い、VLMを再ロードしない。
    report_again = preprocess_dataset(tmp_path, pipeline)
    assert report_again.prompts_ok == 2
    assert events == ["tag", "tag", "caption", "caption"]


def test_one_stage_fails_but_caption_stage_runs_and_no_old_labels_leak(tmp_path):
    make_dataset(tmp_path)
    events: list[str] = []

    @dataclass
    class FailingTags(FakeTags):
        def tag(self, image: bytes, mime_type: str) -> object:
            self.events.append("tag")
            if image == (tmp_path / DatasetManifest.load(tmp_path).images[0].image_path).read_bytes():
                return {"tags": []}
            return super().tag(image, mime_type)

    pipeline = ImageTrainingPreprocessor(
        ImagePromptPipeline(providers=[FakeCaption(events)]),
        TaggingPipeline([FailingTags(events)]),
    )
    report = preprocess_dataset(tmp_path, pipeline)
    assert report.images == 2
    assert report.tags_ok == 1
    assert report.captions_ok == 2
    assert report.prompts_ok == 1
    assert any(stage == "tags" for _, stage, _ in report.failures)
    manifest = DatasetManifest.load(tmp_path)
    failed = [img for img in manifest.images if not img.training_prompt]
    assert len(failed) == 1
    assert failed[0].tags == [] and failed[0].caption == ""
