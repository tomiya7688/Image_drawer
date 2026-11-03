"""画像のみのDataset登録、抽出器の切替、target隔離、権利と分割の規則を確認する。"""

import dataclasses
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from image_drawer.dataset import (
    DatasetManifest,
    ExtractorInfo,
    PolicyInput,
    RewardTarget,
    create_default_registry,
    iter_samples,
    register_images,
)
from image_drawer.dataset.cli import main as cli_main
from image_drawer.dataset.ingest import assign_split, compute_cache_key


# {
#   責務: [
#     solid: 単色の画像ファイルを作る
#   ]
#   引数: [
#     path: 保存先, color: RGB, size: 画素数
#   ]
# }
def solid(path: Path, color, size=(32, 24)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


# {
#   責務: [
#     source_dir: 色の違う3枚と、完全一致のコピー1枚を含む元画像のディレクトリを作る
#   ]
#   戻り値: [
#     directory: 元画像のディレクトリ
#   ]
# }
@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "source"
    directory.mkdir()
    solid(directory / "red.png", (220, 40, 40))
    solid(directory / "blue.png", (50, 80, 220), size=(24, 32))
    solid(directory / "green.png", (50, 170, 70))
    shutil.copy(directory / "red.png", directory / "red_copy.png")
    return directory


def register(source_dir, tmp_path, **options):
    registry = options.pop("registry", create_default_registry())
    options.setdefault("tagger_id", "color-layout-tags")
    options.setdefault("captioner_id", "color-layout-caption")
    return registry, register_images(source_dir, tmp_path / "dataset", registry, **options)


def test_register_normalizes_dedupes_and_writes_a_verifiable_manifest(source_dir, tmp_path):
    _, report = register(source_dir, tmp_path)
    assert (report.added, report.duplicates, report.errors) == (3, 1, [])
    manifest = DatasetManifest.load(tmp_path / "dataset")
    assert len(manifest.images) == 3
    red = next(i for i in manifest.images if "赤" in i.caption)
    assert red.training_prompt.startswith("赤が中心")
    assert (red.width, red.height) == (32, 24)
    assert (tmp_path / "dataset" / red.image_path).is_file()
    assert red.tagger.id == "color-layout-tags" and red.captioner.id == "color-layout-caption"


def test_source_images_are_not_modified(source_dir, tmp_path):
    before = {p.name: p.read_bytes() for p in source_dir.iterdir()}
    register(source_dir, tmp_path)
    assert {p.name: p.read_bytes() for p in source_dir.iterdir()} == before


def test_registering_again_is_idempotent(source_dir, tmp_path):
    register(source_dir, tmp_path)
    first = DatasetManifest.load(tmp_path / "dataset").dataset_hash
    _, report = register(source_dir, tmp_path)
    assert (report.added, report.reextracted) == (0, 0)
    assert DatasetManifest.load(tmp_path / "dataset").dataset_hash == first


class FixedTagger:
    info = ExtractorInfo(id="fixed-tags", version="v1", capabilities=["tag"], model_hash="m1")

    def __init__(self):
        self.calls = 0

    def extract(self, image):
        self.calls += 1
        return [{"label": "固定", "confidence": 0.9}]


def test_switching_the_tag_extractor_reextracts_and_changes_the_dataset(source_dir, tmp_path):
    registry, _ = register(source_dir, tmp_path)
    before = DatasetManifest.load(tmp_path / "dataset")
    tagger = FixedTagger()
    registry.register_tagger(tagger)
    _, report = register(source_dir, tmp_path, registry=registry, tagger_id="fixed-tags")
    after = DatasetManifest.load(tmp_path / "dataset")
    assert (report.added, report.reextracted) == (0, 3)
    assert tagger.calls == 3
    assert after.dataset_hash != before.dataset_hash
    assert all(i.tags == [{"label": "固定", "confidence": 0.9}] for i in after.images)
    assert all(i.captioner.id == "color-layout-caption" for i in after.images)
    # 同じ設定で再登録すると、キャッシュを使い抽出器を呼ばない。
    register(source_dir, tmp_path, registry=registry, tagger_id="fixed-tags")
    assert tagger.calls == 3


def test_cache_key_changes_with_every_extractor_field():
    base = ExtractorInfo(id="a", version="v1", capabilities=["tag"])
    caption = ExtractorInfo(id="c", version="v1", capabilities=["caption"])
    keys = {compute_cache_key("sha", base, caption)}
    for change in ({"id": "b"}, {"version": "v2"}, {"model_hash": "x"}, {"config_hash": "y"}):
        keys.add(compute_cache_key("sha", dataclasses.replace(base, **change), caption))
    keys.add(compute_cache_key("sha2", base, caption))
    assert len(keys) == 6


def test_extractor_failure_becomes_a_warning_without_dropping_the_image(source_dir, tmp_path):
    class Broken:
        info = ExtractorInfo(id="broken", version="v1", capabilities=["caption"])

        def extract(self, image):
            raise RuntimeError("model unavailable")

    registry = create_default_registry()
    registry.register_captioner(Broken())
    _, report = register(source_dir, tmp_path, registry=registry, captioner_id="broken")
    manifest = DatasetManifest.load(tmp_path / "dataset")
    assert report.added == 3 and report.quality_warnings == 3
    codes = {issue.code for issue in manifest.images[0].quality_issues}
    assert {"extractor_failed", "empty_caption"} <= codes
    assert manifest.images[0].training_prompt != ""  # タグだけでpromptを作る


def test_bad_files_are_reported_and_good_files_are_kept(source_dir, tmp_path):
    (source_dir / "broken.png").write_bytes(b"not an image")
    frames = [Image.new("RGB", (8, 8), c) for c in ((255, 0, 0), (0, 0, 255))]
    frames[0].save(source_dir / "anim.png", save_all=True, append_images=frames[1:], duration=50)
    _, report = register(source_dir, tmp_path)
    assert report.added == 3
    assert {Path(e.path).name for e in report.errors} == {"broken.png", "anim.png"}


def test_alpha_is_composited_on_white_and_exif_orientation_is_applied(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(source / "clear.png")
    photo = Image.new("RGB", (20, 10), (200, 30, 30))
    exif = photo.getexif()
    exif[274] = 6  # 90度回転
    photo.save(source / "rotated.jpg", exif=exif)
    _, report = register(source, tmp_path)
    manifest = DatasetManifest.load(tmp_path / "dataset")
    sizes = {i.caption: (i.width, i.height) for i in manifest.images}
    assert report.errors == [] and (10, 20) in sizes.values()
    clear = next(i for i in manifest.images if i.width == 10 and i.height == 10)
    with Image.open(tmp_path / "dataset" / clear.image_path) as stored:
        assert stored.getpixel((0, 0)) == (255, 255, 255)


def test_near_duplicates_share_a_group_and_never_cross_splits(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    base = Image.linear_gradient("L").resize((64, 64)).convert("RGB")
    base.save(source / "a.png")
    Image.eval(base, lambda v: min(255, v + 2)).save(source / "a_bright.png")
    Image.new("RGB", (64, 64), (0, 255, 0)).save(source / "other.png")
    register(source, tmp_path)
    images = DatasetManifest.load(tmp_path / "dataset").images
    assert len(images) == 3
    groups: dict[str, set[str]] = {}
    for image in images:
        groups.setdefault(image.group_id, set()).add(image.split)
    # 派生した2枚は同じグループになり、緑の画像は別グループ。グループ内のsplitは1つだけ。
    assert sorted(len(g) for g in groups.values()) == [1, 1]
    assert len({i.group_id for i in images if i.width == 64}) == 2
    sizes = {}
    for image in images:
        sizes.setdefault(image.group_id, []).append(image.image_path)
    assert sorted(len(v) for v in sizes.values()) == [1, 2]


def test_flat_color_images_with_the_same_structure_are_not_grouped(source_dir, tmp_path):
    # 単色画像は構造のハッシュが全て同じになる。色の署名で区別できなければ無関係な画像が同じグループになる。
    register(source_dir, tmp_path)
    images = DatasetManifest.load(tmp_path / "dataset").images
    assert len({i.group_id for i in images}) == 3


def test_split_assignment_is_deterministic_and_seed_dependent():
    assert assign_split("group-x", 0) == assign_split("group-x", 0)
    values = {assign_split(f"group-{n}", 0) for n in range(200)}
    assert values == {"train", "validation", "test"}


def test_policy_input_cannot_carry_the_target_image(source_dir, tmp_path):
    register(source_dir, tmp_path)
    samples = (
        list(iter_samples(tmp_path / "dataset", "train"))
        + list(iter_samples(tmp_path / "dataset", "validation"))
        + list(iter_samples(tmp_path / "dataset", "test"))
    )
    assert len(samples) == 3
    for policy_input, target in samples:
        assert isinstance(policy_input, PolicyInput) and isinstance(target, RewardTarget)
        assert {f.name for f in dataclasses.fields(PolicyInput)} == {"image_id", "prompt"}
        assert target.image_path not in json.dumps(dataclasses.asdict(policy_input))
        assert Path(target.image_path).is_file()


def test_manifest_tampering_is_detected(source_dir, tmp_path):
    register(source_dir, tmp_path)
    path = tmp_path / "dataset" / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["images"][0]["training_prompt"] = "改ざん"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        DatasetManifest.load(tmp_path / "dataset")


def test_invalid_options_are_rejected(source_dir, tmp_path):
    registry = create_default_registry()
    with pytest.raises(KeyError, match="registered"):
        register_images(
            source_dir,
            tmp_path / "d",
            registry,
            tagger_id="nope",
            captioner_id="color-layout-caption",
        )
    with pytest.raises(ValueError, match="capability"):
        registry.register_tagger(
            type("T", (), {"info": ExtractorInfo(id="x", version="v", capabilities=["caption"])})()
        )


def test_cli_registers_lists_and_shows(source_dir, tmp_path, capsys):
    dataset = str(tmp_path / "dataset")
    assert (
        cli_main(
            [
                "register",
                str(source_dir),
                "--dataset",
                dataset,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["added"] == 3
    assert cli_main(["extractors"]) == 0
    assert "color-layout-tags" in capsys.readouterr().out
    assert cli_main(["show", "--dataset", dataset]) == 0
    assert json.loads(capsys.readouterr().out)["images"] == 3
    assert cli_main(["register", str(source_dir), "--dataset", dataset, "--tagger", "nope"]) == 2


@pytest.mark.parametrize(
    "content",
    [
        "{broken",
        "[]",
        '{"tags": "x", "caption": 1, "issues": []}',
        '{"tags": [{"label": 1}], "caption": "", "issues": []}',
    ],
)
def test_corrupted_cache_is_ignored_and_reextracted(source_dir, tmp_path, content):
    register(source_dir, tmp_path)
    for cache in (tmp_path / "dataset" / "cache").glob("*.json"):
        cache.write_text(content, encoding="utf-8")
    registry = create_default_registry()
    _, report = register(source_dir, tmp_path, registry=registry)
    manifest = DatasetManifest.load(tmp_path / "dataset")
    assert report.errors == [] and all(i.caption for i in manifest.images)
    assert all(
        json.loads(c.read_text(encoding="utf-8"))["caption"]
        for c in (tmp_path / "dataset" / "cache").glob("*.json")
    )


def test_near_duplicate_grouping_is_transitive():
    from image_drawer.dataset.normalize import ImageSignature, group_near_duplicates

    # A-BとB-Cは近く、A-Cは離れている。連結は推移的なので3件とも同じグループになる。
    a = ImageSignature(0, (0,) * 48)
    b = ImageSignature(0b1111, (10,) * 48)
    c = ImageSignature(0b11111111, (20,) * 48)
    groups = group_near_duplicates({"a": a, "b": b, "c": c})
    assert len(set(groups.values())) == 1


def test_invalid_extractor_output_becomes_a_warning_without_dropping_the_image(
    source_dir, tmp_path
):
    class BadTags:
        info = ExtractorInfo(id="bad-tags", version="v1", capabilities=["tag"])

        def extract(self, image):
            return [{"label": 1}]

    registry = create_default_registry()
    registry.register_tagger(BadTags())
    _, report = register(source_dir, tmp_path, registry=registry, tagger_id="bad-tags")
    manifest = DatasetManifest.load(tmp_path / "dataset")
    assert report.added == 3 and len(manifest.images) == 3
    assert all(
        "extractor_invalid_output" in {q.code for q in image.quality_issues}
        for image in manifest.images
    )
    assert all(image.tags == [] and image.training_prompt for image in manifest.images)


def test_tag_extractor_can_be_disabled_and_the_prompt_comes_from_the_caption(source_dir, tmp_path):
    register(source_dir, tmp_path, tagger_id="none")
    manifest = DatasetManifest.load(tmp_path / "dataset")
    assert all(image.tags == [] for image in manifest.images)
    assert all(image.training_prompt == image.caption for image in manifest.images)
    assert all(
        "empty_tags" not in {q.code for q in image.quality_issues} for image in manifest.images
    )


def test_examples_without_a_text_condition_are_skipped_not_requested(source_dir, tmp_path):
    class Silent:
        info = ExtractorInfo(id="silent", version="v1", capabilities=["caption"])

        def extract(self, image):
            return ""

    registry = create_default_registry()
    registry.register_captioner(Silent())
    register(source_dir, tmp_path, registry=registry, tagger_id="none", captioner_id="silent")
    manifest = DatasetManifest.load(tmp_path / "dataset")
    assert len(manifest.images) == 3
    assert all(
        "empty_prompt" in {q.code for q in image.quality_issues} for image in manifest.images
    )
    samples = [
        s
        for split in ("train", "validation", "test")
        for s in iter_samples(tmp_path / "dataset", split)
    ]
    assert samples == []
