"""設計書8.4節の実測: 合成画像でtrain -> evaluateを実行し、判定をそのまま保存する。

  python scripts/run_training_experiment.py --output reports/training/manual [--iterations 200]

数十分かかる。結果(learned/not_learned/inconclusive)は閾値を変えずに報告する。
合成した単純な画像セットでの測定であり、実画像の品質は主張しない。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # このcheckoutのimage_drawerを使う

PALETTE = [(220, 40, 40), (40, 90, 220), (40, 170, 70), (230, 190, 30)]


# {
#   責務: [ make_dataset: 白地に1色の長方形を置いた32x32の合成PNGを登録する(画像のみ) ]
# }
def make_dataset(root: Path, count: int) -> Path:
    from PIL import Image

    from image_drawer.dataset import create_default_registry, register_images

    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = Image.new("RGB", (32, 32), (255, 255, 255))
        left = 2 + (index * 5) % 14
        for x in range(left, left + 12):
            for y in range(8, 24):
                image.putpixel((x, y), PALETTE[index % len(PALETTE)])
        image.save(source / f"img_{index:03d}.png")
    dataset = root / "dataset"
    register_images(
        source, dataset, create_default_registry(),
        tagger_id="color-layout-tags", captioner_id="color-layout-caption",
    )  # fmt: skip
    return dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="出力先(Git管理外)")
    parser.add_argument("--images", type=int, default=150)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    from image_drawer.training.__main__ import main as training_main

    root = Path(args.output)
    dataset = make_dataset(root, args.images)
    config = root / "config.json"
    config.write_text(
        json.dumps({"canvas_side": 32, "summary_side": 8, "max_actions": 96,
                    "text_encoder_id": "hashed-ngram-v1", "curriculum": {"enabled": True}}),
        encoding="utf-8",
    )  # fmt: skip
    experiment = root / "experiment"
    code = training_main(["train", str(dataset), "--config", str(config), "--seed", str(args.seed),
                          "--iterations", str(args.iterations), "--output-dir", str(experiment)])  # fmt: skip
    if code != 0:
        return code
    return training_main(["evaluate", str(experiment), "--dataset", str(dataset)])


if __name__ == "__main__":
    sys.exit(main())
