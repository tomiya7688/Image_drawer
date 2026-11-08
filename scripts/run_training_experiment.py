"""設計書8.4節の実測: 合成画像でtrain -> evaluateを実行し、判定をそのまま保存する。

  python scripts/run_training_experiment.py --output reports/training/manual [--iterations 200]

数十分かかる。結果(learned/not_learned/inconclusive)は閾値を変えずに報告する。
合成した単純な画像セットでの測定であり、実画像の品質は主張しない。
"""

from __future__ import annotations

import argparse
import json
import random
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
        # 位置・大きさが連続的に変わる長方形に、画像ごとに異なる小ブロックを重ねる。
        # 近重複として数群にまとまると、test splitが数枚になり判定が必ずinconclusiveになる
        rng = random.Random(index)
        image = Image.new("RGB", (32, 32), (255, 255, 255))
        width, height = rng.randrange(8, 20), rng.randrange(8, 20)
        left, top = rng.randrange(0, 32 - width), rng.randrange(0, 32 - height)
        for x in range(left, left + width):
            for y in range(top, top + height):
                image.putpixel((x, y), PALETTE[index % len(PALETTE)])
        for _ in range(8):
            x0, y0 = rng.randrange(0, 28), rng.randrange(0, 28)
            block = tuple(rng.randrange(256) for _ in range(3))
            for x in range(x0, x0 + 4):
                for y in range(y0, y0 + 4):
                    image.putpixel((x, y), block)
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
    parser.add_argument("--text-encoder", default="hashed-ngram-v1")
    parser.add_argument(
        "--reward-spec", help="calibrate-rewardが書いたReward仕様JSON(省略時は既定)"
    )
    args = parser.parse_args(argv)
    from image_drawer.training.__main__ import main as training_main

    root = Path(args.output)
    dataset = make_dataset(root, args.images)
    config = root / "config.json"
    config.write_text(
        json.dumps({"canvas_side": 32, "summary_side": 8, "max_actions": 96,
                    "text_encoder_id": args.text_encoder, "curriculum": {"enabled": True}}),
        encoding="utf-8",
    )  # fmt: skip
    experiment = root / "experiment"
    spec_args = ["--reward-spec", args.reward_spec] if args.reward_spec else []
    code = training_main(["train", str(dataset), "--config", str(config), "--seed", str(args.seed),
                          "--iterations", str(args.iterations), "--output-dir", str(experiment), *spec_args])  # fmt: skip
    if code != 0:
        return code
    return training_main(["evaluate", str(experiment), "--dataset", str(dataset), *spec_args])


if __name__ == "__main__":
    sys.exit(main())
