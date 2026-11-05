"""テキスト条件化の独立スモークとフォールバックを確認するCLI。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .text_condition import TextEncodingError, load_text_encoder_from_settings


# {
#   責務: [main: 設定済みTextEncoderでpromptを数値に変換する]
#   処理: [同一モデルのGPU→CPU試行を記録し、特徴空間を表示する]
#   引数: [argv: CLI引数]
#   戻り値: [int: 正常0、失敗1]
# }
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TextEncoderのモデル選択とGPU→CPUを検証")
    parser.add_argument("--settings", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint-space", default=None)
    parser.add_argument("--checkpoint-dim", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        encoder = load_text_encoder_from_settings(args.settings)
        if args.checkpoint_space is not None:
            if args.checkpoint_dim is None:
                parser.error("--checkpoint-spaceには--checkpoint-dimも必要")
            encoder.validate_checkpoint_space(args.checkpoint_space, args.checkpoint_dim)
        result = encoder.encode(args.prompt)
    except (OSError, ValueError, TextEncodingError) as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "feature_space_id": result.feature_space_id,
                "encoder_id": result.encoder_id,
                "model_revision": result.model_revision,
                "dimension": len(result.values),
                "attempts": [asdict(item) for item in result.attempts],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
