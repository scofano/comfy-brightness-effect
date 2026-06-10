import argparse
from pathlib import Path

from brightness_flicker_core import GlobalFlickerSettings, process_global_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a lamp-like flicker to bright areas of a video."
    )
    parser.add_argument("input", type=Path, help="Input video path")
    parser.add_argument("output", type=Path, help="Output video path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=185.0,
        help="Brightness threshold (0-255) where flicker starts to apply",
    )
    parser.add_argument(
        "--softness",
        type=float,
        default=35.0,
        help="Soft transition range above the threshold (0-255)",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=0.28,
        help="Base flicker strength. Typical values: 0.1 to 0.5",
    )
    parser.add_argument(
        "--glow",
        type=float,
        default=0.22,
        help="Extra bloom mixed into bright regions during brighter flicker peaks",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible flicker",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = GlobalFlickerSettings(
        threshold=args.threshold,
        softness=args.softness,
        amount=args.amount,
        glow=args.glow,
        seed=args.seed,
    )
    process_global_video(args.input, args.output, settings)


if __name__ == "__main__":
    main()
