import argparse
from pathlib import Path

from brightness_flicker_core import IndependentFlickerSettings, process_independent_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply independent lamp-like flicker to separate bright areas in a video."
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
        default=0.3,
        help="Base flicker strength. Typical values: 0.1 to 0.5",
    )
    parser.add_argument(
        "--glow",
        type=float,
        default=0.22,
        help="Extra bloom mixed into bright regions during brighter flicker peaks",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum bright blob area in pixels to track as a light source",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=60.0,
        help="Maximum centroid movement in pixels when matching lights between frames",
    )
    parser.add_argument(
        "--max-missed",
        type=int,
        default=8,
        help="How many frames a light may disappear before its track is removed",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for reproducible flicker",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = IndependentFlickerSettings(
        threshold=args.threshold,
        softness=args.softness,
        amount=args.amount,
        glow=args.glow,
        min_area=args.min_area,
        max_distance=args.max_distance,
        max_missed=args.max_missed,
        seed=args.seed,
    )
    process_independent_video(args.input, args.output, settings)


if __name__ == "__main__":
    main()
