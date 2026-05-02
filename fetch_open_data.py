from __future__ import annotations

import argparse
from pathlib import Path

from power_forecasting import DEFAULT_DATA_PATH, UCI_DATA_URL, download_household_power_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the UCI Individual Household Electric Power Consumption open dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Where household_power_consumption.txt will be written.",
    )
    parser.add_argument(
        "--url",
        default=UCI_DATA_URL,
        help="Dataset ZIP URL. Defaults to the official UCI archive URL.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when the output file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        print(f"Dataset already exists: {args.output}")
        return

    download_household_power_data(args.output, args.url)


if __name__ == "__main__":
    main()
