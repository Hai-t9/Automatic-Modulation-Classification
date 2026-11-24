"""Print and save a structural summary of the RadioML dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_loader import DEFAULT_DATA_PATH, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the trusted RadioML pickle file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "reports" / "dataset_summary.json",
        help="Where to save the JSON summary.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    modulations = sorted({str(modulation) for modulation, _ in dataset})
    snrs = sorted({int(snr) for _, snr in dataset})
    entries = []
    total_samples = 0
    shapes = set()
    dtypes = set()

    for (modulation, snr), samples in sorted(
        dataset.items(), key=lambda item: (str(item[0][0]), int(item[0][1]))
    ):
        count = int(samples.shape[0])
        total_samples += count
        shapes.add(tuple(int(dimension) for dimension in samples.shape[1:]))
        dtypes.add(str(samples.dtype))
        entries.append(
            {
                "modulation": str(modulation),
                "snr_db": int(snr),
                "sample_count": count,
                "sample_shape": [int(dimension) for dimension in samples.shape[1:]],
                "dtype": str(samples.dtype),
            }
        )

    summary = {
        "dataset": "RadioML 2016.10a",
        "source_file": str(args.data.expanduser().resolve()),
        "number_of_modulations": len(modulations),
        "modulations": modulations,
        "snr_values_db": snrs,
        "number_of_modulation_snr_pairs": len(entries),
        "total_samples": total_samples,
        "sample_shapes": [list(shape) for shape in sorted(shapes)],
        "dtypes": sorted(dtypes),
        "entries": entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Dataset: {summary['dataset']}")
    print(f"Modulations ({len(modulations)}): {', '.join(modulations)}")
    print(f"SNR values (dB): {', '.join(map(str, snrs))}")
    print(f"Modulation/SNR groups: {len(entries)}")
    print(f"Total examples: {total_samples:,}")
    print(f"Sample shape (channels, timesteps): {summary['sample_shapes']}")
    print(f"Data type(s): {', '.join(sorted(dtypes))}")
    print(f"Detailed summary written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
