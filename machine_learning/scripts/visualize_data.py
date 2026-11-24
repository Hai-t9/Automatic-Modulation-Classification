"""Create exploratory plots for the RadioML 2016.10a dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "figures",
        help="Directory for generated PNG figures.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    modulations = sorted({str(modulation) for modulation, _ in dataset})
    snrs = sorted({int(snr) for _, snr in dataset})
    counts = {
        (str(modulation), int(snr)): int(samples.shape[0])
        for (modulation, snr), samples in dataset.items()
    }

    # Total examples for each modulation class.
    class_counts = [sum(counts.get((modulation, snr), 0) for snr in snrs) for modulation in modulations]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(modulations, class_counts, color="steelblue")
    ax.set(title="Examples per modulation", xlabel="Modulation", ylabel="Number of examples")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    class_path = args.output_dir / "class_distribution.png"
    fig.savefig(class_path, dpi=150)
    plt.close(fig)

    # Number of examples available for each modulation/SNR combination.
    count_matrix = np.array(
        [[counts.get((modulation, snr), 0) for snr in snrs] for modulation in modulations]
    )
    fig, ax = plt.subplots(figsize=(11, max(4, len(modulations) * 0.45)))
    image = ax.imshow(count_matrix, aspect="auto", cmap="viridis")
    ax.set(
        title="Examples by modulation and SNR",
        xlabel="SNR (dB)",
        ylabel="Modulation",
        xticks=np.arange(len(snrs)),
        xticklabels=snrs,
        yticks=np.arange(len(modulations)),
        yticklabels=modulations,
    )
    fig.colorbar(image, ax=ax, label="Number of examples")
    fig.tight_layout()
    snr_path = args.output_dir / "snr_distribution.png"
    fig.savefig(snr_path, dpi=150)
    plt.close(fig)

    # Plot I and Q for one example. Dictionary keys identify modulation and SNR.
    example_key = min(dataset, key=lambda key: (str(key[0]), int(key[1])))
    example = dataset[example_key][0]
    timesteps = np.arange(example.shape[-1])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(timesteps, example[0], label="I (in-phase)", linewidth=1)
    ax.plot(timesteps, example[1], label="Q (quadrature)", linewidth=1)
    ax.set(
        title=f"Example IQ signal: {example_key[0]}, SNR {example_key[1]} dB",
        xlabel="Time step",
        ylabel="Amplitude",
    )
    ax.legend()
    fig.tight_layout()
    signal_path = args.output_dir / "example_iq_signal.png"
    fig.savefig(signal_path, dpi=150)
    plt.close(fig)

    print("Generated figures:")
    for path in (class_path, snr_path, signal_path):
        print(f"- {path.resolve()}")


if __name__ == "__main__":
    main()
