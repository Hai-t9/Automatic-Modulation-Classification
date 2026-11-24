"""Loading and basic validation for the RadioML 2016.10a dataset."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "data" / "RML2016.10a_dict.pkl"


def load_dataset(data_path: str | Path = DEFAULT_DATA_PATH) -> dict[tuple[Any, Any], np.ndarray]:
    """Load and validate a RadioML dictionary pickle.

    Pickle files can execute code while being loaded. Only load a dataset from a
    source you trust. ``encoding="latin1"`` supports the original Python 2 pickle.
    """
    path = Path(data_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset pickle not found: {path}")

    with path.open("rb") as file:
        dataset = pickle.load(file, encoding="latin1")

    if not isinstance(dataset, dict):
        raise TypeError("Expected the pickle to contain a dictionary.")

    if not dataset:
        raise ValueError("The dataset dictionary is empty.")

    for key, samples in dataset.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(
                "Expected each dataset key to be a (modulation, SNR) tuple; "
                f"got {key!r}."
            )
        if not isinstance(samples, np.ndarray):
            raise TypeError(f"Dataset entry {key!r} is not a NumPy array.")
        if samples.ndim != 3 or samples.shape[1] != 2:
            raise ValueError(
                f"Dataset entry {key!r} should have shape "
                f"(samples, 2, timesteps); got {samples.shape}."
            )

    return dataset
