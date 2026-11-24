# RadioML 2016.10a machine-learning analysis

This folder is the starting point for exploring and preparing the RadioML 2016.10a automatic modulation classification dataset. The dataset currently lives at `../data/RML2016.10a_dict.pkl`.

## Dataset overview

RadioML 2016.10a is a collection of synthetic radio signal examples. Its pickle is expected to contain a dictionary keyed by `(modulation, SNR)` pairs. Each value is a NumPy array shaped like `(examples, 2, timesteps)`, where the two channels are I (in-phase) and Q (quadrature). The provided scripts inspect the actual file and report its modulation labels, SNR values, sample counts, shapes, and data types rather than assuming those details.

The source data includes a license at `../data/LICENSE.TXT`. Review and follow its terms before redistributing the dataset or derived materials.

> **Security:** Python pickle files can execute code during loading. Only run these scripts with a pickle from a source you trust.

## Setup

From this directory, create/activate a Python environment and install the analysis dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Python 3.10 or newer is recommended. The pickle is large (about 612 MB), so allow sufficient memory and disk space. Loading it requires memory for the decoded arrays in addition to the file on disk.

## Explore the dataset

From `machine_learning/`:

```bash
python scripts/describe_data.py
python scripts/visualize_data.py
```

The default data location is the repository's `data/RML2016.10a_dict.pkl`. To use a different trusted pickle:

```bash
python scripts/describe_data.py --data /path/to/dataset.pkl
python scripts/visualize_data.py --data /path/to/dataset.pkl
```

The description script prints an overview and writes `outputs/reports/dataset_summary.json`. The visualization script saves these figures under `outputs/figures/`:

- `class_distribution.png` — total examples per modulation.
- `snr_distribution.png` — examples per modulation/SNR combination.
- `example_iq_signal.png` — I and Q traces from one example.

Both scripts accept `--help` for output path options.

## Suggested workflow

1. Load and validate the pickle with `scripts/data_loader.py`.
2. Review the generated summary and plots for class balance, SNR coverage, shapes, and data quality.
3. Define a train/validation/test split and preprocessing strategy that matches the evaluation goal; keep related examples from leaking across splits where applicable.
4. Train and evaluate a classifier, then record metrics and model details here or in a dedicated report.

No classifier or train/test split is imposed yet, so the analysis can guide those choices first.
